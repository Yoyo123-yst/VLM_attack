#!/usr/bin/env python3
"""P0-B runner. Measures energy on P0-A failure states, then a 24-step intervention.

Does not insert QP into the attacker. Refuses sealed queries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CG = HERE.parent
ROOT = CG.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(CG / "src"))

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
os.environ.setdefault("P0_QWEN_KEEP_336", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from p0_qwen.config import load_cfg
from run_p0_qwen import load_model, seed_all
from traceflip.datasets import build_cells
from traceflip.protocol import OPT_SEEDS, assert_seed_split

from cgvsf.p0b import collect_unique_states, intervene_24, measure_state
from cgvsf.protocol import load_frozen, sha256_file
from cgvsf.reporting import append_jsonl, append_run_log, atomic_write_text, now, read_jsonl, write_status
from cgvsf.verdict import p0a_verdict, p0b_verdict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CG / "configs" / "p0b.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def extra_certs_for(rows, cell_id: str, method: str, rnd) -> list[list[int]]:
    out = []
    seen = set()
    for r in rows:
        if r.get("cell_id") != cell_id or r.get("method") != method:
            continue
        if r.get("round") == "summary":
            continue
        try:
            if int(r.get("round") or 0) > int(rnd):
                continue
        except (TypeError, ValueError):
            continue
        ids = r.get("certificate_token_ids")
        h = r.get("certificate_hash")
        if not ids or not h or h in seen:
            continue
        seen.add(h)
        out.append(list(ids))
    return out


def main() -> int:
    args = parse_args()
    assert_seed_split()
    frozen = load_frozen()
    sha = sha256_file(CG / "CGVSF_P0_FROZEN.json")
    jsonl_a = CG / "out" / "p0a" / "p0a_results.jsonl"
    out = CG / "out" / "p0b"
    out.mkdir(parents=True, exist_ok=True)
    jsonl_b = out / "p0b_results.jsonl"
    log = out / "RUN_LOG.md"
    state_dir = CG / "out" / "p0a" / "states"

    if not jsonl_a.exists():
        atomic_write_text(out / "P0B_REPORT.md", "# P0-B\n\nBlocked: P0-A JSONL is missing.\n")
        print("blocked: missing P0-A", flush=True)
        return 2

    rows_a = read_jsonl(jsonl_a)
    gate_a = p0a_verdict(
        [r for r in rows_a if r.get("round") != "summary"],
        hard_cells=list(frozen["scope"]["hard_cells"]),
        frozen_gates=frozen["gates"],
    )
    states = collect_unique_states(rows_a)
    states = [s for s in states if (state_dir / f"{s['state_key']}.pt").exists()]
    if args.limit:
        states = states[: int(args.limit)]

    if not states:
        atomic_write_text(
            out / "P0B_REPORT.md",
            "# P0-B\n\nBlocked: no saved failure states with token_ids + delta. Finish P0-A first.\n",
        )
        print("blocked: no states", flush=True)
        return 2

    existing = read_jsonl(jsonl_b) if args.resume else []
    if not args.resume and jsonl_b.exists():
        jsonl_b.unlink()
        existing = []
    done = {str(r.get("state_key")) for r in existing if r.get("state_key")}
    write_status(
        CG / "out" / "STATUS.md",
        stage="P0-B",
        state="RUNNING",
        frozen_sha=sha,
        git_commit="local",
        started=now(),
        completed=len(done),
        total=len(states),
        next_cmd="python CG-VSF/scripts/render_report.py --stage p0b",
        evidence=f"P0-A {gate_a['verdict']}; QP not in attacker",
    )
    seed_all(OPT_SEEDS[0])
    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="mini")
    wrapper = load_model(cfg, setting="native")
    want = {(s["query_id"], s["carrier_id"]) for s in states}
    cells = {
        (c["query_id"], c["carrier_id"]): c
        for c in build_cells(tuple(dict.fromkeys(q for q, _ in want)), tuple(dict.fromkeys(c for _, c in want)))
        if (c["query_id"], c["carrier_id"]) in want
    }
    n_fail = 0
    try:
        for st in states:
            if st["state_key"] in done:
                continue
            qid = st["query_id"]
            if qid.startswith("h") and 83 <= int(qid[1:]) <= 130:
                raise RuntimeError(f"sealed query {qid}")
            cell = cells[(st["query_id"], st["carrier_id"])]
            blob = torch.load(state_dir / f"{st['state_key']}.pt", map_location="cpu")
            extra = extra_certs_for(rows_a, st["cell_id"], st["method"], st["round"])
            measured = measure_state(
                wrapper,
                cell,
                delta=blob["delta"],
                cert_ids=st["certificate_token_ids"],
                extra_ids=extra,
                frozen=frozen,
            )
            inter = intervene_24(
                wrapper,
                cell,
                measured,
                old_hash=st["certificate_hash"],
                frozen=frozen,
            )
            row = {
                "state_key": st["state_key"],
                "cell_id": st["cell_id"],
                "method": st["method"],
                "round": st["round"],
                "certificate_hash": st["certificate_hash"],
                "certificate_mode": st["certificate_mode"],
                "G": measured["G"],
                "certificate_margin": measured["certificate_margin"],
                "grad_norm": measured["grad_norm"],
                "E_single": measured["E_single"],
                "E_joint": measured["E_joint"],
                "diag_gramian_energy": measured["diag_gramian_energy"],
                "gramian_rank": measured["gramian"].get("rank"),
                "gramian_condition": measured["gramian"].get("condition"),
                "refusal_margin": measured["refusal_margin"],
                "n_certs": measured["n_certs"],
                "eliminated": inter["eliminated"],
                "core_rhc": inter["core_rhc"],
                "after_mode": inter["after_mode"],
                "after_cert_hash": inter["after_cert_hash"],
                "G_after": inter["G_after"],
                "in_model_elim": inter["in_model_elim"],
                "steps_used": inter["steps_used"],
                "delta_linf": inter["delta_linf"],
                "wall_seconds": inter["wall_seconds"],
            }
            for k in ("x0", "base", "delta", "ids_list"):
                measured.pop(k, None)
            append_jsonl(jsonl_b, row)
            done.add(st["state_key"])
            append_run_log(log, "state_done", {"state": st["state_key"], "elim": inter["eliminated"]})
            write_status(
                CG / "out" / "STATUS.md",
                stage="P0-B",
                state="RUNNING",
                frozen_sha=sha,
                git_commit="local",
                started=now(),
                completed=len(done),
                total=len(states),
                latest={
                    "cell_id": st["cell_id"],
                    "method": st["method"],
                    "core_rhc": inter["core_rhc"],
                    "after_mode": inter["after_mode"],
                    "backward_used": inter["steps_used"],
                },
            )
            del measured, inter, blob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception:
        n_fail = 1
        append_run_log(log, "error", {"exception": traceback.format_exc()[-1500:]})
        write_status(
            CG / "out" / "STATUS.md",
            stage="P0-B",
            state="INTERRUPTED",
            frozen_sha=sha,
            git_commit="local",
            started=now(),
            completed=len(done),
            total=len(states),
            failed=n_fail,
        )
        raise

    measured_rows = read_jsonl(jsonl_b)
    gate = p0b_verdict(measured_rows, frozen_gates=frozen["gates"])
    (out / "p0b_verdict.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    lines = [
        "# P0-B report",
        "",
        f"- frozen sha256: `{sha}`",
        f"- P0-A verdict: **{gate_a['verdict']}**",
        f"- states measured: {gate['n_states']}",
        f"- automatic verdict: **{gate['verdict']}**",
        f"- -E_joint AUROC: {gate['auroc'].get('-E_joint')}",
        f"- best simple: {gate['best_simple']} (Δ={gate['delta_vs_simple']})",
        f"- Spearman E vs steps: {gate['spearman_E_vs_steps']}",
        f"- top-quartile lift: {gate['top_quartile_lift']}",
        "",
        gate["note"],
        "",
        "QP/Gramian is **not** in the attacker unless this verdict is GO.",
        "",
    ]
    if gate_a["verdict"] != "GO":
        lines.append("Joint CG-VSF integration is forbidden by P0-A.")
    atomic_write_text(out / "P0B_REPORT.md", "\n".join(lines))
    write_status(
        CG / "out" / "STATUS.md",
        stage="P0-B",
        state="COMPLETE",
        frozen_sha=sha,
        git_commit="local",
        started=now(),
        completed=len(states),
        total=len(states),
        verdict=str(gate["verdict"]),
        evidence=f"n={gate['n_states']} auroc={gate['auroc'].get('-E_joint')}",
        next_cmd="python CG-VSF/scripts/advance_stage.py --require-preregistered-gate",
    )
    print(json.dumps({"verdict": gate["verdict"], "n": gate["n_states"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
