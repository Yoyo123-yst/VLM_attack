#!/usr/bin/env python3
"""P0-A runner. Writes JSONL + Markdown; never overwrites TraceFlip/pilot.json."""

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

from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import load_model, seed_all  # noqa: E402
from traceflip.datasets import build_cells  # noqa: E402
from traceflip.protocol import OPT_SEEDS, EVAL_SEEDS, assert_seed_split  # noqa: E402

from cgvsf.p0a import run_cell_method  # noqa: E402
from cgvsf.protocol import hard_cells, load_frozen, sha256_file  # noqa: E402
from cgvsf.reporting import append_jsonl, append_run_log, now, read_jsonl, write_status  # noqa: E402
from cgvsf.verdict import p0a_verdict  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CG / "configs" / "p0a.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--cells", nargs="*")
    return ap.parse_args()


def _done_keys(rows: list[dict]) -> set[tuple[str, str]]:
    done = set()
    for r in rows:
        if r.get("round") == "summary":
            done.add((str(r.get("cell_id")), str(r.get("method"))))
    return done


def _keep_complete(rows: list[dict]) -> list[dict]:
    done = _done_keys(rows)
    kept = []
    for r in rows:
        key = (str(r.get("cell_id")), str(r.get("method")))
        if key in done:
            kept.append(r)
    return kept


def main() -> int:
    args = parse_args()
    assert_seed_split()
    frozen = load_frozen()
    sha = sha256_file(CG / "CGVSF_P0_FROZEN.json")
    methods = list(args.methods or frozen["methods_p0a"])
    prefer = [
        "targeted_prefix_earlystop",
        "gateflip_fair",
        "refusal_margin",
        "last_certificate",
        "accumulated_certificate",
        "accumulated_exact_only",
    ]
    methods = [m for m in prefer if m in methods] + [m for m in methods if m not in prefer]
    want = [tuple(s.split(":", 1)) for s in (args.cells or frozen["scope"]["hard_cells"])]
    for q, _c in want:
        if q.startswith("h") and 83 <= int(q[1:]) <= 130:
            raise RuntimeError(f"sealed query {q}")
    queries = tuple(dict.fromkeys(q for q, _ in want))
    carriers = tuple(dict.fromkeys(c for _, c in want))
    cells = [
        c
        for c in build_cells(queries, carriers)
        if (c["query_id"], c["carrier_id"]) in set(want)
    ]
    out = CG / "out" / "p0a"
    jsonl = out / "p0a_results.jsonl"
    log = out / "RUN_LOG.md"
    jobs = [(c, m) for m in methods for c in cells]
    print(
        json.dumps(
            {
                "run": "p0a",
                "n_cells": len(cells),
                "n_jobs": len(jobs),
                "methods": methods,
                "frozen_sha": sha,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for c, m in jobs:
            print(f"{c['query_id']}:{c['carrier_id']} {m}", flush=True)
        return 0

    existing = read_jsonl(jsonl) if args.resume else []
    if not args.resume and jsonl.exists():
        jsonl.unlink()
        existing = []
    if args.resume and existing:
        kept = _keep_complete(existing)
        if len(kept) != len(existing):
            jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
            existing = kept
    done = _done_keys(existing) if args.resume else set()
    write_status(
        CG / "out" / "STATUS.md",
        stage="P0-A",
        state="RUNNING",
        frozen_sha=sha,
        git_commit="local",
        started=now(),
        completed=len(done),
        total=len(jobs),
        next_cmd="python CG-VSF/scripts/render_report.py --stage p0a",
    )
    seed_all(OPT_SEEDS[0])
    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="mini")
    wrapper = load_model(cfg, setting="native")
    n_fail = 0
    try:
        for cell, method in jobs:
            key = (f"{cell['query_id']}:{cell['carrier_id']}", method)
            if key in done:
                continue
            append_run_log(
                log,
                "cell_start",
                {"cell": key[0], "method": method, "config_hash": sha},
            )
            rec = run_cell_method(
                wrapper,
                cell,
                method,
                frozen=frozen,
                opt_seed=OPT_SEEDS[0],
                eval_seed=EVAL_SEEDS[0],
            )
            state_dir = out / "states"
            state_dir.mkdir(parents=True, exist_ok=True)
            for row in rec["rounds"]:
                delta = row.pop("delta", None)
                key = row.get("state_key")
                if delta is not None and key:
                    torch.save(
                        {
                            "delta": delta,
                            "certificate_token_ids": row.get("certificate_token_ids"),
                            "certificate_hash": row.get("certificate_hash"),
                            "certificate_mode": row.get("certificate_mode"),
                            "cell_id": rec["cell_id"],
                            "method": method,
                            "round": row.get("round"),
                            "core_rhc": row.get("core_rhc"),
                        },
                        state_dir / f"{key}.pt",
                    )
                append_jsonl(jsonl, row)
            append_jsonl(
                jsonl,
                {
                    "cell_id": rec["cell_id"],
                    "method": method,
                    "round": "summary",
                    "core_rhc": rec["core_rhc"],
                    "robust_core_rhc": rec["robust_core_rhc"],
                    "win_round": rec["win_round"],
                    "clean_mode": rec["clean_mode"],
                    "n_certificates": rec["n_certificates"],
                    "backward_used": rec["budget"]["backward_passes"],
                    "delta_linf": rec["delta_linf"],
                    "wall_seconds": rec["wall_seconds"],
                    "opt_seed": rec["opt_seed"],
                    "eval_seed": rec["eval_seed"],
                },
            )
            done.add(key)
            append_run_log(
                log,
                "cell_done",
                {
                    "cell": key[0],
                    "method": method,
                    "core_rhc": rec["core_rhc"],
                    "backward": rec["budget"]["backward_passes"],
                },
            )
            write_status(
                CG / "out" / "STATUS.md",
                stage="P0-A",
                state="RUNNING",
                frozen_sha=sha,
                git_commit="local",
                started=now(),
                completed=len(done),
                total=len(jobs),
                latest={
                    "cell_id": rec["cell_id"],
                    "method": method,
                    "core_rhc": rec["core_rhc"],
                    "before_mode": rec["clean_mode"],
                    "after_mode": rec["rounds"][-1]["after_mode"] if rec["rounds"] else None,
                    "backward_used": rec["budget"]["backward_passes"],
                },
            )
    except Exception:
        n_fail = 1
        append_run_log(log, "error", {"exception": traceback.format_exc()[-1500:]})
        write_status(
            CG / "out" / "STATUS.md",
            stage="P0-A",
            state="INTERRUPTED",
            frozen_sha=sha,
            git_commit="local",
            started=now(),
            completed=len(done),
            total=len(jobs),
            failed=n_fail,
        )
        raise

    rows = [r for r in read_jsonl(jsonl) if r.get("round") != "summary"]
    gate = p0a_verdict(
        rows,
        hard_cells=list(frozen["scope"]["hard_cells"]),
        frozen_gates=frozen["gates"],
    )
    (out / "p0a_verdict.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_status(
        CG / "out" / "STATUS.md",
        stage="P0-A",
        state="COMPLETE",
        frozen_sha=sha,
        git_commit="local",
        started=now(),
        completed=len(jobs),
        total=len(jobs),
        verdict=str(gate["verdict"]),
        evidence=f"acc {gate['n_accumulated']}/8 vs last {gate['n_last']}/8",
        next_cmd="python CG-VSF/scripts/render_report.py --stage p0a",
    )
    print(json.dumps({"verdict": gate["verdict"], "n_acc": gate["n_accumulated"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
