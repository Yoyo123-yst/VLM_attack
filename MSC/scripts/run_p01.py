#!/usr/bin/env python3
"""P0-1 runner. Refuses to start unless P0-0 verdict is GO."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent
ROOT = MS.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "CG-VSF" / "src"))
sys.path.insert(0, str(MS / "src"))

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
os.environ.setdefault("P0_QWEN_KEEP_336", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from p0_qwen.config import load_cfg
from run_p0_qwen import load_model

from msc.data import build_hard_cells
from msc.p01 import run_cell_method
from msc.phrases import tokenize_banks
from msc.protocol import load_frozen, sha256_file
from msc.reporting import append_jsonl, append_run_log, now, read_jsonl, write_status
from msc.verdict import p01_verdict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MS / "configs" / "p01.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--methods", nargs="*")
    return ap.parse_args()


def _done_keys(rows: list[dict]) -> set[tuple]:
    done = set()
    for r in rows:
        if r.get("round") == "summary":
            done.add((str(r.get("cell_id")), str(r.get("method")), int(r.get("opt_seed"))))
    return done


def main() -> int:
    args = parse_args()
    frozen = load_frozen()
    sha = sha256_file(MS / "MSC_P0_FROZEN.json")
    p00 = MS / "out" / "p00" / "p00_verdict.json"
    if not p00.exists():
        print("missing P0-0 verdict", flush=True)
        return 2
    v0 = json.loads(p00.read_text(encoding="utf-8"))
    if v0.get("verdict") != "GO":
        print(f"refusing P0-1: P0-0 is {v0.get('verdict')}", flush=True)
        return 2
    methods = list(args.methods or frozen["methods_p01"])
    cells = build_hard_cells(frozen)
    seeds = list(frozen["seeds"]["opt"])
    jobs = [(c, m, s) for s in seeds for m in methods for c in cells]
    out = MS / "out" / "p01"
    jsonl = out / "p01_results.jsonl"
    log = out / "RUN_LOG.md"
    print(json.dumps({"run": "p01", "n_jobs": len(jobs), "methods": methods, "frozen_sha": sha}), flush=True)
    if args.dry_run:
        for c, m, s in jobs:
            print(f"{c['query_id']}:{c['carrier_id']} {m} seed={s}", flush=True)
        return 0

    existing = read_jsonl(jsonl) if args.resume else []
    if not args.resume and jsonl.exists():
        jsonl.unlink()
        existing = []
    done = _done_keys(existing) if args.resume else set()
    write_status(
        MS / "out" / "STATUS.md",
        stage="P0-1",
        state="RUNNING",
        frozen_sha=sha,
        git_commit="local",
        started=now(),
        completed=len(done),
        total=len(jobs),
        next_cmd="python MSC/scripts/render_report.py --stage p01",
    )
    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="mini")
    wrapper = load_model(cfg, setting="native")
    banks = tokenize_banks(wrapper.processor.tokenizer, frozen)
    try:
        for cell, method, seed in jobs:
            key = (f"{cell['query_id']}:{cell['carrier_id']}", method, int(seed))
            if key in done:
                continue
            append_run_log(log, "cell_start", {"cell": key[0], "method": method, "opt_seed": seed})
            rec = run_cell_method(
                wrapper,
                cell,
                method,
                frozen=frozen,
                banks=banks,
                opt_seed=int(seed),
            )
            for row in rec["rounds"]:
                append_jsonl(jsonl, row)
            append_jsonl(
                jsonl,
                {
                    "cell_id": rec["cell_id"],
                    "method": method,
                    "round": "summary",
                    "core_rhc": rec["core_rhc"],
                    "robust_core_rhc": rec["robust_core_rhc"],
                    "n_eval_rhc": rec["n_eval_rhc"],
                    "n_eval": rec["n_eval"],
                    "win_round": rec["win_round"],
                    "clean_mode": rec["clean_mode"],
                    "after_mode": rec.get("after_mode"),
                    "backward_used": rec["budget"]["backward_passes"],
                    "delta_linf": rec["delta_linf"],
                    "wall_seconds": rec["wall_seconds"],
                    "opt_seed": rec["opt_seed"],
                },
            )
            done.add(key)
            append_run_log(
                log,
                "cell_done",
                {
                    "cell": key[0],
                    "method": method,
                    "opt_seed": seed,
                    "robust": rec["robust_core_rhc"],
                    "backward": rec["budget"]["backward_passes"],
                },
            )
            write_status(
                MS / "out" / "STATUS.md",
                stage="P0-1",
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
                    "after_mode": rec.get("after_mode"),
                    "backward_used": rec["budget"]["backward_passes"],
                },
            )
    except Exception:
        append_run_log(log, "error", {"exception": traceback.format_exc()[-1500:]})
        write_status(
            MS / "out" / "STATUS.md",
            stage="P0-1",
            state="INTERRUPTED",
            frozen_sha=sha,
            git_commit="local",
            started=now(),
            completed=len(done),
            total=len(jobs),
            failed=1,
        )
        raise

    rows = read_jsonl(jsonl)
    gate = p01_verdict(
        rows,
        hard_cells=list(frozen["scope"]["hard_cells"]),
        frozen_gates=frozen["gates"],
        n_opt_seeds=len(seeds),
        majority_k=int(frozen["success"]["majority_k"]),
    )
    (out / "p01_verdict.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_status(
        MS / "out" / "STATUS.md",
        stage="P0-1",
        state="COMPLETE",
        frozen_sha=sha,
        git_commit="local",
        started=now(),
        completed=len(jobs),
        total=len(jobs),
        verdict=str(gate["verdict"]),
        evidence="; ".join(gate["reasons"]) or f"switched {gate['n_switched']}/8",
        next_cmd="python MSC/scripts/advance_stage.py --require-preregistered-gate",
    )
    print(json.dumps({"verdict": gate["verdict"], "n_switched": gate["n_switched"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
