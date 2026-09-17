#!/usr/bin/env python3
"""P1 recovery runner. Does not start unless P0-1 has a verdict file.

Reuses P0 banks/budget/hard cells. New methods only. Resume by (cell, method, seed).
"""

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

from msc.controller import RECOVERY_METHODS
from msc.data import build_hard_cells
from msc.p01 import run_cell_method
from msc.phrases import tokenize_banks
from msc.protocol import load_frozen, sha256_file
from msc.reporting import append_jsonl, append_run_log, now, read_jsonl, write_status


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--out", default=str(MS / "out" / "p1"))
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _done_keys(rows: list[dict]) -> set[tuple]:
    done = set()
    for r in rows:
        if r.get("round") == "summary":
            done.add((str(r.get("cell_id")), str(r.get("method")), int(r.get("opt_seed"))))
    return done


def main() -> int:
    args = parse_args()
    p01_verdict = MS / "out" / "p01" / "p01_verdict.json"
    if not args.dry_run and not p01_verdict.exists():
        print("refusing P1: P0-1 verdict missing", flush=True)
        return 2
    parent = load_frozen()
    p1 = json.loads((MS / "MSC_P1_FROZEN.json").read_text(encoding="utf-8"))
    if not p1.get("frozen"):
        print("P1 freeze missing", flush=True)
        return 2
    sha = sha256_file(MS / "MSC_P1_FROZEN.json")
    methods = list(args.methods)
    for m in methods:
        if m not in RECOVERY_METHODS:
            print(f"refusing unknown recovery method {m}", flush=True)
            return 2
    want = list(args.cells or p1["fail_cells"])
    cells = [c for c in build_hard_cells(parent) if f"{c['query_id']}:{c['carrier_id']}" in set(want)]
    if len(cells) != len(want):
        print(f"cell filter mismatch have={len(cells)} want={want}", flush=True)
        return 2
    seeds = list(args.seeds or p1["seeds"]["opt"])
    jobs = [(c, m, s) for m in methods for s in seeds for c in cells]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "p1_results.jsonl"
    log = out / "RUN_LOG.md"
    print(
        json.dumps(
            {"run": "p1", "n_jobs": len(jobs), "methods": methods, "cells": want, "frozen_sha": sha},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for c, m, s in jobs:
            print(f"{c['query_id']}:{c['carrier_id']} {m} seed={s}", flush=True)
        return 0
    from run_p0_qwen import load_model

    existing = read_jsonl(jsonl) if args.resume else []
    if not args.resume and jsonl.exists():
        jsonl.unlink()
        existing = []
    done = _done_keys(existing) if args.resume else set()
    write_status(
        MS / "out" / "STATUS.md",
        stage="P1",
        state="RUNNING",
        frozen_sha=sha,
        git_commit="local",
        started=now(),
        completed=len(done),
        total=len(done) + sum(1 for c, m, s in jobs if (f"{c['query_id']}:{c['carrier_id']}", m, int(s)) not in done),
        next_cmd="python MSC/scripts/night_watch_p1.py",
    )
    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="mini")
    wrapper = load_model(cfg, setting="native")
    banks = tokenize_banks(wrapper.processor.tokenizer, parent)
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
                frozen=parent,
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
                stage="P1",
                state="RUNNING",
                frozen_sha=sha,
                git_commit="local",
                started=now(),
                completed=len(done),
                total=max(len(done), len(jobs)),
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
            stage="P1",
            state="INTERRUPTED",
            frozen_sha=sha,
            git_commit="local",
            started=now(),
            completed=len(done),
            total=len(jobs),
            failed=1,
        )
        raise
    print(json.dumps({"p1_batch_done": True, "n_done": len(done)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
