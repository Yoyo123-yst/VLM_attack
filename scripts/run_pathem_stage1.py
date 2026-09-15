#!/usr/bin/env python3
"""PathEM Stage 1 collect + cheap committor fit.

Grid: 5 committor queries × 2 carriers × 2 CR-0 restarts × 4 opt seeds.
Does not scan the 96-cell CR-0 grid. Does not read h83–h130.
Does not print raw generations. Resume-safe.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
os.environ.setdefault("P0_QWEN_KEEP_336", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathem.budget import BudgetLedger  # noqa: E402
from pathem.collect import collect_grid_spec, collect_one, select_stage1_cells  # noqa: E402
from pathem.committor import compare_margin_vs_full  # noqa: E402
from pathem.prefixes import assert_group_split  # noqa: E402
from pathem.protocol import EVAL_SEEDS, STORE_RAW_OUTPUTS, assert_seed_split  # noqa: E402
from pathem.splits import COMMITTOR_TRAIN, COMMITTOR_VAL  # noqa: E402
from p0.datautil import load_json, save_json  # noqa: E402
from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import load_model, seed_all, vram_preflight  # noqa: E402

OUT = ROOT / "artifacts" / "stage1_committor"
TRAJ = OUT / "trajectories.json"
PREFIX = OUT / "prefixes.json"
REPORT = OUT / "collect_report.json"
FIT = OUT / "committor_fit.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_partial() -> tuple[list[dict], list[dict], set[str]]:
    recs, prefs = [], []
    if TRAJ.exists():
        recs = list((load_json(TRAJ) or {}).get("records") or [])
    if PREFIX.exists():
        prefs = list((load_json(PREFIX) or {}).get("prefixes") or [])
    done = {f"{r.get('delta_id')}:{r.get('random_seed')}" for r in recs}
    return recs, prefs, done


def leak_check(rows: list[dict]) -> None:
    blob = json.dumps(rows)
    for rec in rows:
        if "text" in rec or rec.get("output_hash") is None:
            raise RuntimeError("raw text leaked or hash missing")
        if rec.get("query_id") in {"h83", "h91", "h130"}:
            raise RuntimeError("sealed query leaked")
    if "step-by-step" in blob.lower():
        raise RuntimeError("possible raw generation leaked into artifacts")


def main() -> int:
    if STORE_RAW_OUTPUTS:
        raise RuntimeError("store_raw_outputs must be false")
    assert_seed_split()
    spec = collect_grid_spec()
    if set(spec["seeds"]) & set(EVAL_SEEDS):
        raise RuntimeError("eval seeds in collect")
    cells = select_stage1_cells(n_restart=2)
    OUT.mkdir(parents=True, exist_ok=True)

    recs, prefs, done = load_partial()
    planned = [(c, int(s)) for c in cells for s in spec["seeds"]]
    remaining = [(c, s) for c, s in planned if f"{c.get('record_id')}:{s}" not in done]
    print(
        {
            "n_cells": len(cells),
            "n_planned": len(planned),
            "n_done": len(done),
            "n_remaining": len(remaining),
            "seeds": spec["seeds"],
        },
        flush=True,
    )

    ledger = BudgetLedger()
    if remaining:
        cfg = load_cfg(scale="full")
        pre = vram_preflight(10.0)
        print({"vram_preflight": pre}, flush=True)
        if not pre.get("ok"):
            save_json(OUT / "collect_error.json", {"vram": pre, "written": utc_now()})
            return 2
        seed_all(2026)
        wrapper = load_model(cfg, setting="native")
        wrapper.set_setting("native")
        for i, (cell, seed) in enumerate(remaining, 1):
            blob = collect_one(wrapper, cell, int(seed), do_sample=True, ledger=ledger)
            recs.append(blob["full"])
            prefs.extend(blob["prefixes"])
            leak_check([blob["full"]])
            save_json(TRAJ, {"records": recs})
            save_json(PREFIX, {"prefixes": prefs})
            print({"i": i, "n_remaining": len(remaining), "record": blob["record"]}, flush=True)

    assert_group_split(prefs, COMMITTOR_TRAIN, COMMITTOR_VAL)
    leak_check(recs)
    n_rhc = sum(bool(r.get("core_rhc")) for r in recs)
    collect_gate = "GO" if len(recs) >= spec["n_traj_planned"] and n_rhc >= 8 else "STOP"
    report = {
        "stage": "1-collect",
        "written": utc_now(),
        "spec": spec,
        "n_traj": len(recs),
        "n_prefix": len(prefs),
        "n_core_rhc": n_rhc,
        "labels": {
            lab: sum(r.get("terminal_label") == lab for r in recs)
            for lab in sorted({r.get("terminal_label") for r in recs})
        },
        "by_query": {
            q: {
                "n": sum(r.get("query_id") == q for r in recs),
                "n_rhc": sum(r.get("query_id") == q and r.get("core_rhc") for r in recs),
            }
            for q in spec["query_ids"]
        },
        "budget": ledger.as_dict(),
        "store_raw_outputs": False,
        "gate": collect_gate,
        "note": "Collect only. Fit is a separate artifact. No PathEM ASR claim.",
    }
    save_json(TRAJ, {"records": recs})
    save_json(PREFIX, {"prefixes": prefs})
    save_json(REPORT, report)
    print(
        json.dumps(
            {k: report[k] for k in ("stage", "gate", "n_traj", "n_core_rhc", "labels")},
            ensure_ascii=False,
        ),
        flush=True,
    )

    if not prefs:
        return 1 if collect_gate != "GO" else 0
    fit = compare_margin_vs_full(prefs)
    fit["written"] = utc_now()
    fit["n_prefix"] = len(prefs)
    fit["n_traj"] = len(recs)
    fit["collect_gate"] = collect_gate
    save_json(FIT, fit)
    print(
        json.dumps(
            {
                "stage": "1-fit",
                "gate": fit.get("gate"),
                "full_val": fit.get("full", {}).get("val"),
                "margin_val": fit.get("refusal_margin", {}).get("val"),
                "full_not_better_than_margin": fit.get("full_not_better_than_margin"),
                "advisory_070": fit.get("full", {}).get("advisory_auroc_070"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if collect_gate == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())

