#!/usr/bin/env python3
"""PathEM Stage 2 VLM splitting. Token-budget matched. Resume-safe. No eval seeds."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
from pathem.collect import select_stage1_cells  # noqa: E402
from pathem.privacy import console_safe  # noqa: E402
from pathem.protocol import EVAL_SEEDS, OPT_SEEDS, STORE_RAW_OUTPUTS, assert_seed_split  # noqa: E402
from pathem.scorer import fit_live, scorer_from_fit  # noqa: E402
from pathem.vlm_ams import ams_cell, gate_from_cell_rows, naive_cell, stage2_cells  # noqa: E402
from p0.datautil import load_json, save_json  # noqa: E402
from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import load_model, seed_all, vram_preflight  # noqa: E402

OUT = ROOT / "artifacts" / "stage2_splitting"
PREFIX = ROOT / "artifacts" / "stage1_committor" / "prefixes.json"
LIVE = OUT / "live_scorer.json"
CELLS = OUT / "vlm_cells.json"
REPORT = OUT / "vlm_report.json"

M = 8
N_LEVELS = 3
CHUNK = 16
METHODS = ("ams_full", "naive", "ams_refusal", "ams_random")


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def job_key(delta_id: str, method: str) -> str:
    return f"{delta_id}:{method}"


def main() -> int:
    if STORE_RAW_OUTPUTS:
        raise RuntimeError("store_raw_outputs must be false")
    assert_seed_split()
    OUT.mkdir(parents=True, exist_ok=True)
    prefs = list((load_json(PREFIX) or {}).get("prefixes") or [])
    if not prefs:
        raise RuntimeError("missing Stage 1 prefixes")
    live = fit_live(prefs)
    live["written"] = utc_now()
    save_json(LIVE, live)
    print(
        {
            "live_full_val": live["full"].get("val"),
            "live_margin_val": live["refusal_margin"].get("val"),
            "live_full_gate": live["full"].get("gate"),
        },
        flush=True,
    )
    scorer = scorer_from_fit(live, "full")
    cells = stage2_cells(select_stage1_cells(n_restart=2))
    seeds = list(OPT_SEEDS[4:8])  # 20264–20267, unused in Stage 1 collect
    if set(seeds) & set(EVAL_SEEDS):
        raise RuntimeError("eval seeds")
    blob = load_json(CELLS) if CELLS.exists() else {"runs": []}
    runs = list(blob.get("runs") or [])
    done = {job_key(r["delta_id"], r["method"]) for r in runs}
    planned = [(c, m) for c in cells for m in METHODS]
    remaining = [(c, m) for c, m in planned if job_key(c.get("record_id"), m) not in done]
    print(
        {
            "n_cells": len(cells),
            "cell_ids": [c.get("record_id") for c in cells],
            "n_planned": len(planned),
            "n_done": len(done),
            "n_remaining": len(remaining),
            "seeds": seeds,
            "M": M,
        },
        flush=True,
    )
    ledger = BudgetLedger()
    if remaining:
        cfg = load_cfg(scale="full")
        pre = vram_preflight(10.0)
        print({"vram_preflight": pre}, flush=True)
        if not pre.get("ok"):
            save_json(OUT / "vlm_error.json", {"vram": pre, "written": utc_now()})
            return 2
        seed_all(2026)
        wrapper = load_model(cfg, setting="native")
        wrapper.set_setting("native")
        rng = np.random.default_rng(20264)
        token_by_cell: dict[str, int] = {}
        for r in runs:
            if r.get("method") == "ams_full":
                token_by_cell[r["delta_id"]] = int(r.get("tokens") or 0)
        for i, (cell, method) in enumerate(remaining, 1):
            did = cell.get("record_id")
            if method == "naive":
                budget = token_by_cell.get(did)
                out = naive_cell(
                    wrapper,
                    cell,
                    n_traj=M,
                    seeds=list(OPT_SEEDS),
                    ledger=ledger,
                    token_budget=budget,
                )
            else:
                kind = method.replace("ams_", "")
                out = ams_cell(
                    wrapper,
                    cell,
                    kind=kind,
                    scorer=scorer,
                    n_particles=M,
                    n_levels=N_LEVELS,
                    chunk=CHUNK,
                    seeds=seeds,
                    ledger=ledger,
                    rng=rng,
                )
                if method == "ams_full":
                    token_by_cell[did] = int(out["tokens"])
            out["written"] = utc_now()
            pub = {k: out[k] for k in out if k != "particles"}
            leak = json.dumps(pub)
            if "text" in out or "step-by-step" in leak.lower():
                raise RuntimeError("raw leak")
            runs.append(out)
            save_json(CELLS, {"runs": runs})
            print({"i": i, "n_remaining": len(remaining), "record": console_safe({**pub, "status": "ok"}) | {
                "method": out["method"],
                "n_unique_lineages": out["n_unique_lineages"],
                "tokens": out["tokens"],
            }}, flush=True)

    gate = gate_from_cell_rows(runs)
    report = {
        "stage": "2-vlm",
        "written": utc_now(),
        "M": M,
        "n_levels": N_LEVELS,
        "chunk": CHUNK,
        "cells": [c.get("record_id") for c in cells],
        "seeds": seeds,
        "live_full_val": live["full"].get("val"),
        "gate": gate,
        "runs_public": [
            {
                "method": r["method"],
                "query_id": r["query_id"],
                "delta_id": r["delta_id"],
                "n_unique_lineages": r["n_unique_lineages"],
                "n_hit_particles": r["n_hit_particles"],
                "n_clone_hits": r["n_clone_hits"],
                "tokens": r["tokens"],
                "labels": r.get("labels"),
            }
            for r in runs
        ],
        "budget": ledger.as_dict(),
        "store_raw_outputs": False,
        "note": "Not ASR. Unique lineages vs naive/random/refusal. No eval seeds.",
    }
    save_json(REPORT, report)
    print(json.dumps({"stage": "2-vlm", **gate}, ensure_ascii=False), flush=True)
    return 0 if gate["gate"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
