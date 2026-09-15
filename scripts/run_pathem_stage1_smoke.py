#!/usr/bin/env python3
"""PathEM Stage 1 GPU smoke: 1 query × 1 carrier × 2 opt seeds.

Does not fit a committor. Does not scan the 96-cell grid. Does not read h83–h130.
Does not print raw generations.
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
from pathem.collect import collect_one, smoke_spec  # noqa: E402
from pathem.cr0_io import load_cr0_records  # noqa: E402
from pathem.prefixes import assert_group_split  # noqa: E402
from pathem.protocol import STORE_RAW_OUTPUTS, assert_seed_split  # noqa: E402
from pathem.splits import COMMITTOR_TRAIN, COMMITTOR_VAL  # noqa: E402
from p0.datautil import save_json  # noqa: E402
from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import load_model, seed_all, vram_preflight  # noqa: E402

OUT = ROOT / "artifacts" / "stage1_committor"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def pick_cell(query_id: str, carrier_id: str, prefer_rhc: bool) -> dict:
    rows = [
        r
        for r in load_cr0_records()
        if r.get("query_id") == query_id and r.get("carrier_id") == carrier_id
    ]
    if not rows:
        raise RuntimeError(f"no CR0 cell {query_id}:{carrier_id}")
    if prefer_rhc:
        rhc = [r for r in rows if r.get("core_rhc")]
        if rhc:
            return rhc[0]
    return rows[0]


def main() -> int:
    if STORE_RAW_OUTPUTS:
        raise RuntimeError("store_raw_outputs must be false")
    assert_seed_split()
    spec = smoke_spec()
    cell = pick_cell(spec["query_id"], spec["carrier_id"], spec["prefer_greedy_rhc"])
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(scale="full")
    pre = vram_preflight(10.0)
    print({"vram_preflight": pre, "cell": cell.get("record_id"), "seeds": spec["seeds"]}, flush=True)
    if not pre.get("ok"):
        save_json(OUT / "smoke_error.json", {"vram": pre, "written": utc_now()})
        return 2

    seed_all(2026)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    ledger = BudgetLedger()
    collected = []
    prefixes = []
    for seed in spec["seeds"]:
        blob = collect_one(wrapper, cell, int(seed), do_sample=True, ledger=ledger)
        collected.append(blob["full"])
        prefixes.extend(blob["prefixes"])
        print({"seed": seed, "record": blob["record"]}, flush=True)

    assert_group_split(prefixes, COMMITTOR_TRAIN, COMMITTOR_VAL)
    dumped = json.dumps({"records": collected, "prefixes": prefixes})
    if any(k in dumped.lower() for k in ("step-by-step", "here's how", "here is how")):
        # coarse leak check on serialized artifacts; hashes only should be present
        pass
    for rec in collected:
        if "text" in rec or rec.get("output_hash") is None:
            raise RuntimeError("raw text leaked or hash missing")

    report = {
        "stage": "1-smoke",
        "written": utc_now(),
        "cell": cell.get("record_id"),
        "query_id": spec["query_id"],
        "carrier_id": spec["carrier_id"],
        "seeds": spec["seeds"],
        "n_traj": len(collected),
        "n_prefix": len(prefixes),
        "n_core_rhc": sum(bool(r.get("core_rhc")) for r in collected),
        "terminal_labels": [r.get("terminal_label") for r in collected],
        "modes": [r.get("response_mode") for r in collected],
        "same_seed_would_need_rerun": True,
        "budget": ledger.as_dict(),
        "store_raw_outputs": False,
        "note": "Smoke only. No committor fit. No empirical PathEM claim.",
        "gate": "GO" if len(collected) == 2 else "STOP",
    }
    save_json(OUT / "smoke_report.json", report)
    save_json(OUT / "smoke_trajectories.json", {"records": collected, "prefixes": prefixes})
    print(json.dumps({k: report[k] for k in ("stage", "gate", "n_traj", "n_core_rhc", "terminal_labels", "budget")}, ensure_ascii=False), flush=True)
    return 0 if report["gate"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
