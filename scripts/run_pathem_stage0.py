#!/usr/bin/env python3
"""PathEM Stage 0: CPU infrastructure check. No model load, no new PGD."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

import torch

from pathem.budget import BudgetLedger
from pathem.config import load_cfg
from pathem.cr0_io import load_cr0_records, load_delta, public_record
from pathem.privacy import console_safe
from pathem.projection import in_box, project_linf
from pathem.protocol import EVAL_SEEDS, OPT_SEEDS, VAL_SEEDS, assert_seed_split, frozen_eps
from pathem.records import TrajectoryRecord, serialize_record


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    cfg = load_cfg()
    assert_seed_split()
    rows = load_cr0_records(tuple(cfg["query_ids"]), tuple(cfg["carrier_ids"]))
    ledger = BudgetLedger()
    checks = {
        "n_records": len(rows),
        "queries": sorted({r["query_id"] for r in rows}),
        "carriers": sorted({r["carrier_id"] for r in rows}),
        "n_core_rhc": sum(bool(r.get("core_rhc")) for r in rows),
        "n_core_safe": sum(bool(r.get("core_safe_answer")) for r in rows),
        "opt_seeds": list(OPT_SEEDS),
        "val_seeds": list(VAL_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "eps": frozen_eps(),
        "store_raw_outputs": bool(cfg["store_raw_outputs"]),
    }
    # Sample a few deltas without reconstructing full images.
    sample = rows[:8]
    box_ok = True
    publics = []
    for rec in sample:
        delta = load_delta(rec["delta_path"], device="cpu")
        x0 = torch.zeros_like(delta)
        x = project_linf(x0, x0 + delta, frozen_eps())
        ok = in_box(x0, x, frozen_eps())
        box_ok = box_ok and ok
        ledger.add_forward(1)
        pub = public_record(rec)
        publics.append(console_safe({**pub, "status": "ok" if ok else "eps_fail"}))
        rec_obj = TrajectoryRecord(
            run_id="stage0",
            method="infra",
            query_id=rec["query_id"],
            carrier_id=rec["carrier_id"],
            delta_id=rec.get("record_id"),
            random_seed=int(rec.get("seed") or 0),
            trajectory_id=rec.get("record_id") or "",
            terminal_label=str(rec.get("response_mode") or "OTHER"),
            core_rhc=bool(rec.get("core_rhc")),
            response_mode=rec.get("response_mode"),
            safety=rec.get("safety"),
            status="ok" if ok else "eps_fail",
        )
        rec_obj.output_hash = None
        _ = serialize_record(rec_obj)

    out_dir = ROOT / "artifacts" / "stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": 0,
        "written": utc_now(),
        "checks": checks,
        "perturbation_box_ok": box_ok,
        "sample_public_records": publics,
        "budget": ledger.as_dict(),
        "gate": "GO" if box_ok and rows else "STOP",
        "note": "CPU-only. No model load. No empirical attack result.",
    }
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("stage", "gate", "perturbation_box_ok", "budget", "checks")}, ensure_ascii=False))
    return 0 if box_ok and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
