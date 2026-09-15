#!/usr/bin/env python3
"""PathEM Stage 2-Diagnostic. CPU. Logs only. No Stage 3. No eval seeds."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from pathem.cr0_io import load_cr0_records, public_record  # noqa: E402
from pathem.protocol import EVAL_SEEDS, SEALED, STORE_RAW_OUTPUTS, assert_seed_split  # noqa: E402
from pathem.stage2_diag import run_diagnostic  # noqa: E402
from p0.datautil import load_json, save_json  # noqa: E402

S1 = ROOT / "artifacts" / "stage1_committor"
S2 = ROOT / "artifacts" / "stage2_splitting"
OUT_JSON = S2 / "STAGE2_DIAGNOSTIC.json"
OUT_MD = S2 / "STAGE2_DIAGNOSTIC.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    if STORE_RAW_OUTPUTS:
        raise RuntimeError("store_raw_outputs must be false")
    assert_seed_split()
    traj = list((load_json(S1 / "trajectories.json") or {}).get("records") or [])
    prefs = list((load_json(S1 / "prefixes.json") or {}).get("prefixes") or [])
    cells = list((load_json(S2 / "vlm_cells.json") or {}).get("runs") or [])
    report = load_json(S2 / "vlm_report.json") or {}
    collect = load_json(S1 / "collect_report.json") or {}
    toy = load_json(S2 / "toy_ams.json") or {}
    if not traj or not prefs or not cells:
        raise RuntimeError("missing Stage 1/2 artifacts")
    qids = {str(r.get("query_id")) for r in traj + prefs + cells}
    if qids & set(SEALED):
        raise RuntimeError("sealed query in diagnostic inputs")
    if set(EVAL_SEEDS) & set(report.get("seeds") or []):
        raise RuntimeError("eval seeds in vlm report")
    cr0 = [public_record(r) for r in load_cr0_records()]
    blob = run_diagnostic(
        trajectories=traj,
        prefixes=prefs,
        vlm_cells_runs=cells,
        vlm_report=report,
        cr0_records=cr0,
        collect_report=collect,
        toy_gate=str(toy.get("gate") or "STOP"),
    )
    blob["written"] = utc_now()
    md = blob.pop("markdown")
    save_json(OUT_JSON, blob)
    OUT_MD.write_text(md)
    v = blob["verdict"]
    print(
        {
            "decision": v["decision"],
            "stage2b": v["stage2b"],
            "d1": v["gates"]["d1_rare_cells"],
            "d2": v["gates"]["d2_incremental"],
            "d3": v["gates"]["d3_early_resample_not_ranking"],
            "causes": v["causes"],
        },
        flush=True,
    )
    return 0 if v["decision"] == "STAGE2B" else 2


if __name__ == "__main__":
    raise SystemExit(main())
