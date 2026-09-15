#!/usr/bin/env python3
"""P1 conflict diagnosis. Refuses unless P0-C sets p0c_unlock_p1.

P0 GO is not sufficient. Does not launch PGD even when unlocked.
Does not implement MCCA-Lite, patch routing, or extra models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
P0 = ROOT / "outputs" / "mcca" / "p0" / "P0_RESULTS.json"
P0C = ROOT / "outputs" / "mcca" / "p0c" / "P0C_RESULTS.json"
FROZEN = ROOT / "outputs" / "mcca" / "MCCA_FROZEN.json"


def main() -> None:
    if not P0.exists():
        print("P0_RESULTS.json missing. Run scripts/run_mcca_p0.py first.", flush=True)
        raise SystemExit(3)
    blob = json.loads(P0.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    p0_go = bool((blob.get("decision") or {}).get("go_p1"))
    p0c_unlock = False
    p0c_route = None
    if P0C.exists():
        p0c = json.loads(P0C.read_text(encoding="utf-8"))
        p0c_unlock = bool(p0c.get("p0c_unlock_p1"))
        p0c_route = (p0c.get("route") or {}).get("route")
    # P0 GO is not enough. P0-C must set p0c_unlock_p1 (route A or C). Default locked.
    go = bool(p0c_unlock)
    cells = frozen["p1_frozen_not_run"]
    print(
        {
            "go_p1": go,
            "p0_go_p1": p0_go,
            "p0c_unlock_p1": p0c_unlock,
            "p0c_route": p0c_route,
            "reasons": (blob.get("decision") or {}).get("reasons"),
            "pass_cells": cells["pass_cells"],
            "fail_cells": cells["fail_cells"],
            "restarts": cells["restarts"],
            "steps": cells["steps"],
        },
        flush=True,
    )
    if not go:
        print("P1 locked. Need P0-C p0c_unlock_p1=True (route A or C). Do not spend GPU on P1.", flush=True)
        raise SystemExit(2)
    print(
        "P0-C unlock flag is true. This file still does not launch PGD; P1 GPU is a later ticket.",
        flush=True,
    )


if __name__ == "__main__":
    main()
