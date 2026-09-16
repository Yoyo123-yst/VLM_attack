#!/usr/bin/env python3
"""Armijo-style: Rank-PGD with exchange search every step (same 32 pairs)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.p3 import _method_cfg, stage_p1_attack  # noqa: E402
from vcachepoll.report import stage_p3_report  # noqa: E402


def main() -> None:
    cfg = load_cfg(ROOT / "configs" / "p3.yaml")
    c = _method_cfg(cfg, "rank")
    atk = c["attack"]
    atk["log_prefix"] = "p3-armijo"
    atk["search_every"] = 1
    atk["result_path"] = str(ROOT / "out" / "p3_armijo.json")
    atk["progress_path"] = str(ROOT / "out" / "p3_armijo_progress.json")
    atk["delta_dir"] = str(ROOT / "out" / "p3_deltas" / "armijo")
    atk["smoke_path"] = str(ROOT / "out" / "p3_armijo_smoke.json")
    print(stage_p1_attack(c), flush=True)


if __name__ == "__main__":
    main()
