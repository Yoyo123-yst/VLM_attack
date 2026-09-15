#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.config import DEFAULT_CFG, load_cfg  # noqa: E402
from vcachepoll.probe import stage_pairs, stage_probe, stage_screen, stage_smoke  # noqa: E402
from vcachepoll.report import stage_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="V-CachePoll P0 probe")
    p.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    p.add_argument(
        "--stage",
        type=str,
        default="pairs",
        choices=["pairs", "smoke", "screen", "probe", "report", "p0"],
    )
    args = p.parse_args()
    cfg = load_cfg(Path(args.config))
    stage = args.stage
    order = ["pairs", "smoke", "screen", "probe", "report"] if stage == "p0" else [stage]
    # default stepwise path: pairs → screen → probe → report; smoke is optional
    if stage == "p0":
        order = ["pairs", "screen", "probe", "report"]
    for name in order:
        print(f"== {name} ==", flush=True)
        if name == "pairs":
            print(stage_pairs(cfg), flush=True)
        elif name == "smoke":
            print(stage_smoke(cfg), flush=True)
        elif name == "screen":
            print(stage_screen(cfg), flush=True)
        elif name == "probe":
            print(stage_probe(cfg), flush=True)
        elif name == "report":
            print(stage_report(cfg), flush=True)


if __name__ == "__main__":
    main()
