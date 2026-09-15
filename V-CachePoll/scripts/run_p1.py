#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.attack import stage_p1_attack, stage_p1_smoke  # noqa: E402
from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.report import stage_p1_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="V-CachePoll P1 quota+eviction attack")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "p1.yaml"))
    p.add_argument("--stage", type=str, default="smoke", choices=["smoke", "attack", "report", "p1"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    cfg = load_cfg(Path(args.config))
    if args.stage == "smoke":
        print(stage_p1_smoke(cfg), flush=True)
        return
    if args.stage == "attack":
        print(stage_p1_attack(cfg, limit=args.limit), flush=True)
        return
    if args.stage == "report":
        print(stage_p1_report(cfg), flush=True)
        return
    smoke = stage_p1_smoke(cfg)
    print(smoke, flush=True)
    rows = smoke.get("n") or 0
    if rows < 1:
        raise SystemExit("P1 smoke produced no rows; not starting full attack")
    print(stage_p1_attack(cfg, limit=args.limit), flush=True)
    print(stage_p1_report(cfg), flush=True)


if __name__ == "__main__":
    main()
