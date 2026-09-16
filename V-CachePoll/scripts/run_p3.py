#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.p3 import stage_p3_all  # noqa: E402
from vcachepoll.report import stage_p3_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="V-CachePoll P3 same-protocol baselines")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "p3.yaml"))
    p.add_argument("--stage", type=str, default="all", choices=["all", "report"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--methods",
        type=str,
        default="random,task,caa,rank",
        help="comma-separated: random,task,caa,rank",
    )
    args = p.parse_args()
    cfg = load_cfg(Path(args.config))
    if args.stage == "report":
        print(stage_p3_report(cfg), flush=True)
        return
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(stage_p3_all(cfg, limit=args.limit, methods=methods), flush=True)
    print(stage_p3_report(cfg), flush=True)


if __name__ == "__main__":
    main()
