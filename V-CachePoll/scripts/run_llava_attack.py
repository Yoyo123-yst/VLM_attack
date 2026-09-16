#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.llava_attack import stage_llava_attack  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()
    cfg = load_cfg(ROOT / "configs" / "p3.yaml")
    print(stage_llava_attack(cfg, limit=args.limit, steps=args.steps), flush=True)


if __name__ == "__main__":
    main()
