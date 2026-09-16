#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.llava_p0 import stage_llava_probe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "p3.yaml"))
    p.add_argument("--limit", type=int, default=8)
    args = p.parse_args()
    cfg = load_cfg(Path(args.config))
    print(stage_llava_probe(cfg, limit=args.limit), flush=True)


if __name__ == "__main__":
    main()
