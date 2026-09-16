#!/usr/bin/env python3
"""CAGE-B-only: survivor-set disruption on B, same 32 P3 pairs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.p3 import _method_cfg, stage_p1_attack  # noqa: E402


def main() -> None:
    cfg = load_cfg(ROOT / "configs" / "p3.yaml")
    c = _method_cfg(cfg, "cage")
    print(stage_p1_attack(c), flush=True)


if __name__ == "__main__":
    main()
