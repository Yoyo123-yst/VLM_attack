#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.attack import stage_p1_attack, stage_p1_smoke  # noqa: E402
from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.p2 import reassign_u_from_crit, stage_p2_crit, stage_p2_screen  # noqa: E402
from vcachepoll.report import stage_p2_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="V-CachePoll P2: tight budget + critical-token eviction")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "p2.yaml"))
    p.add_argument(
        "--stage",
        type=str,
        default="screen",
        choices=["screen", "crit", "reassign", "smoke", "attack", "report", "p2"],
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    cfg = load_cfg(Path(args.config))
    if args.stage == "screen":
        print(stage_p2_screen(cfg), flush=True)
        return
    if args.stage == "crit":
        print(stage_p2_crit(cfg), flush=True)
        return
    if args.stage == "reassign":
        print(reassign_u_from_crit(cfg), flush=True)
        return
    if args.stage == "smoke":
        print(stage_p1_smoke(cfg), flush=True)
        return
    if args.stage == "attack":
        print(stage_p1_attack(cfg, limit=args.limit), flush=True)
        return
    if args.stage == "report":
        print(stage_p2_report(cfg), flush=True)
        return
    screen = stage_p2_screen(cfg)
    print(screen, flush=True)
    if int(screen.get("n_kept") or 0) < 8:
        raise SystemExit(f"P2 screen kept {screen.get('n_kept')}; not enough for attack")
    print(stage_p2_crit(cfg), flush=True)
    print(stage_p1_smoke(cfg), flush=True)
    print(stage_p1_attack(cfg, limit=args.limit), flush=True)
    print(stage_p2_report(cfg), flush=True)


if __name__ == "__main__":
    main()
