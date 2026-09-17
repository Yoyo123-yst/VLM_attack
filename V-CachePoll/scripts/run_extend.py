#!/usr/bin/env python3
"""P4–P11 entry. GPU stages no-op if MSC/other jobs hold the card."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.extend import (  # noqa: E402
    disk_ok,
    gpu_free_for_vcache,
    gpu_holder,
    next_gpu_commands,
    p10_inventory,
    p11_inventory,
    p12_decision,
    run_gpu_phase,
    stage_p5_pairs,
    stage_p6_transfer_spec,
    stage_p8_from_crit,
    stage_p9_table,
    write_status,
)
from vcachepoll.io import save_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="V-CachePoll P4–P11 integration")
    p.add_argument(
        "--phase",
        default="inventory",
        choices=["inventory", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12"],
    )
    p.add_argument("--stage", default="smoke", type=str)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-aux", dest="n_aux", type=int, default=1)
    args = p.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if args.phase == "inventory":
        blob = {
            "gpu_free": gpu_free_for_vcache(),
            "gpu_holder": gpu_holder(),
            "disk": disk_ok(),
            "p10": p10_inventory(),
            "p11": p11_inventory(),
            "p12": p12_decision(),
            "p6": stage_p6_transfer_spec(load_cfg(ROOT / "configs" / "p6.yaml")),
            "next_gpu": next_gpu_commands(),
        }
        save_json(ROOT / "out" / "extend" / "inventory.json", blob)
        write_status("inventory", json.dumps(blob, indent=2, default=str))
        print(json.dumps(blob, indent=2, default=str), flush=True)
        return

    if args.phase == "p8":
        rec = stage_p8_from_crit()
        write_status("p8", json.dumps(rec, indent=2, default=str))
        print(rec, flush=True)
        return

    if args.phase == "p9" and args.stage in {"table", "report"}:
        rec = stage_p9_table()
        write_status("p9-table", json.dumps({k: rec[k] for k in rec if k != "table"}, indent=2, default=str))
        print(rec["path"], flush=True)
        return

    if args.phase == "p5" and args.stage == "pairs":
        cfg = load_cfg(ROOT / "configs" / "p5.yaml")
        rec = stage_p5_pairs(cfg, n_aux=args.n_aux)
        write_status(f"p5-pairs-m{args.n_aux}", json.dumps(rec, indent=2))
        print(rec, flush=True)
        return

    if args.phase == "p12":
        rec = p12_decision()
        write_status("p12-skip", json.dumps(rec, indent=2))
        print(rec, flush=True)
        return

    rec = run_gpu_phase(args.phase, args.stage, limit=args.limit, n_aux=args.n_aux)
    write_status(f"{args.phase}-{args.stage}", json.dumps(rec, indent=2, default=str))
    print(rec, flush=True)
    if rec.get("skipped"):
        sys.exit(2)


if __name__ == "__main__":
    main()
