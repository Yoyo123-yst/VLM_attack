#!/usr/bin/env python3
"""PathEM Stage 2 toy AMS. No VLM. Unique lineages vs naive / BoN / random / refusal."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from pathem.ams_toy import run_toy  # noqa: E402

OUT = ROOT / "artifacts" / "stage2_splitting" / "toy_ams.json"


def main() -> int:
    summary = run_toy(
        n_rep=80,
        seed=20260,
        H=8,
        T=40,
        p_up=0.48,
        p_decoy=0.75,
        n_particles=24,
        n_levels=5,
    )
    summary["stage"] = "2-toy"
    summary["written"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    summary["no_vlm"] = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    pub = {
        "stage": "2-toy",
        "gate": summary["gate"],
        "beats_random": summary["beats_random"],
        "beats_naive": summary["beats_naive"],
        "beats_refusal": summary["beats_refusal"],
        "only_beats_random_not_refusal": summary["only_beats_random_not_refusal"],
        "methods": {
            k: {
                "unique_mean": v["unique_mean"],
                "tokens_mean": v["tokens_mean"],
                "clone_hits_mean": v["clone_hits_mean"],
            }
            for k, v in summary["methods"].items()
        },
    }
    print(json.dumps(pub))
    return 0 if summary["gate"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
