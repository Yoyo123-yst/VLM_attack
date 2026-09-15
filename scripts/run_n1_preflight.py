#!/usr/bin/env python3
"""N1 preflight. Does not scan residual unless the provenance gate passes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from n1.preflight import build_provenance, verdict, write_preflight_md  # noqa: E402
from p0.datautil import save_json  # noqa: E402

N1 = ROOT / "outputs" / "n1"


def write_skipped_scan(gate: dict) -> None:
    skip = {
        "status": "skipped_preflight_fail",
        "preflight_pass": False,
        "discover_only": True,
        "reason": gate.get("gate"),
        "layers": [4, 8, 12, 16, 20, 24, 27],
        "ranks": [1, 2, 4, 8, 16],
        "token_positions": ["last_prompt", "assistant_start", "first_generated", "semantic_divergence"],
        "note": "Scan not run. Stored last-prompt hidden cannot separate SAFE/RHC on identical pre-generation inputs.",
    }
    save_json(N1 / "representation_scan.json", skip)
    save_json(N1 / "rank_curve.json", {"status": "skipped_preflight_fail", "ranks": [1, 2, 4, 8, 16]})
    save_json(N1 / "nuisance_controls.json", {"status": "skipped_preflight_fail"})
    save_json(
        N1 / "frozen_candidates.json",
        {
            "candidate_count": 0,
            "candidates": [],
            "development_unused_for_selection": True,
            "confirm_unused": True,
            "attack_test_untouched": True,
            "preflight_pass": False,
        },
    )
    (N1 / "N1_LOCALIZATION.md").write_text(
        "\n".join(
            [
                "# N1_LOCALIZATION",
                "",
                "Status: **not run** (`skipped_preflight_fail`).",
                "",
                "N1-A residual scan was not executed. Discover last-prompt hidden states are not a valid "
                "SAFE vs RHC contrast: 24/30 discover pairs share pixels, input_ids, and template, "
                "and differ only by greedy vs sampled decoding. Last-prompt cosine is ≥ 0.999 on those pairs.",
                "",
                "No candidates frozen. Development was not read for selection. Confirm was not used. "
                "Attack-test generations were not read.",
                "",
                "Do not patch. Recollect trajectories before a localization scan.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    N1.mkdir(parents=True, exist_ok=True)
    prov = build_provenance(ROOT)
    gate = verdict(prov)
    save_json(N1 / "pair_provenance.json", {"verdict": gate, "pairs": prov["rows"]})
    md = write_preflight_md(gate, prov)
    (N1 / "N1_PREFLIGHT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if not gate["allow_residual_scan"]:
        write_skipped_scan(gate)
        print("N1_PREFLIGHT_FAIL skip_residual_scan", flush=True)
        raise SystemExit(4)
    print("N1_PREFLIGHT_PASS residual_scan_allowed_not_started_by_this_script", flush=True)


if __name__ == "__main__":
    main()
