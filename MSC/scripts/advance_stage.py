#!/usr/bin/env python3
"""Stage advance from pre-registered JSON verdicts. No --force-go."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent
sys.path.insert(0, str(MS / "src"))

from msc.reporting import atomic_write_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-preregistered-gate", action="store_true")
    args = ap.parse_args()
    if not args.require_preregistered_gate:
        print("refusing: pass --require-preregistered-gate", flush=True)
        return 2
    p00 = MS / "out" / "p00" / "p00_verdict.json"
    if not p00.exists():
        print("missing P0-0 verdict", flush=True)
        return 2
    a = json.loads(p00.read_text(encoding="utf-8"))
    p01_path = MS / "out" / "p01" / "p01_verdict.json"
    b = json.loads(p01_path.read_text(encoding="utf-8")) if p01_path.exists() else {"verdict": "PENDING"}
    table = {
        ("GO", "GO"): "full MSC closed-loop switched token-set controller",
        ("GO", "STOP"): "mechanism / negative result only; no new attack claim",
        ("GO", "PENDING"): "run P0-1 before integration",
        ("STOP", "PENDING"): "terminate MSC; phrase banks are not separable; do not run P0-1",
        ("STOP", "STOP"): "terminate MSC; keep the negative result",
        ("STOP", "GO"): "invalid: P0-1 must not run after P0-0 STOP",
    }
    plan = table.get((a["verdict"], b["verdict"]), "see MSC_EXPERIMENT_ROADMAP.md §6")
    body = (
        "# Integration decision\n\n"
        f"- P0-0: **{a['verdict']}**\n"
        f"- P0-1: **{b['verdict']}**\n"
        f"- Allowed: {plan}\n"
        "- Sealed queries remain sealed.\n"
        "- HSSC P0-2 is not run.\n"
        "- CG-VSF certificate cutting is not revived.\n"
        "- Last-prompt probes are not revived.\n"
    )
    atomic_write_text(MS / "out" / "integration" / "INTEGRATION_REPORT.md", body)
    print(plan, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
