#!/usr/bin/env python3
"""Stage advance from pre-registered JSON verdicts. No --force-go."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CG = HERE.parent
sys.path.insert(0, str(CG / "src"))

from cgvsf.reporting import atomic_write_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-preregistered-gate", action="store_true")
    args = ap.parse_args()
    if not args.require_preregistered_gate:
        print("refusing: pass --require-preregistered-gate", flush=True)
        return 2
    p0a = CG / "out" / "p0a" / "p0a_verdict.json"
    if not p0a.exists():
        print("missing P0-A verdict", flush=True)
        return 2
    a = json.loads(p0a.read_text(encoding="utf-8"))
    p0b_path = CG / "out" / "p0b" / "p0b_verdict.json"
    b = json.loads(p0b_path.read_text(encoding="utf-8")) if p0b_path.exists() else {"verdict": "PENDING"}
    table = {
        ("GO", "GO"): "CG-VSF full (CE + energy + optional QP after Stage 2)",
        ("GO", "CONDITIONAL"): "CGVSF-CE; energy only for budget ranking",
        ("GO", "STOP"): "CGVSF-CE; delete control-energy claims",
        ("GO", "PENDING"): "run P0-B before integration",
        ("CONDITIONAL", "GO"): "one diagnostic rerun of P0-A first; no auto-upgrade",
        ("STOP", "GO"): "measurement paper only; no new attack claim",
        ("STOP", "STOP"): "terminate route; keep the negative result",
        ("STOP", "PENDING"): "P0-A STOP; P0-B optional as analysis",
    }
    plan = table.get((a["verdict"], b["verdict"]), "see roadmap §6; CONDITIONAL never auto-upgrades")
    body = (
        "# Integration decision\n\n"
        f"- P0-A: **{a['verdict']}**\n"
        f"- P0-B: **{b['verdict']}**\n"
        f"- Allowed: {plan}\n"
        "- Sealed queries remain sealed.\n"
    )
    atomic_write_text(CG / "out" / "integration" / "INTEGRATION_REPORT.md", body)
    print(plan, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
