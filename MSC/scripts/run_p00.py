#!/usr/bin/env python3
"""P0-0 runner: tokenizer-only phrase-bank gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent
ROOT = MS.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MS / "src"))

from msc.p00 import run_p00
from msc.protocol import load_frozen, sha256_file
from msc.reporting import atomic_write_text, now, write_status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(MS / "configs" / "p00.yaml"))
    args = ap.parse_args()
    frozen = load_frozen()
    sha = sha256_file(MS / "MSC_P0_FROZEN.json")
    rec = run_p00(frozen=frozen)
    out = MS / "out" / "p00"
    out.mkdir(parents=True, exist_ok=True)
    (out / "p00_tokensets.json").write_text(
        json.dumps(rec["tokensets"], indent=2), encoding="utf-8"
    )
    (out / "p00_verdict.json").write_text(json.dumps(rec["verdict"], indent=2), encoding="utf-8")
    gate = rec["verdict"]
    jac = gate.get("jaccard") or {}
    lines = [
        "# P0-0 report",
        "",
        f"- frozen sha256: `{sha}`",
        f"- automatic verdict: **{gate['verdict']}**",
        f"- Jaccard(REFUSE,FOLLOW) first token: {float(jac.get('refuse_follow') or 0):.3f}",
        f"- Jaccard(DENY,FOLLOW) first token: {float(jac.get('deny_follow') or 0):.3f}",
        f"- n_phrases: {gate.get('n_phrases')}",
        "",
        "## Reasons",
        "",
    ]
    reasons = gate.get("reasons") or ["none"]
    lines.extend(f"- {r}" for r in reasons)
    lines += [
        "",
        "## Allowed next",
        "",
        "- GO: `python MSC/scripts/run_p01.py --resume`",
        "- STOP: do not run P0-1; phrase banks are not separable.",
        "",
    ]
    atomic_write_text(out / "P00_REPORT.md", "\n".join(lines))
    write_status(
        MS / "out" / "STATUS.md",
        stage="P0-0",
        state="COMPLETE",
        frozen_sha=sha,
        git_commit="local",
        started=now(),
        completed=1,
        total=1,
        verdict=str(gate["verdict"]),
        evidence="; ".join(gate.get("reasons") or []) or "phrase banks tokenize",
        next_cmd="python MSC/scripts/run_p01.py --resume"
        if gate["verdict"] == "GO"
        else "python MSC/scripts/advance_stage.py --require-preregistered-gate",
    )
    print(json.dumps({"verdict": gate["verdict"], "jaccard": jac}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
