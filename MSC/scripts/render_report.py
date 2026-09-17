#!/usr/bin/env python3
"""Rebuild Markdown reports from JSONL / verdict JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent
ROOT = MS.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MS / "src"))

from msc.protocol import load_frozen, sha256_file
from msc.reporting import atomic_write_text, read_jsonl
from msc.verdict import p00_verdict, p01_verdict


def render_p00() -> None:
    frozen = load_frozen()
    sha = sha256_file(MS / "MSC_P0_FROZEN.json")
    out = MS / "out" / "p00"
    tokensets = json.loads((out / "p00_tokensets.json").read_text(encoding="utf-8")) if (out / "p00_tokensets.json").exists() else {}
    verdict = json.loads((out / "p00_verdict.json").read_text(encoding="utf-8")) if (out / "p00_verdict.json").exists() else p00_verdict(tokensets, frozen_gates=frozen["gates"])
    jac = verdict.get("jaccard") or {}
    lines = [
        "# P0-0 report",
        "",
        f"- frozen sha256: `{sha}`",
        f"- automatic verdict: **{verdict.get('verdict', 'PENDING')}**",
        f"- Jaccard(REFUSE,FOLLOW): {float(jac.get('refuse_follow') or 0):.3f}",
        f"- Jaccard(DENY,FOLLOW): {float(jac.get('deny_follow') or 0):.3f}",
        f"- n_phrases: {verdict.get('n_phrases')}",
        "",
        "## Reasons",
        "",
    ]
    reasons = verdict.get("reasons") or ["none"]
    lines.extend(f"- {r}" for r in reasons)
    lines += [
        "",
        "## Allowed next",
        "",
        "- GO: `python MSC/scripts/run_p01.py --resume`",
        "- STOP: do not run P0-1.",
        "",
    ]
    atomic_write_text(out / "P00_REPORT.md", "\n".join(lines))


def render_p01() -> None:
    frozen = load_frozen()
    sha = sha256_file(MS / "MSC_P0_FROZEN.json")
    out = MS / "out" / "p01"
    rows = read_jsonl(out / "p01_results.jsonl")
    gate = p01_verdict(
        rows,
        hard_cells=list(frozen["scope"]["hard_cells"]),
        frozen_gates=frozen["gates"],
        n_opt_seeds=len(frozen["seeds"]["opt"]),
        majority_k=int(frozen["success"]["majority_k"]),
    )
    lines = [
        "# P0-1 report",
        "",
        f"- frozen sha256: `{sha}`",
        f"- automatic verdict: **{gate['verdict']}**",
        f"- switched {gate['n_switched']}/8; static {gate['n_static']}/8; margin {gate['n_margin']}/8",
        f"- backward drop vs margin: {gate['backward_drop_vs_margin']:.3f}",
        f"- backward drop vs static: {gate['backward_drop_vs_static']:.3f}",
        "",
        "## Main table",
        "",
        "| method | robust cells | h53:c07 seeds | mean backward | RELATED_SAFE fail | INVALID fail cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in frozen["methods_p01"]:
        s = gate["summary"].get(m) or {}
        lines.append(
            f"| {m} | {s.get('n_rhc', 0)}/{s.get('n', 8)} | {s.get('h53_c07_seeds', 0)} | "
            f"{float(s.get('mean_backward') or 0):.1f} | {float(s.get('related_safe_rate') or 0):.3f} | {s.get('invalid_fail_cells', 0)} |"
        )
    lines += ["", "## Per-cell last mode (switched)", ""]
    by = defaultdict(list)
    for r in rows:
        if r.get("round") == "summary":
            continue
        by[(r.get("cell_id"), r.get("method"), r.get("opt_seed"))].append(r)
    for cell in frozen["scope"]["hard_cells"]:
        lines.append(f"### {cell}")
        for m in frozen["methods_p01"]:
            for seed in frozen["seeds"]["opt"]:
                recs = sorted(by.get((cell, m, seed), []), key=lambda x: int(x.get("round") or 0))
                path = " → ".join(
                    f"r{r['round']}:{r.get('after_mode')}" + ("*" if r.get("core_rhc") else "")
                    for r in recs
                ) or "—"
                lines.append(f"- `{m}` seed={seed}: {path}")
        lines.append("")
    lines += ["## Reasons", ""]
    reasons = gate.get("reasons") or ["none"]
    lines.extend(f"- {r}" for r in reasons)
    lines.append("")
    atomic_write_text(out / "P01_REPORT.md", "\n".join(lines))
    (out / "p01_verdict.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")


def render_p1() -> None:
    p1 = json.loads((MS / "MSC_P1_FROZEN.json").read_text(encoding="utf-8"))
    out = MS / "out" / "p1"
    rows = read_jsonl(out / "p1_results.jsonl") if (out / "p1_results.jsonl").exists() else []
    gate = json.loads((out / "p1_verdict.json").read_text(encoding="utf-8")) if (out / "p1_verdict.json").exists() else {}
    from msc.verdict import p1_method_rescue

    lines = [
        "# P1 INVALID-recovery report",
        "",
        f"- automatic verdict: **{gate.get('verdict', 'PENDING')}**",
        f"- winner: `{gate.get('method')}`",
        f"- P0-1 switched baseline: {gate.get('p01_switched_n', '—')}/8",
        "",
        "## Ladder",
        "",
    ]
    fail = list(p1["fail_cells"])
    for method in p1["ladder"]:
        info = p1_method_rescue(rows, method=method, cells=fail)
        lines.append(
            f"- `{method}` rescued {info['n_rhc']}/{len(fail)} {info['rescued']} seeds={info['per_cell_seeds']}"
        )
    lines += ["", "## Reasons", ""]
    reasons = gate.get("reasons") or ["none"]
    lines.extend(f"- {r}" for r in reasons)
    lines.append("")
    atomic_write_text(out / "P1_REPORT.md", "\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["p00", "p01", "p1"], required=True)
    args = ap.parse_args()
    if args.stage == "p00":
        render_p00()
    elif args.stage == "p01":
        render_p01()
    else:
        render_p1()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
