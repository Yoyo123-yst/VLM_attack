#!/usr/bin/env python3
"""Rebuild Markdown reports from JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CG = HERE.parent
ROOT = CG.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CG / "src"))

from cgvsf.protocol import load_frozen, sha256_file
from cgvsf.reporting import atomic_write_text, read_jsonl
from cgvsf.verdict import p0a_verdict


def render_p0a() -> None:
    frozen = load_frozen()
    rows = [r for r in read_jsonl(CG / "out" / "p0a" / "p0a_results.jsonl") if r.get("round") != "summary"]
    summaries = [r for r in read_jsonl(CG / "out" / "p0a" / "p0a_results.jsonl") if r.get("round") == "summary"]
    gate = p0a_verdict(
        rows,
        hard_cells=list(frozen["scope"]["hard_cells"]),
        frozen_gates=frozen["gates"],
    )
    hard = list(frozen["scope"]["hard_cells"])
    methods = list(frozen["methods_p0a"])
    suc = defaultdict(dict)
    for r in summaries:
        suc[r["method"]][r["cell_id"]] = bool(r.get("core_rhc"))
    lines = [
        "# P0-A report",
        "",
        f"- frozen sha256: `{sha256_file(CG / 'CGVSF_P0_FROZEN.json')}`",
        f"- judge: `{frozen['success']['judge']}`",
        f"- success: `core_rhc` only",
        f"- hard cells: {len(hard)}",
        f"- automatic verdict: **{gate['verdict']}**",
        "",
        "## 1. Frozen protocol",
        "",
        f"- eps 16/255, 5 rounds × 24 backward = 120, decode every round.",
        f"- target-free arms: {', '.join(frozen['target_free_methods'])}.",
        f"- targeted_prefix_earlystop and gateflip_fair use the same check frequency.",
        "",
        "## 2. Main table (hard 8)",
        "",
        "| method | RHC | recurrence | garbage-fail cells |",
        "|---|---|---|---|",
    ]
    for m in methods:
        s = gate["summary"].get(m) or {}
        lines.append(
            f"| {m} | {s.get('n_rhc', 0)}/{s.get('n', 8)} | {float(s.get('recurrence_rate') or 0):.3f} | {s.get('garbage_fail_cells', 0)} |"
        )
    lines += ["", "## 3. Per-cell mode transitions", ""]
    by = defaultdict(list)
    for r in rows:
        by[(r.get("cell_id"), r.get("method"))].append(r)
    trans_lines = ["# Mode transitions", ""]
    for cell in hard:
        lines.append(f"### {cell}")
        trans_lines.append(f"## {cell}")
        for m in methods:
            recs = sorted(by.get((cell, m), []), key=lambda x: int(x.get("round") or 0))
            path = " → ".join(
                f"r{r['round']}:{r.get('after_mode')}" + ("*" if r.get("core_rhc") else "")
                for r in recs
            ) or "—"
            lines.append(f"- `{m}`: {path}")
            trans_lines.append(f"- `{m}`: {path}")
        lines.append("")
        trans_lines.append("")
    lines += [
        "## 4. Recurrence and ablation",
        "",
        f"- accumulated vs last: {gate['n_accumulated']} vs {gate['n_last']} (Δ={gate['beat_last']})",
        f"- recurrence drop: {gate['recurrence_drop']}",
        "",
        "## 5. Verdict (pre-registered, not hand-edited)",
        "",
        f"**{gate['verdict']}**. {gate['note']}",
        "",
        "## 6. Allowed next step",
        "",
    ]
    if gate["verdict"] == "GO":
        lines.append("P0-B may run. Do not open the sealed set. Do not add Sure-prefix to target-free arms.")
    elif gate["verdict"] == "CONDITIONAL":
        lines.append("One pre-registered diagnostic rerun only. No data expansion.")
    else:
        lines.append("STOP CG as an attack method unless P0-B is run as a measurement-only study.")
    lines.append("")
    atomic_write_text(CG / "out" / "p0a" / "P0A_REPORT.md", "\n".join(lines))
    atomic_write_text(CG / "out" / "p0a" / "MODE_TRANSITIONS.md", "\n".join(trans_lines))
    (CG / "out" / "p0a" / "p0a_verdict.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(f"verdict {gate['verdict']}", flush=True)


def render_p0b() -> None:
    frozen = load_frozen()
    path = CG / "out" / "p0b" / "p0b_results.jsonl"
    from cgvsf.verdict import p0b_verdict

    rows = read_jsonl(path) if path.exists() else []
    gate = p0b_verdict(rows, frozen_gates=frozen["gates"])
    lines = [
        "# P0-B report",
        "",
        f"- frozen sha256: `{sha256_file(CG / 'CGVSF_P0_FROZEN.json')}`",
        f"- states: {gate['n_states']} (min {gate['min_states']})",
        f"- automatic verdict: **{gate['verdict']}**",
        f"- -E_joint AUROC: {gate['auroc'].get('-E_joint')}",
        f"- best simple: {gate.get('best_simple')} Δ={gate.get('delta_vs_simple')}",
        f"- Spearman E vs steps: {gate.get('spearman_E_vs_steps')}",
        "",
        str(gate.get("note") or ""),
        "",
        "QP is not in the attacker unless verdict is GO.",
        "",
    ]
    atomic_write_text(CG / "out" / "p0b" / "P0B_REPORT.md", "\n".join(lines))
    (CG / "out" / "p0b" / "p0b_verdict.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"p0b verdict {gate['verdict']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["p0a", "p0b"])
    args = ap.parse_args()
    if args.stage == "p0a":
        render_p0a()
        return 0
    render_p0b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
