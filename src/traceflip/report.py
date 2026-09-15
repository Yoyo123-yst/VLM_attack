"""TraceFlip markdown report. Never emits raw model text."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

METHOD_LABEL = {
    "clean": "clean (no delta)",
    "refusal_margin_pgd": "refusal-margin PGD",
    "targeted_prefix_pgd": "targeted-prefix PGD",
    "traceflip": "TraceFlip",
    "traceflip_no_prefix": "TraceFlip w/o prefix constraint",
    "traceflip_value_only": "TraceFlip (value-only select)",
    "traceflip_cost_only": "TraceFlip (cost-only select)",
    "traceflip_earliest": "TraceFlip (earliest-branch select)",
    "traceflip_random": "TraceFlip (random select)",
    "traceflip_no_backtrack": "TraceFlip (no backtrack)",
    "gateflip": "GateFlip (k* commitment)",
}


def _lab(name: str) -> str:
    return METHOD_LABEL.get(name, name)


def _fmt(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v:.{nd}f}"


def rates_table(agg: Mapping[str, Any]) -> List[str]:
    lines = [
        "| method | n | ASR core_rhc (95% Wilson) | REFUSE | RELATED_SAFE | DENY | mean chars |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in agg.get("methods", {}).items():
        if not r.get("n"):
            continue
        lines.append(
            "| {m} | {n} | {asr} [{lo}, {hi}] | {ref} | {rel} | {deny} | {ch} |".format(
                m=_lab(name),
                n=r["n"],
                asr=_fmt(r["asr_core_rhc"]),
                lo=_fmt(r["asr_wilson_lo"]),
                hi=_fmt(r["asr_wilson_hi"]),
                ref=_fmt(r["refuse_rate"]),
                rel=_fmt(r["related_safe_rate"]),
                deny=_fmt(r["deny_rate"]),
                ch=_fmt(r["mean_chars"], 1),
            )
        )
    return lines


def budget_lines(agg: Mapping[str, Any]) -> List[str]:
    lines = [
        "| method | generated tokens | forward passes | backward passes | victim generations |",
        "|---|---|---|---|---|",
    ]
    for name, b in agg.get("budgets", {}).items():
        lines.append(
            "| {m} | {g} | {f} | {bw} | {v} |".format(
                m=_lab(name),
                g=_fmt(b.get("mean_generated_tokens"), 1),
                f=_fmt(b.get("mean_forward_passes"), 1),
                bw=_fmt(b.get("mean_backward_passes"), 1),
                v=_fmt(b.get("mean_victim_generations"), 2),
            )
        )
    return lines


def compare_lines(cmp: Mapping[str, Any]) -> List[str]:
    rows = list((cmp.get("paired_cells") or {}).items())
    lines = [
        f"Paired against **{_lab(str(cmp.get('baseline')))}** on identical cells:",
        "",
        "| method | pairs | ΔASR | 95% CI | McNemar n10/n01 | p |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in rows:
        ci = f"[{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}]" if "ci_lo" in r else "—"
        mc = (
            f"{r.get('mcnemar_n10')}/{r.get('mcnemar_n01')}"
            if "mcnemar_n10" in r
            else "—"
        )
        lines.append(
            f"| {_lab(name)} | {r.get('n_pairs', '—')} | {_fmt(r.get('diff'))} | {ci} | {mc} | {_fmt(r.get('mcnemar_p'))} |"
        )
    return lines


def diagnostic_lines(agg: Mapping[str, Any]) -> List[str]:
    lines = [
        "**Diagnostics. BFR and branch hits are solver-health numbers and must never be reported as ASR.**",
        "",
        "| method | cells with flip | BFR | mean attempts | prefix broken | flip failed (prefix kept) | in-model feasible but invalid |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, d in agg.get("diagnostics", {}).items():
        if not d.get("n_flip_cells"):
            continue
        lines.append(
            "| {m} | {c}/{n} | {bfr} | {at} | {pb} | {ff} | {fi} |".format(
                m=_lab(name),
                c=d.get("cells_with_flip", 0),
                n=d.get("n_flip_cells", 0),
                bfr=_fmt(d.get("bfr")),
                at=_fmt(d.get("mean_attempts"), 2),
                pb=_fmt(d.get("prefix_broken_rate")),
                ff=_fmt(d.get("flip_failed_after_prefix_kept_rate")),
                fi=_fmt(d.get("in_model_feasible_but_invalid_rate")),
            )
        )
    return lines


def per_query_lines(agg: Mapping[str, Any]) -> List[str]:
    names = [n for n in (agg.get("per_query") or {}) if (agg["per_query"][n] or {})]
    if not names:
        return []
    queries: List[str] = []
    for n in names:
        for q in (agg["per_query"][n] or {}):
            if q not in queries:
                queries.append(q)
    lines = [
        "Per-query ASR on cells (query-level is the unit that matters for rare events):",
        "",
        "| method | " + " | ".join(queries) + " |",
        "|---" * (len(queries) + 1) + "|",
    ]
    for n in names:
        cells = []
        for q in queries:
            rec = (agg["per_query"][n] or {}).get(q) or {}
            cells.append(
                f"{_fmt(rec.get('asr_core_rhc'))} ({rec.get('n_core_rhc', 0)}/{rec.get('n', 0)})"
            )
        lines.append(f"| {_lab(n)} | " + " | ".join(cells) + " |")
    return lines


def render(
    agg: Mapping[str, Any],
    cmp: Mapping[str, Any],
    *,
    meta: Mapping[str, Any] | None = None,
) -> str:
    meta = dict(meta or {})
    out: List[str] = [
        "# TraceFlip — pilot report",
        "",
        "TraceFlip is *constrained repair of an autoregressive decoding path*: "
        "Execute → Probe → Select → Constrained Flip → Re-execute/Backtrack. It never "
        "receives a pre-written target answer; the branch it repairs is read off the "
        "model's own greedy trajectory.",
        "",
        f"- model: `{meta.get('model', 'Qwen2-VL-7B-Instruct')}`",
        f"- eps: {_fmt(meta.get('eps'), 5)} (16/255), decode: greedy",
        f"- queries: {', '.join(meta.get('queries') or [])}",
        f"- carriers: {', '.join(meta.get('carriers') or [])}",
        f"- equal backward budget per cell: {meta.get('max_backward')}",
        f"- judge: `{meta.get('judge', 'n0.axes.four_axes+p0.judge.harmful_compliance')}`",
        f"- success definition: `core_rhc` only (REFUSE / RELATED_SAFE / DENY are not success)",
        "",
        "## 1. Outcome rates",
        "",
        *rates_table(agg),
        "",
        "## 2. Equal-budget comparison",
        "",
        *compare_lines(cmp),
        "",
        "## 3. Budgets actually spent",
        "",
        *budget_lines(agg),
        "",
        "## 4. Solver diagnostics (not ASR)",
        "",
        *diagnostic_lines(agg),
        "",
    ]
    pq = per_query_lines(agg)
    if pq:
        out += ["## 5. Per-query breakdown", "", *pq, ""]
    out += [
        "## Scope",
        "",
        "Pilot cells only. Sealed queries h83–h130 are untouched. No raw model text "
        "is stored; only hashes, labels and counts.",
        "",
    ]
    return "\n".join(out)
