"""Pre-registered P0-0 / P0-1 verdicts. Thresholds cannot be edited after seeing results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from .protocol import BANK_NAMES


def p00_verdict(
    tokensets: Mapping[str, Any],
    *,
    frozen_gates: Mapping[str, Any],
) -> Dict[str, Any]:
    go = frozen_gates["P00_GO"]
    reasons: List[str] = []
    banks = tokensets.get("banks") or {}
    for name in BANK_NAMES:
        bank = banks.get(name) or {}
        n = int(bank.get("n_phrases") or 0)
        if n < int(go["min_phrases_per_bank"]):
            reasons.append(f"{name} has {n} phrases < {go['min_phrases_per_bank']}")
        ntok = list(bank.get("n_tokens") or [])
        if not ntok or any(int(t) < int(go["min_tokens_per_phrase"]) for t in ntok):
            reasons.append(f"{name} has an empty/short phrase")
    jac = tokensets.get("jaccard") or {}
    rf = float(jac.get("refuse_follow") or 0.0)
    df = float(jac.get("deny_follow") or 0.0)
    if rf >= float(go["jaccard_refuse_follow_first_token_max"]):
        reasons.append(f"Jaccard(REFUSE,FOLLOW) {rf:.3f} >= {go['jaccard_refuse_follow_first_token_max']}")
    if df >= float(go["jaccard_deny_follow_first_token_max"]):
        reasons.append(f"Jaccard(DENY,FOLLOW) {df:.3f} >= {go['jaccard_deny_follow_first_token_max']}")
    return {
        "verdict": "STOP" if reasons else "GO",
        "jaccard": jac,
        "n_phrases": {name: int((banks.get(name) or {}).get("n_phrases") or 0) for name in BANK_NAMES},
        "reasons": reasons,
        "note": "Do not edit phrase_banks after seeing P0-1.",
    }


def _seed_success(rows: Sequence[Mapping[str, Any]], majority_k: int) -> Dict[tuple, bool]:
    buckets: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("round") == "summary":
            buckets[(str(r.get("cell_id")), str(r.get("method")), int(r.get("opt_seed")))].append(r)
    out = {}
    for key, recs in buckets.items():
        rec = recs[-1]
        n_ok = int(rec.get("n_eval_rhc") or 0)
        n_eval = int(rec.get("n_eval") or 0)
        if n_eval > 0:
            out[key] = n_ok >= int(majority_k)
        else:
            out[key] = bool(rec.get("core_rhc"))
    return out


def _cell_method_success(
    seed_ok: Mapping[tuple, bool],
    cell: str,
    method: str,
) -> bool:
    hits = [v for (c, m, _s), v in seed_ok.items() if c == cell and m == method]
    return any(hits)


def _mean_backward(rows: Sequence[Mapping[str, Any]], method: str, cells: Sequence[str]) -> float:
    vals = []
    for r in rows:
        if r.get("round") != "summary":
            continue
        if str(r.get("method")) != method:
            continue
        if str(r.get("cell_id")) not in set(cells):
            continue
        vals.append(float(r.get("backward_used") or 0.0))
    return float(sum(vals) / len(vals)) if vals else 0.0


def _related_safe_rate(rows: Sequence[Mapping[str, Any]], method: str, cells: Sequence[str]) -> float:
    fail = [
        r
        for r in rows
        if r.get("round") != "summary"
        and str(r.get("method")) == method
        and str(r.get("cell_id")) in set(cells)
        and not r.get("core_rhc")
    ]
    if not fail:
        return 0.0
    n = sum(1 for r in fail if str(r.get("after_mode")) == "RELATED_SAFE")
    return n / len(fail)


def _invalid_fail_cells(rows: Sequence[Mapping[str, Any]], method: str, cells: Sequence[str]) -> int:
    bad = set()
    suc = set()
    for r in rows:
        if str(r.get("method")) != method or str(r.get("cell_id")) not in set(cells):
            continue
        if r.get("round") == "summary" and r.get("core_rhc"):
            suc.add(str(r.get("cell_id")))
        if r.get("round") != "summary" and str(r.get("after_mode")) in {"INVALID", "GARBAGE"}:
            bad.add(str(r.get("cell_id")))
    return len(bad - suc)


def p01_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    hard_cells: Sequence[str],
    frozen_gates: Mapping[str, Any],
    n_opt_seeds: int = 3,
    majority_k: int = 3,
) -> Dict[str, Any]:
    go = frozen_gates["P01_GO"]
    seed_ok = _seed_success(rows, majority_k)
    methods = sorted({str(r.get("method")) for r in rows if r.get("method")})
    summary: Dict[str, Any] = {}
    for method in methods:
        rescued = [c for c in hard_cells if _cell_method_success(seed_ok, c, method)]
        h53 = sum(1 for (c, m, _s), v in seed_ok.items() if c == "h53:c07" and m == method and v)
        summary[method] = {
            "n_rhc": len(rescued),
            "n": len(hard_cells),
            "rescued": rescued,
            "h53_c07_seeds": h53,
            "mean_backward": _mean_backward(rows, method, hard_cells),
            "related_safe_rate": _related_safe_rate(rows, method, hard_cells),
            "invalid_fail_cells": _invalid_fail_cells(rows, method, hard_cells),
        }

    sw = summary.get("switched") or {}
    st = summary.get("static_joint") or {}
    mg = summary.get("refusal_margin") or {}
    n_sw = int(sw.get("n_rhc") or 0)
    n_st = int(st.get("n_rhc") or 0)
    n_mg = int(mg.get("n_rhc") or 0)
    bw_sw = float(sw.get("mean_backward") or 0.0)
    bw_st = float(st.get("mean_backward") or 0.0)
    bw_mg = float(mg.get("mean_backward") or 0.0)
    bw_drop_mg = ((bw_mg - bw_sw) / bw_mg) if bw_mg > 0 else 0.0
    bw_drop_st = ((bw_st - bw_sw) / bw_st) if bw_st > 0 else 0.0
    h53_ok = int(sw.get("h53_c07_seeds") or 0) >= int(go["h53_c07_min_seeds"])
    rest_ok = True
    for cell in hard_cells:
        if cell == "h53:c07":
            continue
        if _cell_method_success(seed_ok, cell, "refusal_margin") and not _cell_method_success(
            seed_ok, cell, "switched"
        ):
            rest_ok = False
    beat_static = (n_sw - n_st) >= int(go["switched_beat_static_cells"]) or bw_drop_st >= float(
        go["switched_backward_drop_vs_static_min"]
    )
    rs_ok = True
    if float(mg.get("related_safe_rate") or 0.0) > 0:
        rs_ok = float(sw.get("related_safe_rate") or 0.0) <= float(go["related_safe_sub_max_ratio_vs_margin"]) * float(
            mg["related_safe_rate"]
        )
    inv_ok = int(sw.get("invalid_fail_cells") or 0) <= int(mg.get("invalid_fail_cells") or 0) + int(
        go["invalid_increase_max_cells"]
    )
    primary = n_sw >= int(go["robust_rhc_or"]) or (
        n_sw >= int(go["robust_rhc_and_budget"])
        and n_sw >= n_mg
        and bw_drop_mg >= float(go["backward_drop_vs_margin_min"])
    )
    reasons = []
    if not primary:
        reasons.append(f"switched {n_sw}/8 vs margin {n_mg}/8 bw_drop={bw_drop_mg:.3f}")
    if not h53_ok:
        reasons.append(f"h53:c07 seeds {sw.get('h53_c07_seeds')} < {go['h53_c07_min_seeds']}")
    if not rest_ok:
        reasons.append("a non-h53 cell lagged refusal-margin")
    if not beat_static:
        reasons.append(f"switched {n_sw} vs static {n_st} bw_drop_static={bw_drop_st:.3f}")
    if not rs_ok:
        reasons.append("RELATED_SAFE substitution did not drop vs margin")
    if not inv_ok:
        reasons.append("INVALID/GARBAGE increased")
    return {
        "verdict": "GO" if not reasons else "STOP",
        "summary": summary,
        "n_switched": n_sw,
        "n_static": n_st,
        "n_margin": n_mg,
        "backward_drop_vs_margin": bw_drop_mg,
        "backward_drop_vs_static": bw_drop_st,
        "reasons": reasons,
        "note": "7/8 without a 25% backward cut is STOP. Switched==static is STOP.",
    }


def p1_cell_seeds(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    cell: str,
    majority_k: int = 3,
) -> int:
    seed_ok = _seed_success(rows, majority_k)
    return sum(1 for (c, m, _s), v in seed_ok.items() if c == cell and m == method and v)


def p1_method_rescue(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    cells: Sequence[str],
    majority_k: int = 3,
) -> Dict[str, Any]:
    seed_ok = _seed_success(rows, majority_k)
    rescued = [c for c in cells if _cell_method_success(seed_ok, c, method)]
    per_cell = {c: p1_cell_seeds(rows, method=method, cell=c, majority_k=majority_k) for c in cells}
    return {
        "method": method,
        "rescued": rescued,
        "n_rhc": len(rescued),
        "per_cell_seeds": per_cell,
        "mean_backward": _mean_backward(rows, method, cells),
    }


def p1a_expand(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    fail_cells: Sequence[str],
    min_seeds: int,
    majority_k: int = 3,
) -> Dict[str, Any]:
    info = p1_method_rescue(rows, method=method, cells=fail_cells, majority_k=majority_k)
    ok = all(int(info["per_cell_seeds"].get(c) or 0) >= int(min_seeds) for c in fail_cells)
    return {"expand": ok, **info}


def p1b_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    hard_cells: Sequence[str],
    p01_switched_n: int,
    frozen_gates: Mapping[str, Any],
    majority_k: int = 3,
) -> Dict[str, Any]:
    go = frozen_gates["P1B_GO"]
    info = p1_method_rescue(rows, method=method, cells=hard_cells, majority_k=majority_k)
    h53 = p1_cell_seeds(rows, method=method, cell="h53:c07", majority_k=majority_k)
    reasons = []
    if int(info["n_rhc"]) < int(go["robust_rhc_min"]):
        reasons.append(f"{method} {info['n_rhc']}/8 < {go['robust_rhc_min']}")
    if h53 < int(go["h53_c07_min_seeds"]):
        reasons.append(f"h53:c07 seeds {h53} < {go['h53_c07_min_seeds']}")
    if int(info["n_rhc"]) < int(p01_switched_n) + int(go["beat_p01_switched_cells"]):
        reasons.append(f"{method} {info['n_rhc']} did not beat P0-1 switched {p01_switched_n}")
    return {
        "verdict": "GO" if not reasons else "STOP",
        "method": method,
        "reasons": reasons,
        **info,
        "h53_c07_seeds": h53,
        "p01_switched_n": int(p01_switched_n),
        "note": "P1 does not rewrite P0-1. prefix fallback is a hybrid, not pure MSC.",
    }
