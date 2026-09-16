"""Pre-registered P0-A / P0-B verdicts. 6/8 is never GO."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _cell_success(rows: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for r in rows:
        key = str(r.get("cell_id"))
        out[key] = bool(out.get(key)) or bool(r.get("core_rhc"))
    return out


def p0a_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    hard_cells: Sequence[str],
    frozen_gates: Mapping[str, Any],
) -> Dict[str, Any]:
    by_method: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if str(r.get("cell_id")) not in set(hard_cells):
            continue
        by_method[str(r.get("method"))].append(r)

    summary = {}
    for method, recs in by_method.items():
        suc = _cell_success(recs)
        n_hit = sum(1 for c in hard_cells if suc.get(c))
        rec_rounds = [r for r in recs if r.get("round") is not None and not r.get("core_rhc")]
        rec_n = sum(1 for r in rec_rounds if r.get("old_certificate_recurrence"))
        rec_d = len(rec_rounds)
        garbage = sum(
            1
            for c, ok in suc.items()
            if not ok
            and any(str(r.get("after_mode")) == "GARBAGE" and r.get("cell_id") == c for r in recs)
        )
        summary[method] = {
            "n_rhc": n_hit,
            "n": len(hard_cells),
            "recurrence_rate": (rec_n / rec_d) if rec_d else 0.0,
            "n_recurrence_rounds": rec_n,
            "n_fail_rounds": rec_d,
            "garbage_fail_cells": garbage,
            "rescued": sorted(c for c in hard_cells if suc.get(c)),
        }

    acc = summary.get("accumulated_certificate") or {}
    last = summary.get("last_certificate") or {}
    go_cfg = frozen_gates["P0A_GO"]
    cond_cfg = frozen_gates["P0A_CONDITIONAL"]
    n_acc = int(acc.get("n_rhc") or 0)
    n_last = int(last.get("n_rhc") or 0)
    rec_acc = float(acc.get("recurrence_rate") or 0.0)
    rec_last = float(last.get("recurrence_rate") or 0.0)
    rec_drop = None
    if rec_last > 0:
        rec_drop = (rec_last - rec_acc) / rec_last
    elif rec_acc == 0.0:
        rec_drop = 1.0
    else:
        rec_drop = 0.0
    rescued_special = any(
        c in acc.get("rescued", []) for c in cond_cfg.get("must_rescue_one_of") or []
    )
    beat = n_acc - n_last
    verdict = "STOP"
    if (
        n_acc >= int(go_cfg["hard_core_rhc_min"])
        and beat >= int(go_cfg["beat_last_certificate_by"])
        and (rec_drop or 0.0) >= float(go_cfg["recurrence_drop_vs_last"])
        and int(acc.get("garbage_fail_cells") or 0) <= int(go_cfg["garbage_increase_max_cells"])
    ):
        verdict = "GO"
    elif (
        n_acc == int(cond_cfg["hard_core_rhc"])
        and (rec_drop or 0.0) >= float(cond_cfg["recurrence_drop_vs_last"])
        and rescued_special
    ):
        verdict = "CONDITIONAL"

    return {
        "verdict": verdict,
        "summary": summary,
        "n_accumulated": n_acc,
        "n_last": n_last,
        "beat_last": beat,
        "recurrence_drop": rec_drop,
        "note": "6/8 is CONDITIONAL at most; GO requires 7/8 and +1 vs last_certificate.",
    }


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [s for s, y in zip(scores, labels) if int(y) == 1]
    neg = [s for s, y in zip(scores, labels) if int(y) == 0]
    if not pos or not neg:
        return float("nan")
    gt = eq = 0
    for p in pos:
        for n in neg:
            if p > n:
                gt += 1
            elif p == n:
                eq += 1
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def _rank(xs: Sequence[float]) -> List[float]:
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    rx, ry = _rank(list(x)), _rank(list(y))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx <= 0 or dy <= 0:
        return float("nan")
    return num / (dx * dy)


def top_quartile_lift(labels: Sequence[int], scores: Sequence[float]) -> float:
    if not labels:
        return float("nan")
    base = sum(int(y) for y in labels) / len(labels)
    if base <= 0:
        return float("nan")
    n = len(scores)
    k = max(1, n // 4)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    hit = sum(int(labels[i]) for i in order[:k]) / k
    return hit / base


def p0b_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    frozen_gates: Mapping[str, Any],
    min_states: int = 32,
) -> Dict[str, Any]:
    go_cfg = frozen_gates.get("P0B_GO") or {}
    cond_cfg = frozen_gates.get("P0B_CONDITIONAL") or {}
    min_states = int(go_cfg.get("min_states") or min_states)
    usable = [r for r in rows if r.get("eliminated") is not None and r.get("E_joint") is not None]
    n = len(usable)
    out: Dict[str, Any] = {
        "verdict": "WAIT",
        "n_states": n,
        "min_states": min_states,
        "auroc": {},
        "spearman_E_vs_steps": None,
        "top_quartile_lift": None,
        "best_simple": None,
        "delta_vs_simple": None,
        "note": "QP is not inserted into the attacker until P0-B GO.",
    }
    if n < min_states:
        out["note"] = f"Need ≥{min_states} measured states; have {n}."
        return out
    y = [int(bool(r["eliminated"])) for r in usable]
    steps = [float(r.get("steps_used") or 24) for r in usable]
    predictors = {
        "-E_joint": [-float(r["E_joint"]) for r in usable],
        "-E_single": [
            -float(r.get("E_single") if r.get("E_single") is not None else r["E_joint"])
            for r in usable
        ],
        "grad_norm": [float(r.get("grad_norm") or 0.0) for r in usable],
        "certificate_margin": [-float(r.get("certificate_margin") or 0.0) for r in usable],
        "refusal_margin": [-float(r.get("refusal_margin") or 0.0) for r in usable],
    }
    aucs = {k: auroc(y, v) for k, v in predictors.items()}
    simple_keys = ["grad_norm", "certificate_margin", "refusal_margin", "-E_single"]
    best_simple_name = max(simple_keys, key=lambda k: (aucs.get(k) or 0.0))
    best_simple = aucs.get(best_simple_name) or float("nan")
    e_auc = aucs.get("-E_joint") or float("nan")
    sp = spearman([float(r["E_joint"]) for r in usable], steps)
    lift = top_quartile_lift(y, predictors["-E_joint"])
    delta = e_auc - best_simple if e_auc == e_auc and best_simple == best_simple else float("nan")
    verdict = "STOP"
    if (
        e_auc == e_auc
        and e_auc >= float(go_cfg.get("auroc_min") or 0.70)
        and delta == delta
        and delta >= float(go_cfg.get("auroc_delta_vs_best_simple") or 0.05)
        and sp == sp
        and sp >= float(go_cfg.get("spearman_e_vs_steps_min") or 0.40)
    ):
        verdict = "GO"
    elif (
        e_auc == e_auc
        and e_auc >= float(cond_cfg.get("auroc_min") or 0.65)
        and lift == lift
        and lift >= float(cond_cfg.get("top_quartile_lift_min") or 1.5)
    ):
        verdict = "CONDITIONAL"
    out.update(
        {
            "verdict": verdict,
            "auroc": aucs,
            "spearman_E_vs_steps": sp,
            "top_quartile_lift": lift,
            "best_simple": best_simple_name,
            "delta_vs_simple": delta,
        }
    )
    return out

