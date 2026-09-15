"""Stage 2-Diagnostic. CPU only. Reuse Stage 1/2 logs. Not Stage 3."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .committor import FEATURE_KEYS, MARGIN_ONLY, _metrics, _safe_auc, fit_logistic, split_xy
from .scorer import LIVE_KEYS
from .splits import COMMITTOR_TRAIN, COMMITTOR_VAL

Z95 = 1.959963984540054
NON_MARGIN_KEYS = tuple(k for k in FEATURE_KEYS if k not in MARGIN_ONLY)
LENGTH_STAGE_KEYS = (
    "frac",
    "prefix_word_count",
    "full_word_count",
    "token_count",
    "stage_early",
    "stage_mid",
    "stage_late",
    "stage_pre_eos",
)
LIVE_NON_MARGIN = tuple(k for k in LIVE_KEYS if k not in MARGIN_ONLY)


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Return (p_hat, lower, upper) for a binomial proportion."""
    n = int(n)
    k = int(k)
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    rad = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / den
    return float(p), float(max(0.0, center - rad)), float(min(1.0, center + rad))


def rarity_bin(n_rhc: int, n: int) -> str:
    """common / intermediate / rare / unresolved-zero.

    Wilson LCL≥0.20 or p̂≥0.40 → common (AMS not expected to beat naive).
    0 < p̂ ≤ 0.25 → rare-but-nonzero (AMS-suitable candidate).
    Unresolved-zero if no hits. Else intermediate.
    N=16 Wilson UCL<0.25 would mis-label h49 2/16 and h66 3/16 as intermediate.
    """
    n_rhc, n = int(n_rhc), int(n)
    if n <= 0 or n_rhc <= 0:
        return "unresolved-zero"
    p, lo, _hi = wilson_interval(n_rhc, n)
    if lo >= 0.20 or p >= 0.40:
        return "common"
    if n < 8:
        return "underpowered"
    if p <= 0.25:
        return "rare"
    return "intermediate"


def cell_rate_row(
    query_id: str,
    n_rhc: int,
    n: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p, lo, hi = wilson_interval(n_rhc, n)
    row = {
        "query_id": query_id,
        "n": int(n),
        "n_rhc": int(n_rhc),
        "p_hat": p,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "bin": rarity_bin(n_rhc, n),
    }
    if extra:
        row.update(extra)
    return row


def d1_from_trajectories(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        q = str(rec.get("query_id") or "")
        did = str(rec.get("delta_id") or rec.get("record_id") or "")
        by_query.setdefault(q, []).append(rec)
        by_cell.setdefault(did, []).append(rec)
    q_rows = []
    for q, xs in sorted(by_query.items()):
        n_rhc = sum(1 for r in xs if r.get("core_rhc") or r.get("y_rhc"))
        q_rows.append(cell_rate_row(q, n_rhc, len(xs), {"source": "stage1_query"}))
    c_rows = []
    for did, xs in sorted(by_cell.items()):
        q = str(xs[0].get("query_id") or "")
        n_rhc = sum(1 for r in xs if r.get("core_rhc") or r.get("y_rhc"))
        c_rows.append(
            cell_rate_row(
                q,
                n_rhc,
                len(xs),
                {
                    "source": "stage1_cell",
                    "delta_id": did,
                    "carrier_id": xs[0].get("carrier_id"),
                },
            )
        )
    rare_nz = [r for r in c_rows if r["bin"] == "rare"]
    rare_q = [r for r in q_rows if r["bin"] == "rare"]
    common = [r for r in c_rows if r["bin"] == "common"]
    return {
        "by_query": q_rows,
        "by_cell": c_rows,
        "n_rare_nonzero_cells": len(rare_nz),
        "n_rare_nonzero_queries": len(rare_q),
        "n_common_cells": len(common),
        "rare_cells": [r["delta_id"] for r in rare_nz],
        "rare_queries": [r["query_id"] for r in rare_q],
        "common_cells": [r.get("delta_id") for r in common],
        "ams_unsuitable_if_common": True,
        "note": (
            "AMS is for rare events. Common cells (Wilson LCL>=0.20) cannot "
            "prove AMS advantage; they only test whether the method skips resample."
        ),
    }


def d1_from_cr0(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_qc: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (str(rec.get("query_id") or ""), str(rec.get("carrier_id") or ""))
        by_qc.setdefault(key, []).append(rec)
    rows = []
    for (q, c), xs in sorted(by_qc.items()):
        n_rhc = sum(1 for r in xs if r.get("core_rhc"))
        rows.append(
            cell_rate_row(
                q,
                n_rhc,
                len(xs),
                {"source": "cr0_greedy", "carrier_id": c, "decode": "greedy"},
            )
        )
    rare_nz = [r for r in rows if r["bin"] == "rare"]
    return {
        "by_query_carrier": rows,
        "n_rare_nonzero": len(rare_nz),
        "note": "CR-0 is greedy (do_sample=false). Not comparable 1-1 to Stage 2 stochastic naive.",
    }


def d1_from_vlm_naive(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for r in runs:
        if r.get("method") != "naive":
            continue
        n = int(r.get("n_particles") or 0)
        n_rhc = int(r.get("n_hit_particles") or 0)
        row = cell_rate_row(
            str(r.get("query_id") or ""),
            n_rhc,
            n,
            {
                "source": "vlm_naive",
                "delta_id": r.get("delta_id"),
                "n_unique_lineages": r.get("n_unique_lineages"),
            },
        )
        # M=8 cells with a missing particle still use p-hat bins, not n<8 underpowered.
        if row["bin"] == "underpowered" and n >= 6:
            p = row["p_hat"]
            if p >= 0.40:
                row["bin"] = "common"
            elif 0 < p <= 0.25:
                row["bin"] = "rare"
            elif p > 0:
                row["bin"] = "intermediate"
        rows.append(row)
    return {
        "cells": rows,
        "n_rare_nonzero": sum(1 for r in rows if r["bin"] == "rare"),
        "n_common": sum(1 for r in rows if r["bin"] == "common"),
        "note": "VLM naive uses OPT seeds 20264-20267, M=8. Different seeds than Stage 1 collect.",
    }


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or int(np.sum(y)) == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a.astype(np.float64)))
    rb = np.argsort(np.argsort(b.astype(np.float64)))
    return float(np.corrcoef(ra, rb)[0, 1])


def _top_quartile_enrichment(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 4:
        return float("nan")
    prev = float(np.mean(y))
    if prev <= 0:
        return float("nan")
    q = float(np.quantile(p, 0.75))
    mask = p >= q
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y[mask]) / prev)


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), len(a), len(b)))
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    return float(len(ia & ib) / k)


def _ols_residual(target: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(covariate)), covariate.astype(np.float64)])
    beta, *_ = np.linalg.lstsq(x, target.astype(np.float64), rcond=None)
    return target.astype(np.float64) - x @ beta


def _pack_scores(name: str, y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    m = _metrics(y, p)
    m["auprc"] = _auprc(y, p)
    m["top_quartile_enrichment"] = _top_quartile_enrichment(y, p)
    m["name"] = name
    return m


def _val_proba(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit = fit_logistic(rows, keys, name="+".join(keys)[:48] or "empty")
    _, _, x_va, y_va, _, _ = split_xy(rows, keys)
    mean = np.array(fit.get("scaler_mean") or np.zeros(len(keys)), dtype=np.float64)
    scale = np.array(fit.get("scaler_scale") or np.ones(len(keys)), dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    coef = np.array([fit.get("coef", {}).get(k, 0.0) for k in keys], dtype=np.float64)
    intercept = float(fit.get("intercept") or 0.0)
    if len(x_va) == 0:
        p = np.zeros(0)
    else:
        z = (x_va - mean) / scale
        logit = intercept + z @ coef
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
    return y_va, p, fit


def d2_incremental(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Held-out query incremental value of committor vs refusal-only."""
    y, p_full, fit_full = _val_proba(rows, FEATURE_KEYS)
    _, p_margin, fit_margin = _val_proba(rows, MARGIN_ONLY)
    _, p_non, fit_non = _val_proba(rows, NON_MARGIN_KEYS)
    _, p_len, fit_len = _val_proba(rows, LENGTH_STAGE_KEYS)
    _, p_live, fit_live = _val_proba(rows, LIVE_KEYS)
    _, p_live_nm, _ = _val_proba(rows, LIVE_NON_MARGIN)

    full_m = _pack_scores("full", y, p_full)
    mar_m = _pack_scores("refusal_margin", y, p_margin)
    non_m = _pack_scores("non_margin", y, p_non)
    len_m = _pack_scores("length_stage", y, p_len)
    live_m = _pack_scores("live_full", y, p_live)
    live_nm = _pack_scores("live_non_margin", y, p_live_nm)

    resid = _ols_residual(p_full, p_margin) if len(y) else np.zeros(0)
    resid_clip = np.clip(resid, 0.0, 1.0) if len(resid) else resid
    resid_m = _pack_scores("residual_full_perp_margin", y, resid_clip)
    resid_m["auroc"] = _safe_auc(y, resid) if len(y) else float("nan")
    k = max(8, len(y) // 10) if len(y) else 1
    spearman = _spearman(p_full, p_margin)
    overlap = _topk_overlap(p_full, p_margin, k) if len(y) else float("nan")

    d_auroc = float(full_m["auroc"] - mar_m["auroc"]) if full_m["auroc"] == full_m["auroc"] else float("nan")
    d_auprc = float(full_m["auprc"] - mar_m["auprc"]) if full_m["auprc"] == full_m["auprc"] else float("nan")
    d_brier = float(mar_m["brier"] - full_m["brier"])  # >0 means full is better
    resid_auroc = float(resid_m["auroc"])
    length_explains = (
        abs(float(len_m["auroc"]) - float(full_m["auroc"])) <= 0.02
        if len_m["auroc"] == len_m["auroc"]
        else False
    )
    non_margin_beats_full = (
        non_m["auroc"] == non_m["auroc"]
        and full_m["auroc"] == full_m["auroc"]
        and float(non_m["auroc"]) > float(full_m["auroc"]) + 1e-6
    )
    margin_better_enrichment = (
        mar_m["top_quartile_enrichment"] == mar_m["top_quartile_enrichment"]
        and full_m["top_quartile_enrichment"] == full_m["top_quartile_enrichment"]
        and float(mar_m["top_quartile_enrichment"]) > float(full_m["top_quartile_enrichment"])
    )
    # Frozen Stage-1 margin is first-token of the *full* trajectory, copied onto every prefix.
    unique_traj = {r["trajectory_id"] for r in rows}
    margins_per_traj = {}
    for r in rows:
        margins_per_traj.setdefault(r["trajectory_id"], set()).add(round(float(r.get("refusal_margin") or 0.0), 6))
    margin_constant_within_traj = all(len(s) <= 1 for s in margins_per_traj.values())

    incremental = (
        d_auroc == d_auroc
        and d_auroc >= 0.02
        and d_auprc == d_auprc
        and d_auprc > 0.0
        and d_brier == d_brier
        and d_brier > 0.0
        and resid_auroc == resid_auroc
        and resid_auroc > 0.55
    )
    # Kill-switch: increment must not be only length/stage leaking trajectory identity.
    if incremental and length_explains and margin_constant_within_traj:
        incremental = False
        leak_note = "ΔAUROC is explained by length/stage while refusal_margin is constant within trajectory."
    else:
        leak_note = ""

    return {
        "n_val_prefixes": int(len(y)),
        "n_val_pos": int(np.sum(y)) if len(y) else 0,
        "n_train_queries": list(COMMITTOR_TRAIN),
        "n_val_queries": list(COMMITTOR_VAL),
        "models": {
            "full": full_m,
            "refusal_margin": mar_m,
            "non_margin": non_m,
            "length_stage": len_m,
            "live_full": live_m,
            "live_non_margin": live_nm,
            "residual_full_perp_margin": resid_m,
        },
        "delta_auroc_full_minus_margin": d_auroc,
        "delta_auprc_full_minus_margin": d_auprc,
        "delta_brier_margin_minus_full": d_brier,
        "spearman_full_vs_margin": spearman,
        "topk_prefix_overlap": overlap,
        "topk": int(k),
        "residual_auroc": resid_auroc,
        "margin_constant_within_trajectory": bool(margin_constant_within_traj),
        "n_unique_trajectories": len(unique_traj),
        "length_stage_explains_full": bool(length_explains),
        "non_margin_beats_full": bool(non_margin_beats_full),
        "margin_better_top_quartile_enrichment": bool(margin_better_enrichment),
        "incremental_information": bool(incremental),
        "leak_note": leak_note,
        "coef_full": fit_full.get("coef"),
        "coef_margin": fit_margin.get("coef"),
        "coef_non_margin": fit_non.get("coef"),
        "note": (
            "Stage 1 refusal_margin is the trajectory first-token, copied to all "
            "four prefixes. Live AMS used next-token margin. Incremental I(h;RHC|r) "
            "must hold on held-out queries; else PathEM STOP."
        ),
    }


def _ess_success(lineage_ids: Sequence[Any]) -> float:
    if not lineage_ids:
        return 0.0
    ids, counts = np.unique(np.array(list(lineage_ids), dtype=object), return_counts=True)
    w = counts.astype(np.float64)
    w = w / w.sum()
    return float(1.0 / np.sum(w * w))


def _run_lineage_stats(run: dict[str, Any]) -> dict[str, Any]:
    particles = list(run.get("particles") or [])
    hits_p = [p for p in particles if p.get("core_rhc")]
    method = str(run.get("method"))
    n_part = int(run.get("n_particles") or len(particles) or 0)
    n_hit = int(run.get("n_hit_particles") if run.get("n_hit_particles") is not None else len(hits_p))
    n_unique_root = int(run.get("n_unique_lineages") or 0)
    hashes = [str(p.get("output_hash") or "") for p in hits_p if p.get("output_hash")]
    unique_hash = len(set(hashes)) if hashes else (n_unique_root if method == "naive" else 0)
    lin_ids = [p.get("lineage_id") for p in hits_p]
    if not n_unique_root and lin_ids:
        n_unique_root = len(set(lin_ids))
    ess = _ess_success(lin_ids) if lin_ids else (float(n_unique_root) if n_hit else 0.0)
    clone_frac = float(max(0, n_hit - n_unique_root) / n_hit) if n_hit else 0.0
    tok = [int(p.get("token_count") or 0) for p in hits_p]
    ttf = int(min(tok)) if tok else None
    mean_tok_hit = float(np.mean(tok)) if tok else None
    p_any = 1.0 if n_hit > 0 else 0.0
    return {
        "method": method,
        "query_id": run.get("query_id"),
        "delta_id": run.get("delta_id"),
        "n_particles": n_part,
        "n_hit_particles": n_hit,
        "n_unique_roots": n_unique_root,
        "n_unique_terminal_hashes": unique_hash,
        "n_clone_hits": int(run.get("n_clone_hits") or max(0, n_hit - n_unique_root)),
        "clone_fraction": clone_frac,
        "ess_success": ess,
        "tokens": int(run.get("tokens") or 0),
        "tokens_to_first_rhc": ttf,
        "mean_tokens_rhc": mean_tok_hit,
        "p_at_least_one_rhc": p_any,
        "has_particles": bool(particles),
    }


def d3_lineage(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per = [_run_lineage_stats(r) for r in runs]
    methods = sorted({r["method"] for r in per})
    by_method: dict[str, dict[str, Any]] = {}
    for m in methods:
        xs = [r for r in per if r["method"] == m]
        def mean(key: str) -> float:
            vals = [r[key] for r in xs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        by_method[m] = {
            "n_runs": len(xs),
            "hit_particles_mean": mean("n_hit_particles"),
            "unique_roots_mean": mean("n_unique_roots"),
            "unique_hashes_mean": mean("n_unique_terminal_hashes"),
            "clone_fraction_mean": mean("clone_fraction"),
            "ess_success_mean": mean("ess_success"),
            "tokens_mean": mean("tokens"),
            "tokens_to_first_rhc_mean": mean("tokens_to_first_rhc"),
            "p_at_least_one_rhc": mean("p_at_least_one_rhc"),
        }
    u_full = by_method.get("ams_full", {}).get("unique_roots_mean", 0.0)
    u_ref = by_method.get("ams_refusal", {}).get("unique_roots_mean", 0.0)
    u_naive = by_method.get("naive", {}).get("unique_roots_mean", 0.0)
    h_full = by_method.get("ams_full", {}).get("unique_hashes_mean", 0.0)
    ranking_fail = abs(float(u_full or 0) - float(u_ref or 0)) < 1e-9
    resample_collapse = float(by_method.get("ams_full", {}).get("clone_fraction_mean") or 0) >= 0.25
    # Failure is "early resample not ranking" only if ranking differs from refusal
    # but unique-root metric still collapses clones.
    early_resample_not_ranking = (not ranking_fail) and resample_collapse and (
        float(h_full or 0) > float(u_full or 0)
    )
    naive_beats = float(u_naive or 0) > float(u_full or 0) + 1e-9
    return {
        "per_run": per,
        "by_method": by_method,
        "ranking_indistinguishable_from_refusal": bool(ranking_fail),
        "resample_collapse": bool(resample_collapse),
        "unique_hash_gt_unique_root_ams_full": bool(float(h_full or 0) > float(u_full or 0)),
        "failure_is_early_resample_not_ranking": bool(early_resample_not_ranking),
        "naive_beats_ams_full_unique_roots": bool(naive_beats),
        "note": (
            "Unique roots penalize splitting. Also report unique terminal hashes "
            "and ESS_success=1/sum w_a^2 over successful lineages. "
            "If AMS-full == AMS-refusal, the score is ranking-equivalent to refusal."
        ),
    }


def d4_budget(
    vlm_report: dict[str, Any],
    collect_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    budget = dict(vlm_report.get("budget") or {})
    runs = list(vlm_report.get("runs_public") or [])
    tok_by = {}
    gen_match = {}
    for m in ("ams_full", "naive", "ams_refusal", "ams_random"):
        xs = [r for r in runs if r.get("method") == m]
        tok_by[m] = int(sum(int(r.get("tokens") or 0) for r in xs))
    gen_match["naive_minus_ams_full_tokens"] = tok_by.get("naive", 0) - tok_by.get("ams_full", 0)
    gen_match["relative_gap"] = (
        float(gen_match["naive_minus_ams_full_tokens"]) / tok_by["ams_full"]
        if tok_by.get("ams_full")
        else float("nan")
    )
    scoring_forwards = int(budget.get("forward_passes") or 0)
    generated = int(budget.get("generated_tokens") or 0)
    wall = float(budget.get("wall_clock_seconds") or 0.0)
    peak = float(budget.get("peak_gpu_memory") or 0.0)
    # Scoring forwards were ledgered but not converted into generated-token units.
    scoring_not_in_token_match = scoring_forwards > 0
    collect_budget = (collect_report or {}).get("budget") or {}
    collect_note = (
        "Stage 1 collect_report.budget may cover only the resumed tail, not all 80 traj."
        if int(collect_budget.get("victim_generations") or 0) < int((collect_report or {}).get("n_traj") or 0)
        else ""
    )
    return {
        "vlm_budget": budget,
        "tokens_by_method": tok_by,
        "token_match_naive_vs_ams_full": gen_match,
        "scoring_forwards": scoring_forwards,
        "scoring_forwards_not_in_generated_token_match": bool(scoring_not_in_token_match),
        "ams_compute_underestimated_if_ignore_scoring": bool(scoring_not_in_token_match),
        "generated_tokens": generated,
        "wall_clock_seconds": wall,
        "peak_gpu_memory": peak,
        "killed_and_clone_tokens_in_generated": True,
        "collect_budget": collect_budget,
        "collect_budget_note": collect_note,
        "note": (
            "Generated-token matching compares naive vs AMS-full decode tokens. "
            "AMS also paid scoring forwards (live next-token margin) that naive did not. "
            "Clone continuations and killed-particle tokens are included in generated_tokens."
        ),
    }


def verdict(
    d1: dict[str, Any],
    d2: dict[str, Any],
    d3: dict[str, Any],
    d4: dict[str, Any],
    toy_gate: str = "GO",
) -> dict[str, Any]:
    """Stage 2b only if all four diagnostic gates pass. Else terminate PathEM."""
    vlm_naive = d1.get("vlm_naive") or {}
    stage1 = d1.get("stage1") or {}
    n_rare_cells = int(stage1.get("n_rare_nonzero_cells") or 0)
    n_rare_q = int(stage1.get("n_rare_nonzero_queries") or 0)
    # Query-level N=16 is the powered rarity test. N=4 cells are underpowered.
    d1_pass = n_rare_q >= 2
    d2_pass = bool(d2.get("incremental_information"))
    d3_pass = bool(d3.get("failure_is_early_resample_not_ranking"))
    d4_pass = True  # honesty gate: always report; does not invent missing ledgers
    toy_ok = str(toy_gate).upper() == "GO"
    stage2b = bool(d1_pass and d2_pass and d3_pass and toy_ok)
    causes = {
        "cell_selection_not_ams_suitable": (not d1_pass) or int(vlm_naive.get("n_common") or 0) >= 2,
        "committor_no_incremental_info": not d2_pass,
        "resampling_or_metric_mismatch": bool(d3.get("resample_collapse")) and not d3_pass,
        "pathem_core_hypotheses_false": (not d2_pass) and bool(d3.get("ranking_indistinguishable_from_refusal")),
    }
    if stage2b:
        decision = "STAGE2B"
        decision_note = (
            "All diagnostic gates passed. One-shot Stage 2b only: M=16, "
            "stratified resample, max clones per parent, no first-block resample, "
            "resample on dispersion/ESS trigger, common cells fall back to naive. "
            "Do not retune committor+M+block+fraction+stop together."
        )
    else:
        decision = "TERMINATE_PATHEM"
        decision_note = (
            "Stage 2 STOP stands. Do not enter Stage 3. E-step trajectories are "
            "not better than naive; a bad Q would confound the M-step. Next line "
            "is TraceFlip (path localization + constrained flip), not another AMS."
        )
    return {
        "gates": {
            "d1_rare_cells": d1_pass,
            "d2_incremental": d2_pass,
            "d3_early_resample_not_ranking": d3_pass,
            "d4_budget_audited": d4_pass,
            "toy_ams": toy_ok,
        },
        "n_rare_nonzero_stage1_cells": n_rare_cells,
        "n_rare_nonzero_stage1_queries": n_rare_q,
        "n_rare_vlm_naive": int(vlm_naive.get("n_rare_nonzero") or 0),
        "causes": causes,
        "stage2_stop_stands": True,
        "stage3_forbidden": True,
        "stage2b": "GO" if stage2b else "NO-GO",
        "decision": decision,
        "decision_note": decision_note,
        "scoring_forwards_caveat": bool(d4.get("ams_compute_underestimated_if_ignore_scoring")),
    }


def render_markdown(blob: dict[str, Any]) -> str:
    v = blob["verdict"]
    d1 = blob["d1"]
    d2 = blob["d2"]
    d3 = blob["d3"]
    d4 = blob["d4"]
    lines = [
        "# PathEM Stage 2-Diagnostic",
        "",
        "CPU only. Reuse Stage 1/2 logs. **Not ASR.** Stage 2 STOP stands. "
        "Stage 3 is not started.",
        "",
        f"**Decision: `{v['decision']}`** (Stage 2b `{v['stage2b']}`)",
        "",
        v["decision_note"],
        "",
        "## Gates",
        "",
        "| Gate | Pass | Meaning |",
        "|---|---|---|",
        f"| D1 rare-but-nonzero queries | {v['gates']['d1_rare_cells']} | ≥2 Stage-1 queries (N=16) in rare bin |",
        f"| D2 incremental vs refusal | {v['gates']['d2_incremental']} | held-out ΔAUROC/AUPRC/Brier + residual |",
        f"| D3 early-resample not ranking | {v['gates']['d3_early_resample_not_ranking']} | unique-root collapse without refusal-equivalence |",
        f"| D4 budget audited | {v['gates']['d4_budget_audited']} | tokens, forwards, wall; scoring not in token match |",
        f"| Toy AMS (pre-registered) | {v['gates']['toy_ams']} | weights still beat naive+random on toy |",
        "",
        "## Four causes",
        "",
    ]
    for k, val in v["causes"].items():
        lines.append(f"- `{k}`: **{val}**")
    lines += [
        "",
        "## D1 — event rates (Wilson 95%)",
        "",
        "Bins: `common` if Wilson LCL≥0.20 or p̂≥0.40; `rare` if 0<p̂≤0.25; "
        "`unresolved-zero` if 0 hits; `underpowered` if n<8; else `intermediate`. "
        "D1 gate uses query-level N=16, not N=4 cells.",
        "",
        "### Stage 1 by query (N=16 stochastic)",
        "",
        "| query | n_rhc/n | p̂ | Wilson | bin |",
        "|---|---|---|---|---|",
    ]
    for r in (d1.get("stage1") or {}).get("by_query") or []:
        lines.append(
            f"| {r['query_id']} | {r['n_rhc']}/{r['n']} | {r['p_hat']:.3f} | "
            f"[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] | {r['bin']} |"
        )
    lines += [
        "",
        "### VLM naive (Stage 2 cells, M=8)",
        "",
        "| cell | n_rhc/n | p̂ | Wilson | bin | unique roots |",
        "|---|---|---|---|---|---|",
    ]
    for r in (d1.get("vlm_naive") or {}).get("cells") or []:
        lines.append(
            f"| {r.get('delta_id')} | {r['n_rhc']}/{r['n']} | {r['p_hat']:.3f} | "
            f"[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] | {r['bin']} | "
            f"{r.get('n_unique_lineages')} |"
        )
    lines += [
        "",
        f"Stage-1 rare-nonzero queries: **{v.get('n_rare_nonzero_stage1_queries')}**. "
        f"Stage-1 rare-nonzero cells (N=4, mostly underpowered): **{v['n_rare_nonzero_stage1_cells']}**. "
        f"VLM-naive rare cells: **{v['n_rare_vlm_naive']}**.",
        "",
        "h56/h49 on VLM naive are common events (~54% pooled). AMS is expected to "
        "lose unique lineages to independent sampling on common cells.",
        "",
        "## D2 — incremental information (query-disjoint val h64/h72)",
        "",
        "| model | AUROC | AUPRC | Brier | top-quartile enrichment |",
        "|---|---|---|---|---|",
    ]
    for name in (
        "refusal_margin",
        "full",
        "non_margin",
        "length_stage",
        "live_full",
        "residual_full_perp_margin",
    ):
        m = (d2.get("models") or {}).get(name) or {}
        lines.append(
            f"| {name} | {m.get('auroc')} | {m.get('auprc')} | {m.get('brier')} | "
            f"{m.get('top_quartile_enrichment')} |"
        )
    lines += [
        "",
        f"- ΔAUROC full−margin: **{d2.get('delta_auroc_full_minus_margin')}**",
        f"- ΔAUPRC full−margin: **{d2.get('delta_auprc_full_minus_margin')}**",
        f"- ΔBrier (margin−full, >0 better): **{d2.get('delta_brier_margin_minus_full')}**",
        f"- Spearman(full, margin): **{d2.get('spearman_full_vs_margin')}**",
        f"- top-K prefix overlap (K={d2.get('topk')}): **{d2.get('topk_prefix_overlap')}**",
        f"- residual AUROC: **{d2.get('residual_auroc')}**",
        f"- margin constant within trajectory: **{d2.get('margin_constant_within_trajectory')}**",
        f"- length/stage explains full: **{d2.get('length_stage_explains_full')}**",
        f"- non-margin beats full AUROC: **{d2.get('non_margin_beats_full')}**",
        f"- margin better top-quartile enrichment: **{d2.get('margin_better_top_quartile_enrichment')}**",
        f"- incremental_information: **{d2.get('incremental_information')}**",
        "",
        d2.get("note") or "",
        "",
        d2.get("leak_note") or "",
        "",
        "## D3 — lineage vs hashes vs ESS",
        "",
        "| method | hits | unique roots | unique hashes | clone frac | ESS_success | P(≥1 RHC) | tok-to-first |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, row in (d3.get("by_method") or {}).items():
        lines.append(
            f"| {m} | {row.get('hit_particles_mean')} | {row.get('unique_roots_mean')} | "
            f"{row.get('unique_hashes_mean')} | {row.get('clone_fraction_mean')} | "
            f"{row.get('ess_success_mean')} | {row.get('p_at_least_one_rhc')} | "
            f"{row.get('tokens_to_first_rhc_mean')} |"
        )
    lines += [
        "",
        f"- ranking ≡ refusal: **{d3.get('ranking_indistinguishable_from_refusal')}**",
        f"- resample collapse: **{d3.get('resample_collapse')}**",
        f"- unique hashes > unique roots (AMS-full): **{d3.get('unique_hash_gt_unique_root_ams_full')}**",
        f"- failure is early-resample not ranking: **{d3.get('failure_is_early_resample_not_ranking')}**",
        "",
        d3.get("note") or "",
        "",
        "## D4 — budget honesty",
        "",
        f"- generated_tokens: {d4.get('generated_tokens')}",
        f"- scoring forwards: {d4.get('scoring_forwards')} "
        f"(not in generated-token match: {d4.get('scoring_forwards_not_in_generated_token_match')})",
        f"- wall_s: {d4.get('wall_clock_seconds')}",
        f"- peak_gpu_memory: {d4.get('peak_gpu_memory')}",
        f"- naive−AMS-full tokens: {d4.get('token_match_naive_vs_ams_full')}",
        f"- killed/clone tokens included in generated: {d4.get('killed_and_clone_tokens_in_generated')}",
        "",
        d4.get("note") or "",
        "",
        d4.get("collect_budget_note") or "",
        "",
        "## Explicitly not done",
        "",
        "- No Stage 3 M-step.",
        "- No larger-M rescue.",
        "- No eval seeds, no sealed h83–h130.",
        "- No raw generations in this file.",
        "- SMC hits are not ASR.",
        "",
        "## If PathEM is terminated",
        "",
        "TraceFlip is the next research line: localize an accidental RHC decoding "
        "path and constrain/flip it into a stable image-controlled path. Eval remains "
        "standard decoder + EVAL_SEEDS + frozen `core_rhc`. This repo has no TraceFlip code yet.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def run_diagnostic(
    trajectories: Sequence[dict[str, Any]],
    prefixes: Sequence[dict[str, Any]],
    vlm_cells_runs: Sequence[dict[str, Any]],
    vlm_report: dict[str, Any],
    cr0_records: Sequence[dict[str, Any]] | None = None,
    collect_report: dict[str, Any] | None = None,
    toy_gate: str = "GO",
) -> dict[str, Any]:
    stage1 = d1_from_trajectories(trajectories)
    cr0 = d1_from_cr0(cr0_records or [])
    naive = d1_from_vlm_naive(vlm_cells_runs)
    d1 = {"stage1": stage1, "cr0_greedy": cr0, "vlm_naive": naive}
    d2 = d2_incremental(prefixes)
    d3 = d3_lineage(vlm_cells_runs)
    d4 = d4_budget(vlm_report, collect_report)
    v = verdict(d1, d2, d3, d4, toy_gate=toy_gate)
    blob = {
        "stage": "2-diagnostic",
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "verdict": v,
        "store_raw_outputs": False,
        "not_asr": True,
    }
    blob["markdown"] = render_markdown(blob)
    return blob
