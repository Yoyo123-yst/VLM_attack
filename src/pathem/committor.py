"""Stage 1 prefix committor: logistic on cheap features. No hidden-state claim."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .prefixes import PREFIX_STAGES, assert_group_split
from .splits import COMMITTOR_TRAIN, COMMITTOR_VAL

FEATURE_KEYS = (
    "refusal_margin",
    "logit_max",
    "frac",
    "prefix_word_count",
    "full_word_count",
    "token_count",
    "stage_early",
    "stage_mid",
    "stage_late",
    "stage_pre_eos",
)
MARGIN_ONLY = ("refusal_margin",)


def row_x(row: dict[str, Any], keys: Sequence[str] = FEATURE_KEYS) -> np.ndarray:
    return np.array([float(row.get(k) or 0.0) for k in keys], dtype=np.float64)


def split_xy(
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str] = FEATURE_KEYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    train = [r for r in rows if r["query_id"] in set(COMMITTOR_TRAIN)]
    val = [r for r in rows if r["query_id"] in set(COMMITTOR_VAL)]
    assert_group_split(rows, COMMITTOR_TRAIN, COMMITTOR_VAL)
    x_tr = np.stack([row_x(r, keys) for r in train]) if train else np.zeros((0, len(keys)))
    y_tr = np.array([int(r["y_rhc"]) for r in train], dtype=np.int64)
    x_va = np.stack([row_x(r, keys) for r in val]) if val else np.zeros((0, len(keys)))
    y_va = np.array([int(r["y_rhc"]) for r in val], dtype=np.int64)
    return x_tr, y_tr, x_va, y_va, [r["trajectory_id"] for r in train], [r["trajectory_id"] for r in val]


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    prev = float(np.mean(y)) if len(y) else 0.0
    base = np.full_like(p, prev, dtype=np.float64)
    return {
        "n": int(len(y)),
        "n_pos": int(np.sum(y)),
        "prevalence": prev,
        "auroc": _safe_auc(y, p),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "brier_prevalence": float(brier_score_loss(y, base)) if len(y) else float("nan"),
    }


def stage_enrichment(rows: Sequence[dict[str, Any]], p: np.ndarray, mask_q) -> dict[str, float]:
    qset = set(mask_q)
    out: dict[str, float] = {}
    for i, stage in enumerate(PREFIX_STAGES):
        idx = [j for j, r in enumerate(rows) if r["query_id"] in qset and r["stage"] == stage]
        if not idx:
            out[stage] = float("nan")
            continue
        y = np.array([int(rows[j]["y_rhc"]) for j in idx], dtype=np.int64)
        s = p[idx]
        out[stage] = _safe_auc(y, s)
    return out


def fit_logistic(
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str] = FEATURE_KEYS,
    name: str = "full",
) -> dict[str, Any]:
    x_tr, y_tr, x_va, y_va, _, _ = split_xy(rows, keys)
    if len(np.unique(y_tr)) < 2:
        return {"name": name, "gate": "STOP", "reason": "train_single_class", "features": list(keys)}
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_va_s = scaler.transform(x_va) if len(x_va) else x_va
    clf = LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs")
    clf.fit(x_tr_s, y_tr)
    p_tr = clf.predict_proba(x_tr_s)[:, 1]
    p_va = clf.predict_proba(x_va_s)[:, 1] if len(x_va) else np.zeros(0)
    train_m = _metrics(y_tr, p_tr)
    val_m = _metrics(y_va, p_va)
    val_rows = [r for r in rows if r["query_id"] in set(COMMITTOR_VAL)]
    enrich = stage_enrichment(val_rows, p_va, COMMITTOR_VAL) if len(p_va) else {}
    early = enrich.get("early")
    pre_eos = enrich.get("pre_eos")
    auroc = val_m["auroc"]
    brier = val_m["brier"]
    brier_base = val_m["brier_prevalence"]
    better_than_random = auroc == auroc and auroc > 0.5
    brier_ok = brier == brier and brier_base == brier_base and brier < brier_base
    not_only_eos = (
        early == early
        and pre_eos == pre_eos
        and (early > 0.5 or (pre_eos - early) < 0.25)
    )
    advisory_070 = auroc == auroc and auroc >= 0.70
    gate = "GO" if (better_than_random and brier_ok and not_only_eos) else "STOP"
    return {
        "name": name,
        "features": list(keys),
        "coef": {k: float(v) for k, v in zip(keys, clf.coef_.ravel())},
        "intercept": float(clf.intercept_.ravel()[0]),
        "train": train_m,
        "val": val_m,
        "val_auroc_by_stage": enrich,
        "better_than_random": bool(better_than_random),
        "brier_beats_prevalence": bool(brier_ok),
        "not_only_eos": bool(not_only_eos),
        "advisory_auroc_070": bool(advisory_070),
        "gate": gate,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
    }


def compare_margin_vs_full(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    full = fit_logistic(rows, FEATURE_KEYS, name="full")
    margin = fit_logistic(rows, MARGIN_ONLY, name="refusal_margin")
    full_auc = full.get("val", {}).get("auroc")
    mar_auc = margin.get("val", {}).get("auroc")
    hidden_not_better = (
        full_auc == full_auc
        and mar_auc == mar_auc
        and full_auc <= mar_auc + 1e-6
    )
    note = (
        "full features are not last-layer prefix hidden; Stage 1 uses cheap "
        "first-token margin + length + stage indicators."
    )
    if hidden_not_better and full.get("gate") == "GO":
        note += " full ≰ margin; keep full only if it still beats random+Brier."
    return {
        "full": full,
        "refusal_margin": margin,
        "full_not_better_than_margin": bool(hidden_not_better),
        "gate": full.get("gate", "STOP"),
        "note": note,
    }
