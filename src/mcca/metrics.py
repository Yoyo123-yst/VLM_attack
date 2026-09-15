"""CPU metrics for P0. No GPU, no new attack."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _xy(y: Sequence[int], s: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(y, dtype=np.int32), np.asarray(s, dtype=np.float64)


def auc_score(y: Sequence[int], s: Sequence[float]) -> Optional[float]:
    y_arr, s_arr = _xy(y, s)
    pos = s_arr[y_arr == 1]
    neg = s_arr[y_arr == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def auprc_score(y: Sequence[int], s: Sequence[float]) -> Optional[float]:
    """Average precision. None if no positive labels."""
    y_arr, s_arr = _xy(y, s)
    if y_arr.size == 0 or int(y_arr.sum()) == 0:
        return None
    order = np.argsort(-s_arr, kind="mergesort")
    y_s = y_arr[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    precision = tp / np.maximum(tp + fp, 1)
    n_pos = float(tp[-1])
    if n_pos <= 0:
        return None
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def precision_recall_at(y: Sequence[int], s: Sequence[float], tau: float) -> Dict[str, Any]:
    y_arr, s_arr = _xy(y, s)
    pred = s_arr >= float(tau)
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "tau": float(tau),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": int(y_arr.size),
        "precision": float(prec),
        "recall": float(rec),
        "n_pred_pos": int(pred.sum()),
        "n_true_pos": int(y_arr.sum()),
    }


def youden_tau(y: Sequence[int], s: Sequence[float]) -> Dict[str, Any]:
    y_arr, s_arr = _xy(y, s)
    if y_arr.size == 0 or y_arr.min() == y_arr.max():
        return {}
    best: Optional[tuple] = None
    for t in np.unique(s_arr):
        row = precision_recall_at(y_arr, s_arr, float(t))
        tpr = row["recall"]
        fpr = row["fp"] / max(row["fp"] + row["tn"], 1)
        j = tpr - fpr
        if best is None or j > best[0]:
            best = (j, float(t), tpr, fpr, row)
    assert best is not None
    return {"j": best[0], "tau": best[1], "tpr": best[2], "fpr": best[3], "at_tau": best[4]}


def smallest_tau_for_precision(
    y: Sequence[int],
    s: Sequence[float],
    min_precision: float = 0.8,
) -> Dict[str, Any]:
    """Smallest τ with P(y=1 | s≥τ) ≥ min_precision. Sweep unique scores descending."""
    y_arr, s_arr = _xy(y, s)
    if y_arr.size == 0 or y_arr.min() == y_arr.max():
        return {"tau": None, "ok": False, "reason": "need both classes"}
    uniq = np.unique(s_arr)
    uniq = np.sort(uniq)  # ascending: smaller τ is more inclusive
    chosen: Optional[Tuple[float, Dict[str, Any]]] = None
    table: List[Dict[str, Any]] = []
    for t in uniq:
        row = precision_recall_at(y_arr, s_arr, float(t))
        table.append(row)
        if row["precision"] + 1e-12 >= float(min_precision) and row["n_pred_pos"] > 0:
            if chosen is None or float(t) < chosen[0]:
                chosen = (float(t), row)
    if chosen is None:
        # fall back to max-precision τ, then max recall among ties
        best = max(table, key=lambda r: (r["precision"], r["recall"], -r["tau"]))
        return {
            "tau": float(best["tau"]),
            "ok": False,
            "reason": f"no tau reached precision {min_precision}",
            "at_tau": best,
            "n_thresholds": len(table),
        }
    return {
        "tau": chosen[0],
        "ok": True,
        "reason": None,
        "at_tau": chosen[1],
        "n_thresholds": len(table),
    }


def reliability_bins(
    y: Sequence[int],
    s: Sequence[float],
    n_bins: int = 8,
) -> List[Dict[str, Any]]:
    y_arr, s_arr = _xy(y, s)
    if y_arr.size == 0:
        return []
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(s_arr, qs))
    if edges.size < 2:
        return [
            {
                "lo": float(s_arr.min()),
                "hi": float(s_arr.max()),
                "n": int(y_arr.size),
                "mean_s": float(s_arr.mean()),
                "emp_rate": float(y_arr.mean()),
            }
        ]
    out = []
    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == edges.size - 2:
            mask = (s_arr >= lo) & (s_arr <= hi)
        else:
            mask = (s_arr >= lo) & (s_arr < hi)
        if not mask.any():
            continue
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(mask.sum()),
                "mean_s": float(s_arr[mask].mean()),
                "emp_rate": float(y_arr[mask].mean()),
            }
        )
    return out


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 3 or a.size != b.size:
        return None
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        for i, c in enumerate(counts):
            if c > 1:
                idx = np.where(inv == i)[0]
                ranks[idx] = ranks[idx].mean()
        return ranks

    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if den < 1e-12:
        return None
    return float((ra * rb).sum() / den)


def summarize(xs: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(list(xs), dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p10": float(np.quantile(a, 0.10)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }
