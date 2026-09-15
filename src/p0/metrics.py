from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def mean(xs: Iterable[float]) -> float:
    arr = [float(x) for x in xs]
    if not arr:
        return float("nan")
    return float(np.mean(arr))


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def fmt_ci(values: Sequence[float]) -> str:
    m, lo, hi = bootstrap_ci(values)
    if np.isnan(m):
        return "nan"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    """Mean of paired differences a-b with percentile bootstrap CI."""
    x = np.asarray(list(a), dtype=np.float64)
    y = np.asarray(list(b), dtype=np.float64)
    if x.size == 0 or x.size != y.size:
        return float("nan"), float("nan"), float("nan")
    d = x - y
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, d.size, size=d.size)
        means.append(float(d[idx].mean()))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_test(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int, float]:
    """Exact two-sided McNemar on paired binary outcomes.

    Returns (n10, n01, p_value) where n10 = a=1,b=0 and n01 = a=0,b=1.
    """
    x = np.asarray(list(a), dtype=np.int64).reshape(-1)
    y = np.asarray(list(b), dtype=np.int64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        return 0, 0, float("nan")
    n10 = int(np.sum((x == 1) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    # two-sided exact binomial test under p=0.5
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += float(math.comb(n, i))
    p = min(1.0, 2.0 * cdf / (2**n))
    return n10, n01, float(p)


def pearson_corr_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    if a.size < 3 or a.size != b.size:
        return float("nan"), float("nan"), float("nan")
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    corr = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.size, size=a.size)
        aa, bb = a[idx], b[idx]
        if float(aa.std()) < 1e-12 or float(bb.std()) < 1e-12:
            continue
        boots.append(float(np.corrcoef(aa, bb)[0, 1]))
    if not boots:
        return corr, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return corr, float(lo), float(hi)
