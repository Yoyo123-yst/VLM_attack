"""P6 compression-budget helpers. Experiment C (randomized r) is not enabled."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .compressor import image_quotas, isolated_k, shared_k_total, snap_k_to_total


DEFAULT_BUDGETS = (0.1, 0.2, 0.3, 0.5)


def k_per_image_avtp(
    i_bar: torch.Tensor,
    n_per: Sequence[int],
    r_base: float,
    alpha: float = 1.0,
    r_min: float = 0.05,
    r_max: float = 0.9,
) -> torch.Tensor:
    q = image_quotas(i_bar, n_per, r_base=r_base, alpha=alpha, r_min=r_min, r_max=r_max)
    return q.k


def k_never_empty(k: torch.Tensor) -> bool:
    return bool(int((k >= 1).all().item()))


def total_k_monotone(totals: Sequence[int]) -> bool:
    """As r decreases, K_A+K_B should be non-increasing overall (equal totals allowed)."""
    prev = None
    for t in totals:
        if prev is not None and int(t) > int(prev):
            return False
        prev = t
    return True


def budgets_for_cfg(cfg: Dict) -> List[float]:
    raw = (cfg.get("compressor") or {}).get("budget_sweep")
    if raw:
        return [float(x) for x in raw]
    return list(DEFAULT_BUDGETS)
