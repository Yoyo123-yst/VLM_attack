"""Phrase-set losses. generic_pgd minimises these."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

from . import SAFE_MODES
from .phrases import scores_for_banks


def logsumexp_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_zeros(())
    return torch.logsumexp(scores.reshape(-1), dim=0)


def static_joint_loss(scores: Mapping[str, torch.Tensor]) -> torch.Tensor:
    safe = torch.stack([scores[c] for c in SAFE_MODES])
    return logsumexp_scores(safe) - scores["FOLLOW"]


def switched_phrase_loss(
    scores: Mapping[str, torch.Tensor],
    mode: str,
    *,
    lam: float = 0.5,
) -> torch.Tensor:
    follow = scores["FOLLOW"]
    if mode == "FOLLOW":
        return -follow
    if mode not in set(SAFE_MODES):
        return static_joint_loss(scores)
    s_m = scores[mode]
    if mode == "RELATED_SAFE":
        extra = logsumexp_scores(torch.stack([scores["REFUSE"], scores["DENY"]]))
        return s_m - follow + float(lam) * extra
    return s_m - follow


def banks_needed(method: str, mode: Optional[str]) -> tuple[str, ...]:
    if method == "static_joint":
        return ("REFUSE", "DENY", "RELATED_SAFE", "FOLLOW")
    if method != "switched":
        return ()
    if mode == "FOLLOW":
        return ("FOLLOW",)
    if mode == "REFUSE":
        return ("REFUSE", "FOLLOW")
    if mode == "DENY":
        return ("DENY", "FOLLOW")
    if mode == "RELATED_SAFE":
        return ("REFUSE", "DENY", "RELATED_SAFE", "FOLLOW")
    return ("REFUSE", "DENY", "RELATED_SAFE", "FOLLOW")


def make_phrase_objective(
    banks: Mapping[str, Sequence[Sequence[int]]],
    *,
    method: str,
    mode: Optional[str],
    lam: float,
):
    names = banks_needed(method, mode)

    def objective(wrapper, inputs, delta, x0):
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        scores = scores_for_banks(wrapper, inputs, x, banks, names)
        if method == "static_joint":
            return static_joint_loss(scores)
        return switched_phrase_loss(scores, str(mode or ""), lam=lam)

    return objective
