"""P8: three definitions of critical set U, plus restore-U recovery."""

from __future__ import annotations

from typing import List, Sequence

import torch

from .compressor import restore_u


def u_from_score(scores: torch.Tensor, owner: torch.Tensor, keep: torch.Tensor, n_u: int) -> torch.Tensor:
    """U_score: compressor-importance Top-K among A's kept tokens."""
    idx = torch.nonzero((owner == 0) & keep, as_tuple=False).flatten()
    if idx.numel() == 0:
        return idx
    n_use = max(1, min(int(n_u), int(idx.numel())))
    top = torch.topk(scores[idx].float(), k=n_use, largest=True).indices
    return idx[top]


def u_from_loo(loo_hits: Sequence[int], device=None, dtype=torch.long) -> torch.Tensor:
    """U_LOO: tokens whose leave-one-out removal changes the answer."""
    if not loo_hits:
        return torch.zeros((0,), dtype=dtype, device=device)
    return torch.tensor(list(loo_hits), dtype=dtype, device=device)


def u_from_grad(attr: torch.Tensor, owner: torch.Tensor, keep: torch.Tensor, n_u: int) -> torch.Tensor:
    """U_grad: task-gradient attribution among A's kept tokens (|dL/d token|)."""
    idx = torch.nonzero((owner == 0) & keep, as_tuple=False).flatten()
    if idx.numel() == 0:
        return idx
    n_use = max(1, min(int(n_u), int(idx.numel())))
    mag = attr[idx].float().abs()
    top = torch.topk(mag, k=n_use, largest=True).indices
    return idx[top]


def mask_from_idx(n: int, idx: torch.Tensor, device=None) -> torch.Tensor:
    out = torch.zeros((n,), dtype=torch.bool, device=device if device is not None else idx.device)
    if idx.numel():
        out[idx.long()] = True
    return out


def restore_u_keep(keep_adv: torch.Tensor, u_mask: torch.Tensor) -> torch.Tensor:
    return restore_u(keep_adv, u_mask)


def recover_rate(restore_ok: Sequence[bool], failed: Sequence[bool]) -> float:
    """P(recover | restore U) on the compression-only failures."""
    hits = [int(bool(ok)) for ok, bad in zip(restore_ok, failed) if bad]
    if not hits:
        return 0.0
    return float(sum(hits) / len(hits))
