"""Pixel box and L-inf projection. Same epsilon as CR-0."""

from __future__ import annotations

import torch

from .protocol import EPS


def clip_delta(x0: torch.Tensor, delta: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    delta = delta.clamp(-float(eps), float(eps))
    return torch.clamp(x0 + delta, 0.0, 1.0) - x0


def project_linf(
    x0: torch.Tensor,
    x: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    return torch.clamp(x0 + clip_delta(x0, x - x0, eps), 0.0, 1.0)


def linf(delta: torch.Tensor) -> float:
    return float(delta.detach().abs().max().item())


def in_box(x0: torch.Tensor, x: torch.Tensor, eps: float = EPS, atol: float = 1e-6) -> bool:
    if not bool(torch.all(x >= -atol) and torch.all(x <= 1.0 + atol)):
        return False
    return float((x - x0).abs().max().item()) <= float(eps) + atol
