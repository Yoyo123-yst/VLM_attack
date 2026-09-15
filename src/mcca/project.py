"""Mode-feasible content step. Math only; no full attack loop."""

from __future__ import annotations

from typing import Tuple

import torch

XI = 1e-8


def _flat_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.reshape(-1).float() * b.reshape(-1).float()).sum()


def feasible_content_step(
    g_c: torch.Tensor,
    a: torch.Tensor,
    s_mode: float,
    tau: float,
    xi: float = XI,
) -> Tuple[torch.Tensor, str]:
    """Content descent direction that does not decrease s_mode (1st order).

    g_c = ∇_δ L_content (minimize)
    a   = ∇_δ s_mode     (want s_mode ≥ τ)
    d0  = -g_c
    """
    d0 = -g_c
    anorm2 = _flat_dot(a, a) + float(xi)
    adot = _flat_dot(a, d0)
    s = float(s_mode)
    t = float(tau)
    if s < t:
        scale = (t - s - adot) / anorm2
        d = d0 + scale * a
        return d, "min_correction"
    if float(adot.detach().cpu()) < 0.0:
        d = d0 - (adot / anorm2) * a
        return d, "projected"
    return d0, "unconstrained"


def linf_normalize(d: torch.Tensor, eta: float, xi: float = XI) -> torch.Tensor:
    den = d.abs().amax() + float(xi)
    return float(eta) * d / den


def project_box(x0: torch.Tensor, delta: torch.Tensor, eps: float) -> torch.Tensor:
    delta = delta.clamp(-float(eps), float(eps))
    return torch.clamp(x0 + delta, 0.0, 1.0) - x0
