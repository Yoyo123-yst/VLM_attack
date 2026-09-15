"""CRAA pixel-controllability: R_inf / R_2 via one autograd JVP (no explicit Jacobian).

R_inf(u) = ||J^T u||_1 / ||u||_2  is the exact first-order max displacement of
h_l along u under an L-inf pixel budget: max_{||d||_inf<=eps} u^T J d = eps ||J^T u||_1.
J^T u = d(u^T h_l)/dx is a single vector-Jacobian product, never materializing J
(d=3584, p=338688 would be ~4.8GB and unnecessary).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from otw.score import s_mode

EPS = 0.06274509803921569  # 16/255, frozen CR-0 box


def as_unit(u) -> np.ndarray:
    arr = np.asarray(u, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        raise RuntimeError("target direction has near-zero norm")
    return (arr / n).astype(np.float64)


def random_gaussian(dim: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return as_unit(rng.normal(size=dim))


def random_orthogonal(u: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Random unit vector orthogonal to u (in residual space)."""
    uu = as_unit(u)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    v = rng.normal(size=dim)
    v = v - np.dot(v, uu) * uu
    return as_unit(v)


def _r_from_grad(grad: torch.Tensor, u: np.ndarray) -> Dict[str, float]:
    g = grad.detach().float()
    uu = as_unit(u)
    l1 = float(g.abs().sum().item())
    l2 = float(g.norm().item())
    denom = float(np.linalg.norm(uu))
    # One-step max displacement under the L-inf box (per frozen epsilon).
    return {
        "grad_l1": l1,
        "grad_l2": l2,
        "R_inf": l1 / denom,
        "R_2": l2 / denom,
        "max_displacement_linf": EPS * l1 / denom,
    }


def controllability(
    wrapper,
    x01: torch.Tensor,
    input_ids,
    attention_mask,
    grid,
    u,
) -> Dict[str, float]:
    """R_inf / R_2 for a target direction u at a single image x01.

    x01 must require grad (the caller reconstructs the attacked image and sets
    requires_grad_). s_mode = +<h, u> is reused verbatim from the frozen P0-S probe.
    """
    uu = as_unit(u)
    x = x01.detach().requires_grad_(True)
    score = s_mode(wrapper, x, input_ids, attention_mask, grid, uu)
    grad = torch.autograd.grad(score, x, allow_unused=True)[0]
    if grad is None:
        raise RuntimeError("controllability got no gradient to pixels")
    if not torch.isfinite(grad).all():
        raise RuntimeError("controllability got non-finite gradient")
    out = _r_from_grad(grad, uu)
    out["s_mode"] = float(score.detach().item())
    return out


def random_direction_set(u: np.ndarray, dim: int, n: int, base_seed: int) -> List[Dict[str, object]]:
    """n gaussian + n orthogonal norm-matched directions, deterministic seeds."""
    dirs: List[Dict[str, object]] = []
    for i in range(n):
        dirs.append(
            {
                "kind": "gaussian_unit",
                "index": i,
                "seed": base_seed + i,
                "u": random_gaussian(dim, base_seed + i),
            }
        )
    for i in range(n):
        dirs.append(
            {
                "kind": "random_orthogonal",
                "index": i,
                "seed": base_seed + 100000 + i,
                "u": random_orthogonal(u, dim, base_seed + 100000 + i),
            }
        )
    return dirs
