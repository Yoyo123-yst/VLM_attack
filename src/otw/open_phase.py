"""Open phase: PGD on s_mode only."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .score import s_mode


def clip_delta(x0: torch.Tensor, delta: torch.Tensor, eps: float) -> torch.Tensor:
    delta = delta.clamp(-eps, eps)
    return torch.clamp(x0 + delta, 0.0, 1.0) - x0


def signed_ascent_with_backtrack(
    x0: torch.Tensor,
    delta: torch.Tensor,
    grad: torch.Tensor,
    eps: float,
    alpha: float,
    score_fn,
    min_step: float = 1e-5,
):
    """L∞ sign step of size `alpha`, halved until the score does not fall.

    4-bit last-layer logits tolerate alpha=1/255; the L24 inner product does not.
    This is a numerics wrapper around the same signed PGD direction, not a new
    step-size hyperparameter.
    """
    direction = grad.detach().float().sign()
    step = float(alpha)
    s0 = float(score_fn(delta))
    best = delta.detach()
    while step + 1e-16 >= float(min_step):
        cand = clip_delta(x0, delta.detach() + step * direction, eps)
        s = float(score_fn(cand))
        if s > s0:
            return cand, s, step
        step *= 0.5
        used = step
    return best, s0, 0.0


def pgd_open(
    wrapper,
    x0: torch.Tensor,
    input_ids,
    attention_mask,
    grid,
    u: np.ndarray,
    steps: int,
    eps: float,
    alpha: float,
    seed: int,
    init: str = "random",
    delta0: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    g = torch.Generator(device=x0.device)
    g.manual_seed(int(seed))
    if delta0 is not None:
        delta = clip_delta(x0, delta0.detach().to(device=x0.device, dtype=x0.dtype), eps)
    elif init == "zero":
        delta = torch.zeros_like(x0)
    else:
        delta = (torch.rand(x0.shape, generator=g, device=x0.device) * 2.0 - 1.0) * eps
        delta = clip_delta(x0, delta, eps)

    with torch.no_grad():
        s0 = float(s_mode(wrapper, torch.clamp(x0 + delta, 0.0, 1.0), input_ids, attention_mask, grid, u).item())
    scores: List[float] = [s0]
    losses: List[float] = []
    grad_norms: List[float] = []
    linfs: List[float] = [float(delta.abs().max().item())]
    used_steps: List[float] = []
    n_backprop = 0
    for _ in range(int(steps)):
        delta = delta.detach().requires_grad_(True)
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        score = s_mode(wrapper, x, input_ids, attention_mask, grid, u)
        grad = torch.autograd.grad(score, delta, allow_unused=True)[0]
        if grad is None:
            raise RuntimeError("open-phase PGD got no image gradient")
        if not torch.isfinite(grad).all():
            raise RuntimeError("open-phase PGD got non-finite gradient")
        gn = float(grad.detach().float().norm().item())
        if gn <= 0.0:
            raise RuntimeError("open-phase PGD got zero gradient")

        def _score_fn(d):
            with torch.no_grad():
                return float(
                    s_mode(
                        wrapper,
                        torch.clamp(x0 + d, 0.0, 1.0),
                        input_ids,
                        attention_mask,
                        grid,
                        u,
                    ).item()
                )

        delta, s_after, used_step = signed_ascent_with_backtrack(
            x0, delta.detach(), grad, eps, float(alpha), _score_fn
        )
        n_backprop += 1
        scores.append(s_after)
        losses.append(float(-s_after))
        grad_norms.append(gn)
        linfs.append(float(delta.abs().max().item()))
        used_steps.append(float(used_step))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "delta": delta.detach(),
        "s_mode_init": s0,
        "s_mode_trace": scores,
        "s_mode": scores[-1],
        "mode_loss_trace": losses,
        "grad_norm_trace": grad_norms,
        "linf_trace": linfs,
        "used_step_trace": used_steps,
        "n_backprop": n_backprop,
        "grad_ok": True,
    }
