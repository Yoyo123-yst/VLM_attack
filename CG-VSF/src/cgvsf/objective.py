"""Certificate scores and the LogSumExp cutting-plane objective."""

from __future__ import annotations

from typing import Sequence

import time

import torch
import torch.nn.functional as F

from traceflip.trace import forward_path_logits


def logsumexp_scores(scores: torch.Tensor) -> torch.Tensor:
    """L_CE = logsumexp_j G_j. Minimising this pushes every certificate down."""
    if scores.numel() == 0:
        return scores.new_zeros(())
    return torch.logsumexp(scores.reshape(-1), dim=0)


def certificate_score(
    wrapper,
    inputs: dict,
    delta: torch.Tensor,
    x0: torch.Tensor,
    token_ids: Sequence[int],
) -> torch.Tensor:
    """Mean token log-prob of a forced certificate prefix (higher = easier fail)."""
    ids = [int(x) for x in token_ids]
    if not ids:
        return delta.new_zeros(())
    x = torch.clamp(x0 + delta, 0.0, 1.0)
    pv, grid = wrapper.patchify(x)
    logits = forward_path_logits(
        wrapper,
        input_ids=inputs["input_ids"],
        pixel_values=pv,
        image_grid_thw=grid,
        attention_mask=inputs.get("attention_mask"),
        continuation_ids=ids,
    )
    sel = logits[: len(ids)]
    logp = F.log_softmax(sel.float(), dim=-1)
    idx = torch.tensor(ids, device=logp.device, dtype=torch.long)
    return logp[torch.arange(len(ids), device=logp.device), idx].mean()


def accumulated_objective(token_lists: Sequence[Sequence[int]]):
    """Return a generic_pgd objective that LogSumExp-minimises certificate scores."""

    def objective(wrapper, inputs, delta, x0):
        if not token_lists:
            return delta.reshape(-1)[0] * 0.0
        gs = [
            certificate_score(wrapper, inputs, delta, x0, ids) for ids in token_lists if ids
        ]
        if not gs:
            return delta.reshape(-1)[0] * 0.0
        return logsumexp_scores(torch.stack(gs))

    return objective


def sequential_lse_pgd(
    wrapper,
    inputs: dict,
    x0: torch.Tensor,
    delta_init: torch.Tensor,
    token_lists: Sequence[Sequence[int]],
    *,
    steps: int,
    eps: float,
    alpha: float,
) -> dict:
    """LSE-PGD with one certificate graph at a time. Same gradient as stacked LSE."""
    delta = torch.clamp(x0 + delta_init.detach(), 0.0, 1.0) - x0
    delta = delta.clamp(-float(eps), float(eps))
    lists = [list(ids) for ids in token_lists if ids]
    n_backward = 0
    if not lists:
        return {"delta": delta.detach(), "n_backward": 0}
    for i in range(int(steps)):
        t0 = time.time()
        with torch.no_grad():
            raw = []
            for ids in lists:
                g = certificate_score(wrapper, inputs, delta, x0, ids)
                raw.append(float(g.item()))
                del g
        w = torch.softmax(torch.tensor(raw, dtype=torch.float32), dim=0).tolist()
        grad_acc = None
        ok = False
        for wi, ids in zip(w, lists):
            d = delta.detach().requires_grad_(True)
            g = certificate_score(wrapper, inputs, d, x0, ids)
            gj = torch.autograd.grad(g, d, allow_unused=True)[0]
            del g
            if gj is None:
                del d
                continue
            ok = True
            term = gj * float(wi)
            grad_acc = term if grad_acc is None else grad_acc + term
            del gj, term, d
        if not ok or grad_acc is None:
            break
        n_backward += 1
        delta = (delta.detach() - float(alpha) * grad_acc.sign()).clamp(-float(eps), float(eps))
        delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0
        del grad_acc
        print(f"lse_step {i+1}/{steps} n_certs={len(lists)} dt={time.time()-t0:.2f}s", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"delta": delta.detach(), "n_backward": n_backward}


def certificate_jacobian(
    wrapper,
    inputs: dict,
    delta: torch.Tensor,
    x0: torch.Tensor,
    token_ids: Sequence[int],
):
    """G and ∇_δ G for one certificate. Detached outputs."""
    d = delta.detach().to(dtype=x0.dtype).clone().requires_grad_(True)
    g = certificate_score(wrapper, inputs, d, x0, token_ids)
    jac = torch.autograd.grad(g, d, allow_unused=True)[0]
    if jac is None:
        jac = torch.zeros_like(d)
    return g.detach(), jac.detach()
