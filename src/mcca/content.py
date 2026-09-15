"""Core-content NLL and soft-min. Reuses OtW prefix teacher forcing."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from otw.write_phase import prefix_nll, tokenize_prefix

from .cores import core_prefixes, opening_prefixes, strip_openings


def soft_min_nll(nlls: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """L = -β log Σ_k exp(-L_k / β). As β→0 this is min_k L_k."""
    if nlls.numel() == 1:
        return nlls.reshape(())
    b = max(float(beta), 1e-6)
    return -b * torch.logsumexp(-nlls.float() / b, dim=0)


def tokenize_core(wrapper, text: str) -> torch.Tensor:
    cleaned = strip_openings(text)
    if not cleaned:
        raise RuntimeError("core prefix empty after stripping openings")
    return tokenize_prefix(wrapper, cleaned)


def nll_bank(
    wrapper,
    x01: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_attn: Optional[torch.Tensor],
    grid: torch.Tensor,
    texts: Sequence[str],
) -> torch.Tensor:
    vals: List[torch.Tensor] = []
    for t in texts:
        ids = tokenize_core(wrapper, t)
        vals.append(prefix_nll(wrapper, x01, prompt_ids, prompt_attn, grid, ids))
    return torch.stack(vals)


@torch.no_grad()
def content_losses(
    wrapper,
    x01: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_attn: Optional[torch.Tensor],
    grid: torch.Tensor,
    qid: str,
    beta: float = 0.5,
    k: int = 3,
) -> dict:
    cores = core_prefixes(qid, k=k)
    openings = opening_prefixes(k=k)
    nll_c = nll_bank(wrapper, x01, prompt_ids, prompt_attn, grid, cores)
    nll_o = nll_bank(wrapper, x01, prompt_ids, prompt_attn, grid, openings)
    return {
        "cores": cores,
        "nll_core": [float(x) for x in nll_c.detach().cpu().tolist()],
        "nll_opening": [float(x) for x in nll_o.detach().cpu().tolist()],
        "L_content": float(soft_min_nll(nll_c, beta=beta).detach().cpu().item()),
        "L_opening": float(soft_min_nll(nll_o, beta=beta).detach().cpu().item()),
        "beta": float(beta),
    }
