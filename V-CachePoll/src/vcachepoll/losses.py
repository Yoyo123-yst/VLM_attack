"""Quota-poisoning and visual eviction-set losses. Pure tensor math."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .compressor import image_quotas, mean_importance_per_image


def quota_loss(r: torch.Tensor) -> torch.Tensor:
    """Minimize r_A - r_B so B takes budget from A."""
    if r.numel() < 2:
        raise ValueError("need at least two image ratios")
    return r[0] - r[1]


def eviction_loss(
    scores: torch.Tensor,
    u_mask: torch.Tensor,
    v_mask: torch.Tensor,
    kappa: float = 0.0,
    tau: float = 0.15,
) -> torch.Tensor:
    """Softmin over B tokens of the gap that keeps A token u above v.

    g_uv = s_u^A - s_v^B. When g < 0, v ranks above u.
    """
    if int(u_mask.sum()) == 0 or int(v_mask.sum()) == 0:
        return scores.sum() * 0.0
    s = scores.float()
    scale = s.std().clamp(min=1e-6)
    s = (s - s.mean()) / scale
    su = s[u_mask]
    sv = s[v_mask]
    gap = su.unsqueeze(1) - sv.unsqueeze(0)
    sp = F.softplus(gap + float(kappa))
    tau = max(float(tau), 1e-4)
    softmin = -tau * torch.logsumexp(-sp / tau, dim=1)
    return softmin.mean()


def value_loss(h_adv: torch.Tensor, h_clean: torch.Tensor) -> torch.Tensor:
    a = F.normalize(h_adv.float().reshape(-1), dim=0)
    b = F.normalize(h_clean.float().reshape(-1).detach(), dim=0)
    return 1.0 - (a * b).sum()


def combined_loss(
    scores: torch.Tensor,
    owner: torch.Tensor,
    n_per,
    u_mask: torch.Tensor,
    v_mask: torch.Tensor,
    h_adv: torch.Tensor,
    h_clean: torch.Tensor,
    xB: torch.Tensor,
    r_base: float,
    alpha: float,
    r_min: float,
    r_max: float,
    lambda_q: float,
    lambda_e: float,
    lambda_v: float,
    lambda_p: float,
    kappa: float,
    tau: float,
    tv_fn,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    i_bar = mean_importance_per_image(scores, owner, n_images=len(n_per))
    quota = image_quotas(i_bar, n_per, r_base=r_base, alpha=alpha, r_min=r_min, r_max=r_max)
    lq = quota_loss(quota.r)
    le = eviction_loss(scores, u_mask, v_mask, kappa=kappa, tau=tau)
    lv = value_loss(h_adv, h_clean)
    lp = tv_fn(xB)
    loss = (
        float(lambda_q) * lq
        + float(lambda_e) * le
        + float(lambda_v) * lv
        + float(lambda_p) * lp
    )
    aux = {"L_quota": lq, "L_evict": le, "L_value": lv, "L_perc": lp, "r": quota.r, "i_bar": i_bar, "k": quota.k}
    return loss, aux
