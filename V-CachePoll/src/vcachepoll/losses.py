"""Quota-poisoning and visual eviction-set losses. Pure tensor math."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .compressor import image_quotas, mean_importance_per_image


def caa_b_loss(scores: torch.Tensor, owner: torch.Tensor, keep_clean: torch.Tensor) -> torch.Tensor:
    """CAA-B-only: lift currently uninformative B tokens over A's weakest kept score."""
    b = scores[owner == 1].float()
    if b.numel() == 0:
        return scores.sum() * 0.0
    n_low = max(1, b.numel() // 3)
    low = torch.topk(b, k=n_low, largest=False).values
    a_kept = scores[(owner == 0) & keep_clean].float()
    cut = a_kept.min().detach() if a_kept.numel() else b.median().detach()
    return F.softplus(cut - low).mean()


def cage_b_loss(scores: torch.Tensor, owner: torch.Tensor, keep_clean: torch.Tensor) -> torch.Tensor:
    """CAGE-B-only: push currently dropped B tokens over the clean keep cutoff."""
    dropped = scores[(owner == 1) & ~keep_clean].float()
    if dropped.numel() == 0:
        kept_b = scores[(owner == 1) & keep_clean].float()
        if kept_b.numel() == 0:
            return scores.sum() * 0.0
        cut = kept_b.min().detach()
        return F.softplus(cut - kept_b.mean())
    kept = scores[keep_clean].float()
    cut = kept.min().detach() if kept.numel() else dropped.median().detach()
    return F.softplus(cut - dropped).mean()


def quota_loss(r: torch.Tensor) -> torch.Tensor:
    """Minimize r_A - r_B so B takes budget from A."""
    if r.numel() < 2:
        raise ValueError("need at least two image ratios")
    return r[0] - r[1]


def threshold_evict_loss(
    scores: torch.Tensor,
    owner: torch.Tensor,
    u_mask: torch.Tensor,
    k_a: int,
    kappa: float = 0.0,
) -> torch.Tensor:
    """Push victim tokens below the current within-image keep cutoff."""
    if int(u_mask.sum()) == 0:
        return scores.sum() * 0.0
    a_scores = scores[owner == 0].float()
    if a_scores.numel() == 0:
        return scores.sum() * 0.0
    k_use = max(1, min(int(k_a), int(a_scores.numel())))
    cutoff = torch.topk(a_scores, k=k_use, largest=True).values[-1].detach()
    su = scores[u_mask].float()
    return F.softplus(su - cutoff + float(kappa)).mean()


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
    lambda_c: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    i_bar = mean_importance_per_image(scores, owner, n_images=len(n_per))
    quota = image_quotas(i_bar, n_per, r_base=r_base, alpha=alpha, r_min=r_min, r_max=r_max)
    lq = quota_loss(quota.r)
    le = eviction_loss(scores, u_mask, v_mask, kappa=kappa, tau=tau)
    lc = threshold_evict_loss(scores, owner, u_mask, k_a=int(quota.k[0].item()), kappa=kappa)
    lv = value_loss(h_adv, h_clean)
    lp = tv_fn(xB)
    loss = (
        float(lambda_q) * lq
        + float(lambda_e) * le
        + float(lambda_c) * lc
        + float(lambda_v) * lv
        + float(lambda_p) * lp
    )
    aux = {
        "L_quota": lq,
        "L_evict": le,
        "L_crit": lc,
        "L_value": lv,
        "L_perc": lp,
        "r": quota.r,
        "i_bar": i_bar,
        "k": quota.k,
    }
    return loss, aux
