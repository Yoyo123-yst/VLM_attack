"""P4 mechanism logs. No attack-algorithm change; JSON-safe snapshots only."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .compressor import exchange_events, mean_importance_per_image, summarize_keep


def tau_a(scores: torch.Tensor, owner: torch.Tensor, k_a: int) -> torch.Tensor:
    """Retention threshold: min of TopK_{K_A}(s^A)."""
    a = scores[owner == 0].float()
    if a.numel() == 0:
        return scores.sum() * 0.0
    k_use = max(1, min(int(k_a), int(a.numel())))
    return torch.topk(a, k=k_use, largest=True).values[-1]


def margins_u(scores: torch.Tensor, u_mask: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    if int(u_mask.sum()) == 0:
        return scores.new_zeros((0,))
    return scores[u_mask.bool()].float() - tau.float()


def quota_conserved(r: torch.Tensor, r_base: float, atol: float = 1e-4) -> bool:
    target = float(r.numel()) * float(r_base)
    return abs(float(r.sum()) - target) <= float(atol) + 1e-6 * abs(target)


def mechanism_snapshot(
    scores: torch.Tensor,
    owner: torch.Tensor,
    n_per: Sequence[int],
    keep: torch.Tensor,
    keep_clean: torch.Tensor,
    u_mask: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    i_bar: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    n_images = len(n_per)
    if i_bar is None:
        i_bar = mean_importance_per_image(scores, owner, n_images=n_images)
    tau = tau_a(scores, owner, int(k[0].item()) if torch.is_tensor(k) else int(k[0]))
    m = margins_u(scores, u_mask, tau)
    attacker = 1 if int(owner.max().item()) <= 1 else None
    ev = exchange_events(keep_clean, keep, owner, victim=0, attacker=attacker)
    u_out = int((u_mask.bool() & ~keep.bool()).sum().item())
    n_u = int(u_mask.sum().item())
    m_list = [float(x) for x in m.detach().cpu().tolist()]
    return {
        "i_bar": [float(x) for x in i_bar.detach().cpu().flatten().tolist()],
        "r": [float(x) for x in r.detach().cpu().flatten().tolist()],
        "k": [int(x) for x in k.detach().cpu().flatten().tolist()] if torch.is_tensor(k) else [int(x) for x in k],
        "kept": summarize_keep(keep, owner, n_images=n_images),
        "tau_a": float(tau.detach().cpu()),
        "m_u": m_list,
        "m_u_mean": float(m.mean().detach().cpu()) if m.numel() else None,
        "n_u_above": int((m > 0).sum().item()) if m.numel() else 0,
        "n_u_below": int((m < 0).sum().item()) if m.numel() else 0,
        "a_out": ev["a_out"],
        "b_in": ev["b_in"],
        "u_out": u_out,
        "n_u": n_u,
        "events": ev,
        "k_sum": int((k.detach().cpu().flatten() if torch.is_tensor(k) else torch.tensor(k)).sum().item()),
    }


def margin_crossing(clean: Dict[str, Any], adv: Dict[str, Any]) -> Dict[str, Any]:
    """Attack success on U should look like m_u: clean>0 → adv<0."""
    mc = clean.get("m_u") or []
    ma = adv.get("m_u") or []
    n = min(len(mc), len(ma))
    crossed = sum(1 for i in range(n) if mc[i] > 0 and ma[i] < 0)
    return {
        "n_u": n,
        "n_crossed": crossed,
        "frac_crossed": (crossed / n) if n else 0.0,
        "mean_m_clean": clean.get("m_u_mean"),
        "mean_m_adv": adv.get("m_u_mean"),
    }


def linf_ok(delta: torch.Tensor, eps: float, atol: float = 1e-6) -> bool:
    return float(delta.detach().abs().amax().cpu()) <= float(eps) + float(atol)


def delta_a_is_zero(xA: torch.Tensor, xA_adv: torch.Tensor, atol: float = 1e-8) -> bool:
    return bool(torch.allclose(xA.detach(), xA_adv.detach(), atol=atol, rtol=0.0))
