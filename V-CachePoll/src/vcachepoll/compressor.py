"""AVTP-style image quotas and token survivor selection. Pure tensor logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class QuotaResult:
    r: torch.Tensor          # [n_images]
    k: torch.Tensor          # [n_images] int
    i_bar: torch.Tensor      # [n_images]
    i_avg: torch.Tensor      # scalar


def tokens_per_image(grid_thw: torch.Tensor, merge: int = 2) -> List[int]:
    counts = []
    merge2 = merge * merge
    for row in grid_thw.reshape(-1, 3):
        t, h, w = (int(x) for x in row.tolist())
        n = (t * h * w) // merge2
        if n <= 0:
            raise ValueError(f"non-positive token count from grid {(t, h, w)}")
        counts.append(n)
    return counts


def visual_owner(n_per_image: Sequence[int], device=None, dtype=torch.long) -> torch.Tensor:
    parts = [
        torch.full((n,), i, device=device, dtype=dtype) for i, n in enumerate(n_per_image)
    ]
    return torch.cat(parts, dim=0)


def mean_importance_per_image(scores: torch.Tensor, owner: torch.Tensor, n_images: int) -> torch.Tensor:
    if scores.numel() != owner.numel():
        raise ValueError("scores and owner length mismatch")
    means = []
    for i in range(n_images):
        sel = scores[owner == i]
        if sel.numel() == 0:
            raise ValueError(f"image {i} has no visual tokens")
        means.append(sel.mean())
    return torch.stack(means, dim=0)


def image_quotas(
    i_bar: torch.Tensor,
    n_per_image: Sequence[int],
    r_base: float = 0.5,
    alpha: float = 1.0,
    r_min: float = 0.1,
    r_max: float = 0.9,
) -> QuotaResult:
    """Paper-style r_i = r_base + alpha * (I_i - I_avg) after scaling I by its mean.

    Hidden-state L2 variation is not unit-range; relative I keeps alpha=1 meaningful.
    I_avg in the returned struct is the unweighted mean of the raw per-image scores.
    """
    if i_bar.ndim != 1:
        raise ValueError("i_bar must be [n_images]")
    n_images = i_bar.numel()
    if len(n_per_image) != n_images:
        raise ValueError("n_per_image does not match i_bar")
    # Hidden-state L2 variation is not unit-scaled. Use relative deviation
    # so alpha=1 matches AVTP's "importance around the mean" behavior.
    scale = i_bar.mean().clamp(min=1e-6)
    i_rel = i_bar / scale
    i_avg = i_rel.mean()
    r = r_base + alpha * (i_rel - i_avg)
    r = torch.clamp(r, min=r_min, max=r_max)
    # After clip, restore sum of ratios so one image's gain is another's loss.
    s = r.sum()
    target = r_base * n_images
    if float(s) > 0:
        r = r * (target / s)
        r = torch.clamp(r, min=r_min, max=r_max)
    n = torch.tensor(list(n_per_image), dtype=i_bar.dtype, device=i_bar.device)
    k = torch.round(r * n).to(torch.long)
    k = torch.minimum(k, n.to(torch.long))
    k = torch.clamp(k, min=1)
    return QuotaResult(r=r, k=k, i_bar=i_bar, i_avg=i_bar.mean())


def select_within_image(scores: torch.Tensor, owner: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Keep the top-k tokens inside each image. Ties: higher original index wins."""
    keep = torch.zeros_like(scores, dtype=torch.bool)
    n_images = int(k.numel())
    order_bonus = torch.arange(scores.numel(), device=scores.device, dtype=scores.dtype)
    keyed = scores + 1e-12 * order_bonus / max(scores.numel(), 1)
    for i in range(n_images):
        idx = torch.nonzero(owner == i, as_tuple=False).flatten()
        ki = int(k[i].item())
        ki = min(ki, int(idx.numel()))
        if ki <= 0:
            continue
        _, local = torch.topk(keyed[idx], k=ki, largest=True, sorted=False)
        keep[idx[local]] = True
    return keep


def select_global_topk(scores: torch.Tensor, k_total: int) -> torch.Tensor:
    keep = torch.zeros_like(scores, dtype=torch.bool)
    k_total = min(max(int(k_total), 1), int(scores.numel()))
    order_bonus = torch.arange(scores.numel(), device=scores.device, dtype=scores.dtype)
    keyed = scores + 1e-12 * order_bonus / max(scores.numel(), 1)
    _, idx = torch.topk(keyed, k=k_total, largest=True, sorted=False)
    keep[idx] = True
    return keep


def exchange_events(
    keep_clean: torch.Tensor,
    keep_adv: torch.Tensor,
    owner: torch.Tensor,
    victim: int = 0,
    attacker: int | None = 1,
) -> Dict[str, int]:
    a_out = int(((owner == victim) & keep_clean & ~keep_adv).sum().item())
    a_in = int(((owner == victim) & ~keep_clean & keep_adv).sum().item())
    att = (owner != victim) if attacker is None else (owner == int(attacker))
    b_in = int((att & ~keep_clean & keep_adv).sum().item())
    b_out = int((att & keep_clean & ~keep_adv).sum().item())
    a_kept_clean = int(((owner == victim) & keep_clean).sum().item())
    overlap_a = int(((owner == victim) & keep_clean & keep_adv).sum().item())
    return {
        "a_out": a_out,
        "a_in": a_in,
        "b_in": b_in,
        "b_out": b_out,
        "a_kept_clean": a_kept_clean,
        "a_overlap": overlap_a,
        "swap_ba": int(b_in > 0 and a_out > 0),
    }


def restore_victim(keep_adv: torch.Tensor, keep_clean: torch.Tensor, owner: torch.Tensor, victim: int = 0) -> torch.Tensor:
    out = keep_adv.clone()
    victim_clean = (owner == victim) & keep_clean
    out[victim_clean] = True
    return out


def isolated_k(n_per_image: Sequence[int], r_base: float) -> torch.Tensor:
    n = torch.tensor(list(n_per_image), dtype=torch.long)
    k = torch.round(n.to(torch.float32) * r_base).to(torch.long)
    k = torch.minimum(k, n)
    return torch.clamp(k, min=1)


def layer_variation_score(hidden_by_layer: Dict[int, torch.Tensor], score_layers: Sequence[int]) -> torch.Tensor:
    """hidden_by_layer maps 1-indexed LLM layer -> [seq, hidden] (no batch)."""
    diffs = []
    for layer in score_layers:
        if layer not in hidden_by_layer or (layer - 1) not in hidden_by_layer:
            raise KeyError(f"need hidden states for layers {layer - 1} and {layer}")
        a = hidden_by_layer[layer - 1].float()
        b = hidden_by_layer[layer].float()
        diffs.append((b - a).norm(dim=-1))
    return torch.stack(diffs, dim=0).mean(dim=0)


def summarize_keep(keep: torch.Tensor, owner: torch.Tensor, n_images: int) -> List[int]:
    return [int(((owner == i) & keep).sum().item()) for i in range(n_images)]


def shared_k_total(n_per_image: Sequence[int], r_base: float) -> int:
    """Single integer budget shared by every compressor in a fair comparison."""
    n = [int(x) for x in n_per_image]
    if not n or min(n) <= 0:
        raise ValueError(f"bad n_per_image {n_per_image}")
    total = sum(n)
    m = len(n)
    k = int(round(float(r_base) * total))
    k = max(k, m)
    return min(k, total)


def snap_k_to_total(k: torch.Tensor, k_total: int, n_per_image: Sequence[int]) -> torch.Tensor:
    n = torch.tensor(list(n_per_image), dtype=torch.long, device=k.device)
    out = k.to(torch.long).clone()
    out = torch.minimum(out, n)
    out = torch.clamp(out, min=1)
    cap = int(n.sum().item())
    m = int(n.numel())
    target = max(m, min(int(k_total), cap))
    guard = 0
    while int(out.sum().item()) != target and guard < 10000:
        guard += 1
        diff = target - int(out.sum().item())
        if diff > 0:
            slack = n - out
            if int(slack.max().item()) <= 0:
                break
            i = int(torch.argmax(slack).item())
            out[i] = out[i] + 1
        else:
            slack = out - 1
            if int(slack.max().item()) <= 0:
                break
            i = int(torch.argmax(slack).item())
            out[i] = out[i] - 1
    return out


def isolated_k_matched(n_per_image: Sequence[int], r_base: float) -> torch.Tensor:
    n = torch.tensor(list(n_per_image), dtype=torch.long)
    k_total = shared_k_total(n_per_image, r_base)
    raw = n.to(torch.float32) * (float(k_total) / max(float(n.sum().item()), 1.0))
    k = torch.round(raw).to(torch.long)
    return snap_k_to_total(k, k_total, n_per_image)


def select_spatial_uniform(owner: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """FEATHER-style spatial coverage: even strides inside each image, not score Top-K."""
    keep = torch.zeros(owner.shape[0], dtype=torch.bool, device=owner.device)
    n_images = int(k.numel())
    for i in range(n_images):
        idx = torch.nonzero(owner == i, as_tuple=False).flatten()
        ni = int(idx.numel())
        if ni == 0:
            continue
        ki = max(1, min(int(k[i].item()), ni))
        if ki == ni:
            keep[idx] = True
            continue
        pos = torch.linspace(0, ni - 1, steps=ki, device=owner.device)
        sel = pos.round().to(torch.long)
        keep[idx[sel]] = True
    return keep


def query_relevance_scores(visual_h: torch.Tensor, query_h: torch.Tensor) -> torch.Tensor:
    """QuietPrune-style query–visual cosine. visual_h: [N, D], query_h: [D]."""
    v = torch.nn.functional.normalize(visual_h.float(), dim=-1)
    q = torch.nn.functional.normalize(query_h.float().reshape(-1), dim=0)
    return v @ q


def restore_u(keep_adv: torch.Tensor, u_mask: torch.Tensor) -> torch.Tensor:
    out = keep_adv.clone()
    out[u_mask.bool()] = True
    return out


def quota_sum_target(n_images: int, r_base: float) -> float:
    return float(n_images) * float(r_base)
