"""Universal visual attacks: refusal-margin, random subspace, full residual, U-guided."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .datautil import open_image
from .judge import benign_utility, harmful_compliance
from .model import (
    COMPLY_WORDS,
    REFUSE_WORDS,
    LlavaP0,
    image_to_clip_pixels,
    margin_from_logits,
    normalize_clip,
    token_id_list,
    tv_loss,
)
from .subspace import project, state_shift


def _pack(wrapper: LlavaP0, img, query: str):
    inputs = wrapper.encode(img, query)
    return image_to_clip_pixels(img).to(wrapper.device), inputs["input_ids"], inputs.get("attention_mask")


def _eot(x: torch.Tensor, scale_lo: float, scale_hi: float) -> torch.Tensor:
    if scale_lo == 1.0 and scale_hi == 1.0:
        return x
    s = float(torch.empty(1).uniform_(scale_lo, scale_hi))
    if abs(s - 1.0) < 1e-3:
        return x
    x2 = torch.nn.functional.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
    return torch.nn.functional.interpolate(x2, size=x.shape[-2:], mode="bilinear", align_corners=False)


def optimize_universal(
    wrapper: LlavaP0,
    train_items: List[Tuple[Any, Dict[str, Any]]],
    steps: int,
    eps: float,
    alpha: float,
    loss_fn: Callable,
    tv_weight: float,
    eot: Tuple[float, float],
) -> torch.Tensor:
    device = wrapper.device
    delta = torch.zeros(1, 3, 336, 336, device=device)
    n = max(len(train_items), 1)
    for step in tqdm(range(steps), desc="pgd"):
        torch.cuda.empty_cache()
        img, h = train_items[step % n]
        x0, ids, attn = _pack(wrapper, img, h["query"])
        delta = delta.detach().requires_grad_(True)
        x = _eot(torch.clamp(x0 + delta, 0.0, 1.0), eot[0], eot[1])
        pv = normalize_clip(x, device, torch.float16)
        loss = loss_fn(pv, ids, attn) + tv_weight * tv_loss(torch.clamp(x0 + delta, 0, 1))
        grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        del loss, pv, x, x0, ids, attn
        if grad is None:
            raise RuntimeError("attack got no image gradient")
        delta = (delta - alpha * grad.sign()).clamp(-eps, eps)
        del grad
    torch.cuda.empty_cache()
    return delta.detach()


def refusal_loss_fn(wrapper: LlavaP0, refuse_ids, comply_ids):
    def fn(pv, ids, attn):
        logits = wrapper.first_token_logits(pv, ids, attn)
        return margin_from_logits(logits, refuse_ids, comply_ids)

    return fn


def state_loss_fn(wrapper: LlavaP0, layer: int, U: np.ndarray, mu_ref: np.ndarray, mu_jb: np.ndarray):
    u = torch.from_numpy(U.astype(np.float32)).to(wrapper.device)
    mref = torch.from_numpy(mu_ref.astype(np.float32)).to(wrapper.device)
    mjb = torch.from_numpy(mu_jb.astype(np.float32)).to(wrapper.device)

    def fn(pv, ids, attn):
        h = wrapper.hidden_last_user(pv, ids, layer, attn).float()
        z = h @ u
        return torch.norm(z - mjb) - torch.norm(z - mref)

    return fn


def full_match_loss_fn(wrapper: LlavaP0, layer: int, mu_full: np.ndarray):
    target = torch.from_numpy(mu_full.astype(np.float32)).to(wrapper.device)

    def fn(pv, ids, attn):
        h = wrapper.hidden_last_user(pv, ids, layer, attn).float()
        return torch.norm(h - target)

    return fn


def evaluate_delta(
    wrapper: LlavaP0,
    delta: Optional[torch.Tensor],
    items: List[Tuple[Any, Dict[str, Any]]],
    kind: str,
    layer: Optional[int],
    U: Optional[np.ndarray],
    mu_ref: Optional[np.ndarray],
    mu_jb: Optional[np.ndarray],
) -> Dict[str, Any]:
    rows = []
    for img, q in items:
        pv = None
        if delta is not None:
            x0 = image_to_clip_pixels(img).to(wrapper.device)
            pv = normalize_clip(torch.clamp(x0 + delta, 0.0, 1.0), wrapper.device, torch.float16)
        text = wrapper.generate(img, q["query"], pixel_values=pv)
        if kind == "harmful":
            j = harmful_compliance(text, q["topic_keywords"], q["query"])
        else:
            j = benign_utility(text, q["topic_keywords"], q["query"])
        shift = None
        if layer is not None and U is not None and mu_ref is not None and mu_jb is not None:
            hid = wrapper.collect_hidden(img, q["query"], [layer], pixel_values=pv)
            h = hid[f"L{layer}:last_user"].numpy()
            shift = state_shift(h, U, mu_ref, mu_jb)
        rows.append({"query_id": q["id"], "judge": j, "state_shift": shift})
    return {"rows": rows}


def build_train_items(harmful, carriers):
    items = []
    for h in harmful:
        for c in carriers:
            items.append((open_image(c["path"]), h))
    return items
