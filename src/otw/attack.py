"""OtW loop and same-budget baselines for feasibility."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from n1.pilot import pgd_restart
from p0.model import COMPLY_WORDS, REFUSE_WORDS, margin_from_logits, token_id_list

from .gate import decode_axes, is_answer
from .open_phase import clip_delta, pgd_open
from .prototypes import harmful_prototypes, safe_prototypes
from .write_phase import pgd_write, tokenize_prefix


def random_delta(x0: torch.Tensor, eps: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device=x0.device)
    g.manual_seed(int(seed))
    delta = (torch.rand(x0.shape, generator=g, device=x0.device) * 2.0 - 1.0) * eps
    delta = delta.clamp(-eps, eps)
    return torch.clamp(x0 + delta, 0.0, 1.0) - x0


def pack(wrapper, image, question: str):
    x0 = wrapper.image_to_x01(image)
    enc = wrapper.encode(image, question)
    return x0, enc["input_ids"], enc.get("attention_mask"), enc.get("image_grid_thw")


def prefix_bank(wrapper, qid: str):
    h_ids = [tokenize_prefix(wrapper, t) for t in harmful_prototypes(qid)]
    s_ids = [tokenize_prefix(wrapper, t) for t in safe_prototypes(qid)]
    return h_ids, s_ids


def run_open_only(
    wrapper,
    image,
    question,
    u,
    steps,
    eps,
    alpha,
    seed: int,
    delta0=None,
    init: str = "random",
) -> Dict[str, Any]:
    x0, ids, attn, grid = pack(wrapper, image, question)
    out = pgd_open(wrapper, x0, ids, attn, grid, u, steps, eps, alpha, seed, init=init, delta0=delta0)
    return {"delta": out["delta"], "open": out, "write": None, "switched": False, "n_backprop": out["n_backprop"]}


def run_write_on_delta(
    wrapper,
    image,
    question: str,
    qid: str,
    u: np.ndarray,
    delta0: torch.Tensor,
    write_steps: int,
    eps: float,
    alpha: float,
    keep_mode: bool,
    tau: Optional[float] = None,
) -> Dict[str, Any]:
    x0, ids, attn, grid = pack(wrapper, image, question)
    h_ids, s_ids = prefix_bank(wrapper, qid)
    write = pgd_write(
        wrapper,
        x0,
        delta0,
        ids,
        attn,
        grid,
        u,
        h_ids,
        s_ids,
        write_steps,
        eps,
        alpha,
        keep_mode=keep_mode,
        tau=tau,
    )
    return {"delta": write["delta"], "open": None, "write": write, "n_backprop": write["n_backprop"]}


def run_otw(
    wrapper,
    image,
    question: str,
    qid: str,
    keywords,
    u: np.ndarray,
    open_steps: int,
    write_steps: int,
    eps: float,
    alpha: float,
    seed: int,
    keep_mode: bool = True,
    max_new_tokens: Optional[int] = None,
    fill_remaining_with_open: bool = True,
) -> Dict[str, Any]:
    x0, ids, attn, grid = pack(wrapper, image, question)
    opened = pgd_open(wrapper, x0, ids, attn, grid, u, open_steps, eps, alpha, seed)
    x_open = torch.clamp(x0 + opened["delta"], 0.0, 1.0)
    row_open = decode_axes(wrapper, image, question, keywords, x01=x_open, max_new_tokens=max_new_tokens)
    switched = is_answer(row_open)
    write = None
    delta = opened["delta"]
    n_bp = int(opened["n_backprop"])
    if switched and write_steps > 0:
        h_ids, s_ids = prefix_bank(wrapper, qid)
        tau = opened.get("s_mode")
        write = pgd_write(
            wrapper,
            x0,
            delta,
            ids,
            attn,
            grid,
            u,
            h_ids,
            s_ids,
            write_steps,
            eps,
            alpha,
            keep_mode=keep_mode,
            tau=tau,
        )
        delta = write["delta"]
        n_bp += int(write["n_backprop"])
    elif (not switched) and fill_remaining_with_open and write_steps > 0:
        more = pgd_open(
            wrapper, x0, ids, attn, grid, u, write_steps, eps, alpha, seed, delta0=delta, init="zero"
        )
        delta = more["delta"]
        opened["s_mode_trace"] = list(opened.get("s_mode_trace") or []) + list(more.get("s_mode_trace") or [])[1:]
        opened["s_mode"] = more.get("s_mode")
        opened["n_backprop"] = n_bp + int(more["n_backprop"])
        n_bp = int(opened["n_backprop"])
    return {
        "delta": delta,
        "open": opened,
        "write": write,
        "switched": switched,
        "open_decode": row_open,
        "n_backprop": n_bp,
    }


def run_joint_margin(wrapper, image, question, steps, eps, alpha, seed: int) -> Dict[str, Any]:
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    delta = pgd_restart(wrapper, image, question, steps, eps, alpha, refuse_ids, comply_ids, seed)
    return {"delta": delta, "open": None, "write": None, "switched": None, "n_backprop": int(steps)}


def run_joint_from_delta(
    wrapper,
    image,
    question: str,
    delta0: torch.Tensor,
    steps: int,
    eps: float,
    alpha: float,
) -> Dict[str, Any]:
    """Continue first-token refusal-margin PGD from an existing delta."""
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    x0 = wrapper.image_to_x01(image)
    delta = clip_delta(x0, delta0.detach().to(device=x0.device, dtype=x0.dtype), eps)
    inputs = wrapper.encode(image, question)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    n_bp = 0
    for _ in range(int(steps)):
        delta = delta.detach().requires_grad_(True)
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        pv, grid = wrapper.patchify(x)
        logits = wrapper.first_token_logits(pv, input_ids, attn, grid)
        loss = margin_from_logits(logits, refuse_ids, comply_ids)
        grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        if grad is None:
            raise RuntimeError("joint-from-delta PGD got no image gradient")
        delta = clip_delta(x0, delta.detach() - float(alpha) * grad.sign(), eps)
        n_bp += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"delta": delta.detach(), "n_backprop": n_bp}


def apply_delta(x0: torch.Tensor, delta: Optional[torch.Tensor]) -> torch.Tensor:
    if delta is None:
        return x0
    return torch.clamp(x0 + delta, 0.0, 1.0)
