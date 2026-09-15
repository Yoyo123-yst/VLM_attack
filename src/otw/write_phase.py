"""Write phase: query-conditioned short-prefix NLL with optional mode keep."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .open_phase import clip_delta
from .score import s_mode


def tokenize_prefix(wrapper, text: str) -> torch.Tensor:
    tok = wrapper.processor.tokenizer
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids.to(device=wrapper.device)


def write_loss_from_nll(
    nll_h: torch.Tensor,
    nll_s: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """L = -logsumexp(-NLL_H) + alpha logsumexp(-NLL_S)."""
    return -torch.logsumexp(-nll_h, dim=0) + float(alpha) * torch.logsumexp(-nll_s, dim=0)


def prefix_token_logp(
    wrapper,
    x01: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_attn: Optional[torch.Tensor],
    grid: torch.Tensor,
    prefix_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-token log-prob of a short continuation after the official prompt. Shape [n_pref]."""
    pv, g = wrapper.patchify(x01)
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.view(1, -1)
    ids = torch.cat([prompt_ids, prefix_ids], dim=-1)
    if prompt_attn is None:
        attn = torch.ones_like(ids)
    else:
        extra = torch.ones(
            prompt_attn.shape[0], prefix_ids.shape[1], device=prompt_attn.device, dtype=prompt_attn.dtype
        )
        attn = torch.cat([prompt_attn, extra], dim=-1)
    kwargs = {
        "pixel_values": pv,
        "input_ids": ids,
        "attention_mask": attn,
        "image_grid_thw": g,
        "use_cache": False,
        "output_hidden_states": False,
    }
    logits = wrapper.model(**kwargs).logits
    n_prompt = int(prompt_ids.shape[-1])
    n_pref = int(prefix_ids.shape[-1])
    pred = logits[:, n_prompt - 1 : n_prompt - 1 + n_pref, :]
    logp = F.log_softmax(pred.float(), dim=-1)
    token_lp = logp.gather(-1, prefix_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp.reshape(-1)


def prefix_nll(
    wrapper,
    x01: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_attn: Optional[torch.Tensor],
    grid: torch.Tensor,
    prefix_ids: torch.Tensor,
) -> torch.Tensor:
    """Mean token NLL of a short continuation after the official prompt."""
    return -prefix_token_logp(wrapper, x01, prompt_ids, prompt_attn, grid, prefix_ids).mean()


def write_loss(
    wrapper,
    x01: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_attn: Optional[torch.Tensor],
    grid: torch.Tensor,
    harmful_ids: Sequence[torch.Tensor],
    safe_ids: Sequence[torch.Tensor],
    alpha: float = 1.0,
) -> torch.Tensor:
    nll_h = torch.stack(
        [prefix_nll(wrapper, x01, prompt_ids, prompt_attn, grid, p) for p in harmful_ids]
    )
    nll_s = torch.stack(
        [prefix_nll(wrapper, x01, prompt_ids, prompt_attn, grid, p) for p in safe_ids]
    )
    return write_loss_from_nll(nll_h, nll_s, alpha=alpha)


def _cosine(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if a is None or b is None:
        return None
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    den = float(x.norm().item() * y.norm().item())
    if den < 1e-12:
        return None
    return float((x @ y).item() / den)


def _nlls_nograd(
    wrapper,
    x01: torch.Tensor,
    prompt_ids,
    prompt_attn,
    grid,
    harmful_ids,
    safe_ids,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        nll_h = torch.stack(
            [prefix_nll(wrapper, x01, prompt_ids, prompt_attn, grid, p) for p in harmful_ids]
        )
        nll_s = torch.stack(
            [prefix_nll(wrapper, x01, prompt_ids, prompt_attn, grid, p) for p in safe_ids]
        )
    return nll_h.detach(), nll_s.detach()


def _write_grad(
    wrapper,
    x0: torch.Tensor,
    delta: torch.Tensor,
    input_ids,
    attention_mask,
    grid,
    harmful_ids,
    safe_ids,
    write_alpha: float,
    w_h: torch.Tensor,
    w_s: torch.Tensor,
) -> torch.Tensor:
    d = delta.detach().requires_grad_(True)
    if not harmful_ids and not safe_ids:
        raise RuntimeError("write-phase empty prototype bank")
    for i, p in enumerate(harmful_ids):
        x = torch.clamp(x0 + d, 0.0, 1.0)
        nll = prefix_nll(wrapper, x, input_ids, attention_mask, grid, p)
        (float(w_h[i].item()) * nll).backward()
        del nll
    for j, p in enumerate(safe_ids):
        x = torch.clamp(x0 + d, 0.0, 1.0)
        nll = prefix_nll(wrapper, x, input_ids, attention_mask, grid, p)
        ((-float(write_alpha) * float(w_s[j].item())) * nll).backward()
        del nll
    grad = d.grad
    if grad is None:
        raise RuntimeError("write-phase PGD got no image gradient")
    if not torch.isfinite(grad).all():
        raise RuntimeError("write-phase PGD got non-finite gradient")
    return grad.detach()


def _mode_grad(
    wrapper,
    x0: torch.Tensor,
    delta: torch.Tensor,
    input_ids,
    attention_mask,
    grid,
    u: np.ndarray,
    tau: Optional[float],
) -> Tuple[torch.Tensor, float]:
    d = delta.detach().requires_grad_(True)
    x = torch.clamp(x0 + d, 0.0, 1.0)
    score = s_mode(wrapper, x, input_ids, attention_mask, grid, u)
    loss = -score
    if tau is not None:
        loss = loss + torch.relu(score.new_tensor(float(tau)) - score)
    grad = torch.autograd.grad(loss, d, allow_unused=True)[0]
    if grad is None:
        raise RuntimeError("write-phase mode constraint got no image gradient")
    return grad.detach(), float(score.detach().item())


def pgd_write(
    wrapper,
    x0: torch.Tensor,
    delta0: torch.Tensor,
    input_ids,
    attention_mask,
    grid,
    u: np.ndarray,
    harmful_ids: Sequence[torch.Tensor],
    safe_ids: Sequence[torch.Tensor],
    steps: int,
    eps: float,
    alpha: float,
    keep_mode: bool = True,
    mode_weight: float = 0.5,
    tau: Optional[float] = None,
    write_alpha: float = 1.0,
) -> Dict[str, Any]:
    delta = clip_delta(x0, delta0.detach().to(device=x0.device, dtype=x0.dtype), eps)
    s_trace: List[float] = []
    w_trace: List[float] = []
    h_trace: List[float] = []
    ssafe_trace: List[float] = []
    cos_trace: List[Optional[float]] = []
    viol = 0
    n_backprop = 0
    tau_val = None if tau is None else float(tau)
    for _ in range(int(steps)):
        x_det = torch.clamp(x0 + delta.detach(), 0.0, 1.0)
        nll_h, nll_s = _nlls_nograd(wrapper, x_det, input_ids, attention_mask, grid, harmful_ids, safe_ids)
        lw = write_loss_from_nll(nll_h, nll_s, alpha=write_alpha)
        w_h = torch.softmax(-nll_h.float(), dim=0)
        w_s = torch.softmax(-nll_s.float(), dim=0)
        g_w = _write_grad(
            wrapper, x0, delta, input_ids, attention_mask, grid, harmful_ids, safe_ids, write_alpha, w_h, w_s
        )
        g_m, score = _mode_grad(wrapper, x0, delta, input_ids, attention_mask, grid, u, tau_val if keep_mode else None)
        n_backprop += 1
        cos = _cosine(g_w, g_m)
        grad = g_w
        if keep_mode:
            grad = g_w + float(mode_weight) * g_m
        cand = clip_delta(x0, delta.detach() - float(alpha) * grad.sign(), eps)
        if keep_mode and tau_val is not None:
            with torch.no_grad():
                s_new = float(
                    s_mode(
                        wrapper, torch.clamp(x0 + cand, 0.0, 1.0), input_ids, attention_mask, grid, u
                    ).item()
                )
            if s_new < tau_val:
                viol += 1
            else:
                delta = cand
                score = s_new
        else:
            delta = cand
            with torch.no_grad():
                score = float(
                    s_mode(
                        wrapper, torch.clamp(x0 + delta, 0.0, 1.0), input_ids, attention_mask, grid, u
                    ).item()
                )
        s_trace.append(score)
        w_trace.append(float(lw.item()))
        h_trace.append(float(nll_h.min().item()))
        ssafe_trace.append(float(nll_s.min().item()))
        cos_trace.append(cos)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "delta": delta.detach(),
        "s_mode_trace": s_trace,
        "write_loss_trace": w_trace,
        "nll_harmful_min_trace": h_trace,
        "nll_safe_min_trace": ssafe_trace,
        "cos_gmode_gwrite": cos_trace,
        "constraint_violations": viol,
        "s_mode": s_trace[-1] if s_trace else None,
        "n_backprop": n_backprop,
    }
