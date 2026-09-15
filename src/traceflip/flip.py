"""Step 4 — Constrained Flip.

The watershed of the method. "Try to keep the prefix" is not enough; the
prefix must be an explicit, checkable constraint:

    g_i(delta)      = log p(y_i | y_<i, x+delta) - max_{v != y_i} log p(v | ...)
                      >= KEEP_KAPPA            for all i < t*

    g_flip(delta)   = log p(a* | y_<t*, x+delta) - max_{v != a*} log p(v | y_<t*, x+delta)
                      >= FLIP_KAPPA

    min_delta  -g_flip(delta) + lambda * sum_{i<t*} [KEEP_KAPPA - g_i(delta)]_+
    s.t.       ||delta||_inf <= eps

Two things about ``g_flip`` are easy to get wrong and both matter:

* it must also be a **margin over the best competitor**, not over the incumbent
  token. Comparing against the incumbent alone lets the solver "win" against a
  token that is not actually the strongest rival, so ``g_flip`` can be several
  nats positive while the token at the first position is still something else and
  the greedy re-decode never flips. The constraint is satisfied, the decode does
  not change: the classic in-model-feasible-but-invalid cell.
* ``KEEP_KAPPA`` is reported honestly even for position 0. At ``i = 0`` there is
  no prefix to protect, so "position 0 was kept" is not evidence the constraint
  did anything; the log records ``min_prefix_margin = inf`` (no prefix
  positions) rather than a comfortable-looking number.

Explicit PGD/APGD on the constraint violation, not a soft regulariser on a
cross-entropy. ``violations`` returns per-index margins so the caller can log
exactly which prefix position broke (useful because "the constraint held
in-model" and "greedy re-decode reproduced the prefix" are different claims:
the re-execute stage decides that one).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .protocol import (
    ALPHA,
    EPS,
    FLIP_KAPPA,
    INNER_STEPS,
    KEEP_KAPPA,
    LAMBDA_KEEP,
    OUTER_STEPS,
)
from .trace import forward_logits, forward_path_logits, margin_nats


def _sequence_ids(inputs: Dict[str, torch.Tensor], prefix_ids: Sequence[int]) -> torch.Tensor:
    base = inputs["input_ids"]
    if not prefix_ids:
        return base
    extra = torch.tensor(
        [[int(x) for x in prefix_ids]], device=base.device, dtype=base.dtype
    )
    return torch.cat([base, extra], dim=-1)


def _attn(inputs: Dict[str, torch.Tensor], ids: torch.Tensor) -> Optional[torch.Tensor]:
    if "attention_mask" not in inputs:
        return None
    am = inputs["attention_mask"]
    pad = int(ids.shape[-1]) - int(am.shape[-1])
    if pad <= 0:
        return am
    extra = torch.ones(
        am.shape[0], pad, device=am.device, dtype=am.dtype
    )
    return torch.cat([am, extra], dim=-1)


def prefix_margins(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    prefix_ids: Sequence[int],
    eos_exclude: Sequence[int] = (),
) -> torch.Tensor:
    """``g_i`` for every ``i`` in the given prefix, in nats. Shape [n_prefix].

    The competitor set is the **full** vocab (EOS included). Reproduction under
    greedy means "the token is the raw argmax", and greedy takes the raw argmax
    over every id, EOS included. Excluding EOS here would let the constraint hold
    while the decoder actually terminates, i.e. the prefix looks kept in-model
    but is not reproduced. ``eos_exclude`` is accepted for call-site symmetry but
    is deliberately unused: EOS is a real competitor for reproduction.
    """
    prefix = [int(v) for v in prefix_ids]
    if not prefix:
        return torch.zeros(0, device=x.device)
    pv, grid = wrapper.patchify(x)
    # Follow the decoder incrementally.  A one-shot teacher-forced prefill can
    # disagree with Qwen2-VL's cached path at deeper adversarial positions.
    logits = forward_path_logits(
        wrapper,
        input_ids=inputs["input_ids"],
        pixel_values=pv,
        image_grid_thw=grid,
        attention_mask=inputs.get("attention_mask"),
        continuation_ids=prefix,
    )
    out = []
    for i, tok in enumerate(prefix):
        out.append(margin_nats(logits[i], int(tok)))
    return torch.stack(out)


def flip_margin(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    prefix_ids: Sequence[int],
    branch_token: int,
    incumbent: int,
    eos_exclude: Sequence[int] = (),
) -> torch.Tensor:
    """``g_flip`` in nats: log p(a*) - max_{v != a*} log p(v) at the prefix end.

    The competitor is the **best other token over the full vocab**, incumbent or
    not, EOS included. Two easy mistakes are ruled out here:

    * comparing against the incumbent alone, which lets an unreachable branch
      look feasible whenever the incumbent is not the strongest rival;
    * comparing against the best non-EOS token, which lets the branch "win" while
      greedy actually emits EOS. Reproduction is the raw argmax, so EOS is a real
      competitor. ``eos_exclude`` is accepted for call-site symmetry but unused.
    """
    pv, grid = wrapper.patchify(x)
    logits = forward_path_logits(
        wrapper,
        input_ids=inputs["input_ids"],
        pixel_values=pv,
        image_grid_thw=grid,
        attention_mask=inputs.get("attention_mask"),
        continuation_ids=prefix_ids,
    )[-1]
    return margin_nats(logits, int(branch_token))


def constraint_loss(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    delta: torch.Tensor,
    x0: torch.Tensor,
    prefix_ids: Sequence[int],
    branch_token: int,
    incumbent: int,
    keep_kappa: float = KEEP_KAPPA,
    flip_kappa: float = FLIP_KAPPA,
    lam: float = LAMBDA_KEEP,
    eos_exclude: Sequence[int] = (),
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    x = torch.clamp(x0 + delta, 0.0, 1.0)
    g_pref = prefix_margins(wrapper, inputs, x, prefix_ids, eos_exclude=eos_exclude)
    g_flip = flip_margin(
        wrapper, inputs, x, prefix_ids, branch_token, incumbent, eos_exclude=eos_exclude
    )
    keep_viol = torch.clamp(float(keep_kappa) - g_pref, min=0.0)
    flip_viol = torch.clamp(float(flip_kappa) - g_flip, min=0.0)
    loss = -g_flip + float(lam) * keep_viol.sum()
    info = {
        "g_flip": float(g_flip.detach().item()),
        "flip_violation": float(flip_viol.detach().item()),
        "keep_violation": float(keep_viol.detach().sum().item()),
        "n_keep_broken": int((keep_viol.detach() > 0).sum().item()),
        # No prefix positions means the prefix constraint is vacuous, not
        # trivially satisfied. Report it as inf so it cannot be mistaken for
        # evidence that the constraint was doing work.
        "n_prefix_positions": int(g_pref.numel()),
        "min_prefix_margin": float(g_pref.detach().min().item())
        if g_pref.numel()
        else float("inf"),
        "keep_violations": [float(v) for v in keep_viol.detach().tolist()],
    }
    return loss, info


def constrained_flip(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    x0: torch.Tensor,
    delta_init: torch.Tensor,
    prefix_ids: Sequence[int],
    branch_token: int,
    incumbent: int,
    *,
    eps: float = EPS,
    alpha: float = ALPHA,
    outer_steps: int = OUTER_STEPS,
    inner_steps: int = INNER_STEPS,
    keep_kappa: float = KEEP_KAPPA,
    flip_kappa: float = FLIP_KAPPA,
    lam: float = LAMBDA_KEEP,
    tv_weight: float = 0.0,
    track_clip: bool = True,
    eos_exclude: Sequence[int] = (),
) -> Dict[str, Any]:
    """Solve the constrained-flip program by projected sign descent on the violation.

    ``track_clip`` is the ablation switch: when False the iteration never
    restores the prefix, i.e. it degrades to ordinary token-level target PGD.
    """
    delta = delta_init.detach().clone()
    delta = _box(x0, delta, eps)
    history: List[Dict[str, Any]] = []
    n_backward = 0
    effective_lam = float(lam) if track_clip else 0.0
    for _ in range(int(outer_steps)):
        for _ in range(int(inner_steps)):
            delta = delta.detach().requires_grad_(True)
            loss, info = constraint_loss(
                wrapper,
                inputs,
                delta,
                x0,
                prefix_ids,
                branch_token,
                incumbent,
                keep_kappa=keep_kappa,
                flip_kappa=flip_kappa,
                lam=effective_lam,
                eos_exclude=eos_exclude,
            )
            if tv_weight:
                from p0_qwen.vision import tv_loss

                loss = loss + float(tv_weight) * tv_loss(torch.clamp(x0 + delta, 0.0, 1.0))
            grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
            if grad is None:
                history.append({**info, "grad": "none"})
                break
            n_backward += 1
            delta = (delta - float(alpha) * grad.sign()).detach()
            delta = _box(x0, delta, eps)
            if not track_clip:
                info = {**info, "note": "no_keep_lam"}
            history.append(info)
            del grad, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final_loss, final_info = constraint_loss(
        wrapper,
        inputs,
        delta,
        x0,
        prefix_ids,
        branch_token,
        incumbent,
        keep_kappa=keep_kappa,
        flip_kappa=flip_kappa,
        lam=effective_lam,
        eos_exclude=eos_exclude,
    )
    return {
        "delta": delta.detach(),
        "history": history,
        "final": final_info,
        "n_backward": n_backward,
        # Feasible in-model now means the branch token is the best EOS-excluded
        # continuation at the flip position, which is exactly what the greedy
        # re-decode will test. It is still logged separately from ``valid``.
        "feasible_in_model": bool(
            final_info["flip_violation"] <= 0.0
            and (not track_clip or final_info["keep_violation"] <= 0.0)
        ),
    }


def _box(x0: torch.Tensor, delta: torch.Tensor, eps: float) -> torch.Tensor:
    delta = delta.clamp(-float(eps), float(eps))
    return torch.clamp(x0 + delta, 0.0, 1.0) - x0


# --------------------------------------------------- baseline objectives ----
def refusal_margin_objective(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    delta: torch.Tensor,
    x0: torch.Tensor,
    refuse_ids: Sequence[int],
    comply_ids: Sequence[int],
) -> torch.Tensor:
    from p0.model import margin_from_logits

    x = torch.clamp(x0 + delta, 0.0, 1.0)
    pv, grid = wrapper.patchify(x)
    logits = forward_logits(
        wrapper,
        input_ids=inputs["input_ids"],
        pixel_values=pv,
        image_grid_thw=grid,
        attention_mask=inputs.get("attention_mask"),
        use_cache=False,
    )[0, -1]
    return margin_from_logits(logits, list(refuse_ids), list(comply_ids))


def targeted_prefix_objective(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    delta: torch.Tensor,
    x0: torch.Tensor,
    prefix_ids: Sequence[int],
) -> torch.Tensor:
    """Cross-entropy on a preset prefix. Baseline 2/3; uses a target answer."""
    x = torch.clamp(x0 + delta, 0.0, 1.0)
    pv, grid = wrapper.patchify(x)
    logits = forward_path_logits(
        wrapper,
        input_ids=inputs["input_ids"],
        pixel_values=pv,
        image_grid_thw=grid,
        attention_mask=inputs.get("attention_mask"),
        continuation_ids=prefix_ids,
    )
    tok_ids = torch.tensor([int(x) for x in prefix_ids], device=logits.device)
    sel = logits[: len(prefix_ids)]
    return F.cross_entropy(sel, tok_ids)


def generic_pgd(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    x0: torch.Tensor,
    delta_init: torch.Tensor,
    objective,
    *,
    steps: int,
    eps: float = EPS,
    alpha: float = ALPHA,
    tv_weight: float = 0.0,
) -> Dict[str, Any]:
    """Budget-matched token/graph PGD for the baselines."""
    delta = _box(x0, delta_init.detach(), eps)
    n_backward = 0
    for _ in range(int(steps)):
        delta = delta.detach().requires_grad_(True)
        loss = objective(wrapper, inputs, delta, x0)
        if tv_weight:
            from p0_qwen.vision import tv_loss

            loss = loss + float(tv_weight) * tv_loss(torch.clamp(x0 + delta, 0.0, 1.0))
        grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        if grad is None:
            break
        n_backward += 1
        delta = _box(x0, (delta - float(alpha) * grad.sign()).detach(), eps)
        del grad, loss
    return {"delta": delta.detach(), "n_backward": n_backward}