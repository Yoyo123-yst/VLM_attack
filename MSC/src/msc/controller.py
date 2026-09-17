"""Closed-loop switched controller over frozen phrase banks. No probe / POMDP."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from p0.model import COMPLY_WORDS, REFUSE_WORDS, token_id_list
from traceflip.flip import generic_pgd, refusal_margin_objective, targeted_prefix_objective
from traceflip.gate import encode_gate_prefix

from .objective import make_phrase_objective

PHRASE_METHODS = frozenset({"switched", "static_joint"})
RECOVERY_METHODS = frozenset(
    {"switched_keep", "switched_follow", "switched_margin", "switched_prefix"}
)


def resolve_control(method: str, mode: str) -> Dict[str, Any]:
    """Map (method, decode-mode) to an inner optimiser. P0-1 `switched` still skips INVALID."""
    m = str(method)
    s = str(mode)
    if m == "switched" and s == "INVALID":
        return {"inner_method": m, "inner_mode": s, "skip": True, "reason": "p01_invalid_zero_step"}
    if m == "switched_keep":
        return {"inner_method": "switched", "inner_mode": s, "skip": False, "reason": "invalid_uses_static_joint"}
    if m == "switched_follow":
        inner_mode = "FOLLOW" if s == "INVALID" else s
        return {"inner_method": "switched", "inner_mode": inner_mode, "skip": False, "reason": "invalid_push_follow"}
    if m == "switched_margin":
        if s == "INVALID":
            return {"inner_method": "refusal_margin", "inner_mode": s, "skip": False, "reason": "invalid_fallback_margin"}
        return {"inner_method": "switched", "inner_mode": s, "skip": False, "reason": "mode_switch"}
    if m == "switched_prefix":
        if s == "INVALID":
            return {"inner_method": "targeted_prefix", "inner_mode": s, "skip": False, "reason": "invalid_fallback_prefix"}
        return {"inner_method": "switched", "inner_mode": s, "skip": False, "reason": "mode_switch"}
    return {"inner_method": m, "inner_mode": s, "skip": False, "reason": "identity"}


def _baseline_objective(method: str, wrapper, *, generic_prefix: str):
    if method == "refusal_margin":
        refuse = token_id_list(wrapper.processor, REFUSE_WORDS)
        comply = token_id_list(wrapper.processor, COMPLY_WORDS)

        def objective(w, inputs, delta, x0):
            return refusal_margin_objective(w, inputs, delta, x0, refuse, comply)

        return objective
    if method == "targeted_prefix":
        ids = encode_gate_prefix(wrapper.processor, generic_prefix)

        def objective(w, inputs, delta, x0):
            return targeted_prefix_objective(w, inputs, delta, x0, ids)

        return objective
    raise KeyError(method)


def run_controller(
    wrapper,
    inputs: Mapping[str, torch.Tensor],
    x0: torch.Tensor,
    delta: torch.Tensor,
    *,
    method: str,
    mode: str,
    banks: Mapping[str, Sequence[Sequence[int]]],
    steps: int,
    eps: float,
    alpha: float,
    lam: float,
    generic_prefix: str,
) -> Dict[str, Any]:
    if int(steps) <= 0:
        return {
            "delta": delta.detach(),
            "n_backward": 0,
            "skipped": True,
            "control_mode": str(mode),
            "inner_method": str(method),
            "inner_mode": str(mode),
        }
    plan = resolve_control(str(method), str(mode))
    inner_method = str(plan["inner_method"])
    inner_mode = str(plan["inner_mode"])
    if plan["skip"]:
        return {
            "delta": delta.detach(),
            "n_backward": 0,
            "skipped": True,
            "control_mode": str(mode),
            "inner_method": inner_method,
            "inner_mode": inner_mode,
            "skip_reason": plan["reason"],
        }
    if inner_method in PHRASE_METHODS:
        obj = make_phrase_objective(banks, method=inner_method, mode=inner_mode, lam=lam)
    else:
        obj = _baseline_objective(inner_method, wrapper, generic_prefix=generic_prefix)
    done = generic_pgd(
        wrapper,
        inputs,
        x0,
        delta,
        obj,
        steps=int(steps),
        eps=float(eps),
        alpha=float(alpha),
    )
    done["skipped"] = False
    done["control_mode"] = str(mode)
    done["inner_method"] = inner_method
    done["inner_mode"] = inner_mode
    done["skip_reason"] = plan["reason"]
    done["switched"] = str(method).startswith("switched")
    return done
