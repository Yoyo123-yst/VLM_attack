"""Step 2 — Counterfactual Probe, and Step 3 — Select (value x reachability).

Two distinct quantities, deliberately kept apart:

``V_t(a)``  counterfactual **value**: how much forcing token ``a`` at position
            ``t`` and rolling out ``H`` further greedy steps moves a cheap,
            frozen proxy toward RHC. Value is a *selector only*; it never
            decides success.

``C_t(a)``  **reachability cost**: the L-inf budget fraction the pixel
            perturbation would have to spend to flip that branch in a
            first-order model,
            ``C = [-m]_+ / (eps * ||grad_delta m||_1 + eps0)``.

            The plan writes the denominator as ``||grad||_1 + eps0``. Dividing
            by ``eps`` makes ``C`` dimensionless (0 = branch already flippable,
            1 = needs the entire budget, >1 = linearly infeasible) so that
            ``S = V / (C + eps0)`` mixes terms with sane units. The ordering of
            candidates is unchanged; only interpretability differs.

``S_t(a) = max(V,0) / (C + eps0)`` and only branches that clear a local
gradient check are kept.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from p0.model import COMPLY_WORDS, REFUSE_WORDS, margin_from_logits, token_id_list

from .protocol import (
    EPS,
    EPS0,
    GRAD_CHECK_FRAC,
    GRAD_CHECK_MIN,
    PROBE_HORIZON,
)
from .trace import _eos_ids, forward_path_logits, margin_nats


# ------------------------------------------------------------------ proxy ---
def refusal_margin_nats(logits: torch.Tensor, refuse_ids: Sequence[int], comply_ids: Sequence[int]) -> float:
    """Frozen refusal-minus-compliance scalar (same probe used by CR-0 / OtW)."""
    return float(margin_from_logits(logits, list(refuse_ids), list(comply_ids)).detach().item())


class ValueProxy:
    """Teacher-forced value proxy ``J`` over a token sequence.

    ``J(seq) = - mean_i refusal_margin(logits at position i)``, i.e. higher means
    the sequence sits deeper in the compliance region. It is computed from the
    model's own next-token distribution over the given tokens, so it is
    deterministic, query-agnostic and free of any harmful-text judge.
    """

    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper
        self.refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
        self.comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)

    @torch.no_grad()
    def score(
        self,
        inputs: Dict[str, torch.Tensor],
        continuation_ids: Sequence[int],
    ) -> float:
        cont = [int(x) for x in continuation_ids]
        if not cont:
            return 0.0
        logits = forward_path_logits(
            self.wrapper,
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs.get("attention_mask"),
            continuation_ids=cont,
        )
        vals = []
        for i in range(len(cont)):
            vals.append(refusal_margin_nats(logits[i], self.refuse_ids, self.comply_ids))
        if not vals:
            return 0.0
        return -float(sum(vals) / len(vals))


# ------------------------------------------------------------------ probe ---
@torch.no_grad()
def force_and_rollout(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    prefix_ids: Sequence[int],
    forced_token: int,
    horizon: int,
) -> Dict[str, Any]:
    """Teacher-force ``prefix_ids`` + ``forced_token``, then greedy ``horizon`` steps."""
    base = inputs["input_ids"]
    seq = [int(x) for x in prefix_ids] + [int(forced_token)]
    cur = base
    model_kwargs: Dict[str, Any] = {k: v for k, v in inputs.items() if k != "input_ids"}
    model_kwargs["use_cache"] = True
    cache_position = torch.arange(base.shape[1], device=base.device)
    # Qwen2-VL indexes cache_position before any update, so seed it here exactly
    # as QwenP0._greedy does.
    model_kwargs["cache_position"] = cache_position
    # Seed the KV cache explicitly. On transformers 4.46.3 a bare
    # ``use_cache=True`` is not enough for Qwen2-VL: with ``past_key_values=None``
    # the decoder never returns a cache, every step re-feeds the whole sequence,
    # ``prepare_inputs_for_generation`` drops ``pixel_values`` once
    # ``cache_position[0] != 0``, and the M-RoPE positions shift by
    # ``cache_position[0] + rope_deltas``. The rollout would then measure a
    # different function than the one the branch value is supposed to predict
    # (measured divergence: the same cell decodes to a different token path).
    # An empty DynamicCache keeps this rollout on the decoder's real path.
    if model_kwargs.get("past_key_values") is None:
        from transformers.cache_utils import DynamicCache

        model_kwargs["past_key_values"] = DynamicCache()
    n_forward = 0
    out = None
    # Reproduce the forced prefix incrementally.  Feeding prompt+prefix as one
    # prefill can disagree with the real cached decode at deep positions.
    for token in seq:
        model_inputs = wrapper.model.prepare_inputs_for_generation(cur, **model_kwargs)
        out = wrapper.model(**model_inputs, return_dict=True)
        n_forward += 1
        cur = torch.cat(
            [cur, torch.tensor([[token]], device=cur.device, dtype=cur.dtype)], dim=-1
        )
        model_kwargs = wrapper.model._update_model_kwargs_for_generation(
            out, model_kwargs, is_encoder_decoder=False
        )
        if "cache_position" not in model_kwargs:
            model_kwargs["cache_position"] = cache_position[-1:] + 1
        cache_position = model_kwargs["cache_position"]
    # Consume the forced token so the latest logits predict the first rollout
    # token rather than the forced token itself.
    model_inputs = wrapper.model.prepare_inputs_for_generation(cur, **model_kwargs)
    out = wrapper.model(**model_inputs, return_dict=True)
    n_forward += 1
    eos_set = _eos_ids(wrapper)
    cont: List[int] = []
    margins: List[float] = []
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    for rollout_index in range(int(horizon)):
        assert out is not None
        logits = out.logits[:, -1].float()[0]
        nxt = int(logits.argmax().item())
        margins.append(refusal_margin_nats(logits, refuse_ids, comply_ids))
        cont.append(nxt)
        cur = torch.cat(
            [cur, torch.tensor([[nxt]], device=cur.device, dtype=cur.dtype)], dim=-1
        )
        if nxt in eos_set:
            break
        if rollout_index + 1 < int(horizon):
            model_kwargs = wrapper.model._update_model_kwargs_for_generation(
                out, model_kwargs, is_encoder_decoder=False
            )
            if "cache_position" not in model_kwargs:
                model_kwargs["cache_position"] = cache_position[-1:] + 1
            cache_position = model_kwargs["cache_position"]
            model_inputs = wrapper.model.prepare_inputs_for_generation(cur, **model_kwargs)
            out = wrapper.model(**model_inputs, return_dict=True)
            n_forward += 1
    return {
        "forced_token": int(forced_token),
        "prefix_ids": [int(x) for x in prefix_ids],
        "continuation_ids": cont,
        "n_generated": len(cont),
        "n_forward": int(n_forward),
        "refusal_margins": margins,
        "eos_hit": bool(cont and cont[-1] in eos_set),
    }


def probe_value(
    proxy: ValueProxy,
    inputs: Dict[str, torch.Tensor],
    rollout: Dict[str, Any],
    baseline_j: float,
) -> float:
    seq = list(rollout["prefix_ids"]) + [int(rollout["forced_token"])] + list(
        rollout["continuation_ids"]
    )
    return float(proxy.score(inputs, seq) - float(baseline_j))


# ----------------------------------------------------------- reachability ---
def branch_grads(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    prefix_ids: Sequence[int],
    x01: torch.Tensor,
    targets: Sequence[int],
    incumbent: int,
    chunk: int = 4,
    eos_exclude: Sequence[int] = (),
) -> Dict[int, torch.Tensor]:
    """``grad_delta (logp(a) - max_{v != a} logp(v))`` for each ``a``.

    Teacher-forces ``prefix_ids`` on the *differentiable* pixels ``x01`` and
    backprops through the vision tower. The competitor is the best other token
    over the full vocab (matching ``flip.flip_margin``), incumbent or not.
    ``eos_exclude`` is accepted for call-site symmetry but unused: EOS is a real
    competitor for the greedy argmax. Gradients for several ``a`` share one
    forward pass; ``chunk`` controls how many simultaneous backward graphs are
    held at once to stay inside 32 GB.
    """
    grads: Dict[int, torch.Tensor] = {}
    base = inputs["input_ids"]
    n_target = len(targets)
    for start in range(0, n_target, max(1, int(chunk))):
        part = [int(a) for a in list(targets)[start : start + int(chunk)]]
        delta = torch.zeros_like(x01, requires_grad=True)
        x = torch.clamp(x01 + delta, 0.0, 1.0)
        pv, grid = wrapper.patchify(x)
        logits = forward_path_logits(
            wrapper,
            input_ids=base,
            pixel_values=pv,
            image_grid_thw=grid,
            attention_mask=inputs.get("attention_mask"),
            continuation_ids=prefix_ids,
        )[-1].float()
        lp = F.log_softmax(logits, dim=-1)
        # The competitor must be the best *other* token, matching the margin the
        # solver optimises (flip.flip_margin) and the ``margin_nats`` recorded by
        # the trace. Grading against the incumbent instead would estimate the
        # reachability of a different quantity than the one the flip step tries
        # to satisfy. ``max_{v != a}`` is top1 unless a *is* top1, in which case
        # it is top2; the argmax indices are detached, which is the correct
        # subgradient of a max.
        with torch.no_grad():
            order = torch.argsort(lp, descending=True)
            ranked = [int(v) for v in order.tolist()]
            top1 = ranked[0]
            top2 = ranked[1] if len(ranked) > 1 else ranked[0]
        outs = [lp[a] - lp[top2 if int(a) == top1 else top1] for a in part]
        for a, val in zip(part, outs):
            g = torch.autograd.grad(val, delta, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                grads[a] = g.detach()
        del logits, lp
    return grads


def reachability_cost(grad: torch.Tensor, flip_margin: float, eps: float = EPS) -> float:
    """``C = [-m]_+ / (eps * ||grad||_1 + eps0)``; 0 if the branch already wins."""
    deficit = max(0.0, -float(flip_margin))
    if deficit <= 0.0:
        return 0.0
    g1 = float(grad.detach().float().abs().sum().item())
    return float(deficit / (float(eps) * g1 + EPS0))


def grad_ok(grad: Optional[torch.Tensor], eps: float = EPS) -> bool:
    """Local gradient check. A branch with a dead pixel gradient is unreachable.

    Threshold is relative to the number of pixels: a mean |grad| below
    ``GRAD_CHECK_FRAC * eps`` cannot move the branch inside the budget even if
    every pixel moved in the gradient's sign direction for one step.
    """
    if grad is None:
        return False
    g = grad.detach().float()
    if not torch.isfinite(g).all():
        return False
    n = float(g.numel())
    if n <= 0:
        return False
    if float(g.norm().item()) < GRAD_CHECK_MIN:
        return False
    mean_abs = float(g.abs().mean().item())
    return mean_abs > float(GRAD_CHECK_FRAC) * float(eps)


def select_branch(
    candidates: Sequence[Dict[str, Any]],
    eps0: float = EPS0,
) -> Optional[Dict[str, Any]]:
    """``argmax S = max(V,0) / (C + eps0)`` over gradient-checked candidates.

    Ties are resolved by ``(t, a)`` order, never by dict insertion, so the
    selection is reproducible.
    """
    scored: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for c in candidates:
        if not c.get("grad_ok"):
            continue
        v = max(float(c.get("value") or 0.0), 0.0)
        cost = float(c.get("cost") or 0.0)
        s = v / (cost + float(eps0))
        c = dict(c)
        c["score"] = float(s)
        scored.append((float(s), int(c["t"]), int(c["token"]), c))
    if not scored:
        return None
    scored.sort(key=lambda z: (-z[0], z[1], z[2]))
    return scored[0][3]


# ---------------------------------------------------------------- helpers ---
def branch_table(
    wrapper,
    proxy: ValueProxy,
    inputs: Dict[str, torch.Tensor],
    x01: torch.Tensor,
    trace: Dict[str, Any],
    positions: Sequence[int],
    topk: int,
    horizon: int,
    baseline_j: float,
    max_rollouts: Optional[int] = None,
    ledger: Any = None,
    max_backward: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Enumerate ``(t, a)`` for the first ``positions`` steps and score them.

    ``max_backward`` is an *additional budget for this call*, not an absolute
    ceiling on the shared cell ledger.  Budget is charged where the compute
    happens and candidate chunks are truncated before execution, so the probe
    cannot overshoot its allowance.
    """
    token_ids = [int(x) for x in trace["token_ids"]]
    steps = {int(s["step"]): s for s in trace["steps"]}
    eos_set = _eos_ids(wrapper)
    rows: List[Dict[str, Any]] = []
    used = 0
    start_backward = int(ledger.backward_passes) if ledger is not None else 0

    def backward_left() -> Optional[int]:
        if max_backward is None or ledger is None:
            return None
        spent_here = int(ledger.backward_passes) - start_backward
        return int(max_backward) - spent_here

    position_list = [int(t) for t in positions]
    for position_index, t in enumerate(position_list):
        t = int(t)
        if t >= len(token_ids):
            continue
        step = steps.get(t)
        if not step:
            continue
        incumbent = int(token_ids[t])
        cands = [int(a) for a in step["top_k"] if int(a) != incumbent and int(a) not in eos_set]
        cands = cands[: int(topk)]
        if not cands:
            continue
        prefix = token_ids[:t]
        left = backward_left()
        if left is not None and left <= 0:
            # No gradient budget left for this position; still record the branch
            # as unreachable rather than silently skipping it.
            for a in cands:
                rows.append(
                    {
                        "t": t,
                        "token": int(a),
                        "incumbent": incumbent,
                        "flip_margin_nats": float(step["margin_nats"].get(a, float("-inf"))),
                        "cost": float("inf"),
                        "grad_ok": False,
                        "grad_norm": 0.0,
                        "value": None,
                        "continuation_ids": None,
                        "n_generated": 0,
                        "eos_hit": False,
                        "budget_exhausted": True,
                    }
                )
                used += 1
            continue
        # One autograd call is charged per target.  Truncate before computing,
        # otherwise a final top-k chunk can overshoot both the probe allowance
        # and the cell's hard backward budget.
        if left is None:
            active = cands
        else:
            positions_left = max(1, len(position_list) - position_index)
            # Share a tight probe allowance across branch positions instead of
            # spending an entire top-k chunk at the first position.
            position_quota = max(1, -(-int(left) // positions_left))
            active = cands[:position_quota]
        deferred = cands[len(active) :]
        for a in deferred:
            rows.append(
                {
                    "t": t,
                    "token": int(a),
                    "incumbent": incumbent,
                    "flip_margin_nats": float(step["margin_nats"].get(a, float("-inf"))),
                    "cost": float("inf"),
                    "grad_ok": False,
                    "grad_norm": 0.0,
                    "value": None,
                    "continuation_ids": None,
                    "n_generated": 0,
                    "eos_hit": False,
                    "budget_exhausted": True,
                }
            )
            used += 1
        if not active:
            continue
        grads = branch_grads(
            wrapper,
            inputs,
            prefix,
            x01,
            active,
            incumbent,
            eos_exclude=sorted(eos_set),
        )
        if ledger is not None:
            ledger.add_forward(1)
            ledger.add_backward(len(active))
        for a in active:
            ga = grads.get(a)
            # Flip margin at the prefix end: logp(a) - max_{v != a} logp(v),
            # the same margin the flip solver must satisfy. Already measured
            # EOS-excluded in the trace's ``margin_nats``.
            flip_margin = float(step["margin_nats"].get(a, float("-inf")))
            cost = reachability_cost(ga, flip_margin) if ga is not None else float("inf")
            rows.append(
                {
                    "t": t,
                    "token": int(a),
                    "incumbent": incumbent,
                    "flip_margin_nats": flip_margin,
                    "cost": cost,
                    "grad_ok": grad_ok(ga),
                    "grad_norm": float(ga.detach().float().abs().sum().item()) if ga is not None else 0.0,
                    "value": None,
                    "continuation_ids": None,
                    "n_generated": 0,
                    "eos_hit": False,
                    "budget_exhausted": False,
                }
            )
            used += 1
    # Value pass: rollouts ordered by (cost, t, a) so the truncation is stable.
    if max_rollouts is not None and used > int(max_rollouts):
        rows.sort(key=lambda r: (r["cost"], r["t"], r["token"]))
        rows = rows[: int(max_rollouts)]
    for r in rows:
        # Rollout/value scoring consumes generation and forward compute but no
        # backward pass.  It must still run after the gradient allowance is
        # exactly exhausted; coupling it to ``backward_left`` silently left all
        # values as None and collapsed the value-based selectors.
        if not r.get("grad_ok"):
            continue
        roll = force_and_rollout(
            wrapper,
            inputs,
            token_ids[: int(r["t"])],
            int(r["token"]),
            int(horizon),
        )
        if ledger is not None:
            ledger.add_generation(tokens=int(roll["n_generated"]))
            ledger.add_forward(int(roll.get("n_forward") or 0))
        r["continuation_ids"] = roll["continuation_ids"]
        r["value"] = probe_value(proxy, inputs, roll, baseline_j)
        r["n_generated"] = int(roll["n_generated"])
        r["eos_hit"] = bool(roll["eos_hit"])
    return rows
