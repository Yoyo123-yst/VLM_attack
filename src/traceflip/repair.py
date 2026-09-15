"""TraceFlip main loop: Execute -> Probe -> Select -> Constrained Flip
-> Re-execute / Backtrack.

Design decisions worth stating explicitly, because reviewers will ask:

1. **Success is never estimated.** The loop only reads ``core_rhc`` from a full
   greedy decode of the *actual* perturbed image on the unmodified decoder.
   ``V`` and ``S`` are selectors.
2. **A flip is valid only if re-decode reproduces both conditions**
   ``y'_i = y_i for i < t*`` and ``y'_{t*} = a*``. In-model feasibility is
   reported separately (``feasible_in_model``) and never counted.
3. **Budget is honoured**, not reported after the fact: the loop stops inserting
   gradient steps when the backward-pass budget is exhausted, so the
   equal-backward comparison with refusal-margin PGD is exact.
4. **Backtracking** re-selects among the remaining candidates; a rejected flip
   also restores the previous delta, so a bad solve cannot leak into the result.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from pathem.budget import BudgetLedger

from .flip import (
    constrained_flip,
    generic_pgd,
    refusal_margin_objective,
    targeted_prefix_objective,
)
from .probe import (
    ValueProxy,
    branch_table,
    select_branch,
)
from .protocol import (
    ALPHA,
    EPS,
    FLIP_KAPPA,
    INNER_STEPS,
    KEEP_KAPPA,
    LAMBDA_KEEP,
    MAX_NEW_TOKENS,
    MAX_POSITIONS,
    MIN_BRANCH_POSITION,
    OUTER_STEPS,
    PROBE_HORIZON,
    TOPK,
)
from .trace import _eos_ids, execute


# ------------------------------------------------------------- selectors ---
def select_reachability(rows: Sequence[Dict[str, Any]], eps0: float = 1e-3) -> Optional[Dict[str, Any]]:
    return select_branch(rows, eps0=eps0)


def select_value_only(rows: Sequence[Dict[str, Any]], **_: Any) -> Optional[Dict[str, Any]]:
    ok = [r for r in rows if r.get("grad_ok")]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-float(r.get("value") or 0.0), r["t"], r["token"]))[0]


def select_cost_only(rows: Sequence[Dict[str, Any]], **_: Any) -> Optional[Dict[str, Any]]:
    ok = [r for r in rows if r.get("grad_ok")]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (float(r.get("cost") or 0.0), r["t"], r["token"]))[0]


def select_earliest(rows: Sequence[Dict[str, Any]], **_: Any) -> Optional[Dict[str, Any]]:
    """First-token bias control: the top-1 gap branch at the smallest t."""
    ok = [r for r in rows if r.get("grad_ok")]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (r["t"], -float(r.get("value") or 0.0), r["token"]))[0]


def select_random(seed: int) -> Callable[..., Optional[Dict[str, Any]]]:
    def fn(rows: Sequence[Dict[str, Any]], **_: Any) -> Optional[Dict[str, Any]]:
        ok = [r for r in rows if r.get("grad_ok")]
        if not ok:
            return None
        g = torch.Generator().manual_seed(int(seed))
        idx = int(torch.randint(0, len(ok), (1,), generator=g).item())
        return ok[idx]

    return fn


SELECTORS: Dict[str, Callable[..., Optional[Dict[str, Any]]]] = {
    "reachability": select_reachability,
    "value_only": select_value_only,
    "cost_only": select_cost_only,
    "earliest": select_earliest,
}


# --------------------------------------------------------------- helpers ---
def with_pixels(wrapper, base: Dict[str, torch.Tensor], x01: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = dict(base)
    pv, grid = wrapper.patchify(x01)
    out["pixel_values"] = pv
    out["image_grid_thw"] = grid
    return out


def candidate_positions(
    trace: Dict[str, Any],
    max_positions: int = MAX_POSITIONS,
    order: str = "gap_asc",
    min_position: int = MIN_BRANCH_POSITION,
) -> List[int]:
    """Positions to probe, preferring locally unstable non-vacuous prefixes.

    Position zero has no prefix and therefore cannot exercise TraceFlip's
    defining keep constraint.  It is excluded by default instead of allowing
    the method to silently degenerate into a first-token targeted attack.
    """
    steps = list(trace.get("steps") or [])
    if not steps:
        return []
    usable = [
        s for s in steps
        if s["top_k"] and int(s.get("step", -1)) >= int(min_position)
    ]
    if order == "index":
        return [int(s["step"]) for s in usable[: int(max_positions)]]
    usable.sort(key=lambda s: (float(s.get("top1_gap_nats") or 0.0), int(s["step"])))
    # Preserve instability priority.  Sorting these indices back into trajectory
    # order lets a tight probe budget get consumed by the earliest position and
    # recreates the very early-token bias this ranking is meant to avoid.
    return [int(s["step"]) for s in usable[: int(max_positions)]]


def accept_flip(
    new_trace: Dict[str, Any],
    old_trace: Dict[str, Any],
    t: int,
    token: int,
) -> Dict[str, Any]:
    """Validity test for one branch flip. Both conditions are mandatory."""
    old_ids = [int(x) for x in old_trace["token_ids"]]
    new_ids = [int(x) for x in new_trace["token_ids"]]
    prefix_ok = len(new_ids) > t and new_ids[:t] == old_ids[:t]
    branch_ok = len(new_ids) > t and int(new_ids[t]) == int(token)
    return {
        "prefix_kept": bool(prefix_ok),
        "branch_flipped": bool(branch_ok),
        "valid": bool(prefix_ok and branch_ok),
        "new_prefix": new_ids[:t],
        "new_token": int(new_ids[t]) if len(new_ids) > t else None,
    }


# ------------------------------------------------------------------- cell ---
def probe_budget_for(max_backward: Optional[int], probe_frac: float) -> Optional[int]:
    """Gradient budget the exploration stage may spend.

    The fairness claim is *equal backward passes*, so the probe must not eat the
    solver's share: the probe is capped at ``probe_frac`` of the cell budget and
    the remainder is reserved for constrained flips. Without this cap a
    12-position x 9-target probe would consume the whole budget and the method
    could never solve anything while still "using" the same budget as PGD.
    """
    if max_backward is None:
        return None
    return max(0, int(round(float(probe_frac) * int(max_backward))))


def split_backward_budget(
    left: Optional[int],
    outer_steps: int,
    inner_steps: int,
) -> Tuple[int, int]:
    """Split ``left`` remaining backward passes into (outer, inner).

    Instantaneous step count is kept as close to the frozen schedule as the
    remaining budget allows; the product never exceeds ``left``, so the cell can
    never spend more gradients than the PGD baseline was given.
    """
    outer_steps = max(1, int(outer_steps))
    inner_steps = max(1, int(inner_steps))
    if left is None:
        return outer_steps, inner_steps
    left = int(left)
    if left <= 0:
        return 0, 0
    inner = max(1, min(inner_steps, -(-left // outer_steps)))
    outer = max(1, min(outer_steps, left // inner))
    while outer > 1 and outer * inner > left:
        outer -= 1
    if outer * inner > left:
        inner = left
        outer = 1
    return int(outer), int(inner)


def traceflip_cell(
    wrapper,
    item: Dict[str, Any],
    carrier_id: str,
    image,
    x0: torch.Tensor,
    *,
    eps: float = EPS,
    alpha: float = ALPHA,
    max_new_tokens: int = MAX_NEW_TOKENS,
    topk: int = TOPK,
    horizon: int = PROBE_HORIZON,
    max_positions: int = MAX_POSITIONS,
    outer_steps: int = OUTER_STEPS,
    inner_steps: int = INNER_STEPS,
    backtrack: bool = True,
    track_clip: bool = True,
    selector: str = "reachability",
    selector_seed: int = 20260,
    max_backward: Optional[int] = None,
    probe_frac: float = 0.25,
    keep_kappa: float = KEEP_KAPPA,
    flip_kappa: float = FLIP_KAPPA,
    lam: float = LAMBDA_KEEP,
    tv_weight: float = 0.0,
    max_retries: int = 6,
    method_name: Optional[str] = None,
) -> Dict[str, Any]:
    """One TraceFlip cell = one (query, carrier) pair, delta initialised to 0.

    ``max_backward`` caps gradient steps for the *whole cell*, matching the
    step count given to the PGD baselines, so the ASR comparison is
    equal-backward rather than equal-iteration.
    """
    base = wrapper.encode(image, item["query"])
    proxy = ValueProxy(wrapper)
    ledger = BudgetLedger()
    eos_ids = sorted(_eos_ids(wrapper))
    sel = SELECTORS.get(selector) if selector in SELECTORS else select_random(selector_seed)

    t0 = time.time()
    delta_used = torch.zeros_like(x0)
    trace = execute(wrapper, item, x0, image, max_new_tokens=max_new_tokens, top_k=topk)
    ledger.add_generation(tokens=int(trace["n_tokens"]), wall_s=float(trace["wall_seconds"]))
    ledger.add_forward(int(trace["n_tokens"]))

    attempts: List[Dict[str, Any]] = []
    flip_log: List[Dict[str, Any]] = []

    def backward_left() -> Optional[int]:
        if max_backward is None:
            return None
        return int(max_backward) - int(ledger.backward_passes)

    def probe_budget() -> Optional[int]:
        return probe_budget_for(max_backward, probe_frac)

    probe_spent = 0

    if not trace["core_rhc"]:
        for outer in range(int(outer_steps)):
            if trace["core_rhc"]:
                break
            x_cur = torch.clamp(x0 + delta_used, 0.0, 1.0)
            inputs = with_pixels(wrapper, base, x_cur)
            baseline_j = proxy.score(inputs, trace["token_ids"])
            ledger.add_forward(1)
            positions = candidate_positions(trace, max_positions=max_positions)
            if not positions:
                break
            rounds_left = max(1, int(outer_steps) - int(outer))
            total_probe_cap = probe_budget()
            if total_probe_cap is None:
                probe_allowance = None
            else:
                probe_remaining = max(0, int(total_probe_cap) - int(probe_spent))
                # Spread the cell-wide probe allowance across repair rounds.
                # Giving the full 25% to round zero makes later trajectories
                # impossible to probe and turns the repair loop into one shot.
                probe_allowance = min(
                    probe_remaining,
                    -(-probe_remaining // rounds_left),
                    max(0, int(backward_left() or 0)),
                )
            before_probe = int(ledger.backward_passes)
            rows = branch_table(
                wrapper,
                proxy,
                inputs,
                x_cur,
                trace,
                positions,
                topk=topk,
                horizon=horizon,
                baseline_j=baseline_j,
                max_rollouts=max(4, 3 * int(topk)),
                ledger=ledger,
                max_backward=probe_allowance,
            )
            probe_spent += int(ledger.backward_passes) - before_probe
            if not rows:
                break
            tried: set[Tuple[int, int]] = set()
            accepted_any = False
            while True:
                pool = [
                    r
                    for r in rows
                    if (int(r["t"]), int(r["token"])) not in tried and r.get("grad_ok")
                ]
                if not pool:
                    break
                if len(tried) >= int(max_retries):
                    break
                cand = sel(pool, eps0=1e-3)
                if cand is None:
                    break
                t_star = int(cand["t"])
                a_star = int(cand["token"])
                tried.add((t_star, a_star))
                left = backward_left()
                if left is None:
                    inner_eff = int(inner_steps)
                else:
                    # One repair round performs at most INNER_STEPS updates.
                    # The former nested OUTER_STEPS x INNER_STEPS solve spent
                    # nearly the whole cell budget on the first candidate.
                    inner_eff = min(
                        int(inner_steps),
                        max(0, int(left) // max(1, rounds_left)),
                    )
                if inner_eff <= 0:
                    break
                prefix_ids = [int(v) for v in trace["token_ids"][:t_star]]
                incumbent = int(cand["incumbent"])
                solved = constrained_flip(
                    wrapper,
                    inputs,
                    x0,
                    delta_used,
                    prefix_ids,
                    a_star,
                    incumbent,
                    eps=eps,
                    alpha=alpha,
                    outer_steps=1,
                    inner_steps=inner_eff,
                    keep_kappa=keep_kappa,
                    flip_kappa=flip_kappa,
                    lam=lam,
                    tv_weight=tv_weight,
                    track_clip=bool(track_clip),
                    eos_exclude=eos_ids,
                )
                ledger.add_backward(int(solved["n_backward"]))
                x_new = torch.clamp(x0 + solved["delta"], 0.0, 1.0)
                new_trace = execute(
                    wrapper, item, x_new, image, max_new_tokens=max_new_tokens, top_k=topk
                )
                ledger.add_generation(
                    tokens=int(new_trace["n_tokens"]), wall_s=float(new_trace["wall_seconds"])
                )
                ledger.add_forward(int(new_trace["n_tokens"]))
                check = accept_flip(new_trace, trace, t_star, a_star)
                j_new = proxy.score(with_pixels(wrapper, base, x_new), new_trace["token_ids"])
                ledger.add_forward(1)
                improved = float(j_new) >= float(baseline_j) - 1e-9
                accepted = bool(check["valid"] and improved)
                attempts.append(
                    {
                        "outer": int(outer),
                        "t": t_star,
                        "token": a_star,
                        "incumbent": incumbent,
                        "score": float(cand.get("score") or 0.0),
                        "value": float(cand.get("value") or 0.0),
                        "cost": float(cand.get("cost") or 0.0),
                        "flip_margin_nats": float(cand.get("flip_margin_nats") or 0.0),
                        "feasible_in_model": bool(solved["feasible_in_model"]),
                        "g_flip_final": float(solved["final"]["g_flip"]),
                        "min_prefix_margin": float(solved["final"]["min_prefix_margin"]),
                        "n_prefix_positions": int(solved["final"].get("n_prefix_positions", 0)),
                        "n_keep_broken": int(solved["final"]["n_keep_broken"]),
                        "prefix_kept": bool(check["prefix_kept"]),
                        "branch_flipped": bool(check["branch_flipped"]),
                        "valid": bool(check["valid"]),
                        "j_old": float(baseline_j),
                        "j_new": float(j_new),
                        "improved": bool(improved),
                        "accepted": accepted,
                        "accepted_despite_value": bool(check["valid"] and not improved),
                        "n_backward": int(solved["n_backward"]),
                        "terminal_label": new_trace["terminal_label"],
                        "core_rhc": bool(new_trace["core_rhc"]),
                        "delta_linf": float(solved["delta"].abs().max().item()),
                    }
                )
                if accepted:
                    delta_used = solved["delta"].detach()
                    trace = new_trace
                    accepted_any = True
                    flip_log.append(attempts[-1])
                    break
                if not backtrack:
                    break
            if not accepted_any:
                # No admissible candidate this round: keep the delta we have and
                # stop churning rather than burn budget on rejections.
                break

    wins = [a for a in flip_log if a["accepted"]]
    # Report the caller's method name verbatim. Deriving it from ``track_clip``
    # alone silently collapsed every selector / backtrack ablation onto the
    # plain "traceflip" label, which would make the G4-G6 ablations uncomputable.
    resolved_method = method_name or ("traceflip" if track_clip else "traceflip_no_prefix")
    return {
        "method": resolved_method,
        "selector": selector,
        "query_id": item["id"],
        "carrier_id": carrier_id,
        "eps": float(eps),
        "decode": "greedy",
        "delta": delta_used.detach(),
        "delta_linf": float(delta_used.detach().abs().max().item()),
        "trace": trace,
        "terminal_label": trace["terminal_label"],
        "core_rhc": bool(trace["core_rhc"]),
        "core_safe_answer": bool(trace["core_safe_answer"]),
        "response_mode": trace["response_mode"],
        "safety": trace["safety"],
        "grounding": trace["grounding"],
        "quality": trace["quality"],
        "chars": int(trace["chars"]),
        "n_attempts": len(attempts),
        "n_valid_flips": len(wins),
        "bfr": (len(wins) / len(attempts)) if attempts else 0.0,
        "attempts": attempts,
        "budget": ledger.as_dict(),
        "wall_seconds": float(time.time() - t0),
    }


# ------------------------------------------------------------- baselines ---
def baseline_cell(
    wrapper,
    item: Dict[str, Any],
    carrier_id: str,
    image,
    x0: torch.Tensor,
    method: str,
    *,
    eps: float = EPS,
    alpha: float = ALPHA,
    steps: int = 40,
    max_new_tokens: int = 96,
    tv_weight: float = 0.0,
    prefix_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Budget-matched baselines: refusal-margin PGD and preset-prefix PGD."""
    from p0.model import COMPLY_WORDS, REFUSE_WORDS, token_id_list

    base = wrapper.encode(image, item["query"])
    ledger = BudgetLedger()
    if method == "refusal_margin_pgd":
        refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
        comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)

        def objective(w, inputs, delta, x0_):
            return refusal_margin_objective(w, inputs, delta, x0_, refuse_ids, comply_ids)

    elif method == "targeted_prefix_pgd":
        if not prefix_ids:
            raise ValueError("targeted_prefix_pgd needs prefix_ids")
        ids = list(prefix_ids)

        def objective(w, inputs, delta, x0_):
            return targeted_prefix_objective(w, inputs, delta, x0_, ids)

    else:
        raise KeyError(method)
    done = generic_pgd(
        wrapper,
        base,
        x0,
        torch.zeros_like(x0),
        objective,
        steps=steps,
        eps=eps,
        alpha=alpha,
        tv_weight=tv_weight,
    )
    ledger.add_backward(int(done["n_backward"]))
    x_new = torch.clamp(x0 + done["delta"], 0.0, 1.0)
    trace = execute(wrapper, item, x_new, image, max_new_tokens=max_new_tokens, top_k=TOPK)
    ledger.add_generation(tokens=int(trace["n_tokens"]), wall_s=float(trace["wall_seconds"]))
    ledger.add_forward(int(trace["n_tokens"]))
    return {
        "method": method,
        "query_id": item["id"],
        "carrier_id": carrier_id,
        "eps": float(eps),
        "delta": done["delta"].detach(),
        "delta_linf": float(done["delta"].detach().abs().max().item()),
        "trace": trace,
        "terminal_label": trace["terminal_label"],
        "core_rhc": bool(trace["core_rhc"]),
        "core_safe_answer": bool(trace["core_safe_answer"]),
        "response_mode": trace["response_mode"],
        "safety": trace["safety"],
        "grounding": trace["grounding"],
        "quality": trace["quality"],
        "chars": int(trace["chars"]),
        "budget": ledger.as_dict(),
    }


def clean_cell(
    wrapper,
    item: Dict[str, Any],
    carrier_id: str,
    image,
    x0: torch.Tensor,
    max_new_tokens: int = 96,
) -> Dict[str, Any]:
    ledger = BudgetLedger()
    trace = execute(wrapper, item, x0, image, max_new_tokens=max_new_tokens, top_k=TOPK)
    ledger.add_generation(tokens=int(trace["n_tokens"]), wall_s=float(trace["wall_seconds"]))
    ledger.add_forward(int(trace["n_tokens"]))
    return {
        "method": "clean",
        "query_id": item["id"],
        "carrier_id": carrier_id,
        "eps": 0.0,
        "delta_linf": 0.0,
        "trace": trace,
        "terminal_label": trace["terminal_label"],
        "core_rhc": bool(trace["core_rhc"]),
        "core_safe_answer": bool(trace["core_safe_answer"]),
        "response_mode": trace["response_mode"],
        "safety": trace["safety"],
        "grounding": trace["grounding"],
        "quality": trace["quality"],
        "chars": int(trace["chars"]),
        "budget": ledger.as_dict(),
    }
