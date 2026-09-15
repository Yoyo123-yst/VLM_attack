"""Step 1 — Concrete Execute.

Decode greedily from the current image ``x + delta`` and record the actual
token path plus the quantities the later stages need:

* token ids of the full trajectory, with ``<eos>`` handling identical to the
  frozen wrapper,
* per-position top-``K`` alternatives and their **nats** margins,
* local stability (top-1 gap) so an unstable branch can be recognised,
* the four-axis terminal label of the decoded text.

Nothing here optimises. This module is the "concrete execution" half of the
concrete/counterfactual pair.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from pathem.labels import label_text

from .protocol import MAX_NEW_TOKENS


def _eos_ids(wrapper) -> set[int]:
    gc = getattr(wrapper.model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None)
    if eos is None:
        return {int(wrapper.processor.tokenizer.eos_token_id)}
    if isinstance(eos, (list, tuple)):
        return {int(x) for x in eos}
    return {int(eos)}


def margin_nats(logits: torch.Tensor, target: int, exclude: Sequence[int] = ()) -> torch.Tensor:
    """``log p(target) - max_{v != target} log p(v)``, in nats.

    This is ``g`` from the plan, with ``z`` = logits (so it is a log-prob gap).
    The competitor set is *everything except the target itself*: the quantity is
    positive exactly when ``target`` is the argmax among the allowed tokens, and
    its magnitude is the buffer by which it wins. Comparing against a competitor
    set that still contains the target would cap the value at 0 and make a
    strictly-positive threshold such as ``KEEP_KAPPA`` unsatisfiable.

    `exclude` removes structural ids (e.g. the EOS family) from the competitor
    set, because "flipping to EOS" is a termination branch, not a content one.
    """
    lp = F.log_softmax(logits.float(), dim=-1)
    keep = torch.ones_like(lp, dtype=torch.bool)
    if exclude:
        # Ignore ids outside the vocab so a mismatched EOS table cannot crash the
        # trace (harmless on the real model, but keeps stubs and sliced logits safe).
        valid = [int(e) for e in exclude if 0 <= int(e) < lp.shape[-1]]
        if valid:
            keep[valid] = False
    # The competitor set is v != target; the target never competes with itself.
    keep[int(target)] = False
    if not bool(keep.any()):
        # Degenerate vocab (nothing else to compare to): target is trivially best.
        return torch.zeros((), device=lp.device, dtype=lp.dtype)
    best_other = lp.masked_fill(~keep, float("-inf")).max()
    return lp[int(target)] - best_other


def topk_candidates(
    logits: torch.Tensor,
    k: int,
    exclude: Sequence[int] = (),
    block_target: Optional[int] = None,
) -> List[int]:
    lp = logits.float()
    order = torch.argsort(lp, descending=True)
    out: List[int] = []
    excluded = set(int(x) for x in exclude)
    if block_target is not None:
        excluded.add(int(block_target))
    for idx in order.tolist():
        if int(idx) in excluded:
            continue
        out.append(int(idx))
        if len(out) >= int(k):
            break
    return out


def forward_logits(
    wrapper,
    *,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """Score a sequence through ``prepare_inputs_for_generation``; return logits.

    This is the *only* correct way to read logits on Qwen2-VL, and every
    constraint / probe / objective forward in TraceFlip must go through it so the
    solver optimises the same function the decoder evaluates.

    A bare ``wrapper.model(input_ids=..., pixel_values=..., image_grid_thw=...)``
    call omits the 3D **M-RoPE** ``position_ids`` that the decoder gets for free
    from ``prepare_inputs_for_generation`` (which computes them via
    ``get_rope_index`` keyed on ``image_grid_thw``). Without them the model falls
    back to 1D positions and the logits shift by several nats -- measured
    ``max|Δlogit| ≈ 4.2`` on the real Qwen2-VL-7B checkpoint, enough to change the
    argmax. A constraint measured on the bare path therefore does not predict the
    greedy re-decode: ``g_flip > 0`` (bare) can coexist with a non-flip under the
    decoder, i.e. ``feasible_in_model`` lies. ``use_cache`` is *irrelevant* to the
    logits here (measured ``Δ = 0`` between the two), so this helper keeps
    ``use_cache=False`` for the single-shot teacher-forced forwards and returns
    logits for every position.

    ``cache_position`` must be seeded before the very first call because
    Qwen2-VL's ``prepare_inputs_for_generation`` indexes it unconditionally.
    """
    model_kwargs: Dict[str, Any] = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "use_cache": bool(use_cache),
    }
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask
    model_kwargs["cache_position"] = torch.arange(
        input_ids.shape[-1], device=input_ids.device
    )
    model_inputs = wrapper.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
    return wrapper.model(**model_inputs, return_dict=True).logits.float()


def forward_path_logits(
    wrapper,
    *,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    continuation_ids: Sequence[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return true cached-decoder next-token logits along a forced path.

    Row 0 predicts the first continuation token and row ``i+1`` predicts the
    token after ``continuation_ids[i]``.  Qwen2-VL's one-shot teacher-forced
    prefill is not numerically/path-equivalent to incremental generation at
    deeper positions after adversarial optimisation, so branch constraints
    must follow the same cache updates as :func:`greedy_trace`.
    """
    from transformers.cache_utils import DynamicCache

    cur = input_ids
    model_kwargs: Dict[str, Any] = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "use_cache": True,
        "past_key_values": DynamicCache(),
        "cache_position": torch.arange(input_ids.shape[-1], device=input_ids.device),
    }
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask

    rows: List[torch.Tensor] = []
    forced = [int(x) for x in continuation_ids]
    for index in range(len(forced) + 1):
        model_inputs = wrapper.model.prepare_inputs_for_generation(cur, **model_kwargs)
        outputs = wrapper.model(**model_inputs, return_dict=True)
        rows.append(outputs.logits[:, -1].float()[0])
        if index >= len(forced):
            break
        token = torch.tensor(
            [[forced[index]]], device=cur.device, dtype=cur.dtype
        )
        cur = torch.cat([cur, token], dim=-1)
        previous_cache_position = model_kwargs["cache_position"]
        model_kwargs = wrapper.model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )
        if "cache_position" not in model_kwargs:
            model_kwargs["cache_position"] = previous_cache_position[-1:] + 1
    return torch.stack(rows, dim=0)


@torch.no_grad()
def greedy_trace(
    wrapper,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int = MAX_NEW_TOKENS,
    top_k: int = 8,
    n_positions: Optional[int] = None,
) -> Dict[str, Any]:
    """Greedy decode with per-position branch bookkeeping.

    ``inputs`` is a prepared encoder output (``input_ids``, ``pixel_values``,
    ``image_grid_thw``, optional ``attention_mask``). It must already carry the
    adversarial pixels.
    """
    input_ids = inputs["input_ids"]
    model_kwargs: Dict[str, Any] = {k: v for k, v in inputs.items() if k != "input_ids"}
    model_kwargs["use_cache"] = True
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    # Qwen2-VL's prepare_inputs_for_generation indexes cache_position
    # unconditionally, so it must be present before the first call. This mirrors
    # QwenP0._greedy; missing it is a hard TypeError on the real model.
    model_kwargs["cache_position"] = cache_position
    # The KV cache must be *seeded*, not merely requested. On transformers
    # 4.46.3 `use_cache=True` alone is not enough for this model: if
    # `past_key_values` arrives as ``None`` the decoder block never builds the
    # cache, the output carries no ``past_key_values`` field at all, and every
    # subsequent step silently re-feeds the whole sequence. Two things then go
    # wrong at once and both corrupt the decode:
    #   1. `prepare_inputs_for_generation` leaves `pixel_values=None` whenever
    #      `cache_position[0] != 0`, so the image tokens lose their visual
    #      embeddings after step 0;
    #   2. the M-RoPE positions are shifted by ``cache_position[0] +
    #      rope_deltas`` (measured +182 on the prompt), so the *unmodified*
    #      decoder no longer produces the trace of the frozen model.
    # Measured on h64:c07, the unseeded loop decodes
    # ``[40, 2776, 14589, 11, ...]`` while ``model.generate`` decodes
    # ``[40, 2776, 11889, 311, ...]``: the paths diverge at token 2. Seeding an
    # empty DynamicCache makes ``_update_model_kwargs_for_generation`` return a
    # real cache, the per-step fed length drops to 1, and the manual loop then
    # matches ``model.generate`` token-for-token. Everything downstream
    # (labels, core_rhc, the solver's `forward_logits`) is only meaningful on
    # this path, so it is not optional.
    if model_kwargs.get("past_key_values") is None:
        from transformers.cache_utils import DynamicCache

        model_kwargs["past_key_values"] = DynamicCache()
    eos_set = _eos_ids(wrapper)

    n_in = int(input_ids.shape[-1])
    n_want = int(max_new_tokens if n_positions is None else n_positions)
    steps: List[Dict[str, Any]] = []
    t0 = time.time()
    for step in range(int(max_new_tokens)):
        model_inputs = wrapper.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = wrapper.model(**model_inputs, return_dict=True)
        logits = outputs.logits[:, -1].float()[0]
        top1 = int(logits.argmax().item())
        if step < n_want:
            topk = topk_candidates(logits, top_k)
            lp = F.log_softmax(logits, dim=-1)
            second = float(lp.masked_fill(
                torch.arange(lp.shape[0], device=lp.device) == top1, float("-inf")
            ).max().item())
            steps.append(
                {
                    "step": int(step),
                    "token_id": int(top1),
                    "top_k": [int(x) for x in topk],
                    # Canonical margin per candidate against the full vocab
                    # (EOS included), from the same function the solver uses, so
                    # the probe ledger and the flip objective cannot drift apart.
                    # EOS is excluded from the candidate *set* above, but it is a
                    # real competitor for the greedy argmax.
                    "margin_nats": {
                        int(a): float(margin_nats(logits, int(a)).item()) for a in topk
                    },
                    "top1_gap_nats": float(lp[top1].item() - second),
                    "logit_max": float(logits.max().item()),
                }
            )
        next_token = torch.tensor([[top1]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        model_kwargs = wrapper.model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )
        if "cache_position" not in model_kwargs:
            model_kwargs["cache_position"] = cache_position[-1:] + 1
        cache_position = model_kwargs["cache_position"]
        if top1 in eos_set:
            break
    wall = time.time() - t0
    gen_ids = input_ids[0, n_in:].tolist()
    text = wrapper.processor.batch_decode(input_ids[:, n_in:], skip_special_tokens=True)[0].strip()
    token_text = wrapper.processor.tokenizer.convert_ids_to_tokens([int(x) for x in gen_ids])
    return {
        "n_input_tokens": n_in,
        "token_ids": [int(x) for x in gen_ids],
        "token_text": [str(x) for x in token_text],
        "text": text,
        "text_hash": _hash(text),
        "n_tokens": len(gen_ids),
        "steps": steps,
        "wall_seconds": float(wall),
        "greedy_steps_run": int(min(max_new_tokens, len(steps) + (len(gen_ids) > len(steps)))),
        "eos_hit": bool(len(gen_ids) and int(gen_ids[-1]) in eos_set),
    }


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def forward_tokens(n_input: int, n_generated: int) -> int:
    """Teacher-forced forward cost of a full decode over ``n_generated`` tokens.

    One prefill over the prompt plus one forward per generated token. Charged
    identically for every method so that "generated tokens" is comparable.
    """
    return int(n_input) + int(n_generated)


def label_trace(trace: Dict[str, Any], keywords: Sequence[str], query: str) -> Dict[str, Any]:
    """Attach the frozen four-axis label. Raw text never leaves this call."""
    return label_text(trace.get("text") or "", list(keywords or []), query or "")


def execute(
    wrapper,
    item: Dict[str, Any],
    x01: torch.Tensor,
    image,
    max_new_tokens: int = MAX_NEW_TOKENS,
    top_k: int = 8,
    n_positions: Optional[int] = None,
) -> Dict[str, Any]:
    """One concrete execution of the current image for one catalog item."""
    inputs = wrapper.encode(image, item["query"])
    inputs = wrapper._apply_pixels(inputs, x01=x01)
    trace = greedy_trace(
        wrapper,
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        n_positions=n_positions,
    )
    lab = label_trace(trace, item.get("topic_keywords") or [], item["query"])
    out = dict(trace)
    out.update(
        {
            "query_id": item["id"],
            "carrier_id": item.get("carrier_id"),
            "terminal_label": lab["terminal_label"],
            "core_rhc": bool(lab["core_rhc"]),
            "core_safe_answer": bool(lab["core_safe_answer"]),
            "safety": lab["safety"],
            "response_mode": lab["response_mode"],
            "grounding": lab["grounding"],
            "quality": lab["quality"],
            "chars": int(lab["chars"]),
            "relevance": float(lab["relevance"]),
        }
    )
    return out


def public_trace(trace: Dict[str, Any], keep_steps: int = 4) -> Dict[str, Any]:
    """Trace without raw text; only hashes, labels and the first steps."""
    out = {k: v for k, v in trace.items() if k not in {"text", "token_text"}}
    out["steps"] = list(out.get("steps") or [])[: int(keep_steps)]
    return out
