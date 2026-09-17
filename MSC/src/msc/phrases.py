"""Frozen class-level phrase banks. Token ids and hashes only in logs."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .protocol import BANK_NAMES, load_frozen, phrase_banks

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - CPU tests never call the cache path
    DynamicCache = None


def phrase_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def encode_phrase(tokenizer, text: str, max_tokens: int) -> List[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    out = [int(x) for x in ids if int(x) >= 0][: int(max_tokens)]
    if not out:
        raise RuntimeError(f"empty tokenisation for phrase hash {phrase_hash(text)}")
    return out


def tokenize_banks(tokenizer, frozen: Mapping[str, Any] | None = None) -> Dict[str, List[List[int]]]:
    blob = dict(frozen or load_frozen())
    max_tokens = int(blob["budget"]["max_phrase_tokens"])
    banks = phrase_banks(blob)
    return {name: [encode_phrase(tokenizer, p, max_tokens) for p in phrases] for name, phrases in banks.items()}


def first_token_set(bank: Sequence[Sequence[int]]) -> set[int]:
    return {int(ids[0]) for ids in bank if ids}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def public_tokensets(
    tokenizer,
    frozen: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    blob = dict(frozen or load_frozen())
    banks_text = phrase_banks(blob)
    encoded = tokenize_banks(tokenizer, blob)
    first = {name: sorted(first_token_set(ids)) for name, ids in encoded.items()}
    return {
        "max_phrase_tokens": int(blob["budget"]["max_phrase_tokens"]),
        "banks": {
            name: {
                "n_phrases": len(banks_text[name]),
                "phrase_hashes": [phrase_hash(p) for p in banks_text[name]],
                "n_tokens": [len(ids) for ids in encoded[name]],
                "token_ids": encoded[name],
                "first_tokens": first[name],
            }
            for name in BANK_NAMES
        },
        "jaccard": {
            "refuse_follow": jaccard(set(first["REFUSE"]), set(first["FOLLOW"])),
            "deny_follow": jaccard(set(first["DENY"]), set(first["FOLLOW"])),
            "related_safe_follow": jaccard(set(first["RELATED_SAFE"]), set(first["FOLLOW"])),
        },
    }


def _clone_cache(cache):
    if cache is None:
        return None
    if DynamicCache is not None and hasattr(cache, "key_cache"):
        new = DynamicCache()
        for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
            if k is None or v is None:
                continue
            new.update(k.clone(), v.clone(), layer_idx=i)
        return new
    return copy.deepcopy(cache)


def _clone_kwargs(model_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in model_kwargs.items():
        if k == "past_key_values":
            out[k] = _clone_cache(v)
        elif torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def _prompt_forward(wrapper, inputs: Mapping[str, torch.Tensor], x: torch.Tensor):
    from transformers.cache_utils import DynamicCache as DC

    pv, grid = wrapper.patchify(x)
    cur = inputs["input_ids"]
    model_kwargs: Dict[str, Any] = {
        "pixel_values": pv,
        "image_grid_thw": grid,
        "use_cache": True,
        "past_key_values": DC(),
        "cache_position": torch.arange(cur.shape[-1], device=cur.device),
    }
    if "attention_mask" in inputs:
        model_kwargs["attention_mask"] = inputs["attention_mask"]
    model_inputs = wrapper.model.prepare_inputs_for_generation(cur, **model_kwargs)
    outputs = wrapper.model(**model_inputs, return_dict=True)
    logits0 = outputs.logits[:, -1].float()[0]
    prev = model_kwargs.get("cache_position")
    updated = wrapper.model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=False
    )
    if "cache_position" not in updated and prev is not None:
        updated["cache_position"] = prev[-1:] + 1
    return logits0, cur, updated


def _phrase_mean_nll(
    wrapper,
    logits0: torch.Tensor,
    cur: torch.Tensor,
    model_kwargs: Mapping[str, Any],
    ids: Sequence[int],
) -> torch.Tensor:
    ids = [int(x) for x in ids]
    logp = F.log_softmax(logits0, dim=-1)
    nlls = [-logp[ids[0]]]
    if len(ids) == 1:
        return nlls[0]
    kw = _clone_kwargs(model_kwargs)
    seq = torch.cat([cur, torch.tensor([[ids[0]]], device=cur.device, dtype=cur.dtype)], dim=-1)
    for j in range(1, len(ids)):
        model_inputs = wrapper.model.prepare_inputs_for_generation(seq, **kw)
        outputs = wrapper.model(**model_inputs, return_dict=True)
        lp = F.log_softmax(outputs.logits[:, -1].float()[0], dim=-1)
        nlls.append(-lp[int(ids[j])])
        seq = torch.cat(
            [seq, torch.tensor([[ids[j]]], device=seq.device, dtype=seq.dtype)],
            dim=-1,
        )
        prev = kw.get("cache_position")
        kw = wrapper.model._update_model_kwargs_for_generation(
            outputs, kw, is_encoder_decoder=False
        )
        if "cache_position" not in kw and prev is not None:
            kw["cache_position"] = prev[-1:] + 1
    return torch.stack(nlls).mean()


def phrase_set_score(
    wrapper,
    inputs: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    bank: Sequence[Sequence[int]],
    *,
    prompt_state: Optional[tuple] = None,
) -> torch.Tensor:
    """s_T = LogSumExp_p (-mean NLL of phrase p), truncated to frozen length."""
    if not bank:
        raise RuntimeError("empty phrase bank")
    if prompt_state is None:
        logits0, cur, kw = _prompt_forward(wrapper, inputs, x)
    else:
        logits0, cur, kw = prompt_state
    scores = []
    for ids in bank:
        nll = _phrase_mean_nll(wrapper, logits0, cur, kw, ids)
        scores.append(-nll)
    return torch.logsumexp(torch.stack(scores), dim=0)


def scores_for_banks(
    wrapper,
    inputs: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    banks: Mapping[str, Sequence[Sequence[int]]],
    names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    logits0, cur, kw = _prompt_forward(wrapper, inputs, x)
    state = (logits0, cur, kw)
    return {
        name: phrase_set_score(wrapper, inputs, x, banks[name], prompt_state=state)
        for name in names
    }
