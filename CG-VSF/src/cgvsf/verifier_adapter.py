"""Official greedy decode + four-axis labels. Raw text never leaves this module."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from pathem.labels import label_text
from traceflip.protocol import MAX_NEW_TOKENS, TOPK
from traceflip.repair import with_pixels
from traceflip.trace import execute, public_trace

from .certificates import certificate_mode


def decode_prefix_text(processor, token_ids: Sequence[int]) -> str:
    ids = [int(x) for x in token_ids]
    if not ids:
        return ""
    return processor.tokenizer.decode(ids, skip_special_tokens=True)


def label_prefix(processor, token_ids, keywords, query: str) -> Dict[str, Any]:
    text = decode_prefix_text(processor, token_ids)
    return dict(label_text(text, list(keywords or []), query or ""))


def official_decode(wrapper, item, x01, image, max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
    tr = execute(
        wrapper,
        item,
        x01,
        image,
        max_new_tokens=max_new_tokens,
        top_k=TOPK,
    )
    pub = public_trace(tr)
    pub["certificate_mode"] = certificate_mode(tr)
    return pub


def pixels_for(wrapper, image, query: str, x01):
    base = wrapper.encode(image, query)
    return with_pixels(wrapper, base, x01)
