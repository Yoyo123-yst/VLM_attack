"""Greedy decode + four-axis labels. Mode gate uses response_mode only."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from n0.axes import four_axes
from p0.judge import harmful_compliance


@torch.no_grad()
def decode_axes(
    wrapper,
    image,
    question: str,
    keywords,
    x01: Optional[torch.Tensor] = None,
    max_new_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    text = wrapper.generate(image, question, x01=x01, do_sample=False, max_new_tokens=max_new_tokens)
    judge = harmful_compliance(text, keywords, question)
    ax = four_axes(text, keywords, question, judge)
    return {
        "text": text,
        "legacy_label": judge.get("label"),
        "safety": ax["safety"],
        "response_mode": ax["response_mode"],
        "grounding": ax["grounding"],
        "quality": ax["quality"],
        "core_rhc": bool(ax["core_rhc"]),
        "core_safe_answer": bool(ax["core_safe_answer"]),
        "theme_denial": bool(ax["theme_denial"]),
        "refuse_then_advice": bool(ax.get("refuse_then_advice")),
        "chars": int(ax["chars"]),
    }


def is_answer(row: Dict[str, Any]) -> bool:
    return row.get("response_mode") == "ANSWER"


def strip_text(row: Dict[str, Any], keep_chars: int = 240) -> Dict[str, Any]:
    out = dict(row)
    text = out.get("text") or ""
    out["text_head"] = text[:keep_chars]
    return out
