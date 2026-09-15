"""Live prefix scores for VLM AMS. Next-token margin, not frozen first-token."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from p0.model import COMPLY_WORDS, REFUSE_WORDS, margin_from_logits, token_id_list

from .committor import MARGIN_ONLY, fit_logistic
from .prefixes import PREFIX_STAGES

LIVE_KEYS = (
    "refusal_margin",
    "logit_max",
    "frac",
    "prefix_word_count",
    "stage_early",
    "stage_mid",
    "stage_late",
    "stage_pre_eos",
)


def stage_from_frac(frac: float) -> str:
    if frac < 0.375:
        return "early"
    if frac < 0.625:
        return "mid"
    if frac < 0.875:
        return "late"
    return "pre_eos"


def live_features(
    refusal_margin: float,
    logit_max: float,
    n_words: int,
    max_tokens: int = 96,
) -> dict[str, float]:
    n = max(int(n_words), 0)
    frac = float(n) / float(max(max_tokens, 1))
    stage = stage_from_frac(frac)
    row = {
        "refusal_margin": float(refusal_margin),
        "logit_max": float(logit_max),
        "frac": frac,
        "prefix_word_count": float(n),
        "full_word_count": float(max_tokens),
        "token_count": float(max_tokens),
    }
    for s in PREFIX_STAGES:
        row[f"stage_{s}"] = float(s == stage)
    row["stage"] = stage
    return row


class LinearScorer:
    def __init__(
        self,
        keys: Sequence[str],
        coef: Sequence[float],
        intercept: float,
        mean: Sequence[float],
        scale: Sequence[float],
        name: str,
    ) -> None:
        self.keys = list(keys)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.scale = np.where(self.scale == 0.0, 1.0, self.scale)
        self.name = name

    def predict_proba(self, row: dict[str, Any]) -> float:
        x = np.array([float(row.get(k) or 0.0) for k in self.keys], dtype=np.float64)
        z = (x - self.mean) / self.scale
        logit = float(self.intercept + np.dot(self.coef, z))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


def fit_live(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    full = fit_logistic(rows, LIVE_KEYS, name="live_full")
    margin = fit_logistic(rows, MARGIN_ONLY, name="live_margin")
    return {"full": full, "refusal_margin": margin, "features": list(LIVE_KEYS)}


def scorer_from_fit(blob: dict[str, Any], which: str = "full") -> LinearScorer:
    rec = blob[which]
    return LinearScorer(
        rec["features"],
        [rec["coef"][k] for k in rec["features"]],
        rec["intercept"],
        rec["scaler_mean"],
        rec["scaler_scale"],
        rec.get("name", which),
    )


def append_prefix_ids(wrapper, inputs: dict, prefix_text: str) -> dict:
    out = dict(inputs)
    text = (prefix_text or "").strip()
    if not text:
        return out
    extra = wrapper.processor.tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    )
    extra_ids = extra["input_ids"].to(out["input_ids"].device)
    out["input_ids"] = torch.cat([out["input_ids"], extra_ids], dim=-1)
    ones = torch.ones_like(extra_ids)
    if "attention_mask" in out:
        out["attention_mask"] = torch.cat([out["attention_mask"], ones], dim=-1)
    return out


@torch.no_grad()
def next_token_margin(wrapper, img, query: str, x01: torch.Tensor, prefix_text: str) -> dict[str, float]:
    packed = wrapper.encode(img, query)
    packed = wrapper._apply_pixels(packed, x01=x01)
    packed = append_prefix_ids(wrapper, packed, prefix_text)
    logits = wrapper.first_token_logits(
        packed["pixel_values"],
        packed["input_ids"],
        packed.get("attention_mask"),
        packed.get("image_grid_thw"),
    )
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    margin = margin_from_logits(logits, refuse_ids, comply_ids)
    return {
        "refusal_margin": float(margin.detach().item()),
        "logit_max": float(logits.detach().float().max().item()),
    }


@torch.no_grad()
def generate_continue(
    wrapper,
    img,
    query: str,
    x01: torch.Tensor,
    prefix_text: str,
    *,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> str:
    """Sample up to max_new_tokens after an assistant prefix. No raw logging."""
    if max_new_tokens <= 0:
        return ""
    inputs = wrapper.encode(img, query)
    inputs = wrapper._apply_pixels(inputs, x01=x01)
    inputs = append_prefix_ids(wrapper, inputs, prefix_text)
    n_in = int(inputs["input_ids"].shape[-1])
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    gen_kw = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        use_cache=True,
    )
    out = wrapper.model.generate(**inputs, **gen_kw)
    piece = wrapper.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (piece or "").strip()
