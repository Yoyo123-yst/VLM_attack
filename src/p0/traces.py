"""Collect REF / JB / FAIL / BENIGN traces.

JB is sampled by per-sample visual refusal-margin PGD with a relaxed budget.
That sampler is not the P0 method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from pathlib import Path

from .datautil import open_image
from .judge import benign_utility, harmful_compliance
from .model import (
    COMPLY_WORDS,
    REFUSE_WORDS,
    LlavaP0,
    image_to_clip_pixels,
    margin_from_logits,
    normalize_clip,
    token_id_list,
)


def _pgd_sampler(
    wrapper: LlavaP0,
    image,
    question: str,
    steps: int,
    eps: float,
    alpha: float,
    refuse_ids: List[int],
    comply_ids: List[int],
) -> torch.Tensor:
    x0 = image_to_clip_pixels(image).to(wrapper.device)
    delta = torch.zeros_like(x0)
    inputs = wrapper.encode(image, question)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        pv = normalize_clip(x, wrapper.device, torch.float16)
        logits = wrapper.first_token_logits(pv, input_ids, attn)
        loss = margin_from_logits(logits, refuse_ids, comply_ids)
        grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        if grad is None:
            raise RuntimeError("sampler got no image gradient; vision tower is likely quantized")
        delta = (delta - alpha * grad.sign()).clamp(-eps, eps)
        delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0
    return delta.detach()


def _trim_judge(j: Dict[str, Any], n: int = 400) -> Dict[str, Any]:
    out = dict(j)
    if "text" in out and out["text"] is not None:
        out["text"] = out["text"][:n]
    return out


def collect_traces(
    wrapper: LlavaP0,
    cfg: Dict[str, Any],
    harmful: List[Dict[str, Any]],
    benign_map: Dict[str, Dict[str, Any]],
    carriers: List[Dict[str, Any]],
    layers: List[int],
    delta_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    steps = int(cfg["attack"]["sampler_steps"])
    eps = float(cfg["attack"]["sampler_eps"])
    alpha = float(cfg["attack"]["alpha"])
    records = []

    pairs = [(h, c) for h in harmful for c in carriers]
    for h, c in tqdm(pairs, desc="traces"):
        img = open_image(c["path"])
        clean_text = wrapper.generate(img, h["query"])
        clean_j = harmful_compliance(clean_text, h["topic_keywords"], h["query"])
        hid = {
            k: v.tolist()
            for k, v in wrapper.collect_hidden(img, h["query"], layers).items()
        }
        rec: Dict[str, Any] = {
            "query_id": h["id"],
            "category": h["category"],
            "split": h.get("split", "discover"),
            "carrier_id": c["id"],
            "carrier_split": c["split"],
            "query": h["query"],
            "clean": _trim_judge(clean_j),
            "clean_hidden": hid,
            "jb": None,
            "fail": None,
        }

        if clean_j["label"] != "REF":
            rec["skip_reason"] = "clean_not_refusal"
            records.append(rec)
            continue

        try:
            delta = _pgd_sampler(wrapper, img, h["query"], steps, eps, alpha, refuse_ids, comply_ids)
        except RuntimeError as exc:
            rec["skip_reason"] = str(exc)
            records.append(rec)
            continue

        x0 = image_to_clip_pixels(img).to(wrapper.device)
        pv = normalize_clip(torch.clamp(x0 + delta, 0.0, 1.0), wrapper.device, torch.float16)
        atk_text = wrapper.generate(img, h["query"], pixel_values=pv)
        atk_j = harmful_compliance(atk_text, h["topic_keywords"], h["query"])
        atk_h = {
            k: v.tolist()
            for k, v in wrapper.collect_hidden(img, h["query"], layers, pixel_values=pv).items()
        }
        rec["attack_eval"] = _trim_judge(atk_j)
        rec["delta_linf"] = float(delta.abs().max().item())
        rec["delta_path"] = None
        if delta_dir is not None:
            delta_dir.mkdir(parents=True, exist_ok=True)
            dpath = delta_dir / f"{h['id']}_{c['id']}.pt"
            torch.save(delta.detach().float().cpu(), dpath)
            rec["delta_path"] = str(dpath)
        if atk_j["label"] == "JB":
            rec["jb"] = {"judge": _trim_judge(atk_j), "hidden": atk_h}
        elif atk_j["label"] == "FAIL":
            rec["fail"] = {"judge": _trim_judge(atk_j), "hidden": atk_h}
        else:
            rec["sampler_still_refused"] = True
        records.append(rec)
        torch.cuda.empty_cache()

    benign_recs = []
    for h in harmful:
        b = benign_map.get(h["benign_id"])
        if b is None:
            continue
        c = carriers[0]
        img = open_image(c["path"])
        text = wrapper.generate(img, b["query"])
        j = benign_utility(text, b["topic_keywords"], b["query"])
        hid = {
            k: v.tolist()
            for k, v in wrapper.collect_hidden(img, b["query"], layers).items()
        }
        benign_recs.append(
            {
                "query_id": b["id"],
                "harmful_id": h["id"],
                "carrier_id": c["id"],
                "query": b["query"],
                "judge": _trim_judge(j),
                "hidden": hid,
            }
        )

    n_ref = sum(1 for r in records if r["clean"]["label"] == "REF")
    n_jb = sum(1 for r in records if r.get("jb") is not None)
    if n_jb < 4:
        hard_eps = min(eps * 2.0, 32.0 / 255.0)
        hard_steps = steps + 30
        for rec in tqdm(records, desc="traces-retry"):
            if rec.get("jb") is not None or rec["clean"]["label"] != "REF":
                continue
            h = next(x for x in harmful if x["id"] == rec["query_id"])
            c = next(x for x in carriers if x["id"] == rec["carrier_id"])
            img = open_image(c["path"])
            try:
                delta = _pgd_sampler(wrapper, img, h["query"], hard_steps, hard_eps, alpha, refuse_ids, comply_ids)
            except RuntimeError:
                continue
            x0 = image_to_clip_pixels(img).to(wrapper.device)
            pv = normalize_clip(torch.clamp(x0 + delta, 0.0, 1.0), wrapper.device, torch.float16)
            atk_text = wrapper.generate(img, h["query"], pixel_values=pv)
            atk_j = harmful_compliance(atk_text, h["topic_keywords"], h["query"])
            rec["attack_eval"] = _trim_judge(atk_j)
            rec["delta_linf"] = float(delta.abs().max().item())
            rec["retry_eps"] = hard_eps
            if delta_dir is not None:
                dpath = delta_dir / f"{h['id']}_{c['id']}.pt"
                torch.save(delta.detach().float().cpu(), dpath)
                rec["delta_path"] = str(dpath)
            if atk_j["label"] == "JB":
                rec["jb"] = {
                    "judge": _trim_judge(atk_j),
                    "hidden": {
                        k: v.tolist()
                        for k, v in wrapper.collect_hidden(img, h["query"], layers, pixel_values=pv).items()
                    },
                }
            n_jb = sum(1 for r in records if r.get("jb") is not None)
    return {
        "records": records,
        "benign": benign_recs,
        "summary": {
            "n_pairs": len(records),
            "n_clean_refusal": n_ref,
            "n_jb": n_jb,
            "n_fail": sum(1 for r in records if r.get("fail") is not None),
            "n_benign": len(benign_recs),
        },
    }


def jb_pairs(trace_blob: Dict[str, Any], split: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for rec in trace_blob["records"]:
        if rec.get("jb") is None:
            continue
        if split and rec.get("split") != split:
            continue
        out.append(rec)
    return out
