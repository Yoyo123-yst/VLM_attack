#!/usr/bin/env python3
"""Per-sample visual attack: U-guided vs refusal-margin at the sampler budget.

Universal 8/255 failed for every method, so this checks whether U has
attack value when the perturbation is allowed to be instance-specific.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from p0.attacks import refusal_loss_fn, state_loss_fn  # noqa: E402
from p0.config import load_cfg  # noqa: E402
from p0.datautil import load_carriers, load_json, open_image  # noqa: E402
from p0.judge import harmful_compliance  # noqa: E402
from p0.model import (  # noqa: E402
    COMPLY_WORDS,
    REFUSE_WORDS,
    LlavaP0,
    image_to_clip_pixels,
    normalize_clip,
    token_id_list,
    tv_loss,
)
from p0.subspace import state_shift  # noqa: E402


def pgd_one(wrapper, img, query, steps, eps, alpha, loss_fn, tvw):
    x0, ids, attn = image_to_clip_pixels(img).to(wrapper.device), None, None
    packed = wrapper.encode(img, query)
    ids, attn = packed["input_ids"], packed.get("attention_mask")
    delta = torch.zeros_like(x0)
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        x = torch.clamp(x0 + delta, 0, 1)
        pv = normalize_clip(x, wrapper.device, torch.float16)
        loss = loss_fn(pv, ids, attn) + tvw * tv_loss(x)
        g = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        if g is None:
            raise RuntimeError("no grad")
        delta = (delta - alpha * g.sign()).clamp(-eps, eps)
    return delta.detach()


def main() -> None:
    cfg = load_cfg(scale="mini")
    out = Path(cfg["output_dir"])
    probe = load_json(out / "probe.json")
    causal = load_json(out / "causal.json")
    sub = load_json(out / "subspace.json")
    layer = int(causal["best"]["layer"])
    rank = int(cfg["subspace"]["ranks"][0])
    spec = sub["layers"][str(layer)][str(rank)]
    U = np.asarray(spec["U"], dtype=np.float64)
    mu_ref = np.asarray(spec["mu_ref"], dtype=np.float64)
    mu_jb = np.asarray(spec["mu_jb"], dtype=np.float64)

    from p0.catalog import all_pairs

    hold_ids = set(probe["holdout"])
    queries = [dict(p, split="holdout") for p in all_pairs() if p["id"] in hold_ids][:6]
    _, test_c = load_carriers(cfg)
    carriers = test_c[:1]
    wrapper = LlavaP0(cfg)
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    steps = 40
    eps = float(cfg["attack"]["sampler_eps"])
    alpha = float(cfg["attack"]["alpha"])
    tvw = float(cfg["attack"]["tv_weight"])

    rows = []
    for q in queries:
        for c in carriers:
            img = open_image(c["path"])
            clean = wrapper.generate(img, q["query"])
            clean_j = harmful_compliance(clean, q["topic_keywords"], q["query"])
            rec = {"query_id": q["id"], "carrier_id": c["id"], "clean": clean_j["label"]}
            for name, fn in [
                ("pgd16", refusal_loss_fn(wrapper, refuse_ids, comply_ids)),
                ("u16", state_loss_fn(wrapper, layer, U, mu_ref, mu_jb)),
            ]:
                torch.cuda.empty_cache()
                delta = pgd_one(wrapper, img, q["query"], steps, eps, alpha, fn, tvw)
                x0 = image_to_clip_pixels(img).to(wrapper.device)
                pv = normalize_clip(torch.clamp(x0 + delta, 0, 1), wrapper.device, torch.float16)
                text = wrapper.generate(img, q["query"], pixel_values=pv)
                j = harmful_compliance(text, q["topic_keywords"], q["query"])
                hid = wrapper.collect_hidden(img, q["query"], [layer], pixel_values=pv)
                rec[name] = {
                    "label": j["label"],
                    "R": j["R"],
                    "refusal": j["refusal"],
                    "shift": state_shift(hid[f"L{layer}:last_user"].numpy(), U, mu_ref, mu_jb),
                }
            rows.append(rec)
            print(rec, flush=True)
    summary = {}
    for name in ("pgd16", "u16"):
        labs = [r[name]["label"] for r in rows]
        summary[name] = {
            "asr": sum(x == "JB" for x in labs) / len(labs),
            "refusal": sum(r[name]["refusal"] for r in rows) / len(rows),
            "shift": float(np.mean([r[name]["shift"] for r in rows])),
        }
    blob = {"rows": rows, "summary": summary, "layer": layer, "eps": eps, "steps": steps}
    (out / "per_sample.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
