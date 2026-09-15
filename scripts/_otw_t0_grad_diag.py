#!/usr/bin/env python3
"""Engineering diagnostic: does autograd ascent on +s_mode actually raise s_mode?"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")

from n1.preflight import all_carrier_index
from otw.attack import pack
from otw.open_phase import clip_delta
from otw.score import load_u_refusal_p0s, s_mode
from p0.catalog import all_pairs
from p0.datautil import open_image
from p0_qwen.config import load_cfg
from run_p0_qwen import load_model, seed_all


def main() -> None:
    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="full")
    seed_all(2026)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    wrapper.model.eval()
    u = load_u_refusal_p0s()
    catalog = {p["id"]: p for p in all_pairs()}
    img = open_image(all_carrier_index()["c05"]["path"])
    q = catalog["h01"]["query"]
    x0, ids, attn, grid = pack(wrapper, img, q)
    eps = 16 / 255
    alpha = 1 / 255
    g = torch.Generator(device=x0.device)
    g.manual_seed(2026)
    delta = (torch.rand(x0.shape, generator=g, device=x0.device) * 2 - 1) * eps
    delta = clip_delta(x0, delta, eps)

    def score_of(d):
        x = torch.clamp(x0 + d, 0, 1)
        return s_mode(wrapper, x, ids, attn, grid, u)

    d = delta.detach().requires_grad_(True)
    s = score_of(d)
    grad = torch.autograd.grad(s, d, allow_unused=True)[0]
    print(
        {
            "s": float(s.detach()),
            "grad_none": grad is None,
            "grad_abs_mean": None if grad is None else float(grad.abs().mean()),
            "grad_finite": None if grad is None else bool(torch.isfinite(grad).all()),
        },
        flush=True,
    )
    assert grad is not None
    plus = clip_delta(x0, delta.detach() + alpha * grad.sign(), eps)
    minus = clip_delta(x0, delta.detach() - alpha * grad.sign(), eps)
    with torch.no_grad():
        s0 = float(score_of(delta.detach()).item())
        sp = float(score_of(plus).item())
        sm = float(score_of(minus).item())
    print({"s0": s0, "ascent_+sign(ds)": sp, "descent_-sign(ds)": sm, "d_plus": sp - s0, "d_minus": sm - s0}, flush=True)
    # also check first-token margin grad exists (known-good path)
    from p0.model import COMPLY_WORDS, REFUSE_WORDS, margin_from_logits, token_id_list

    d2 = delta.detach().requires_grad_(True)
    x = torch.clamp(x0 + d2, 0, 1)
    pv, gth = wrapper.patchify(x)
    logits = wrapper.first_token_logits(pv, ids, attn, gth)
    m = margin_from_logits(logits, token_id_list(wrapper.processor, REFUSE_WORDS), token_id_list(wrapper.processor, COMPLY_WORDS))
    gm = torch.autograd.grad(m, d2, allow_unused=True)[0]
    print({"margin": float(m.detach()), "margin_grad_abs_mean": None if gm is None else float(gm.abs().mean())}, flush=True)

    # tiny steps: local linearization
    for scale in (1e-4, 1e-3, 1e-2):
        p = clip_delta(x0, delta.detach() + scale * grad.sign(), eps)
        n = clip_delta(x0, delta.detach() - scale * grad.sign(), eps)
        with torch.no_grad():
            print(
                {
                    "scale": scale,
                    "d+": float(score_of(p).item()) - s0,
                    "d-": float(score_of(n).item()) - s0,
                },
                flush=True,
            )

    # output_hidden_states path (same last token)
    from otw.score import LAYER, s_mode_from_hidden

    d3 = delta.detach().requires_grad_(True)
    x = torch.clamp(x0 + d3, 0, 1)
    pv, gth = wrapper.patchify(x)
    kwargs = {
        "pixel_values": pv,
        "input_ids": ids,
        "attention_mask": attn,
        "image_grid_thw": gth,
        "use_cache": False,
        "output_hidden_states": True,
    }
    hs = wrapper.model(**kwargs).hidden_states
    # embeddings + 28 layers; hook on layers[24] == hidden_states[25]
    h = hs[LAYER + 1][0, -1, :]
    s_hs = s_mode_from_hidden(h, u)
    ghs = torch.autograd.grad(s_hs, d3, allow_unused=True)[0]
    plus_hs = clip_delta(x0, delta.detach() + alpha * ghs.sign(), eps)
    with torch.no_grad():
        xh = torch.clamp(x0 + plus_hs, 0, 1)
        pv2, g2 = wrapper.patchify(xh)
        kwargs["pixel_values"] = pv2
        kwargs["image_grid_thw"] = g2
        h2 = wrapper.model(**kwargs).hidden_states[LAYER + 1][0, -1, :]
        s_hs2 = float(s_mode_from_hidden(h2, u).item())
    print(
        {
            "s_hs": float(s_hs.detach()),
            "s_hs_after_+sign": s_hs2,
            "d_hs": s_hs2 - float(s_hs.detach()),
            "ghs_abs_mean": float(ghs.abs().mean()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
