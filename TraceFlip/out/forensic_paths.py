#!/usr/bin/env python3
"""Pin down which forward path is authoritative for M-RoPE positions at t>0.

The replay in forensic_tpos.py may itself be wrong, so this script instruments
the REAL greedy decode loop (a copy of trace.greedy_trace plus position logging)
and compares its position_ids with the solver's single-shot forward at
[prompt + prefix]. Three candidate forwards at the same pixels and prefix:

  1. decoder_real   : incremental greedy with KV cache (what execute() runs)
  2. solver_single  : forward_logits([prompt + prefix]), the solver's path
  3. prefill_single : cached prefill of [prompt + prefix] in one shot

Reports the last position_id, the argmax, and pairwise max|dlogit|.

    TRACEFLIP_QUERY=h64 TRACEFLIP_CARRIER=c07 python TraceFlip/out/forensic_paths.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
os.environ.setdefault("P0_QWEN_KEEP_336", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import load_model, seed_all  # noqa: E402
from traceflip.datasets import build_cells  # noqa: E402
from traceflip.protocol import OPT_SEEDS  # noqa: E402
from traceflip.repair import with_pixels  # noqa: E402
from traceflip.trace import execute, forward_logits  # noqa: E402

OUT = HERE
QUERY = os.environ.get("TRACEFLIP_QUERY", "h64")
CARRIER = os.environ.get("TRACEFLIP_CARRIER", "c07")
GEN = int(os.environ.get("TRACEFLIP_MAX_NEW_TOKENS", "24"))


def decoder_real_with_positions(wrapper, base, x, n_steps: int):
    """Copy of trace.greedy_trace for the first ``n_steps`` tokens, recording the
    position_ids handed to the model at every step and the logits of each step."""
    inputs = wrapper._apply_pixels(dict(base), x01=x)
    input_ids = inputs["input_ids"]
    model_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    model_kwargs["use_cache"] = True
    model_kwargs["cache_position"] = torch.arange(input_ids.shape[1], device=input_ids.device)
    steps = []
    for step in range(n_steps):
        cp = model_kwargs.get("cache_position")
        rd = model_kwargs.get("rope_deltas")
        pkw = model_kwargs.get("past_key_values")
        mi = wrapper.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        pos = mi.get("position_ids")
        fed = mi.get("input_ids")
        out = wrapper.model(**mi, return_dict=True)
        logits = out.logits[0, -1].float()
        steps.append(
            {
                "step": int(step),
                "input_ids_len": int(input_ids.shape[1]),
                "fed_input_len": int(fed.shape[1]) if fed is not None else None,
                "fed_pos_len": int(pos.shape[-1]) if pos is not None else None,
                "has_past_kv": bool(pkw is not None),
                "cp_shape": int(cp.shape[0]) if cp is not None else None,
                "cache_position_0": int(cp[0].item()) if cp is not None else None,
                "pos_last": int(pos[0, 0, -1].item()) if pos is not None else None,
                "pos_last_all3": [int(pos[k, 0, -1].item()) for k in range(3)]
                if pos is not None
                else None,
                "argmax": int(logits.argmax().item()),
                "rope_deltas": int(out.rope_deltas.item())
                if getattr(out, "rope_deltas", None) is not None
                else None,
                "rope_deltas_before": (
                    int(rd.item()) if rd is not None and torch.is_tensor(rd) else None
                ),
            }
        )
        top1 = int(logits.argmax().item())
        nxt = torch.tensor([[top1]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, nxt], dim=-1)
        model_kwargs = wrapper.model._update_model_kwargs_for_generation(
            out, model_kwargs, is_encoder_decoder=False
        )
    return steps


def single_forward_at(wrapper, inputs, x, prefix_ids, *, cached: bool):
    ids = inputs["input_ids"]
    if prefix_ids:
        ids = torch.cat(
            [ids, torch.tensor([list(prefix_ids)], device=ids.device, dtype=ids.dtype)], dim=-1
        )
    pv, grid = wrapper.patchify(x)
    kw = {
        "pixel_values": pv,
        "image_grid_thw": grid,
        "use_cache": bool(cached),
    }
    if "attention_mask" in inputs:
        am = inputs["attention_mask"]
        pad = ids.shape[-1] - am.shape[-1]
        if pad > 0:
            am = torch.cat(
                [am, torch.ones(am.shape[0], pad, device=am.device, dtype=am.dtype)], dim=-1
            )
        kw["attention_mask"] = am
    kw["cache_position"] = torch.arange(ids.shape[-1], device=ids.device)
    mi = wrapper.model.prepare_inputs_for_generation(ids, **kw)
    pos = mi.get("position_ids")
    out = wrapper.model(**mi, return_dict=True)
    pos_last = int(pos[0, 0, -1].item()) if pos is not None else None
    pos_all3 = [int(pos[k, 0, -1].item()) for k in range(3)] if pos is not None else None
    return out.logits[0, -1].float(), pos_last, pos_all3


def main() -> int:
    seed_all(OPT_SEEDS[0])
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="mini")
    wrapper = load_model(cfg, setting="native")
    cells = build_cells((QUERY,), (CARRIER,))
    cell = cells[0]
    item = dict(cell["item"])
    item.setdefault("carrier_id", cell["carrier_id"])
    image = cell["image"]
    x0 = wrapper.image_to_x01(image)
    base = wrapper.encode(image, item["query"])

    trace = execute(wrapper, item, x0, image, max_new_tokens=GEN, top_k=8)
    ids = [int(v) for v in trace["token_ids"]]
    inputs0 = with_pixels(wrapper, base, x0)

    rec = {
        "query": QUERY,
        "carrier": CARRIER,
        "clean_label": trace["terminal_label"],
        "n_input_tokens": int(trace["n_input_tokens"]),
        "n_generated": len(ids),
    }

    # Trace the REAL decoder positions for the first 10 steps.
    real = decoder_real_with_positions(wrapper, base, x0, n_steps=min(10, len(ids) + 1))
    rec["decoder_real_steps"] = real
    print("--- REAL decoder (greedy_trace) positions ---")
    for s in real:
        print(
            f"  step={s['step']:2d} in_len={s['input_ids_len']:4d} "
            f"fed_len={s['fed_input_len']} fed_pos_len={s['fed_pos_len']} "
            f"has_kv={s['has_past_kv']} cp_len={s['cp_shape']} "
            f"cp0={s['cache_position_0']} pos_last={s['pos_last']} "
            f"pos3={s['pos_last_all3']} argmax={s['argmax']} rope_deltas={s['rope_deltas']}"
        )

    # Compare the solver's single-shot forward with the real decoder at t values.
    print("\n--- solver single-shot vs real decoder ---")
    rows = []
    for t in range(0, min(9, len(ids))):
        prefix = ids[:t]
        if t == 0:
            # the real decoder step 0 predicts token 0
            dec_logits = None
            dec_pos = real[0]["pos_last"]
            dec_argmax = real[0]["argmax"]
        else:
            dec_logits = None
            dec_pos = real[t]["pos_last"]
            dec_argmax = real[t]["argmax"]
        lp_s, pos_s, pos3_s = single_forward_at(wrapper, inputs0, x0, prefix, cached=False)
        lp_p, pos_p, pos3_p = single_forward_at(wrapper, inputs0, x0, prefix, cached=True)
        am_s = int(lp_s.argmax())
        am_p = int(lp_p.argmax())
        row = {
            "t": int(t),
            "prefix_len": len(prefix),
            "decoder_pos": dec_pos,
            "solver_pos": pos_s,
            "prefill_pos": pos_p,
            "solver_pos3": pos3_s,
            "prefill_pos3": pos3_p,
            "decoder_argmax": dec_argmax,
            "solver_argmax": am_s,
            "prefill_argmax": am_p,
            "solver_matches_decoder": bool(am_s == dec_argmax),
            "prefill_matches_decoder": bool(am_p == dec_argmax),
        }
        rows.append(row)
        print(
            f"  t={t:2d} len={len(prefix):2d} | pos dec={dec_pos} solver={pos_s} "
            f"prefill={pos_p} | argmax dec={dec_argmax:6d} solver={am_s:6d} "
            f"prefill={am_p:6d} | s==d:{row['solver_matches_decoder']} "
            f"p==d:{row['prefill_matches_decoder']}"
        )
        del lp_s, lp_p
    rec["rows"] = rows

    (OUT / "forensic_paths.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'forensic_paths.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
