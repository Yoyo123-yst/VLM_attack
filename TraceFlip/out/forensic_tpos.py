#!/usr/bin/env python3
"""Forensics for the honesty-invariant violation.

Observation from the 12-cell pilot:

    t == 0  : 42 attempts, 32 feasible_in_model, 32 valid   (100% predictive)
    t  > 0  : 42 attempts, 13 feasible_in_model,  0 valid   (0% predictive)

Every t>0 attempt has prefix_kept=True, branch_flipped=False. The solver says
the branch token wins by several nats at the post-prefix position, but the
greedy re-decode does not emit it. This script measures, at the SAME pixels and
the SAME prefix, whether the solver's single-shot forward really agrees with the
decoder's incremental forward at the moment the flip position is predicted.

Sections:
  A. solver single-shot forward at [prompt + prefix]  -> argmax + margin
  B. decoder incremental forward, replaying the clean
     token path token-by-token with a KV cache        -> argmax + margin at step t*
  C. elementwise logit agreement A vs B
  D. position_ids used by each path (to expose an M-RoPE mismatch)
  E. the same comparison for several t values, including t=0 as a control

Labels and counts only; no raw model text is printed.

    TRACEFLIP_QUERY=h64 TRACEFLIP_CARRIER=c07 python TraceFlip/out/forensic_tpos.py
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

import torch.nn.functional as F  # noqa: E402

from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import load_model, seed_all  # noqa: E402
from traceflip.datasets import build_cells  # noqa: E402
from traceflip.flip import flip_margin  # noqa: E402
from traceflip.protocol import OPT_SEEDS  # noqa: E402
from traceflip.repair import with_pixels  # noqa: E402
from traceflip.trace import execute, forward_logits, margin_nats  # noqa: E402
from traceflip.protocol import MAX_NEW_TOKENS  # noqa: E402

OUT = HERE
QUERY = os.environ.get("TRACEFLIP_QUERY", "h64")
CARRIER = os.environ.get("TRACEFLIP_CARRIER", "c07")
GEN = int(os.environ.get("TRACEFLIP_MAX_NEW_TOKENS", "24"))


def decoder_logits_at_step(wrapper, base, x, n_generated: int):
    """Replay the decoder incrementally and return (logits, position_ids) at the
    step that PREDICTS token index ``n_generated`` (0 = first generated token).

    This mirrors ``greedy_trace`` exactly: cached prefill, then one cached
    forward per emitted token, with the same cache_position / rope_deltas
    bookkeeping transformers uses.
    """
    inputs = wrapper._apply_pixels(dict(base), x01=x)
    input_ids = inputs["input_ids"]
    model_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    model_kwargs["use_cache"] = True
    model_kwargs["cache_position"] = torch.arange(input_ids.shape[1], device=input_ids.device)

    # Replay the clean prefix: at each step take the argmax (greedy), which is
    # what ``execute`` did to produce the trace we are matching against.
    for step in range(n_generated + 1):
        model_inputs = wrapper.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        out = wrapper.model(**model_inputs, return_dict=True)
        pos = model_inputs.get("position_ids")
        if step == n_generated:
            last_pos = None if pos is None else int(pos[0, 0, -1].item())
            return out.logits[:, -1].float()[0], last_pos
        top1 = int(out.logits[0, -1].argmax())
        nxt = torch.tensor([[top1]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, nxt], dim=-1)
        model_kwargs = wrapper.model._update_model_kwargs_for_generation(
            out, model_kwargs, is_encoder_decoder=False
        )
        if "cache_position" not in model_kwargs:
            model_kwargs["cache_position"] = model_kwargs.get("cache_position")
        cache_position = model_kwargs["cache_position"]
    raise RuntimeError("unreachable")


def solver_logits_at(wrapper, inputs, x, prefix_ids):
    ids = inputs["input_ids"]
    if prefix_ids:
        ids = torch.cat(
            [ids, torch.tensor([list(prefix_ids)], device=ids.device, dtype=ids.dtype)], dim=-1
        )
    pv, grid = wrapper.patchify(x)
    model_kwargs = {
        "pixel_values": pv,
        "image_grid_thw": grid,
        "use_cache": False,
        "cache_position": torch.arange(ids.shape[-1], device=ids.device),
    }
    if "attention_mask" in inputs:
        am = inputs["attention_mask"]
        pad = ids.shape[-1] - am.shape[-1]
        if pad > 0:
            am = torch.cat(
                [am, torch.ones(am.shape[0], pad, device=am.device, dtype=am.dtype)], dim=-1
            )
        model_kwargs["attention_mask"] = am
    mi = wrapper.model.prepare_inputs_for_generation(ids, **model_kwargs)
    pos = mi.get("position_ids")
    last = None if pos is None else int(pos[0, 0, -1].item())
    logits = wrapper.model(**mi, return_dict=True).logits[0, -1].float()
    return logits, last


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
    n_in = int(trace["n_input_tokens"])
    inputs0 = with_pixels(wrapper, base, x0)

    rec = {
        "query": QUERY,
        "carrier": CARRIER,
        "clean_label": trace["terminal_label"],
        "n_input_tokens": n_in,
        "n_generated": len(ids),
        "eos_hit": trace["eos_hit"],
        "rows": [],
    }

    # Sweep t including the control t=0 and the failing t=6.
    for t in (0, 1, 2, 3, 4, 5, 6, 7, 8):
        if t >= len(ids):
            continue
        prefix = ids[:t]
        incumbent = ids[t]
        # a* = the branch token the pilot reported for this cell (646 at t=6),
        # otherwise the top alternative to the incumbent.
        if t == 6:
            a_star = 646
        else:
            step = next((s for s in trace["steps"] if int(s["step"]) == t), None)
            a_star = None
            if step:
                for cand in step["top_k"]:
                    if int(cand) != incumbent:
                        a_star = int(cand)
                        break
        if a_star is None:
            continue

        lp_solver, pos_solver = solver_logits_at(wrapper, inputs0, x0, prefix)
        lp_dec, pos_dec = decoder_logits_at_step(wrapper, base, x0, t)

        am_solver = int(lp_solver.argmax())
        am_dec = int(lp_dec.argmax())
        g_solver = float(margin_nats(lp_solver, a_star))
        g_dec = float(margin_nats(lp_dec, a_star))
        dmax = float((lp_solver - lp_dec).abs().max())
        d_a = float((lp_solver[a_star] - lp_dec[a_star]).abs())

        row = {
            "t": int(t),
            "incumbent": int(incumbent),
            "a_star": int(a_star),
            "solver_argmax": am_solver,
            "decoder_argmax": am_dec,
            "solver_matches_incumbent": bool(am_solver == incumbent),
            "decoder_matches_incumbent": bool(am_dec == incumbent),
            "solver_argmax_is_a_star": bool(am_solver == a_star),
            "decoder_argmax_is_a_star": bool(am_dec == a_star),
            "g_flip_solver_nats": g_solver,
            "g_flip_decoder_nats": g_dec,
            "max_abs_logit_diff_all_vocab": dmax,
            "abs_logit_diff_at_a_star": d_a,
            "position_ids_solver": pos_solver,
            "position_ids_decoder": pos_dec,
            "position_ids_agree": bool(pos_solver == pos_dec),
        }
        rec["rows"].append(row)
        print(
            f"t={t:2d} inc={incumbent:6d} a*={a_star:6d} "
            f"argmax solver={am_solver:6d} dec={am_dec:6d} "
            f"g_solver={g_solver:+7.3f} g_dec={g_dec:+7.3f} "
            f"max|dlogit|={dmax:8.4f} pos solver={pos_solver} dec={pos_dec}",
            flush=True,
        )

    (OUT / "forensic_tpos.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'forensic_tpos.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
