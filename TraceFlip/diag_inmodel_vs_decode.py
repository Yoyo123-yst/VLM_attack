#!/usr/bin/env python3
"""Diagnose the in-model vs re-decode disagreement seen in the smoke run.

The smoke cell reported ``feasible_in_model: true`` with ``g_flip_final`` of
several nats, yet the greedy re-decode did not produce the branch token. The
root cause was **not** the KV cache: it was the Qwen2-VL **M-RoPE position
encoding**. The decoder gets its 3D ``position_ids`` for free from
``prepare_inputs_for_generation`` (via ``get_rope_index`` keyed on
``image_grid_thw``), while a bare ``model(...)`` call omits them and falls back
to 1D positions, shifting logits by up to ~4.2 nats and changing the argmax.

This script measures three forwards at the same pixels:

  (1) ``solver_forward_argmax``  — the decoder's own path (``forward_logits``);
      this is what the solver now uses, so it MUST match the reload.
  (2) ``bare_forward_argmax``    — the old, wrong bare ``model(...)`` path, kept
      only to show it is the one that diverges.
  (3) ``cached_prefill_argmax``  — the cached, ``use_cache=True`` prefill the
      decoder actually runs.

It reports the numbers at delta = 0 and at the solved delta. Labels and counts
only; no raw text.

    python TraceFlip/diag_inmodel_vs_decode.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
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
from traceflip.flip import constrained_flip, flip_margin, prefix_margins  # noqa: E402
from traceflip.protocol import ALPHA, EPS, OPT_SEEDS  # noqa: E402
from traceflip.repair import with_pixels  # noqa: E402
from traceflip.trace import execute, forward_logits  # noqa: E402

OUT = HERE / "out"
QUERY = os.environ.get("TRACEFLIP_QUERY", "h49")
CARRIER = os.environ.get("TRACEFLIP_CARRIER", "c07")
GEN = int(os.environ.get("TRACEFLIP_MAX_NEW_TOKENS", "24"))
SOLVE_OUTER = int(os.environ.get("TRACEFLIP_OUTER", "4"))
SOLVE_INNER = int(os.environ.get("TRACEFLIP_INNER", "24"))
POSITION = int(os.environ.get("TRACEFLIP_T", "0"))


def _seq_ids(inputs, prefix_ids):
    ids = inputs["input_ids"]
    if prefix_ids:
        ids = torch.cat(
            [ids, torch.tensor([list(prefix_ids)], device=ids.device, dtype=ids.dtype)], dim=-1
        )
    return ids


def _attn(inputs, ids):
    if "attention_mask" not in inputs:
        return None
    am = inputs["attention_mask"]
    extra = ids.shape[-1] - am.shape[-1]
    if extra <= 0:
        return am
    return torch.cat(
        [am, torch.ones(am.shape[0], extra, device=am.device, dtype=am.dtype)], dim=-1
    )


def solver_forward_argmax(wrapper, inputs, x, prefix_ids):
    """The forward the solver now uses: ``forward_logits`` (the decoder's own
    ``prepare_inputs_for_generation`` path, M-RoPE ``position_ids`` included).
    This must match the greedy re-decode; that is the whole point of the fix."""
    ids = _seq_ids(inputs, prefix_ids)
    pv, grid = wrapper.patchify(x)
    logits = forward_logits(
        wrapper,
        input_ids=ids,
        pixel_values=pv,
        image_grid_thw=grid,
        attention_mask=_attn(inputs, ids),
        use_cache=False,
    )[0, -1].float()
    return int(logits.argmax().item()), logits


def bare_forward_argmax(wrapper, inputs, x, prefix_ids):
    """The old, WRONG solver forward: a bare ``model(...)`` call with no
    ``position_ids``. Kept only as a contrast, to show it is the path that
    diverges from the decoder."""
    ids = _seq_ids(inputs, prefix_ids)
    pv, grid = wrapper.patchify(x)
    kw = {
        "pixel_values": pv,
        "image_grid_thw": grid,
        "input_ids": ids,
        "use_cache": False,
        "output_hidden_states": False,
    }
    am = _attn(inputs, ids)
    if am is not None:
        kw["attention_mask"] = am
    logits = wrapper.model(**kw).logits[0, -1].float()
    return int(logits.argmax().item()), logits


def cached_prefill_argmax(wrapper, inputs, x, prefix_ids):
    """First-step argmax exactly as ``greedy_trace`` computes it: cached prefill
    via ``prepare_inputs_for_generation`` with ``use_cache=True``."""
    ids = _seq_ids(inputs, prefix_ids)
    inputs = wrapper._apply_pixels(dict(inputs), x01=x)
    model_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    if "attention_mask" in model_kwargs:
        model_kwargs["attention_mask"] = _attn(inputs, ids)
    model_kwargs["use_cache"] = True
    model_kwargs["cache_position"] = torch.arange(ids.shape[1], device=ids.device)
    model_inputs = wrapper.model.prepare_inputs_for_generation(ids, **model_kwargs)
    out = wrapper.model(**model_inputs, return_dict=True)
    logits = out.logits[:, -1].float()[0]
    return int(logits.argmax().item()), logits


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
    if POSITION < 0 or POSITION >= len(trace["token_ids"]):
        raise RuntimeError(f"TRACEFLIP_T={POSITION} outside trace length {len(trace['token_ids'])}")
    prefix = [int(x) for x in trace["token_ids"][:POSITION]]
    inc0 = int(trace["token_ids"][POSITION])
    step0 = trace["steps"][POSITION]
    # Reproduce the smoke choice exactly: the reachability selector picked token
    # 785 at t=0 in the smoke run. Fall back to the first non-incumbent candidate.
    cand0 = int(os.environ.get("TRACEFLIP_CAND", "0")) or next(
        int(a) for a in step0["top_k"] if int(a) != inc0
    )

    inputs0 = with_pixels(wrapper, base, x0)

    eos_ids = sorted(
        int(x)
        for x in (
            wrapper.model.generation_config.eos_token_id
            if isinstance(wrapper.model.generation_config.eos_token_id, (list, tuple))
            else [wrapper.model.generation_config.eos_token_id]
        )
    )

    # (1) solver forward (prepare path) vs bare forward vs cached prefill, same pixels.
    inc_cached, lp_cached = cached_prefill_argmax(wrapper, inputs0, x0, prefix)
    am_solver, logits_solver = solver_forward_argmax(wrapper, inputs0, x0, prefix)
    am_bare, logits_bare = bare_forward_argmax(wrapper, inputs0, x0, prefix)
    g0_nats = float(flip_margin(wrapper, inputs0, x0, prefix, cand0, inc0).item())
    pm = prefix_margins(wrapper, inputs0, x0, prefix)
    top5_solver = [int(v) for v in torch.argsort(logits_solver, descending=True)[:5].tolist()]
    top5_cached = [int(v) for v in torch.argsort(lp_cached, descending=True)[:5].tolist()]
    rec = {
        "query": QUERY,
        "carrier": CARRIER,
        "position": POSITION,
        "clean_label": trace["terminal_label"],
        "clean_n_tokens": trace["n_tokens"],
        "incumbent_t0": inc0,
        "candidate_t0": cand0,
        "candidate_is_eos": bool(cand0 in eos_ids),
        "incumbent_is_eos": bool(inc0 in eos_ids),
        "cached_greedy_argmax_t0": inc_cached,
        "solver_forward_argmax_t0": am_solver,
        "bare_forward_argmax_t0": am_bare,
        "cached_vs_solver_agree": bool(am_solver == inc_cached),
        "top5_solver_at_delta0": top5_solver,
        "top5_cached_at_delta0": top5_cached,
        "g_flip_nats_at_delta0": g0_nats,
        "prefix_margin_nats_t0_at_delta0": float(pm[0].item()) if pm.numel() else None,
        "topk_t0": [int(x) for x in step0["top_k"]],
    }

    # (2) Solve the flip and compare in-model margin against the real re-decode.
    solved = constrained_flip(
        wrapper,
        inputs0,
        x0,
        torch.zeros_like(x0),
        prefix,
        cand0,
        inc0,
        eps=EPS,
        alpha=ALPHA,
        outer_steps=SOLVE_OUTER,
        inner_steps=SOLVE_INNER,
        eos_exclude=eos_ids,
    )
    x_new = torch.clamp(x0 + solved["delta"], 0.0, 1.0)
    t_new = execute(wrapper, item, x_new, image, max_new_tokens=GEN, top_k=8)
    g_solved = float(flip_margin(wrapper, inputs0, x_new, prefix, cand0, inc0).item())
    am_solved_solver, logits_solved = solver_forward_argmax(wrapper, inputs0, x_new, prefix)
    am_solved_bare, _ = bare_forward_argmax(wrapper, inputs0, x_new, prefix)
    inc_solved_cached, logits_solved_cached = cached_prefill_argmax(wrapper, inputs0, x_new, prefix)
    lp_solved = F.log_softmax(logits_solved, dim=-1)
    # Competitor is the best other token over the FULL vocab (EOS included), the
    # same set the solver's flip_margin uses after the fix.
    keep = torch.ones_like(lp_solved, dtype=torch.bool)
    keep[cand0] = False
    best_other = int(lp_solved.masked_fill(~keep, float("-inf")).argmax().item())
    top5_solved_solver = [
        int(v) for v in torch.argsort(logits_solved, descending=True)[:5].tolist()
    ]
    top5_solved_cached = [
        int(v) for v in torch.argsort(logits_solved_cached, descending=True)[:5].tolist()
    ]
    rec.update(
        {
            "solved_delta_linf": float(solved["delta"].abs().max().item()),
            "solved_n_backward": int(solved["n_backward"]),
            "solved_feasible_in_model": bool(solved["feasible_in_model"]),
            "g_flip_nats_after_solve": g_solved,
            "solver_forward_argmax_after_solve": am_solved_solver,
            "bare_forward_argmax_after_solve": am_solved_bare,
            "cached_prefill_argmax_after_solve": inc_solved_cached,
            "best_competitor_at_solved": best_other,
            "top5_solver_at_solved": top5_solved_solver,
            "top5_cached_at_solved": top5_solved_cached,
            "redecode_token_t0_after_solve": int(t_new["token_ids"][POSITION]),
            "redecode_label": t_new["terminal_label"],
            "redecode_matches_candidate": bool(int(t_new["token_ids"][POSITION]) == cand0),
            "inmodel_agrees_with_redecode": bool(
                am_solved_solver == int(t_new["token_ids"][POSITION])
            ),
            "cachedpath_agrees_with_redecode": bool(
                inc_solved_cached == int(t_new["token_ids"][POSITION])
            ),
            # Fresh forward reproducibility: re-run solver forward twice and
            # confirm it is deterministic (no hidden cache state).
            "redecode_reproducible": bool(
                int(t_new["token_ids"][POSITION])
                == int(execute(wrapper, item, x_new, image, max_new_tokens=GEN, top_k=8)["token_ids"][POSITION])
            ),
            # The max-margin claim: g_flip > 0 exactly when the candidate is the
            # solver-forward argmax.
            "g_flip_positive_iff_argmax": bool(
                (g_solved > 0.0) == (am_solved_solver == cand0)
            ),
        }
    )
    # Direct max-margin from the solved solver forward.
    rec["solver_forward_margin_nats_at_solved"] = float(
        (lp_solved[cand0] - lp_solved[best_other]).item()
    )
    rec["eos_ids"] = eos_ids
    rec["candidate_is_eos"] = bool(cand0 in rec["eos_ids"])
    rec["incumbent_is_eos"] = bool(inc0 in rec["eos_ids"])

    (OUT / "diag_inmodel_vs_decode.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)
    if not rec["cached_vs_solver_agree"]:
        print("DIAG: solver forward (prepare path) disagrees with cached prefill at delta=0", flush=True)
        return 2
    if not rec["inmodel_agrees_with_redecode"]:
        print("DIAG: solver-forward argmax disagrees with re-decode at solved delta", flush=True)
        return 3
    if not rec["cachedpath_agrees_with_redecode"]:
        print("DIAG: cached-prefill argmax disagrees with re-decode at solved delta", flush=True)
        return 5
    if not rec["g_flip_positive_iff_argmax"]:
        print("DIAG: g_flip sign disagrees with the argmax (margin semantics broken)", flush=True)
        return 4
    print("DIAG OK: solver forward and re-decode agree", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
