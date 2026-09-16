"""Cheap FastV-style probe: last-token attention over visual tokens,  no PGD."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from .attack import _empty, _load_kept, prepare_sample
from .compressor import exchange_events, summarize_keep
from .io import save_json
from .model import QwenMultiImage
from .config import out_dir
from .probe import _ok, _select


def _attn_keep(wrapper: QwenMultiImage, packed, k_total: int) -> Optional[torch.Tensor]:
    tensors = {**packed.tensors}
    try:
        out = wrapper.model(
            **tensors,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
        )
    except Exception:
        return None
    att = out.attentions
    if not att:
        return None
    last = att[-1][0].mean(dim=0)[-1]
    vis = packed.vis_index.to(last.device)
    scores = last[vis]
    k_use = max(1, min(int(k_total), int(scores.numel())))
    top = torch.topk(scores, k=k_use, largest=True).indices
    keep = torch.zeros(int(scores.numel()), dtype=torch.bool, device=scores.device)
    keep[top] = True
    del out
    return keep


def _try_eager(wrapper: QwenMultiImage) -> bool:
    """SDPA often cannot return attentions; FastV needs eager or a similar kernel."""
    try:
        wrapper.model.config._attn_implementation = "eager"
        if hasattr(wrapper.model, "set_attn_implementation"):
            wrapper.model.set_attn_implementation("eager")
        return True
    except Exception:
        return False


def stage_fastv_probe(cfg: Dict[str, Any], limit: int = 8) -> Dict[str, Any]:
    kept = _load_kept(cfg)[: int(limit)]
    wrapper = QwenMultiImage(cfg)
    rows = []
    t0 = time.time()
    supported = True
    for pair in kept:
        state = prepare_sample(wrapper, pair, cfg)
        packed = state["packed"]
        owner = state["owner"]
        n_per = packed.n_per_image
        quota, keep_avtp = _select(wrapper.visual_scores(packed), owner, n_per, cfg, "avtp")
        k_total = int(quota.k.sum().item())
        keep_fv = _attn_keep(wrapper, packed, k_total)
        if keep_fv is None and _try_eager(wrapper):
            keep_fv = _attn_keep(wrapper, packed, k_total)
        if keep_fv is None:
            supported = False
            rows.append({"pair_id": pair["pair_id"], "error": "output_attentions unavailable"})
            break
        ev = exchange_events(keep_avtp, keep_fv, owner)
        ans_fv = wrapper.generate(packed, keep_visual=keep_fv)
        gold = state["gold"]
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "k_total": k_total,
                "kept": summarize_keep(keep_fv, owner, n_images=len(n_per)),
                "events_vs_avtp": ev,
                "ans_fastv": ans_fv,
                "fastv_ok": bool(_ok(ans_fv, pair, gold=gold)),
                "avtp_ok": bool(_ok(wrapper.generate(packed, keep_visual=keep_avtp), pair, gold=gold)),
            }
        )
        _empty()
    blob = {
        "n": len(rows),
        "supported": supported and all(r.get("error") is None for r in rows),
        "elapsed_s": time.time() - t0,
        "rows": rows,
        "note": "FastV last-token attention Top-K vs AVTP at the same K. Not an attack.",
    }
    dest = out_dir(cfg) / "fastv_probe.json"
    save_json(dest, blob)
    return {"n": blob["n"], "supported": blob["supported"], "path": str(dest)}
