"""LLaVA-1.5 transfer: does shared AVTP still couple A and B?"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from .attack import _empty, _load_kept, _metrics_from_scores
from .config import out_dir
from .io import save_json
from .llava_model import LlavaMultiImage
from .model import load_pair_images, perturb_linf
from .probe import _ok, _select


def _one(wrapper: LlavaMultiImage, pair: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    a, b = load_pair_images(pair)
    packed = wrapper.pack([a, b], pair["prompt"])
    owner = wrapper.owner(packed)
    scores = wrapper.visual_scores(packed)
    quota, keep = _select(scores, owner, packed.n_per_image, cfg, "avtp")
    gold = pair.get("ans_a_only") or pair.get("screen", {}).get("ans_a_only")
    ans_full = wrapper.generate(packed, keep_visual=None)
    ans_avtp = wrapper.generate(packed, keep_visual=keep)
    keep_iso = _select(scores, owner, packed.n_per_image, cfg, "isolated")[1]
    ans_iso = wrapper.generate(packed, keep_visual=keep_iso)
    eps = float(cfg["probe"]["eps"])
    b_noisy = perturb_linf(b, eps=eps, seed=int(cfg["seed"]) + sum(ord(c) for c in pair["pair_id"]))
    packed_n = wrapper.pack([a, b_noisy], pair["prompt"])
    scores_n = wrapper.visual_scores(packed_n)
    met = _metrics_from_scores(scores_n, wrapper.owner(packed_n), packed_n.n_per_image, cfg, keep)
    ans_n = wrapper.generate(packed_n, keep_visual=met["keep"])
    return {
        "pair_id": pair["pair_id"],
        "n_per": packed.n_per_image,
        "r": [float(x) for x in quota.r.detach().cpu().tolist()],
        "k": [int(x) for x in quota.k.detach().cpu().tolist()],
        "r_noise": met["r"],
        "k_noise": met["k"],
        "delta_r": [met["r"][i] - float(quota.r[i].detach().cpu()) for i in range(2)],
        "events": met["events"],
        "ans_full": ans_full,
        "ans_avtp": ans_avtp,
        "ans_isolated": ans_iso,
        "ans_noise_avtp": ans_n,
        "full_ok": bool(_ok(ans_full, pair, gold=gold)),
        "avtp_ok": bool(_ok(ans_avtp, pair, gold=gold)),
        "isolated_ok": bool(_ok(ans_iso, pair, gold=gold)),
        "noise_avtp_ok": bool(_ok(ans_n, pair, gold=gold)),
        "gold": gold,
    }


def stage_llava_probe(cfg: Dict[str, Any], limit: Optional[int] = 8) -> Dict[str, Any]:
    kept = _load_kept(cfg)
    if limit is not None:
        kept = kept[: int(limit)]
    wrapper = LlavaMultiImage(cfg)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    dest = out_dir(cfg) / "llava_probe.json"
    for i, pair in enumerate(kept, start=1):
        print(f"== llava {pair['pair_id']} ({i}/{len(kept)}) ==", flush=True)
        try:
            row = _one(wrapper, pair, cfg)
            row["error"] = None
            print(
                f"llava {pair['pair_id']} rA={row['r'][0]:.3f}->{row['r_noise'][0]:.3f} "
                f"a_out={row['events']['a_out']} full={row['full_ok']} avtp={row['avtp_ok']}",
                flush=True,
            )
        except Exception as exc:
            print(f"llava {pair['pair_id']} FAILED {type(exc).__name__}: {exc}", flush=True)
            row = {"pair_id": pair["pair_id"], "error": f"{type(exc).__name__}: {exc}"}
            _empty()
        rows.append(row)
        save_json(dest, {"n": len(rows), "elapsed_s": time.time() - t0, "rows": rows})
        _empty()
    ok = [r for r in rows if not r.get("error")]
    n = max(len(ok), 1)
    summary = {
        "n": len(rows),
        "n_ok": len(ok),
        "frac_full_ok": sum(int(r.get("full_ok")) for r in ok) / n if ok else 0.0,
        "frac_avtp_ok": sum(int(r.get("avtp_ok")) for r in ok) / n if ok else 0.0,
        "mean_abs_delta_rA": sum(abs(r["delta_r"][0]) for r in ok) / n if ok else 0.0,
        "frac_a_out": sum(int(r["events"]["a_out"] > 0) for r in ok) / n if ok else 0.0,
        "mean_a_out": sum(r["events"]["a_out"] for r in ok) / n if ok else 0.0,
        "elapsed_s": time.time() - t0,
        "path": str(dest),
    }
    save_json(dest, {"n": len(rows), "elapsed_s": summary["elapsed_s"], "summary": summary, "rows": rows})
    return summary
