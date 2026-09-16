"""Small LLaVA PGD transfer: quota+evict on B, 20 steps, clean AVTP pairs."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from .attack import (
    _J,
    _empty,
    _log_prefix,
    _metrics_from_scores,
    evaluate_final,
)
from .config import out_dir
from .io import load_json, save_json
from .llava_model import LlavaMultiImage
from .llava_p0 import stage_llava_probe
from .losses import combined_loss
from .model import load_pair_images
from .vision import clip_delta, tv_loss


def _prepare(wrapper: LlavaMultiImage, pair: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    a, b = load_pair_images(pair)
    packed = wrapper.pack([a, b], pair["prompt"])
    xA = wrapper.x01_from_pil(a)
    xB0 = wrapper.x01_from_pil(b)
    pv = wrapper.pixels_from_x01(xA, xB0)
    packed.tensors["pixel_values"] = pv
    owner = wrapper.owner(packed)
    scores = wrapper.visual_scores(packed)
    from .probe import _select

    quota, keep = _select(scores, owner, packed.n_per_image, cfg, "avtp")
    u_idx = pair.get("u_idx")
    if u_idx:
        # Qwen token indices do not map onto LLaVA's 576-grid. Use low-score A kept.
        u_idx = None
    a_keep = torch.nonzero((owner == 0) & keep, as_tuple=False).flatten()
    n_bot = max(1, int(round(0.25 * int(a_keep.numel()))))
    bot = torch.topk(scores[a_keep], k=min(n_bot, int(a_keep.numel())), largest=False).indices
    u_mask = torch.zeros_like(keep)
    u_mask[a_keep[bot]] = True
    gold = pair.get("ans_a_only") or pair.get("screen", {}).get("ans_a_only")
    from .attack import _gold_ids

    # tokenizer API differs; skip gold_ids
    return {
        "pair": pair,
        "packed": packed,
        "xA": xA.detach(),
        "xB0": xB0.detach(),
        "owner": owner,
        "keep_clean": keep.detach(),
        "u_mask": u_mask.detach(),
        "v_mask": (owner == 1).detach(),
        "h_clean": scores.detach().mean().reshape(-1),  # placeholder; value loss disabled
        "gold": gold,
        "r_clean": [float(x) for x in quota.r.detach().cpu().tolist()],
        "k_clean": [int(x) for x in quota.k.detach().cpu().tolist()],
        "gold_ids": None,
    }


def _forward(wrapper, state, delta, cfg):
    xB = torch.clamp(state["xB0"] + delta, 0.0, 1.0)
    pv = wrapper.pixels_from_x01(state["xA"], xB)
    scores, h_adv = wrapper.visual_scores_grad(state["packed"], pv)
    atk = cfg["attack"]
    loss, aux = combined_loss(
        scores=scores,
        owner=state["owner"],
        n_per=state["packed"].n_per_image,
        u_mask=state["u_mask"],
        v_mask=state["v_mask"],
        h_adv=h_adv,
        h_clean=h_adv.detach(),
        xB=xB,
        r_base=float(cfg["compressor"]["r_base"]),
        alpha=float(cfg["compressor"]["alpha"]),
        r_min=float(cfg["compressor"]["r_min"]),
        r_max=float(cfg["compressor"]["r_max"]),
        lambda_q=float(atk["lambda_q"]),
        lambda_e=float(atk["lambda_e"]),
        lambda_v=0.0,
        lambda_p=float(atk.get("lambda_p", 0.05)),
        kappa=float(atk["kappa"]),
        tau=float(atk["tau"]),
        tv_fn=tv_loss,
        lambda_c=float(atk.get("lambda_c", 1.5)),
    )
    return loss, aux, scores, xB, pv


def _eval(wrapper, state, delta, cfg):
    """Same interventions as Qwen evaluate_final, using LLaVA generate."""
    from .compressor import restore_victim
    from .probe import _ok, _select

    pair = state["pair"]
    gold = state["gold"]
    xB = torch.clamp(state["xB0"] + delta, 0.0, 1.0)
    pv = wrapper.pixels_from_x01(state["xA"], xB).detach()
    packed = state["packed"]
    packed.tensors["pixel_values"] = pv
    scores = wrapper.visual_scores(packed)
    owner = state["owner"]
    n_per = packed.n_per_image
    avtp = _metrics_from_scores(scores, owner, n_per, cfg, state["keep_clean"], "avtp")
    ans_full = wrapper.generate(packed, keep_visual=None)
    ans_avtp = wrapper.generate(packed, keep_visual=avtp["keep"])
    keep_iso = _select(scores, owner, n_per, cfg, "isolated")[1]
    ans_iso = wrapper.generate(packed, keep_visual=keep_iso)
    keep_restore = restore_victim(avtp["keep"], state["keep_clean"], owner, victim=0)
    ans_restore = wrapper.generate(packed, keep_visual=keep_restore)
    full_ok = _ok(ans_full, pair, gold=gold)
    avtp_ok = _ok(ans_avtp, pair, gold=gold)
    u_mask = state["u_mask"].to(avtp["keep"].device)
    u_out = int((u_mask & ~avtp["keep"]).sum().item())
    n_u = int(u_mask.sum().item())
    return {
        "ans_full": ans_full,
        "ans_avtp": ans_avtp,
        "ans_isolated": ans_iso,
        "ans_restore_a": ans_restore,
        "full_ok": bool(full_ok),
        "avtp_ok": bool(avtp_ok),
        "isolated_ok": bool(_ok(ans_iso, pair, gold=gold)),
        "restore_a_ok": bool(_ok(ans_restore, pair, gold=gold)),
        "avtp": {k: v for k, v in avtp.items() if k != "keep"},
        "linf": float(delta.abs().amax().cpu()),
        "comp_only_fail": bool(full_ok and not avtp_ok),
        "n_u": n_u,
        "u_out": u_out,
        "u_evict_rate": (u_out / n_u) if n_u else 0.0,
    }


def stage_llava_attack(cfg: Dict[str, Any], limit: int = 6, steps: int = 20) -> Dict[str, Any]:
    probe_path = out_dir(cfg) / "llava_probe.json"
    if not probe_path.is_file():
        stage_llava_probe(cfg, limit=8)
    blob = load_json(probe_path)
    pairs = [
        r for r in blob.get("rows") or []
        if r.get("full_ok") and r.get("avtp_ok") and not r.get("error")
    ]
    # attach original pair fields from screen
    from .attack import _load_kept

    by = {p["pair_id"]: p for p in _load_kept(cfg)}
    kept = []
    for r in pairs:
        if r["pair_id"] in by:
            kept.append(by[r["pair_id"]])
    kept = kept[: int(limit)]
    wrapper = LlavaMultiImage(cfg)
    dest = out_dir(cfg) / f"llava_attack_{steps}.json"
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    eps = float(cfg["attack"]["eps"])
    alpha = float(cfg["attack"]["alpha"])
    for i, pair in enumerate(kept, start=1):
        print(f"== llava-atk {pair['pair_id']} ({i}/{len(kept)}) ==", flush=True)
        try:
            state = _prepare(wrapper, pair, cfg)
            delta = torch.zeros_like(state["xB0"])
            for step in range(1, steps + 1):
                delta = delta.detach().requires_grad_(True)
                loss, aux, scores, _xB, _pv = _forward(wrapper, state, delta, cfg)
                grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
                if grad is None:
                    raise RuntimeError("no image gradient")
                delta = clip_delta(state["xB0"], delta.detach() - alpha * grad.sign(), eps)
                met = _metrics_from_scores(
                    scores.detach(), state["owner"], state["packed"].n_per_image, cfg, state["keep_clean"]
                )
                u_out = int((state["u_mask"].to(met["keep"].device) & ~met["keep"]).sum().item())
                print(
                    f"llava-atk {pair['pair_id']} {step}/{steps} L={float(loss.detach()):.3f} "
                    f"rA={met['r'][0]:.3f} a_out={met['events']['a_out']} u_out={u_out}",
                    flush=True,
                )
                del loss, aux, scores, grad
                _empty()
            ev = _eval(wrapper, state, delta, cfg)
            out = {
                "pair_id": pair["pair_id"],
                "steps": steps,
                "eval": ev,
                "error": None,
                "elapsed_s": time.time() - t0,
            }
        except Exception as exc:
            print(f"llava-atk {pair['pair_id']} FAILED {type(exc).__name__}: {exc}", flush=True)
            out = {"pair_id": pair["pair_id"], "error": f"{type(exc).__name__}: {exc}", "eval": None}
            _empty()
        rows.append(out)
        save_json(dest, {"n": len(rows), "rows": rows, "elapsed_s": time.time() - t0})
        _empty()
    ok = [r for r in rows if r.get("eval")]
    n = max(len(ok), 1)
    summary = {
        "n": len(ok),
        "frac_full_hold": sum(int(r["eval"]["full_ok"]) for r in ok) / n if ok else 0.0,
        "frac_comp_only_fail": sum(int(r["eval"]["comp_only_fail"]) for r in ok) / n if ok else 0.0,
        "mean_u_evict_rate": sum(float(r["eval"]["u_evict_rate"]) for r in ok) / n if ok else 0.0,
        "mean_a_out": sum(r["eval"]["avtp"]["events"]["a_out"] for r in ok) / n if ok else 0.0,
        "path": str(dest),
    }
    save_json(dest, {"n": len(rows), "summary": summary, "rows": rows, "elapsed_s": time.time() - t0})
    return summary
