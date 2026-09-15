"""P1: quota poisoning + eviction-set PGD on auxiliary image B."""

from __future__ import annotations

import gc
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .compressor import exchange_events, restore_victim, summarize_keep
from .config import out_dir
from .io import load_json, save_json
from .losses import combined_loss
from .model import QwenMultiImage, load_pair_images
from .probe import _ok, _select
from .vision import clip_delta, tv_loss, x01_to_pil


def _gpu_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 ** 3))


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _empty() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _select_cfg(scores, owner, n_per, cfg, mode: str):
    return _select(scores, owner, n_per, cfg, mode)


def _metrics_from_scores(scores, owner, n_per, cfg, keep_clean, mode: str = "avtp") -> Dict[str, Any]:
    quota, keep = _select_cfg(scores, owner, n_per, cfg, mode)
    ev = exchange_events(keep_clean, keep, owner)
    return {
        "r": [float(x) for x in quota.r.detach().cpu().tolist()],
        "k": [int(x) for x in quota.k.detach().cpu().tolist()],
        "i_bar": [float(x) for x in quota.i_bar.detach().cpu().tolist()],
        "kept": summarize_keep(keep, owner, n_images=len(n_per)),
        "events": ev,
        "keep": keep.detach(),
    }


def _J(ev: Dict[str, int], r: List[float], w1: float = 2.0, w2: float = 2.0, w3: float = 8.0) -> float:
    return float(w1 * ev["a_out"] + w2 * ev["b_in"] + w3 * (r[1] - r[0]))


def prepare_sample(wrapper: QwenMultiImage, pair: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    a, b = load_pair_images(pair)
    packed = wrapper.pack([a, b], pair["prompt"])
    grid = packed.tensors["image_grid_thw"]
    xA = wrapper.x01_for_grid(a, grid[0]).detach()
    xB0 = wrapper.x01_for_grid(b, grid[1]).detach()
    pv = wrapper.pixels_from_x01(xA, xB0).detach()
    packed = wrapper.with_pixels(packed, pv)
    n_from_pv = int(pv.shape[0]) // (2 * 2)
    if n_from_pv != sum(packed.n_per_image):
        raise RuntimeError(
            f"patchify tokens {n_from_pv} != packed visual {sum(packed.n_per_image)}"
        )
    owner = wrapper.owner(packed)
    with torch.no_grad():
        scores, h_clean, out = wrapper._forward_hidden(packed, pixel_values=pv)
        del out
    quota, keep = _select_cfg(scores, owner, packed.n_per_image, cfg, "avtp")
    u_mask = (owner == 0) & keep
    v_mask = owner == 1
    gold = pair.get("ans_a_only") or pair.get("screen", {}).get("ans_a_only")
    return {
        "pair": pair,
        "packed": packed,
        "xA": xA,
        "xB0": xB0,
        "owner": owner,
        "keep_clean": keep.detach(),
        "u_mask": u_mask.detach(),
        "v_mask": v_mask.detach(),
        "h_clean": h_clean.detach(),
        "gold": gold,
        "r_clean": [float(x) for x in quota.r.detach().cpu().tolist()],
        "k_clean": [int(x) for x in quota.k.detach().cpu().tolist()],
        "i_bar_clean": [float(x) for x in quota.i_bar.detach().cpu().tolist()],
    }


def _forward_loss(wrapper, state, delta, cfg):
    atk = cfg["attack"]
    xB = torch.clamp(state["xB0"] + delta, 0.0, 1.0)
    pv = wrapper.pixels_from_x01(state["xA"], xB)
    scores, h_adv = wrapper.visual_scores_grad(state["packed"], pv)
    loss, aux = combined_loss(
        scores=scores,
        owner=state["owner"],
        n_per=state["packed"].n_per_image,
        u_mask=state["u_mask"],
        v_mask=state["v_mask"],
        h_adv=h_adv,
        h_clean=state["h_clean"],
        xB=xB,
        r_base=float(cfg["compressor"]["r_base"]),
        alpha=float(cfg["compressor"]["alpha"]),
        r_min=float(cfg["compressor"]["r_min"]),
        r_max=float(cfg["compressor"]["r_max"]),
        lambda_q=float(atk["lambda_q"]),
        lambda_e=float(atk["lambda_e"]),
        lambda_v=float(atk["lambda_v"]),
        lambda_p=float(atk["lambda_p"]),
        kappa=float(atk["kappa"]),
        tau=float(atk["tau"]),
        tv_fn=tv_loss,
    )
    return loss, aux, scores, xB, pv


def _log_step(pair_id: str, step: int, steps: int, loss, aux, ev, elapsed: float) -> Dict[str, Any]:
    r = [float(x) for x in aux["r"].detach().cpu().tolist()]
    row = {
        "step": step,
        "loss": float(loss.detach().cpu()),
        "L_quota": float(aux["L_quota"].detach().cpu()),
        "L_evict": float(aux["L_evict"].detach().cpu()),
        "L_value": float(aux["L_value"].detach().cpu()),
        "L_perc": float(aux["L_perc"].detach().cpu()),
        "r": r,
        "k": [int(x) for x in aux["k"].detach().cpu().tolist()],
        "a_out": ev["a_out"],
        "b_in": ev["b_in"],
        "swap": ev["swap_ba"],
        "J": _J(ev, r),
        "elapsed_s": elapsed,
        "mem_gb": _gpu_mem_gb(),
    }
    print(
        f"p1 {pair_id} {step}/{steps} L={row['loss']:.3f} "
        f"Lq={row['L_quota']:+.4f} Le={row['L_evict']:.3f} "
        f"rA={r[0]:.3f} rB={r[1]:.3f} a_out={ev['a_out']} b_in={ev['b_in']} "
        f"mem={row['mem_gb']}",
        flush=True,
    )
    return row


def exchange_search(wrapper, state, delta, cfg) -> torch.Tensor:
    atk = cfg["attack"]
    eps = float(atk["eps"])
    scales = [float(s) for s in atk.get("search_scales", [0.5, 0.75, 1.0, 1.25, 1.5])]
    xB0 = state["xB0"]
    best = clip_delta(xB0, delta.detach(), eps)
    best_j = -1e9
    linf = float(best.abs().amax().clamp(min=1e-8))
    cands = [clip_delta(xB0, best * s, eps) for s in scales]
    cands.append(clip_delta(xB0, best * (eps / linf), eps))
    with torch.no_grad():
        for cand in cands:
            xB = torch.clamp(xB0 + cand, 0.0, 1.0)
            pv = wrapper.pixels_from_x01(state["xA"], xB)
            scores = wrapper.visual_scores(wrapper.with_pixels(state["packed"], pv))
            met = _metrics_from_scores(scores, state["owner"], state["packed"].n_per_image, cfg, state["keep_clean"])
            j = _J(met["events"], met["r"])
            if j > best_j:
                best_j = j
                best = cand
    return best


def evaluate_final(wrapper, state, delta, cfg) -> Dict[str, Any]:
    pair = state["pair"]
    gold = state["gold"]
    xB = torch.clamp(state["xB0"] + delta, 0.0, 1.0)
    pv = wrapper.pixels_from_x01(state["xA"], xB).detach()
    packed = wrapper.with_pixels(state["packed"], pv)
    scores = wrapper.visual_scores(packed)
    owner = state["owner"]
    n_per = packed.n_per_image
    avtp = _metrics_from_scores(scores, owner, n_per, cfg, state["keep_clean"], "avtp")
    glob = _metrics_from_scores(scores, owner, n_per, cfg, state["keep_clean"], "global")
    ans_full = wrapper.generate(packed, keep_visual=None)
    ans_avtp = wrapper.generate(packed, keep_visual=avtp["keep"])
    keep_iso = _select_cfg(scores, owner, n_per, cfg, "isolated")[1]
    ans_iso = wrapper.generate(packed, keep_visual=keep_iso)
    keep_restore = restore_victim(avtp["keep"], state["keep_clean"], owner, victim=0)
    ans_restore = wrapper.generate(packed, keep_visual=keep_restore)
    full_ok = _ok(ans_full, pair, gold=gold)
    avtp_ok = _ok(ans_avtp, pair, gold=gold)
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
        "global": {k: v for k, v in glob.items() if k != "keep"},
        "linf": float(delta.abs().amax().cpu()),
        "comp_only_fail": bool(full_ok and not avtp_ok),
    }


def pgd_one(
    wrapper: QwenMultiImage,
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    steps: Optional[int] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    atk = cfg["attack"]
    eps = float(atk["eps"])
    alpha = float(atk["alpha"])
    steps = int(steps if steps is not None else atk["steps"])
    search_every = int(atk.get("search_every", 10))
    pair_id = state["pair"]["pair_id"]
    delta = torch.zeros_like(state["xB0"])
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    oom_fallback = False
    step = 0
    while step < steps:
        step += 1
        delta = delta.detach().requires_grad_(True)
        try:
            with torch.enable_grad():
                loss, aux, scores, _xB, _pv = _forward_loss(wrapper, state, delta, cfg)
            grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        except torch.cuda.OutOfMemoryError:
            _empty()
            if oom_fallback:
                raise
            oom_fallback = True
            print(f"p1 {pair_id} OOM at step {step}; retry with score_layers=[1]", flush=True)
            wrapper.score_layers = [1]
            step -= 1
            continue
        if grad is None:
            raise RuntimeError(f"{pair_id}: no image gradient")
        gnorm = float(grad.detach().float().abs().mean().cpu())
        delta = clip_delta(state["xB0"], delta.detach() - alpha * grad.sign(), eps)
        if search_every > 0 and step % search_every == 0:
            delta = exchange_search(wrapper, state, delta, cfg)

        with torch.no_grad():
            ev = _metrics_from_scores(
                scores.detach(), state["owner"], state["packed"].n_per_image, cfg, state["keep_clean"]
            )["events"]
        row = _log_step(pair_id, step, steps, loss, aux, ev, time.time() - t0)
        row["grad_abs_mean"] = gnorm
        row["oom_fallback_layer1"] = oom_fallback
        history.append(row)
        if progress_cb is not None:
            progress_cb(row)
        del loss, aux, scores, grad
        _empty()

    delta = exchange_search(wrapper, state, delta, cfg)
    evaled = evaluate_final(wrapper, state, delta, cfg)
    return {
        "pair_id": pair_id,
        "steps": steps,
        "history": history,
        "eval": evaled,
        "delta_linf": float(delta.abs().amax().cpu()),
        "oom_fallback_layer1": oom_fallback,
        "elapsed_s": time.time() - t0,
        "delta": delta.detach().cpu(),
        "xB_hw": list(state["xB0"].shape),
    }


def _load_kept(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = out_dir(cfg) / "screen.json"
    if not path.is_file():
        raise FileNotFoundError("P1 needs P0 screen.json")
    kept = load_json(path)["kept"]
    if not kept:
        raise RuntimeError("screen kept zero pairs")
    return kept


def _write_progress(cfg: Dict[str, Any], blob: Dict[str, Any]) -> Path:
    dest = Path(cfg["attack"]["progress_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_json(dest, blob)
    return dest


def _eta(done: int, total: int, elapsed: float) -> Optional[float]:
    if done <= 0:
        return None
    return (elapsed / done) * (total - done)


def stage_p1_smoke(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kept = _load_kept(cfg)[: int(cfg["attack"].get("n_smoke", 1))]
    Path(cfg["attack"]["delta_dir"]).mkdir(parents=True, exist_ok=True)
    wrapper = QwenMultiImage(cfg)
    rows = []
    t0 = time.time()
    for pair in kept:
        print(f"== p1 smoke {pair['pair_id']} ==", flush=True)
        state = prepare_sample(wrapper, pair, cfg)

        def _cb(row):
            _write_progress(
                cfg,
                {
                    "stage": "smoke",
                    "updated_at": _now(),
                    "pair_id": pair["pair_id"],
                    "step": row["step"],
                    "last": row,
                    "elapsed_s": time.time() - t0,
                },
            )

        out = pgd_one(wrapper, state, cfg, steps=int(cfg["attack"]["smoke_steps"]), progress_cb=_cb)
        delta = out.pop("delta")
        torch.save(delta, Path(cfg["attack"]["delta_dir"]) / f"{pair['pair_id']}_smoke.pt")
        ev = out["eval"]["avtp"]["events"]
        rows.append({k: v for k, v in out.items()})
        _write_progress(
            cfg,
            {
                "stage": "smoke",
                "updated_at": _now(),
                "elapsed_s": time.time() - t0,
                "n": len(rows),
                "last": {"pair_id": pair["pair_id"], "a_out": ev["a_out"], "b_in": ev["b_in"], "full_ok": out["eval"]["full_ok"]},
                "rows": rows,
            },
        )
        _empty()
    dest = out_dir(cfg) / "p1_smoke.json"
    save_json(dest, {"n": len(rows), "elapsed_s": time.time() - t0, "rows": rows})
    return {"n": len(rows), "path": str(dest), "elapsed_s": time.time() - t0}


def stage_p1_attack(cfg: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
    kept = _load_kept(cfg)
    if limit is not None:
        kept = kept[: int(limit)]
    dest = out_dir(cfg) / "p1_attack.json"
    done_ids = set()
    rows: List[Dict[str, Any]] = []
    if dest.is_file():
        prev = load_json(dest)
        prev_rows = list(prev.get("rows") or [])
        rows = [r for r in prev_rows if r.get("eval") and not r.get("error")]
        done_ids = {r["pair_id"] for r in rows}
        print(f"resume {len(done_ids)} finished samples", flush=True)
    pending = [p for p in kept if p["pair_id"] not in done_ids]
    wrapper = QwenMultiImage(cfg)
    t0 = time.time()
    total = len(kept)
    already = len(done_ids)
    Path(cfg["attack"]["delta_dir"]).mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pending, start=1):
        print(f"== p1 attack {pair['pair_id']} ({already + i}/{total}) ==", flush=True)
        sample_t0 = time.time()
        state = None
        try:
            state = prepare_sample(wrapper, pair, cfg)

            def _cb(row, _pair=pair, _done=already + i, _total=total, _t0=t0):
                eta = _eta(_done - already, max(len(pending), 1), time.time() - _t0)
                _write_progress(
                    cfg,
                    {
                        "stage": "attack",
                        "updated_at": _now(),
                        "pair_id": _pair["pair_id"],
                        "done": _done - 1,
                        "total": _total,
                        "step": row["step"],
                        "eta_s": eta,
                        "last": row,
                    },
                )

            out = pgd_one(wrapper, state, cfg, progress_cb=_cb)
            delta = out.pop("delta")
            torch.save(delta, Path(cfg["attack"]["delta_dir"]) / f"{pair['pair_id']}.pt")
            out["error"] = None
        except Exception as exc:
            print(f"p1 {pair['pair_id']} FAILED {type(exc).__name__}: {exc}", flush=True)
            out = {
                "pair_id": pair["pair_id"],
                "error": f"{type(exc).__name__}: {exc}",
                "eval": None,
                "elapsed_s": time.time() - sample_t0,
            }
            _empty()
        rows.append(out)
        elapsed = time.time() - t0
        done = already + i
        eta = _eta(i, len(pending), elapsed)
        blob = {
            "stage": "attack",
            "updated_at": _now(),
            "done": done,
            "total": total,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "last": {
                "pair_id": pair["pair_id"],
                "error": out.get("error"),
                "eval": None if out.get("eval") is None else {
                    "full_ok": out["eval"]["full_ok"],
                    "avtp_ok": out["eval"]["avtp_ok"],
                    "comp_only_fail": out["eval"]["comp_only_fail"],
                    "a_out": out["eval"]["avtp"]["events"]["a_out"],
                    "b_in": out["eval"]["avtp"]["events"]["b_in"],
                    "delta_r": [
                        out["eval"]["avtp"]["r"][j] - state["r_clean"][j]
                        for j in range(2)
                    ]
                    if out.get("eval") is not None
                    else None,
                },
            },
            "rows": rows,
        }
        # drop keep tensors already stripped
        save_json(dest, {"n": len(rows), "elapsed_s": elapsed, "rows": _strip(rows)})
        _write_progress(cfg, _strip(blob))
        _empty()
    save_json(dest, {"n": len(rows), "elapsed_s": time.time() - t0, "rows": _strip(rows)})
    return {"n": len(rows), "path": str(dest)}


def _strip(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in {"keep", "delta", "packed", "xA", "xB0", "owner", "u_mask", "v_mask", "h_clean", "keep_clean"}}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj
