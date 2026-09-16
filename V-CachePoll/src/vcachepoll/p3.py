"""P3 same-protocol baselines on the P2 32-pair pool."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .attack import (
    _J,
    _empty,
    _eta,
    _load_kept,
    _log_prefix,
    _metrics_from_scores,
    _now,
    _result_path,
    _strip,
    _write_progress,
    evaluate_final,
    pgd_one,
    prepare_sample,
    stage_p1_attack,
)
from .config import out_dir
from .io import save_json
from .model import QwenMultiImage
from .vision import clip_delta


def _method_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    c = deepcopy(cfg)
    atk = c["attack"]
    root = out_dir(c)
    atk["log_prefix"] = f"p3-{name}"
    atk["result_path"] = str(root / f"p3_{name}.json")
    atk["progress_path"] = str(root / f"p3_{name}_progress.json")
    atk["delta_dir"] = str(root / "p3_deltas" / name)
    atk["smoke_path"] = str(root / f"p3_{name}_smoke.json")
    if name == "random":
        atk["loss_kind"] = "random"
        atk["search_every"] = 0
    elif name == "task":
        atk["loss_kind"] = "task"
        atk["search_every"] = 0
    elif name == "caa":
        atk["loss_kind"] = "caa"
        atk["search_every"] = 0
    elif name == "cage":
        atk["loss_kind"] = "cage"
        atk["search_every"] = 0
    elif name == "rank":
        atk["loss_kind"] = "rank"
        atk["search_every"] = 0
        atk["lambda_c"] = 0.0
    elif name == "vcache":
        atk["loss_kind"] = "vcache"
        atk["search_every"] = int(cfg["attack"].get("search_every", 5))
    return c


def random_one(wrapper: QwenMultiImage, state: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    atk = cfg["attack"]
    eps = float(atk["eps"])
    steps = int(atk["steps"])
    pair_id = state["pair"]["pair_id"]
    xB0 = state["xB0"]
    best = torch.zeros_like(xB0)
    best_j = -1e9
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    seed0 = int(cfg["seed"]) + sum(ord(ch) for ch in pair_id)
    for step in range(1, steps + 1):
        g = torch.Generator(device=xB0.device)
        g.manual_seed(seed0 + step)
        noise = (torch.rand(xB0.shape, generator=g, device=xB0.device) * 2.0 - 1.0) * eps
        cand = clip_delta(xB0, noise, eps)
        with torch.no_grad():
            pv = wrapper.pixels_from_x01(state["xA"], torch.clamp(xB0 + cand, 0.0, 1.0))
            scores = wrapper.visual_scores(wrapper.with_pixels(state["packed"], pv))
            met = _metrics_from_scores(scores, state["owner"], state["packed"].n_per_image, cfg, state["keep_clean"])
            u_out = int((state["u_mask"].to(met["keep"].device) & ~met["keep"]).sum().item())
            j = _J(met["events"], met["r"], u_out=u_out)
        if j > best_j:
            best_j = j
            best = cand
        row = {
            "step": step,
            "loss": 0.0,
            "L_quota": 0.0,
            "L_evict": 0.0,
            "L_value": 0.0,
            "L_perc": 0.0,
            "r": met["r"],
            "k": met["k"],
            "a_out": met["events"]["a_out"],
            "b_in": met["events"]["b_in"],
            "u_out": u_out,
            "swap": met["events"]["swap_ba"],
            "J": j,
            "best_j": best_j,
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        print(
            f"{_log_prefix(cfg)} {pair_id} {step}/{steps} J={j:.1f} best={best_j:.1f} "
            f"a_out={row['a_out']} u_out={u_out} rA={met['r'][0]:.3f}",
            flush=True,
        )
        _empty()
    evaled = evaluate_final(wrapper, state, best, cfg)
    return {
        "pair_id": pair_id,
        "steps": steps,
        "history": history,
        "eval": evaled,
        "delta_linf": float(best.abs().amax().cpu()),
        "elapsed_s": time.time() - t0,
        "delta": best.detach().cpu(),
        "method": "random",
    }


def stage_p3_random(cfg: Dict[str, Any], wrapper: QwenMultiImage, limit: Optional[int] = None) -> Dict[str, Any]:
    kept = _load_kept(cfg)
    if limit is not None:
        kept = kept[: int(limit)]
    dest = _result_path(cfg, "result_path", "p3_random.json")
    rows: List[Dict[str, Any]] = []
    done_ids = set()
    if dest.is_file():
        from .io import load_json

        prev = load_json(dest)
        rows = [r for r in (prev.get("rows") or []) if r.get("eval") and not r.get("error")]
        done_ids = {r["pair_id"] for r in rows}
        print(f"resume {len(done_ids)} random samples", flush=True)
    pending = [p for p in kept if p["pair_id"] not in done_ids]
    Path(cfg["attack"]["delta_dir"]).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = len(kept)
    already = len(done_ids)
    for i, pair in enumerate(pending, start=1):
        print(f"== {_log_prefix(cfg)} {pair['pair_id']} ({already + i}/{total}) ==", flush=True)
        try:
            state = prepare_sample(wrapper, pair, cfg)
            out = random_one(wrapper, state, cfg)
            delta = out.pop("delta")
            torch.save(delta, Path(cfg["attack"]["delta_dir"]) / f"{pair['pair_id']}.pt")
            out["error"] = None
        except Exception as exc:
            print(f"{_log_prefix(cfg)} {pair['pair_id']} FAILED {type(exc).__name__}: {exc}", flush=True)
            out = {"pair_id": pair["pair_id"], "error": f"{type(exc).__name__}: {exc}", "eval": None}
            _empty()
        rows.append(out)
        elapsed = time.time() - t0
        save_json(dest, {"n": len(rows), "method": "random", "elapsed_s": elapsed, "rows": _strip(rows)})
        _write_progress(
            cfg,
            {
                "stage": "random",
                "updated_at": _now(),
                "done": already + i,
                "total": total,
                "eta_s": _eta(i, len(pending), elapsed),
                "last": {"pair_id": pair["pair_id"], "error": out.get("error")},
            },
        )
        _empty()
    return {"n": len(rows), "path": str(dest)}


def stage_p3_all(cfg: Dict[str, Any], limit: Optional[int] = None, methods: Optional[List[str]] = None) -> Dict[str, Any]:
    methods = methods or ["random", "task", "caa", "rank"]
    wrapper = QwenMultiImage(cfg)
    summary = {}
    for name in methods:
        print(f"==== P3 METHOD {name} ====", flush=True)
        c = _method_cfg(cfg, name)
        if name == "random":
            summary[name] = stage_p3_random(c, wrapper, limit=limit)
        else:
            summary[name] = stage_p1_attack(c, limit=limit, wrapper=wrapper)
        _empty()
    return summary
