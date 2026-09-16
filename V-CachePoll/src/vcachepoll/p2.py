"""Identify task-critical A tokens by drop / leave-one-out tests."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import torch

from .attack import _empty, _load_kept, _now, _result_path, _write_progress, prepare_sample
from .config import out_dir
from .io import save_json
from .model import QwenMultiImage
from .probe import _ok


def _ranked_a(scores: torch.Tensor, owner: torch.Tensor, keep: torch.Tensor, largest: bool) -> torch.Tensor:
    idx = torch.nonzero((owner == 0) & keep, as_tuple=False).flatten()
    if idx.numel() == 0:
        return idx
    order = torch.argsort(scores[idx], descending=largest)
    return idx[order]


def _drop(keep: torch.Tensor, owner: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    out = keep.clone()
    if drop.numel():
        out[drop] = False
    if int(((owner == 0) & out).sum().item()) == 0:
        a_idx = torch.nonzero(owner == 0, as_tuple=False).flatten()
        if a_idx.numel():
            out[a_idx[0]] = True
    return out


def _gen_ok(wrapper, packed, keep, pair, gold) -> Dict[str, Any]:
    ans = wrapper.generate(packed, keep_visual=keep)
    return {"ok": bool(_ok(ans, pair, gold=gold)), "ans": ans}


def probe_critical_one(wrapper: QwenMultiImage, pair: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    state = prepare_sample(wrapper, pair, cfg)
    packed = state["packed"]
    keep = state["keep_clean"]
    owner = state["owner"]
    gold = state["gold"]
    with torch.no_grad():
        scores = wrapper.visual_scores(packed)
    top = _ranked_a(scores, owner, keep, largest=True)
    bot = _ranked_a(scores, owner, keep, largest=False)
    m = int(cfg["attack"].get("crit_drop_m", 10))
    loo_n = int(cfg["attack"].get("crit_loo_n", 12))
    m = min(m, int(top.numel()))
    loo_n = min(loo_n, int(top.numel()))

    clean = _gen_ok(wrapper, packed, keep, pair, gold)
    drop_top = _gen_ok(wrapper, packed, _drop(keep, owner, top[:m]), pair, gold) if m else dict(clean)
    drop_bot = _gen_ok(wrapper, packed, _drop(keep, owner, bot[:m]), pair, gold) if m else dict(clean)
    half = max(m, int(top.numel() // 2))
    drop_top_half = _gen_ok(wrapper, packed, _drop(keep, owner, top[:half]), pair, gold) if half else dict(clean)

    loo_hits: List[int] = []
    loo_rows = []
    for i in range(loo_n):
        tok = int(top[i].item())
        rec = _gen_ok(wrapper, packed, _drop(keep, owner, top[i : i + 1]), pair, gold)
        loo_rows.append({"idx": tok, "ok": rec["ok"], "ans": rec["ans"]})
        if clean["ok"] and not rec["ok"]:
            loo_hits.append(tok)

    if loo_hits:
        u_idx, u_src = loo_hits, "loo"
    elif clean["ok"] and not drop_top["ok"]:
        u_idx, u_src = [int(x) for x in top[:m].tolist()], "drop_top"
    elif clean["ok"] and not drop_bot["ok"]:
        u_idx, u_src = [int(x) for x in bot[:m].tolist()], "drop_bot"
    else:
        n_top = max(1, int(round(0.25 * max(int(bot.numel()), 1))))
        # Quota steal evicts the lowest A survivors first.
        u_idx, u_src = [int(x) for x in bot[:n_top].tolist()], "bot_frac"

    return {
        "pair_id": pair["pair_id"],
        "clean_ok": clean["ok"],
        "ans_clean": clean["ans"],
        "drop_top_ok": drop_top["ok"],
        "drop_bot_ok": drop_bot["ok"],
        "drop_top_half_ok": drop_top_half["ok"],
        "ans_drop_top": drop_top["ans"],
        "ans_drop_bot": drop_bot["ans"],
        "k_a": state["k_clean"][0],
        "n_keep_a": int(((owner == 0) & keep).sum().item()),
        "loo_n": loo_n,
        "loo_hits": loo_hits,
        "top_idx": [int(x) for x in top[: max(m, loo_n)].tolist()],
        "bot_idx": [int(x) for x in bot[:m].tolist()],
        "u_idx": u_idx,
        "u_src": u_src,
        "n_u": len(u_idx),
        "localized": bool(loo_hits) or (clean["ok"] and (not drop_top["ok"] or not drop_bot["ok"])),
    }


def stage_p2_screen(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keep P0 clean-correct pairs that still answer correctly at this r_base."""
    src = out_dir(cfg) / "screen.json"
    from .io import load_json

    kept0 = load_json(src)["kept"]
    wrapper = QwenMultiImage(cfg)
    kept, rejected = [], []
    t0 = time.time()
    dest = _result_path(cfg, "screen_path", "p2_screen.json")
    for pair in kept0:
        state = prepare_sample(wrapper, pair, cfg)
        rec = _gen_ok(wrapper, state["packed"], state["keep_clean"], pair, state["gold"])
        item = {
            **pair,
            "p2_screen": {
                "avtp_ok": rec["ok"],
                "ans": rec["ans"],
                "r": state["r_clean"],
                "k": state["k_clean"],
            },
        }
        if rec["ok"]:
            kept.append(item)
        else:
            rejected.append({"pair_id": pair["pair_id"], "ans": rec["ans"], "k": state["k_clean"]})
        print(
            f"p2 screen {pair['pair_id']} ok={rec['ok']} k={state['k_clean']} r={ [round(x,3) for x in state['r_clean']] }",
            flush=True,
        )
        save_json(dest, {"n_kept": len(kept), "n_rejected": len(rejected), "kept": kept, "rejected": rejected})
        _write_progress(cfg, {"stage": "screen", "updated_at": _now(), "n_kept": len(kept), "n_rejected": len(rejected)})
        _empty()
    save_json(dest, {"n_kept": len(kept), "n_rejected": len(rejected), "kept": kept, "rejected": rejected})
    return {"n_kept": len(kept), "n_rejected": len(rejected), "path": str(dest), "elapsed_s": time.time() - t0}


def stage_p2_crit(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kept = _load_kept(cfg)
    wrapper = QwenMultiImage(cfg)
    rows = []
    t0 = time.time()
    dest = out_dir(cfg) / "p2_crit.json"
    for i, pair in enumerate(kept, start=1):
        print(f"== p2 crit {pair['pair_id']} ({i}/{len(kept)}) ==", flush=True)
        rec = probe_critical_one(wrapper, pair, cfg)
        rows.append(rec)
        print(
            f"p2 crit {pair['pair_id']} localized={rec['localized']} u_src={rec['u_src']} "
            f"n_u={rec['n_u']} drop_top={rec['drop_top_ok']} drop_bot={rec['drop_bot_ok']} loo={len(rec['loo_hits'])}",
            flush=True,
        )
        save_json(dest, {"n": len(rows), "rows": rows})
        _write_progress(
            cfg,
            {"stage": "crit", "updated_at": _now(), "done": i, "total": len(kept), "last": rec, "elapsed_s": time.time() - t0},
        )
        _empty()
    n_loc = sum(1 for r in rows if r["localized"])
    save_json(dest, {"n": len(rows), "n_localized": n_loc, "elapsed_s": time.time() - t0, "rows": rows})
    screen_path = _result_path(cfg, "screen_path", "p2_screen.json")
    from .io import load_json

    screen = load_json(screen_path)
    screen["kept"] = attach_u_idx(screen["kept"], dest)
    save_json(screen_path, screen)
    return {"n": len(rows), "n_localized": n_loc, "path": str(dest)}


def reassign_u_from_crit(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild U from saved drop/LOO flags plus a cheap score ranking. No generate."""
    from .io import load_json

    dest = out_dir(cfg) / "p2_crit.json"
    blob = load_json(dest)
    kept = _load_kept(cfg)
    by_pair = {p["pair_id"]: p for p in kept}
    wrapper = QwenMultiImage(cfg)
    m = int(cfg["attack"].get("crit_drop_m", 10))
    rows = []
    for rec in blob.get("rows") or []:
        pair = by_pair.get(rec["pair_id"])
        if pair is None:
            rows.append(rec)
            continue
        state = prepare_sample(wrapper, pair, cfg)
        with torch.no_grad():
            scores = wrapper.visual_scores(state["packed"])
        top = _ranked_a(scores, state["owner"], state["keep_clean"], largest=True)
        bot = _ranked_a(scores, state["owner"], state["keep_clean"], largest=False)
        mm = min(m, int(bot.numel()), int(top.numel()))
        loo_hits = list(rec.get("loo_hits") or [])
        if loo_hits:
            u_idx, u_src = loo_hits, "loo"
        elif rec.get("clean_ok") and not rec.get("drop_top_ok"):
            u_idx, u_src = [int(x) for x in top[:mm].tolist()], "drop_top"
        elif rec.get("clean_ok") and not rec.get("drop_bot_ok"):
            u_idx, u_src = [int(x) for x in bot[:mm].tolist()], "drop_bot"
        else:
            n_bot = max(1, int(round(0.25 * max(int(bot.numel()), 1))))
            u_idx, u_src = [int(x) for x in bot[:n_bot].tolist()], "bot_frac"
        rec = dict(rec)
        rec["u_idx"] = u_idx
        rec["u_src"] = u_src
        rec["n_u"] = len(u_idx)
        rec["bot_idx"] = [int(x) for x in bot[:mm].tolist()]
        rec["top_idx"] = [int(x) for x in top[:mm].tolist()]
        rec["localized"] = bool(loo_hits) or (
            bool(rec.get("clean_ok")) and (not rec.get("drop_top_ok") or not rec.get("drop_bot_ok"))
        )
        rows.append(rec)
        print(f"p2 reassign {rec['pair_id']} u_src={u_src} n_u={len(u_idx)} localized={rec['localized']}", flush=True)
        _empty()
    n_loc = sum(1 for r in rows if r["localized"])
    save_json(dest, {"n": len(rows), "n_localized": n_loc, "rows": rows, "reassigned": True})
    screen_path = _result_path(cfg, "screen_path", "p2_screen.json")
    screen = load_json(screen_path)
    screen["kept"] = attach_u_idx(screen["kept"], dest)
    save_json(screen_path, screen)
    return {"n": len(rows), "n_localized": n_loc}


def attach_u_idx(pairs: Sequence[Dict[str, Any]], crit_path) -> List[Dict[str, Any]]:
    from .io import load_json

    blob = load_json(crit_path)
    by = {r["pair_id"]: r for r in blob.get("rows") or []}
    out = []
    for p in pairs:
        rec = by.get(p["pair_id"])
        item = dict(p)
        if rec:
            item["u_idx"] = rec["u_idx"]
            item["u_src"] = rec["u_src"]
        out.append(item)
    return out
