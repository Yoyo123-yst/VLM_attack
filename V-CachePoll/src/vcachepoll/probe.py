from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .coco_pairs import answers_agree, mentions_source, novel_words, write_pairs
from .compressor import (
    exchange_events,
    image_quotas,
    isolated_k,
    mean_importance_per_image,
    restore_victim,
    select_global_topk,
    select_within_image,
    summarize_keep,
)
from .config import out_dir
from .io import load_json, save_json
from .model import QwenMultiImage, load_pair_images, perturb_linf


def _ok(answer: str, pair: Dict[str, Any], gold: Optional[str] = None) -> bool:
    prompt = pair["prompt"]
    source = pair["source"]
    if gold:
        return answers_agree(answer, gold, prompt, source=source)
    return mentions_source(answer, source)


def _tensor_list(t: torch.Tensor) -> List[float]:
    return [float(x) for x in t.detach().cpu().flatten().tolist()]
    return [float(x) for x in t.detach().cpu().flatten().tolist()]


def _select(scores: torch.Tensor, owner: torch.Tensor, n_per, cfg: Dict[str, Any], mode: str):
    r_base = float(cfg["compressor"]["r_base"])
    if mode == "avtp":
        i_bar = mean_importance_per_image(scores, owner, n_images=len(n_per))
        quota = image_quotas(
            i_bar,
            n_per,
            r_base=r_base,
            alpha=float(cfg["compressor"]["alpha"]),
            r_min=float(cfg["compressor"]["r_min"]),
            r_max=float(cfg["compressor"]["r_max"]),
        )
        keep = select_within_image(scores, owner, quota.k)
        return quota, keep
    if mode == "global":
        k_total = max(1, int(round(r_base * int(scores.numel()))))
        dummy = mean_importance_per_image(scores, owner, n_images=len(n_per))
        quota = image_quotas(
            dummy,
            n_per,
            r_base=r_base,
            alpha=float(cfg["compressor"]["alpha"]),
            r_min=float(cfg["compressor"]["r_min"]),
            r_max=float(cfg["compressor"]["r_max"]),
        )
        keep = select_global_topk(scores, k_total)
        return quota, keep
    if mode == "isolated":
        i_bar = mean_importance_per_image(scores, owner, n_images=len(n_per))
        quota = image_quotas(i_bar, n_per, r_base=r_base, alpha=0.0)
        keep = select_within_image(scores, owner, isolated_k(n_per, r_base))
        return quota, keep
    raise KeyError(mode)


def _pack_record(quota, keep, owner, n_per, mode: str) -> Dict[str, Any]:
    return {
        "mode": mode,
        "i_bar": _tensor_list(quota.i_bar),
        "i_avg": float(quota.i_avg),
        "r": _tensor_list(quota.r),
        "k": [int(x) for x in quota.k.tolist()],
        "n": [int(x) for x in n_per],
        "kept": summarize_keep(keep, owner, n_images=len(n_per)),
        "keep_idx": torch.nonzero(keep, as_tuple=False).flatten().cpu().tolist(),
    }


def stage_pairs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dest = out_dir(cfg) / "pairs.json"
    pairs = write_pairs(cfg, dest)
    return {"n": len(pairs), "path": str(dest)}


def _load_pairs(cfg: Dict[str, Any], n: Optional[int] = None) -> List[Dict[str, Any]]:
    path = out_dir(cfg) / "pairs.json"
    if not path.is_file():
        stage_pairs(cfg)
    blob = load_json(path)
    pairs = blob["pairs"]
    if n is not None:
        pairs = pairs[:n]
    return pairs


def _eval_pair(
    wrapper: QwenMultiImage,
    pair: Dict[str, Any],
    cfg: Dict[str, Any],
    b_image=None,
    modes=("avtp", "global"),
    with_full: bool = True,
    gold_answer: Optional[str] = None,
) -> Dict[str, Any]:
    a, b = load_pair_images(pair, b_image=b_image)
    packed = wrapper.pack([a, b], pair["prompt"])
    owner = wrapper.owner(packed)
    scores = wrapper.visual_scores(packed)
    gold = gold_answer
    rec: Dict[str, Any] = {
        "pair_id": pair["pair_id"],
        "source": pair["source"],
        "n_per_image": packed.n_per_image,
        "compress": {},
    }
    if with_full:
        rec["ans_full"] = wrapper.generate(packed, keep_visual=None)
        rec["full_ok"] = _ok(rec["ans_full"], pair, gold=gold)
    for mode in modes:
        quota, keep = _select(scores, owner, packed.n_per_image, cfg, mode)
        ans = wrapper.generate(packed, keep_visual=keep)
        rec["compress"][mode] = {
            **_pack_record(quota, keep, owner, packed.n_per_image, mode),
            "answer": ans,
            "ok": _ok(ans, pair, gold=gold),
            "keep": keep.cpu(),
            "owner": owner.cpu(),
        }
    rec["_scores"] = scores.cpu()
    rec["_packed_n"] = packed.n_per_image
    return rec


def stage_smoke(cfg: Dict[str, Any]) -> Dict[str, Any]:
    n = int(cfg["data"]["n_smoke"])
    pairs = _load_pairs(cfg, n=n)
    wrapper = QwenMultiImage(cfg)
    rows = []
    for pair in pairs:
        a, _ = load_pair_images(pair)
        packed_a = wrapper.pack([a], pair["prompt"])
        ans_a = wrapper.generate(packed_a, keep_visual=None)
        rec = _eval_pair(wrapper, pair, cfg, gold_answer=ans_a)
        row = {
            "pair_id": rec["pair_id"],
            "source": rec["source"],
            "n_per_image": rec["n_per_image"],
            "ans_a_only": ans_a,
            "a_only_ok": bool(novel_words(ans_a, pair["prompt"]) or mentions_source(ans_a, pair["source"])),
            "ans_full": rec.get("ans_full"),
            "full_ok": rec.get("full_ok"),
            "avtp": {k: v for k, v in rec["compress"]["avtp"].items() if k not in {"keep", "owner"}},
            "global": {k: v for k, v in rec["compress"]["global"].items() if k not in {"keep", "owner"}},
        }
        rows.append(row)
        print(
            f"smoke {pair['pair_id']} a_ok={row['a_only_ok']} full_ok={row['full_ok']} "
            f"avtp_ok={row['avtp']['ok']} r={row['avtp']['r']}",
            flush=True,
        )
    dest = out_dir(cfg) / "smoke.json"
    save_json(dest, {"n": len(rows), "rows": rows})
    return {"n": len(rows), "path": str(dest)}


def stage_screen(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pairs = _load_pairs(cfg, n=int(cfg["data"]["n_screen_max"]))
    wrapper = QwenMultiImage(cfg)
    kept = []
    rejected = []
    for pair in pairs:
        a, _ = load_pair_images(pair)
        packed_a = wrapper.pack([a], pair["prompt"])
        ans_a = wrapper.generate(packed_a, keep_visual=None)
        a_has_content = bool(novel_words(ans_a, pair["prompt"]) or mentions_source(ans_a, pair["source"]))
        rec = _eval_pair(wrapper, pair, cfg, modes=("avtp",), gold_answer=ans_a)
        avtp_ok = bool(rec["compress"]["avtp"]["ok"])
        full_ok = bool(rec.get("full_ok"))
        item = {
            "pair_id": pair["pair_id"],
            "source": pair["source"],
            "a_only_ok": a_has_content,
            "full_ok": full_ok,
            "avtp_ok": avtp_ok,
            "ans_a_only": ans_a,
            "ans_full": rec.get("ans_full"),
            "ans_avtp": rec["compress"]["avtp"]["answer"],
            "r": rec["compress"]["avtp"]["r"],
            "kept": rec["compress"]["avtp"]["kept"],
        }
        if a_has_content and full_ok and avtp_ok:
            kept.append({**pair, "ans_a_only": ans_a, "screen": item})
        else:
            rejected.append(item)
        print(
            f"screen {pair['pair_id']} keep={a_has_content and full_ok and avtp_ok} "
            f"a={a_has_content} full={full_ok} avtp={avtp_ok}",
            flush=True,
        )
        save_json(
            out_dir(cfg) / "screen.json",
            {"n_kept": len(kept), "n_rejected": len(rejected), "kept": kept, "rejected": rejected},
        )
    return {"n_kept": len(kept), "n_rejected": len(rejected)}


def stage_probe(cfg: Dict[str, Any]) -> Dict[str, Any]:
    screen_path = out_dir(cfg) / "screen.json"
    if not screen_path.is_file():
        raise FileNotFoundError("run screen before probe")
    kept = load_json(screen_path)["kept"]
    if not kept:
        raise RuntimeError("screen kept zero pairs")
    wrapper = QwenMultiImage(cfg)
    eps = float(cfg["probe"]["eps"])
    seed = int(cfg["seed"])
    rows = []
    for pair in kept:
        gold = pair.get("ans_a_only") or pair.get("screen", {}).get("ans_a_only")
        clean = _eval_pair(wrapper, pair, cfg, modes=("avtp", "global"), gold_answer=gold)
        b_adv = perturb_linf(
            load_pair_images(pair)[1],
            eps=eps,
            seed=seed + _hash_pair(pair["pair_id"]),
        )
        adv = _eval_pair(
            wrapper, pair, cfg, b_image=b_adv, modes=("avtp", "global", "isolated"), gold_answer=gold
        )
        row: Dict[str, Any] = {
            "pair_id": pair["pair_id"],
            "source": pair["source"],
            "full_clean_ok": clean.get("full_ok"),
            "full_adv_ok": adv.get("full_ok"),
            "ans_full_clean": clean.get("ans_full"),
            "ans_full_adv": adv.get("ans_full"),
        }
        for mode in ("avtp", "global"):
            c = clean["compress"][mode]
            a = adv["compress"][mode]
            ev = exchange_events(c["keep"], a["keep"], c["owner"])
            dr = [float(a["r"][i] - c["r"][i]) for i in range(len(c["r"]))]
            row[mode] = {
                "clean": {k: v for k, v in c.items() if k not in {"keep", "owner"}},
                "adv": {k: v for k, v in a.items() if k not in {"keep", "owner"}},
                "delta_r": dr,
                "events": ev,
                "comp_adv_ok": a["ok"],
                "comp_clean_ok": c["ok"],
            }
        # Causal interventions on AVTP when A tokens actually left.
        avtp_ev = row["avtp"]["events"]
        row["causal"] = None
        if avtp_ev["a_out"] > 0:
            packed = wrapper.pack(list(load_pair_images(pair, b_image=b_adv)), pair["prompt"])
            owner = wrapper.owner(packed)
            scores = wrapper.visual_scores(packed)
            quota, keep_adv = _select(scores, owner, packed.n_per_image, cfg, "avtp")
            keep_clean = clean["compress"]["avtp"]["keep"].to(keep_adv.device)
            keep_iso = _select(scores, owner, packed.n_per_image, cfg, "isolated")[1]
            keep_restore = restore_victim(keep_adv, keep_clean, owner, victim=0)
            ans_iso = wrapper.generate(packed, keep_visual=keep_iso)
            ans_restore = wrapper.generate(packed, keep_visual=keep_restore)
            row["causal"] = {
                "isolated_ok": _ok(ans_iso, pair, gold=gold),
                "restore_a_ok": _ok(ans_restore, pair, gold=gold),
                "ans_isolated": ans_iso,
                "ans_restore_a": ans_restore,
                "full_adv_ok": adv.get("full_ok"),
            }
        rows.append({k: v for k, v in row.items()})
        # drop tensors before save
        serial = _strip_tensors(row)
        save_json(out_dir(cfg) / "probe.json", {"n": len(rows), "eps": eps, "rows": [_strip_tensors(r) for r in rows]})
        print(
            f"probe {pair['pair_id']} d_rA={row['avtp']['delta_r'][0]:+.3f} "
            f"a_out={avtp_ev['a_out']} b_in={avtp_ev['b_in']} "
            f"full_adv={row['full_adv_ok']} avtp_adv={row['avtp']['comp_adv_ok']}",
            flush=True,
        )
        _ = serial
    save_json(out_dir(cfg) / "probe.json", {"n": len(rows), "eps": eps, "rows": [_strip_tensors(r) for r in rows]})
    return {"n": len(rows)}


def _hash_pair(pair_id: str) -> int:
    return sum(ord(c) for c in pair_id)


def _strip_tensors(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _strip_tensors(v) for k, v in obj.items() if k not in {"keep", "owner"} and not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_tensors(x) for x in obj]
    return obj
