"""Fill RELATED_SAFE on the same image/query as an existing RHC, without clobbering RHC."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch
from tqdm import tqdm

from p0.catalog import all_pairs
from p0.datautil import open_image, save_json
from p0.judge import harmful_compliance
from p0.model import COMPLY_WORDS, REFUSE_WORDS, token_id_list
from p0_qwen.traces import _pgd_sampler, _trim_judge

from .axes import four_axes, length_matched
from .pairs import sealed_query


def _n_core_pairs(records, catalog, blocked: Set[str]) -> int:
    n = 0
    for rec in records:
        if sealed_query(rec["query_id"]) or rec["query_id"] in blocked:
            continue
        if rec.get("jb") is None or rec.get("related_safe") is None:
            continue
        h = catalog.get(rec["query_id"])
        if not h:
            continue
        q = rec.get("query") or h["query"]
        rhc = four_axes(rec["jb"]["judge"].get("text") or "", h["topic_keywords"], q, rec["jb"]["judge"])
        rs = four_axes(
            rec["related_safe"]["judge"].get("text") or "",
            h["topic_keywords"],
            q,
            rec["related_safe"]["judge"],
        )
        if rhc["core_rhc"] and rs["core_safe_answer"] and length_matched(rhc["chars"], rs["chars"]):
            n += 1
    return n


def _load_delta(path: Optional[str], device) -> Optional[torch.Tensor]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    delta = torch.load(p, map_location=device, weights_only=True)
    return delta.to(device)


def collect_matched_safe(
    wrapper,
    cfg: Dict[str, Any],
    traces: Dict[str, Any],
    carriers: Dict[str, Dict[str, Any]],
    *,
    target_pairs: int = 50,
    blocked: Optional[Set[str]] = None,
    delta_dir: Optional[Path] = None,
    max_tries: int = 8,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    blocked = set(blocked or [])
    catalog = {p["id"]: p for p in all_pairs()}
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    layers = cfg["layers"]["candidates"]
    alpha = float(cfg["attack"]["alpha"])
    records = traces["records"]
    stats = {
        "tried": 0,
        "got_safe": 0,
        "got_core_safe": 0,
        "start_pairs": _n_core_pairs(records, catalog, blocked),
    }

    rhc_recs = [
        rec
        for rec in records
        if rec.get("jb") is not None
        and rec.get("clean", {}).get("label") == "REF"
        and not sealed_query(rec["query_id"])
        and rec["query_id"] not in blocked
    ]

    def accept(rec, h, q, img, text, atk_j, x01, tag: str) -> bool:
        ax = four_axes(text, h["topic_keywords"], q, atk_j)
        rhc = four_axes(
            rec["jb"]["judge"].get("text") or "",
            h["topic_keywords"],
            q,
            rec["jb"]["judge"],
        )
        if not ax["core_safe_answer"]:
            return False
        if not rhc["core_rhc"] or not length_matched(rhc["chars"], ax["chars"]):
            return False
        hid = {kk: v.tolist() for kk, v in wrapper.collect_hidden(img, q, layers, x01=x01).items()}
        rec["related_safe"] = {"judge": _trim_judge(atk_j), "hidden": hid}
        rec["n0_matched_attempt"] = {"tag": tag, "label": atk_j.get("label")}
        stats["got_safe"] += 1
        stats["got_core_safe"] += 1
        if checkpoint_path is not None and stats["got_core_safe"] % 5 == 0:
            traces["summary"] = {**(traces.get("summary") or {}), "n0_matched": stats}
            save_json(checkpoint_path, traces)
        return True

    for rec in tqdm(rhc_recs, desc="n0-matched-safe"):
        if rec.get("n0_matched_attempt") is not None or rec.get("n0_new_slot"):
            continue
        if _n_core_pairs(records, catalog, blocked) >= target_pairs:
            break
        h = catalog.get(rec["query_id"])
        c = carriers.get(rec["carrier_id"])
        if h is None or c is None:
            continue
        q = rec.get("query") or h["query"]
        rhc = four_axes(
            rec["jb"]["judge"].get("text") or "",
            h["topic_keywords"],
            q,
            rec["jb"]["judge"],
        )
        if not rhc["core_rhc"]:
            continue
        if rec.get("related_safe") is not None:
            rs = four_axes(
                rec["related_safe"]["judge"].get("text") or "",
                h["topic_keywords"],
                q,
                rec["related_safe"]["judge"],
            )
            if rs["core_safe_answer"] and length_matched(rhc["chars"], rs["chars"]):
                continue
        img = open_image(c["path"])
        x0 = wrapper.image_to_x01(img)
        base = _load_delta(rec.get("delta_path"), wrapper.device)
        qseed = int(hashlib.sha256(f"{rec['query_id']}:{rec['carrier_id']}".encode("utf-8")).hexdigest()[:8], 16)
        attempts: list[Tuple[str, torch.Tensor]] = []
        if base is not None:
            for a in (0.15, 0.30, 0.45, 0.60, 0.75):
                attempts.append((f"scale_{a}", (base * a).clamp(-0.08, 0.08)))
        rng = torch.Generator(device=x0.device)
        rng.manual_seed(2026 + (qseed % 100000))
        for eps in (4 / 255, 8 / 255):
            noise = (torch.rand(x0.shape, generator=rng, device=x0.device) * 2 - 1) * eps
            attempts.append((f"noise_{eps:.4f}", noise))
        attempts = attempts[: max(1, max_tries)]

        found = False
        for tag, delta in attempts:
            stats["tried"] += 1
            x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
            text = wrapper.generate(img, q, x01=x01)
            atk_j = harmful_compliance(text, h["topic_keywords"], q)
            rec["n0_matched_attempt"] = {"tag": tag, "label": atk_j.get("label")}
            if accept(rec, h, q, img, text, atk_j, x01, tag):
                if delta_dir is not None:
                    delta_dir.mkdir(parents=True, exist_ok=True)
                    dpath = delta_dir / f"{rec['query_id']}_{rec['carrier_id']}_n0safe.pt"
                    torch.save(delta.detach().float().cpu(), dpath)
                    rec["related_safe_delta_path"] = str(dpath)
                found = True
                break
            torch.cuda.empty_cache()

        if not found:
            for k, (eps, steps) in enumerate(((8 / 255, 8), (4 / 255, 12))):
                if stats["tried"] >= max_tries * len(rhc_recs):
                    break
                stats["tried"] += 1
                try:
                    delta = _pgd_sampler(wrapper, img, q, steps, eps, alpha, refuse_ids, comply_ids)
                    rng2 = torch.Generator(device=delta.device)
                    rng2.manual_seed(2026 + 19 * k + (qseed % 100000))
                    noise = (torch.rand(delta.shape, generator=rng2, device=delta.device) * 2 - 1) * (eps * 0.25)
                    delta = (delta + noise).clamp(-eps, eps)
                except RuntimeError:
                    continue
                x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
                text = wrapper.generate(img, q, x01=x01)
                atk_j = harmful_compliance(text, h["topic_keywords"], q)
                if accept(rec, h, q, img, text, atk_j, x01, f"pgd_{eps}_{steps}"):
                    if delta_dir is not None:
                        delta_dir.mkdir(parents=True, exist_ok=True)
                        dpath = delta_dir / f"{rec['query_id']}_{rec['carrier_id']}_n0safe.pt"
                        torch.save(delta.detach().float().cpu(), dpath)
                        rec["related_safe_delta_path"] = str(dpath)
                    break
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    # Same image as existing RHC: sample alternate trajectories (not pixel PGD).
    n_sample = 0
    for rec in tqdm(rhc_recs, desc="n0-sample-safe"):
        if rec.get("n0_sample_attempt") is not None:
            continue
        if _n_core_pairs(records, catalog, blocked) >= target_pairs:
            break
        h = catalog.get(rec["query_id"])
        c = carriers.get(rec["carrier_id"])
        if h is None or c is None:
            continue
        q = rec.get("query") or h["query"]
        rhc = four_axes(rec["jb"]["judge"].get("text") or "", h["topic_keywords"], q, rec["jb"]["judge"])
        if not rhc["core_rhc"]:
            continue
        if rec.get("related_safe") is not None:
            rs = four_axes(
                rec["related_safe"]["judge"].get("text") or "",
                h["topic_keywords"],
                q,
                rec["related_safe"]["judge"],
            )
            if rs["core_safe_answer"] and length_matched(rhc["chars"], rs["chars"]):
                continue
        img = open_image(c["path"])
        x0 = wrapper.image_to_x01(img)
        base = _load_delta(rec.get("delta_path"), wrapper.device)
        x01 = torch.clamp(x0 + base, 0.0, 1.0) if base is not None else x0
        qseed = int(hashlib.sha256(f"{rec['query_id']}:{rec['carrier_id']}:sample".encode("utf-8")).hexdigest()[:8], 16)
        for k in range(8):
            stats["tried"] += 1
            n_sample += 1
            try:
                text = wrapper.generate(
                    img,
                    q,
                    x01=x01,
                    do_sample=True,
                    temperature=0.95,
                    top_p=0.92,
                    seed=2026 + k + (qseed % 10000),
                )
            except Exception:
                torch.manual_seed(2026 + k + (qseed % 10000))
                text = wrapper.generate(img, q, x01=x01, do_sample=True, temperature=0.95, top_p=0.92)
            atk_j = harmful_compliance(text, h["topic_keywords"], q)
            rec["n0_sample_attempt"] = {"k": k, "label": atk_j.get("label")}
            if accept(rec, h, q, img, text, atk_j, x01, f"sample_{k}"):
                rec["related_safe_from_sample"] = True
                break
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    stats["n_sample_tries"] = n_sample

    # Same image as existing SAFE: add RHC without clobbering RELATED_SAFE.
    n_rhc_fill = 0
    safe_recs = []
    for rec in records:
        if sealed_query(rec["query_id"]) or rec["query_id"] in blocked:
            continue
        h = catalog.get(rec["query_id"])
        c = carriers.get(rec["carrier_id"])
        if h is None or c is None or rec.get("related_safe") is None:
            continue
        q = rec.get("query") or h["query"]
        rs = four_axes(rec["related_safe"]["judge"].get("text") or "", h["topic_keywords"], q, rec["related_safe"]["judge"])
        if not rs["core_safe_answer"]:
            continue
        rhc = None
        if rec.get("jb") is not None:
            rhc = four_axes(rec["jb"]["judge"].get("text") or "", h["topic_keywords"], q, rec["jb"]["judge"])
            if rhc["core_rhc"] and length_matched(rhc["chars"], rs["chars"]):
                continue
        safe_recs.append(rec)
    steps = int(cfg["attack"]["sampler_steps"])
    eps = float(cfg["attack"]["sampler_eps"])
    for rec in tqdm(safe_recs, desc="n0-fill-rhc"):
        if rec.get("n0_rhc_fill") is not None:
            continue
        if _n_core_pairs(records, catalog, blocked) >= target_pairs:
            break
        h = catalog[rec["query_id"]]
        c = carriers[rec["carrier_id"]]
        q = rec.get("query") or h["query"]
        img = open_image(c["path"])
        try:
            delta = _pgd_sampler(wrapper, img, q, steps, eps, alpha, refuse_ids, comply_ids)
        except RuntimeError:
            continue
        n_rhc_fill += 1
        stats["tried"] += 1
        x0 = wrapper.image_to_x01(img)
        x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
        text = wrapper.generate(img, q, x01=x01)
        atk_j = harmful_compliance(text, h["topic_keywords"], q)
        ax = four_axes(text, h["topic_keywords"], q, atk_j)
        rec["n0_rhc_fill"] = {"label": atk_j.get("label")}
        if ax["core_rhc"] and rec.get("jb") is None:
            hid = {kk: v.tolist() for kk, v in wrapper.collect_hidden(img, q, layers, x01=x01).items()}
            rec["jb"] = {"judge": _trim_judge(atk_j), "hidden": hid}
            if delta_dir is not None:
                delta_dir.mkdir(parents=True, exist_ok=True)
                dpath = delta_dir / f"{rec['query_id']}_{rec['carrier_id']}_n0rhc.pt"
                torch.save(delta.detach().float().cpu(), dpath)
                rec["delta_path"] = rec.get("delta_path") or str(dpath)
                rec["n0_rhc_delta_path"] = str(dpath)
            stats["got_rhc_fill"] = int(stats.get("got_rhc_fill") or 0) + 1
            if checkpoint_path is not None:
                traces["summary"] = {**(traces.get("summary") or {}), "n0_matched": stats}
                save_json(checkpoint_path, traces)
        torch.cuda.empty_cache()
    stats["n_rhc_fill_tries"] = n_rhc_fill

    # New (query, train-carrier) slots: collect both trajectories on the same unused image.
    existing = {(r["query_id"], r["carrier_id"]) for r in records}
    qids = sorted(
        {
            r["query_id"]
            for r in records
            if not sealed_query(r["query_id"]) and r["query_id"] not in blocked
        }
    )
    new_slots = []
    for qid in qids:
        h = catalog.get(qid)
        if not h:
            continue
        for cid in ("c01", "c02", "c03", "c04"):
            if (qid, cid) in existing:
                continue
            c = carriers.get(cid)
            if c is None:
                continue
            new_slots.append((qid, h, c))
    stats["n_new_slots"] = 0
    stats["n_new_pairs"] = 0
    for qid, h, c in tqdm(new_slots, desc="n0-new-slots"):
        if _n_core_pairs(records, catalog, blocked) >= 52:
            break
        img = open_image(c["path"])
        q = h["query"]
        clean_text = wrapper.generate(img, q)
        clean_j = harmful_compliance(clean_text, h["topic_keywords"], q)
        hid_clean = {kk: v.tolist() for kk, v in wrapper.collect_hidden(img, q, layers).items()}
        rec = {
            "query_id": qid,
            "category": h.get("category"),
            "split": "discover",
            "carrier_id": c["id"],
            "carrier_split": c.get("split"),
            "query": q,
            "clean": _trim_judge(clean_j),
            "clean_hidden": hid_clean,
            "jb": None,
            "fail": None,
            "n0_new_slot": True,
        }
        records.append(rec)
        existing.add((qid, c["id"]))
        stats["n_new_slots"] += 1
        stats["tried"] += 1
        if clean_j["label"] != "REF":
            rec["skip_reason"] = "clean_not_refusal"
            torch.cuda.empty_cache()
            continue
        try:
            delta = _pgd_sampler(wrapper, img, q, steps, eps, alpha, refuse_ids, comply_ids)
        except RuntimeError as exc:
            rec["skip_reason"] = str(exc)
            torch.cuda.empty_cache()
            continue
        x0 = wrapper.image_to_x01(img)
        x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
        rhc_text = wrapper.generate(img, q, x01=x01)
        rhc_j = harmful_compliance(rhc_text, h["topic_keywords"], q)
        rhc_ax = four_axes(rhc_text, h["topic_keywords"], q, rhc_j)
        rec["n0_new_rhc"] = {"label": rhc_j.get("label")}
        if rhc_ax["core_rhc"]:
            hid = {kk: v.tolist() for kk, v in wrapper.collect_hidden(img, q, layers, x01=x01).items()}
            rec["jb"] = {"judge": _trim_judge(rhc_j), "hidden": hid}
            if delta_dir is not None:
                delta_dir.mkdir(parents=True, exist_ok=True)
                dpath = delta_dir / f"{qid}_{c['id']}_n0new.pt"
                torch.save(delta.detach().float().cpu(), dpath)
                rec["delta_path"] = str(dpath)
            qseed = int(hashlib.sha256(f"{qid}:{c['id']}:new".encode("utf-8")).hexdigest()[:8], 16)
            for k in range(8):
                stats["tried"] += 1
                try:
                    stext = wrapper.generate(
                        img, q, x01=x01, do_sample=True, temperature=0.95, top_p=0.92, seed=2026 + k + (qseed % 10000)
                    )
                except Exception:
                    torch.manual_seed(2026 + k + (qseed % 10000))
                    stext = wrapper.generate(img, q, x01=x01, do_sample=True, temperature=0.95, top_p=0.92)
                sj = harmful_compliance(stext, h["topic_keywords"], q)
                if accept(rec, h, q, img, stext, sj, x01, f"new_sample_{k}"):
                    stats["n_new_pairs"] += 1
                    rec["related_safe_from_sample"] = True
                    break
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if checkpoint_path is not None and stats["n_new_slots"] % 5 == 0:
            traces["summary"] = {**(traces.get("summary") or {}), "n0_matched": stats}
            save_json(checkpoint_path, traces)

    stats["end_pairs"] = _n_core_pairs(records, catalog, blocked)
    traces["summary"] = {**(traces.get("summary") or {}), "n0_matched": stats}
    return traces
