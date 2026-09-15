"""Collect Qwen REF / JB / FAIL / BENIGN traces with a per-sample visual sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from p0.datautil import open_image
from p0.judge import benign_utility, harmful_compliance, is_success_label
from p0.model import COMPLY_WORDS, REFUSE_WORDS, margin_from_logits, token_id_list

from .model import QwenP0


def _pgd_sampler(
    wrapper: QwenP0,
    image,
    question: str,
    steps: int,
    eps: float,
    alpha: float,
    refuse_ids: List[int],
    comply_ids: List[int],
) -> torch.Tensor:
    x0 = wrapper.image_to_x01(image)
    delta = torch.zeros_like(x0)
    inputs = wrapper.encode(image, question)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        pv, grid = wrapper.patchify(x)
        logits = wrapper.first_token_logits(pv, input_ids, attn, grid)
        loss = margin_from_logits(logits, refuse_ids, comply_ids)
        grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        if grad is None:
            raise RuntimeError("sampler got no image gradient; vision tower is likely quantized")
        delta = (delta - alpha * grad.sign()).clamp(-eps, eps)
        delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0
    return delta.detach()


def _trim_judge(j: Dict[str, Any], n: int = 400) -> Dict[str, Any]:
    out = dict(j)
    if "text" in out and out["text"] is not None:
        out["text"] = out["text"][:n]
    return out


def collect_traces(
    wrapper: QwenP0,
    cfg: Dict[str, Any],
    harmful: List[Dict[str, Any]],
    benign_map: Dict[str, Dict[str, Any]],
    carriers: List[Dict[str, Any]],
    layers: List[int],
    delta_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    steps = int(cfg["attack"]["sampler_steps"])
    eps = float(cfg["attack"]["sampler_eps"])
    alpha = float(cfg["attack"]["alpha"])
    records = []

    pairs = [(h, c) for h in harmful for c in carriers]
    for h, c in tqdm(pairs, desc="traces"):
        img = open_image(c["path"])
        clean_text = wrapper.generate(img, h["query"])
        clean_j = harmful_compliance(clean_text, h["topic_keywords"], h["query"])
        hid = {
            k: v.tolist()
            for k, v in wrapper.collect_hidden(img, h["query"], layers).items()
        }
        rec: Dict[str, Any] = {
            "query_id": h["id"],
            "category": h["category"],
            "split": h.get("split", "discover"),
            "carrier_id": c["id"],
            "carrier_split": c["split"],
            "query": h["query"],
            "clean": _trim_judge(clean_j),
            "clean_hidden": hid,
            "jb": None,
            "fail": None,
        }

        if clean_j["label"] != "REF":
            rec["skip_reason"] = "clean_not_refusal"
            records.append(rec)
            continue

        try:
            delta = _pgd_sampler(wrapper, img, h["query"], steps, eps, alpha, refuse_ids, comply_ids)
        except RuntimeError as exc:
            rec["skip_reason"] = str(exc)
            records.append(rec)
            continue

        x0 = wrapper.image_to_x01(img)
        x01 = torch.clamp(x0 + delta, 0.0, 1.0)
        atk_text = wrapper.generate(img, h["query"], x01=x01)
        atk_j = harmful_compliance(atk_text, h["topic_keywords"], h["query"])
        atk_h = {
            k: v.tolist()
            for k, v in wrapper.collect_hidden(img, h["query"], layers, x01=x01).items()
        }
        rec["attack_eval"] = _trim_judge(atk_j)
        rec["delta_linf"] = float(delta.abs().max().item())
        rec["delta_path"] = None
        if delta_dir is not None:
            delta_dir.mkdir(parents=True, exist_ok=True)
            dpath = delta_dir / f"{h['id']}_{c['id']}.pt"
            torch.save(delta.detach().float().cpu(), dpath)
            rec["delta_path"] = str(dpath)
        if is_success_label(atk_j["label"]):
            rec["jb"] = {"judge": _trim_judge(atk_j), "hidden": atk_h}
        elif atk_j["label"] == "FAIL":
            rec["fail"] = {"judge": _trim_judge(atk_j), "hidden": atk_h}
        elif atk_j["label"] == "RELATED_SAFE":
            rec["related_safe"] = {"judge": _trim_judge(atk_j), "hidden": atk_h}
        else:
            rec["sampler_still_refused"] = True
        records.append(rec)
        torch.cuda.empty_cache()

    benign_recs = []
    for h in harmful:
        b = benign_map.get(h["benign_id"])
        if b is None:
            continue
        c = carriers[0]
        img = open_image(c["path"])
        text = wrapper.generate(img, b["query"])
        j = benign_utility(text, b["topic_keywords"], b["query"])
        hid = {
            k: v.tolist()
            for k, v in wrapper.collect_hidden(img, b["query"], layers).items()
        }
        benign_recs.append(
            {
                "query_id": b["id"],
                "harmful_id": h["id"],
                "carrier_id": c["id"],
                "query": b["query"],
                "judge": _trim_judge(j),
                "hidden": hid,
            }
        )

    n_ref = sum(1 for r in records if r["clean"]["label"] == "REF")
    n_jb = sum(1 for r in records if r.get("jb") is not None)
    if n_jb < int(cfg["subspace"].get("min_jb_pairs", 30)):
        hard_eps = min(eps * 2.0, 32.0 / 255.0)
        hard_steps = steps + 30
        for rec in tqdm(records, desc="traces-retry"):
            if rec.get("jb") is not None or rec["clean"]["label"] != "REF":
                continue
            h = next(x for x in harmful if x["id"] == rec["query_id"])
            c = next(x for x in carriers if x["id"] == rec["carrier_id"])
            img = open_image(c["path"])
            try:
                delta = _pgd_sampler(wrapper, img, h["query"], hard_steps, hard_eps, alpha, refuse_ids, comply_ids)
            except RuntimeError:
                continue
            x0 = wrapper.image_to_x01(img)
            x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
            atk_text = wrapper.generate(img, h["query"], x01=x01)
            atk_j = harmful_compliance(atk_text, h["topic_keywords"], h["query"])
            rec["attack_eval"] = _trim_judge(atk_j)
            rec["delta_linf"] = float(delta.abs().max().item())
            rec["retry_eps"] = hard_eps
            if delta_dir is not None:
                dpath = delta_dir / f"{h['id']}_{c['id']}.pt"
                torch.save(delta.detach().float().cpu(), dpath)
                rec["delta_path"] = str(dpath)
            if is_success_label(atk_j["label"]):
                rec["jb"] = {
                    "judge": _trim_judge(atk_j),
                    "hidden": {
                        k: v.tolist()
                        for k, v in wrapper.collect_hidden(img, h["query"], layers, x01=x01).items()
                    },
                }
            elif atk_j["label"] == "FAIL" and rec.get("fail") is None:
                rec["fail"] = {
                    "judge": _trim_judge(atk_j),
                    "hidden": {
                        k: v.tolist()
                        for k, v in wrapper.collect_hidden(img, h["query"], layers, x01=x01).items()
                    },
                }
            elif atk_j["label"] == "RELATED_SAFE" and rec.get("related_safe") is None:
                rec["related_safe"] = {
                    "judge": _trim_judge(atk_j),
                    "hidden": {
                        k: v.tolist()
                        for k, v in wrapper.collect_hidden(img, h["query"], layers, x01=x01).items()
                    },
                }
            n_jb = sum(1 for r in records if r.get("jb") is not None)
            if n_jb >= int(cfg["subspace"].get("min_jb_pairs", 30)) + 10:
                break
    return {
        "records": records,
        "benign": benign_recs,
        "summary": {
            "n_pairs": len(records),
            "n_clean_refusal": n_ref,
            "n_jb": sum(1 for r in records if r.get("jb") is not None),
            "n_rhc": sum(1 for r in records if r.get("jb") is not None),
            "n_fail": sum(1 for r in records if r.get("fail") is not None),
            "n_related_safe": sum(1 for r in records if r.get("related_safe") is not None),
            "n_benign": len(benign_recs),
        },
    }


def jb_pairs(trace_blob: Dict[str, Any], split: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for rec in trace_blob["records"]:
        if rec.get("jb") is None:
            continue
        if split and rec.get("split") != split:
            continue
        out.append(rec)
    return out


def related_safe_pairs(trace_blob: Dict[str, Any], split: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for rec in trace_blob["records"]:
        if rec.get("related_safe") is None:
            continue
        if split and rec.get("split") != split:
            continue
        out.append(rec)
    return out


def theme_denial_pairs(trace_blob: Dict[str, Any], split: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for rec in trace_blob["records"]:
        fail = rec.get("fail")
        if fail is None:
            continue
        if float((fail.get("judge") or {}).get("theme_denial") or 0.0) < 0.5:
            continue
        if split and rec.get("split") != split:
            continue
        out.append(rec)
    return out


def fail_pairs(trace_blob: Dict[str, Any], split: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for rec in trace_blob["records"]:
        if rec.get("fail") is None:
            continue
        if split and rec.get("split") != split:
            continue
        out.append(rec)
    return out


def resample_missing_rhc(
    wrapper: QwenP0,
    cfg: Dict[str, Any],
    traces: Dict[str, Any],
    harmful: List[Dict[str, Any]],
    carriers: List[Dict[str, Any]],
    layers: List[int],
    delta_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """GPU retry of clean-REF records that still lack RHC after reclean."""
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    steps = int(cfg["attack"]["sampler_steps"]) + 30
    eps = min(float(cfg["attack"]["eps_list"][-1]) * 2.0, 32.0 / 255.0)
    alpha = float(cfg["attack"]["alpha"])
    min_jb = int(cfg["subspace"].get("min_jb_pairs", 30))
    records = traces["records"]
    for rec in tqdm(records, desc="resample-rhc"):
        if rec.get("jb") is not None:
            continue
        if rec.get("clean", {}).get("label") != "REF":
            continue
        h = next(x for x in harmful if x["id"] == rec["query_id"])
        c = next(x for x in carriers if x["id"] == rec["carrier_id"])
        img = open_image(c["path"])
        try:
            delta = _pgd_sampler(wrapper, img, h["query"], steps, eps, alpha, refuse_ids, comply_ids)
        except RuntimeError:
            continue
        x0 = wrapper.image_to_x01(img)
        x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
        atk_text = wrapper.generate(img, h["query"], x01=x01)
        atk_j = harmful_compliance(atk_text, h["topic_keywords"], h["query"])
        rec["attack_eval"] = _trim_judge(atk_j)
        rec["resample_eps"] = eps
        if delta_dir is not None:
            delta_dir.mkdir(parents=True, exist_ok=True)
            dpath = delta_dir / f"{h['id']}_{c['id']}.pt"
            torch.save(delta.detach().float().cpu(), dpath)
            rec["delta_path"] = str(dpath)
        hid = {
            k: v.tolist()
            for k, v in wrapper.collect_hidden(img, h["query"], layers, x01=x01).items()
        }
        if is_success_label(atk_j["label"]):
            rec["jb"] = {"judge": _trim_judge(atk_j), "hidden": hid}
            rec.pop("reclean_now_ref", None)
        elif atk_j["label"] == "FAIL":
            rec["fail"] = {"judge": _trim_judge(atk_j), "hidden": hid}
        elif atk_j["label"] == "RELATED_SAFE":
            rec["related_safe"] = {"judge": _trim_judge(atk_j), "hidden": hid}
        n_jb = sum(1 for r in records if r.get("jb") is not None)
        if n_jb >= min_jb + 10:
            break
    traces["summary"] = {
        **(traces.get("summary") or {}),
        "n_jb": sum(1 for r in records if r.get("jb") is not None),
        "n_rhc": sum(1 for r in records if r.get("jb") is not None),
        "n_fail": sum(1 for r in records if r.get("fail") is not None),
        "n_related_safe": sum(1 for r in records if r.get("related_safe") is not None),
        "resampled": True,
    }
    return traces


def _store_mode_slot(rec: Dict[str, Any], atk_j: Dict[str, Any], hid: Dict[str, Any]) -> str:
    """Fill related_safe / fail / jb without clobbering an existing RHC slot."""
    if is_success_label(atk_j["label"]):
        if rec.get("jb") is None:
            rec["jb"] = {"judge": _trim_judge(atk_j), "hidden": hid}
            rec.pop("reclean_now_ref", None)
        return "RHC"
    if atk_j["label"] == "RELATED_SAFE":
        if rec.get("related_safe") is None:
            rec["related_safe"] = {"judge": _trim_judge(atk_j), "hidden": hid}
        return "RELATED_SAFE"
    if atk_j["label"] == "FAIL":
        if rec.get("fail") is None:
            rec["fail"] = {"judge": _trim_judge(atk_j), "hidden": hid}
        return "FAIL"
    rec["sampler_still_refused"] = True
    return "REF"


def collect_related_safe_gap(
    wrapper: QwenP0,
    cfg: Dict[str, Any],
    traces: Dict[str, Any],
    harmful: List[Dict[str, Any]],
    extra_carriers: List[Dict[str, Any]],
    layers: List[int],
    *,
    target_n: int = 30,
    split: str = "discover",
    blocked_ids: Optional[set] = None,
    delta_dir: Optional[Path] = None,
    retry_carriers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Mild extra sampling on discover only. Does not use confirm or attack-test ids."""
    blocked = set(blocked_ids or [])
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    steps = max(12, int(cfg["attack"]["sampler_steps"]) // 3)
    eps = float(cfg["attack"]["eps_list"][0])
    alpha = float(cfg["attack"]["alpha"])
    h_by_id = {h["id"]: h for h in harmful}
    records = traces["records"]
    existing_pairs = {(r["query_id"], r["carrier_id"]) for r in records}
    retry_index = {c["id"]: c for c in (retry_carriers or extra_carriers)}

    def n_rs() -> int:
        return sum(1 for r in records if r.get("split") == split and r.get("related_safe") is not None)

    for rec in tqdm(records, desc="p0s-mild-retry"):
        if n_rs() >= target_n:
            break
        if rec.get("split") != split or rec["query_id"] in blocked:
            continue
        if rec.get("related_safe") is not None:
            continue
        if rec.get("clean", {}).get("label") != "REF":
            continue
        h = h_by_id.get(rec["query_id"])
        c = retry_index.get(rec["carrier_id"])
        if h is None or c is None:
            continue
        img = open_image(c["path"])
        try:
            delta = _pgd_sampler(wrapper, img, h["query"], steps, eps, alpha, refuse_ids, comply_ids)
        except RuntimeError:
            continue
        x0 = wrapper.image_to_x01(img)
        x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
        atk_j = harmful_compliance(wrapper.generate(img, h["query"], x01=x01), h["topic_keywords"], h["query"])
        hid = {k: v.tolist() for k, v in wrapper.collect_hidden(img, h["query"], layers, x01=x01).items()}
        _store_mode_slot(rec, atk_j, hid)
        rec["p0s_mild_retry"] = True
        torch.cuda.empty_cache()

    # 2) new carriers on discover clean-REF queries
    disc_qids = []
    seen_q = set()
    for rec in records:
        if rec.get("split") != split or rec["query_id"] in blocked:
            continue
        if rec.get("clean", {}).get("label") != "REF":
            continue
        if rec["query_id"] in seen_q:
            continue
        seen_q.add(rec["query_id"])
        disc_qids.append(rec["query_id"])

    for qid in tqdm(disc_qids, desc="p0s-extra-carriers"):
        if n_rs() >= target_n:
            break
        h = h_by_id.get(qid)
        if h is None:
            continue
        for c in extra_carriers:
            if n_rs() >= target_n:
                break
            if (qid, c["id"]) in existing_pairs:
                continue
            img = open_image(c["path"])
            clean_text = wrapper.generate(img, h["query"])
            clean_j = harmful_compliance(clean_text, h["topic_keywords"], h["query"])
            hid_clean = {k: v.tolist() for k, v in wrapper.collect_hidden(img, h["query"], layers).items()}
            rec = {
                "query_id": qid,
                "category": h.get("category"),
                "split": split,
                "carrier_id": c["id"],
                "carrier_split": c.get("split"),
                "query": h["query"],
                "clean": _trim_judge(clean_j),
                "clean_hidden": hid_clean,
                "jb": None,
                "fail": None,
                "p0s_extra_carrier": True,
            }
            existing_pairs.add((qid, c["id"]))
            if clean_j["label"] != "REF":
                rec["skip_reason"] = "clean_not_refusal"
                records.append(rec)
                continue
            try:
                delta = _pgd_sampler(wrapper, img, h["query"], steps, eps, alpha, refuse_ids, comply_ids)
            except RuntimeError as exc:
                rec["skip_reason"] = str(exc)
                records.append(rec)
                continue
            x0 = wrapper.image_to_x01(img)
            x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
            atk_j = harmful_compliance(wrapper.generate(img, h["query"], x01=x01), h["topic_keywords"], h["query"])
            hid = {k: v.tolist() for k, v in wrapper.collect_hidden(img, h["query"], layers, x01=x01).items()}
            rec["attack_eval"] = _trim_judge(atk_j)
            if delta_dir is not None:
                delta_dir.mkdir(parents=True, exist_ok=True)
                dpath = delta_dir / f"{qid}_{c['id']}_p0s.pt"
                torch.save(delta.detach().float().cpu(), dpath)
                rec["delta_path"] = str(dpath)
            _store_mode_slot(rec, atk_j, hid)
            records.append(rec)
            torch.cuda.empty_cache()

    traces["summary"] = {
        **(traces.get("summary") or {}),
        "n_pairs": len(records),
        "n_jb": sum(1 for r in records if r.get("jb") is not None),
        "n_rhc": sum(1 for r in records if r.get("jb") is not None),
        "n_fail": sum(1 for r in records if r.get("fail") is not None),
        "n_related_safe": sum(1 for r in records if r.get("related_safe") is not None),
        "n_related_safe_discover": n_rs(),
        "p0s_related_safe_target": target_n,
    }
    return traces
