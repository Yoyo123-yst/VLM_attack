"""Build same-image same-query matched RHC / SAFE-answer pairs from traces."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from p0.catalog import all_pairs, attack_test_pairs

from .axes import four_axes, length_matched

SEALED_CONFIRM = tuple(f"h{i:02d}" for i in range(83, 91))
SEALED_ATTACK = tuple(p["id"] for p in attack_test_pairs())
OLD_SEALED = set(SEALED_CONFIRM) | set(SEALED_ATTACK)


def _slot_text(rec: Dict[str, Any], slot: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if slot == "jb":
        blob = rec.get("jb") or {}
        j = blob.get("judge") or {}
    elif slot == "related_safe":
        blob = rec.get("related_safe") or {}
        j = blob.get("judge") or {}
    elif slot == "fail":
        blob = rec.get("fail") or {}
        j = blob.get("judge") or {}
    elif slot == "clean":
        j = rec.get("clean") or {}
        blob = {"judge": j, "hidden": rec.get("clean_hidden")}
    else:
        return None, ""
    return blob, str(j.get("text") or "")


def _axes_for_slot(rec: Dict[str, Any], slot: str, catalog: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    blob, text = _slot_text(rec, slot)
    if blob is None:
        return None
    j = blob.get("judge") or blob
    if not text and not j:
        return None
    item = catalog.get(rec["query_id"])
    if item is None:
        return None
    ax = four_axes(text, item.get("topic_keywords") or [], rec.get("query") or item.get("query") or "", j)
    ax["slot"] = slot
    ax["trace_id"] = f"{rec['query_id']}:{rec['carrier_id']}:{slot}"
    return ax


def sealed_query(qid: str) -> bool:
    return qid in OLD_SEALED


def iter_controls(records: Sequence[Dict[str, Any]], catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for rec in records:
        if sealed_query(rec.get("query_id") or ""):
            continue
        for slot in ("clean", "jb", "related_safe", "fail"):
            ax = _axes_for_slot(rec, slot, catalog)
            if ax is None:
                continue
            if ax["control_kind"] in {"REFUSE", "DENY", "UNGROUNDED", "GARBAGE"}:
                rows.append(
                    {
                        "query_id": rec["query_id"],
                        "image_id": rec.get("carrier_id"),
                        "trace_id": ax["trace_id"],
                        "split_hint": rec.get("split"),
                        **{k: ax[k] for k in ("safety", "response_mode", "grounding", "quality", "control_kind")},
                    }
                )
    return rows


def matched_pairs(records: Sequence[Dict[str, Any]], catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Same query_id and carrier_id; one core RHC and one core SAFE answer."""
    pairs = []
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in records:
        qid = rec.get("query_id") or ""
        if sealed_query(qid):
            continue
        key = (qid, rec.get("carrier_id") or "")
        by_key.setdefault(key, rec)
        # prefer record that has both slots
        if rec.get("jb") is not None and rec.get("related_safe") is not None:
            by_key[key] = rec
    for (qid, cid), rec in sorted(by_key.items()):
        rhc_ax = _axes_for_slot(rec, "jb", catalog) if rec.get("jb") else None
        rs_ax = _axes_for_slot(rec, "related_safe", catalog) if rec.get("related_safe") else None
        if not rhc_ax or not rs_ax:
            continue
        if not rhc_ax["core_rhc"] or not rs_ax["core_safe_answer"]:
            continue
        if not length_matched(rhc_ax["chars"], rs_ax["chars"]):
            continue
        item = catalog[qid]
        rhc_j = rec["jb"]["judge"]
        rs_j = rec["related_safe"]["judge"]
        pair_id = f"{qid}:{cid}"
        pairs.append(
            {
                "pair_id": pair_id,
                "query_id": qid,
                "image_id": cid,
                "trace_id_rhc": rhc_ax["trace_id"],
                "trace_id_safe": rs_ax["trace_id"],
                "query": item["query"],
                "category": item.get("category"),
                "template": "qwen2vl_official_native",
                "old_trace_split": rec.get("split"),
                "rhc_chars": rhc_ax["chars"],
                "safe_chars": rs_ax["chars"],
                "rhc_axes": {k: rhc_ax[k] for k in ("safety", "response_mode", "grounding", "quality")},
                "safe_axes": {k: rs_ax[k] for k in ("safety", "response_mode", "grounding", "quality")},
                "rhc_text": (rhc_j.get("text") or "")[:400],
                "safe_text": (rs_j.get("text") or "")[:400],
                "legacy_rhc": rhc_j.get("label"),
                "legacy_safe": rs_j.get("label"),
                "has_hidden_rhc": bool((rec.get("jb") or {}).get("hidden")),
                "has_hidden_safe": bool((rec.get("related_safe") or {}).get("hidden")),
            }
        )
    return pairs


def pollution_rates(
    records: Sequence[Dict[str, Any]],
    catalog: Dict[str, Any],
    pairs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    n_legacy_rs = 0
    n_rs_refuse = 0
    n_rs_deny = 0
    n_core_safe = 0
    for rec in records:
        if sealed_query(rec.get("query_id") or ""):
            continue
        if rec.get("related_safe") is None:
            continue
        n_legacy_rs += 1
        ax = _axes_for_slot(rec, "related_safe", catalog)
        if ax is None:
            continue
        if ax["response_mode"] == "REFUSE" or ax["refuse_then_advice"]:
            n_rs_refuse += 1
        if ax["response_mode"] == "DENY":
            n_rs_deny += 1
        if ax["core_safe_answer"]:
            n_core_safe += 1
    n = max(n_legacy_rs, 1)
    n_pair = len(pairs or [])
    n_pair_ref = 0
    n_pair_deny = 0
    for p in pairs or []:
        ax = p.get("safe_axes") or {}
        if ax.get("response_mode") == "REFUSE":
            n_pair_ref += 1
        if ax.get("response_mode") == "DENY":
            n_pair_deny += 1
    pair_n = max(n_pair, 1)
    return {
        "n_legacy_related_safe": n_legacy_rs,
        "n_core_safe_answer": n_core_safe,
        "refuse_or_advice_frac": n_rs_refuse / n,
        "theme_denial_frac": n_rs_deny / n,
        "legacy_refuse_or_deny_frac": (n_rs_refuse + n_rs_deny) / n,
        "n_pairs": n_pair,
        "pair_refuse_or_deny_frac": (n_pair_ref + n_pair_deny) / pair_n if n_pair else 0.0,
        "refuse_or_deny_frac": (n_pair_ref + n_pair_deny) / pair_n if n_pair else 0.0,
    }


def catalog_index() -> Dict[str, Any]:
    return {p["id"]: p for p in all_pairs()}


def sha256_ids(ids: Sequence[str]) -> str:
    blob = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def duplicate_rates(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [p["pair_id"] for p in pairs]
    keys = [(p["query_id"], p["image_id"]) for p in pairs]
    n_id = len(ids) - len(set(ids))
    n_key = len(keys) - len(set(keys))
    n = max(len(pairs), 1)
    return {
        "n_pairs": len(pairs),
        "duplicate_pair_id": n_id,
        "duplicate_query_image": n_key,
        "duplicate_pair_id_frac": n_id / n,
        "duplicate_query_image_frac": n_key / n,
    }
