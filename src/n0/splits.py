"""Mutually exclusive N0 splits. Does not consume sealed P0 confirm/attack IDs as fit data."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .pairs import SEALED_ATTACK, SEALED_CONFIRM, sha256_ids

TARGETS = {"discover": 50, "development": 30, "confirm": 30, "attack_test": 40}
FLOORS = {"discover": 30, "development": 20, "confirm": 25, "attack_test": 30, "benign": 30}


def assign_pair_splits(pairs: Sequence[Dict[str, Any]], seed: int = 2026) -> Dict[str, List[Dict[str, Any]]]:
    """Query-exclusive splits. Images follow their query. Deterministic sort then greedily fill floors."""
    by_query: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_query.setdefault(p["query_id"], []).append(p)
    qids = sorted(by_query)
    # stable shuffle
    rng_q = list(qids)
    # Fisher-Yates with explicit LCG so we do not depend on hash randomization
    s = int(seed)
    for i in range(len(rng_q) - 1, 0, -1):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        j = s % (i + 1)
        rng_q[i], rng_q[j] = rng_q[j], rng_q[i]
    buckets = {"discover": [], "development": [], "confirm": []}
    order = ("discover", "development", "confirm")
    floors = (FLOORS["discover"], FLOORS["development"], FLOORS["confirm"])
    targets = (TARGETS["discover"], TARGETS["development"], TARGETS["confirm"])
    # first pass: floors
    for name, floor in zip(order, floors):
        for qid in list(rng_q):
            if len(buckets[name]) >= floor:
                break
            if qid not in by_query:
                continue
            buckets[name].extend(by_query.pop(qid))
            rng_q.remove(qid)
    # second pass: targets
    for name, tgt in zip(order, targets):
        for qid in list(rng_q):
            if len(buckets[name]) >= tgt:
                break
            if qid not in by_query:
                continue
            buckets[name].extend(by_query.pop(qid))
            rng_q.remove(qid)
    leftover_q = [qid for qid in rng_q if qid in by_query]
    for qid in leftover_q:
        group = by_query.pop(qid)
        if len(buckets["discover"]) < TARGETS["discover"]:
            buckets["discover"].extend(group)
        elif len(buckets["development"]) < TARGETS["development"]:
            buckets["development"].extend(group)
        else:
            buckets["confirm"].extend(group)
    for name in buckets:
        buckets[name] = sorted(buckets[name], key=lambda x: x["pair_id"])
        for p in buckets[name]:
            p["n0_split"] = name
    return buckets


def reserved_attack_test() -> Dict[str, Any]:
    """Reserve old catalog h91–h130 as future N4 IDs without loading generations."""
    ids = list(SEALED_ATTACK)
    return {
        "ids": ids,
        "n": len(ids),
        "role": "reserved_n4_attack_test",
        "generations_loaded": False,
        "note": "Catalog IDs only. P0 attack generations were not read.",
    }


def split_record(name: str, pairs: List[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    qids = sorted({p["query_id"] for p in pairs})
    images = sorted({p["image_id"] for p in pairs})
    traces = sorted({p["trace_id_rhc"] for p in pairs} | {p["trace_id_safe"] for p in pairs})
    rec = {
        "split": name,
        "n_pairs": len(pairs),
        "n_queries": len(qids),
        "n_images": len(images),
        "query_ids": qids,
        "image_ids": images,
        "trace_ids": traces,
        "query_ids_sha256": sha256_ids(qids),
        "image_ids_sha256": sha256_ids(images),
        "trace_ids_sha256": sha256_ids(traces),
        "pairs": pairs,
        "floor": FLOORS.get(name),
        "target": TARGETS.get(name),
        "meets_floor": len(pairs) >= int(FLOORS.get(name) or 0) if name in FLOORS else None,
    }
    if extra:
        rec.update(extra)
    return rec


def disjoint(a: Sequence[str], b: Sequence[str]) -> bool:
    return set(a).isdisjoint(set(b))


def intersection_report(splits: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    names = ["discover", "development", "confirm", "attack_test"]
    out = {}
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            qx = set(splits[x].get("query_ids") or [])
            qy = set(splits[y].get("query_ids") or [])
            ix = set(splits[x].get("image_ids") or [])
            iy = set(splits[y].get("image_ids") or [])
            out[f"query_{x}_x_{y}"] = sorted(qx & qy)
            out[f"image_{x}_x_{y}"] = sorted(ix & iy)
    out["sealed_confirm_used"] = sorted(
        set(splits["discover"]["query_ids"] + splits["development"]["query_ids"] + splits["confirm"]["query_ids"])
        & set(SEALED_CONFIRM)
    )
    out["sealed_attack_used_in_pairs"] = sorted(
        set(splits["discover"]["query_ids"] + splits["development"]["query_ids"] + splits["confirm"]["query_ids"])
        & set(SEALED_ATTACK)
    )
    return out
