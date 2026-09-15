"""Cell construction for TraceFlip.

A cell is ``(query, carrier)``. TraceFlip never reuses a CR-0 delta as an
initialisation: the delta starts at zero so the method is measured on repairing
the *clean* trajectory, not on polishing someone else's attack. CR-0 records are
only read for their labels (the clean baseline) and for the allowed-cell list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from p0.datautil import CARRIER_JSON, load_carriers, load_json, open_image

from .protocol import PILOT_CARRIERS, PILOT_QUERIES, assert_query_allowed

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
CR0_CANDIDATES = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"
CATEGORY_JSON = ROOT / "outputs" / "causal_reach" / "cr0" / "CR0_FROZEN.json"


def load_queries(query_ids: Tuple[str, ...] = PILOT_QUERIES) -> Dict[str, Dict[str, Any]]:
    """Catalog entries keyed by id, restricted to the allowed pilot queries."""
    from p0.catalog import all_pairs

    want = set(query_ids)
    out: Dict[str, Dict[str, Any]] = {}
    for p in all_pairs():
        qid = str(p.get("id"))
        if qid not in want:
            continue
        assert_query_allowed(qid, tuple(query_ids))
        out[qid] = {
            "id": qid,
            "query": p["query"],
            "topic_keywords": list(p.get("topic_keywords") or []),
            "category": p.get("category"),
        }
    missing = sorted(want - set(out))
    if missing:
        raise RuntimeError(f"queries missing from catalog: {missing}")
    return out


def load_carriers_by_id(carrier_ids: Tuple[str, ...] = PILOT_CARRIERS) -> Dict[str, Dict[str, Any]]:
    spec = load_json(CARRIER_JSON)
    root = Path(spec["root"])
    out: Dict[str, Dict[str, Any]] = {}
    for row in spec.get("images") or []:
        cid = str(row.get("id"))
        if cid not in set(carrier_ids):
            continue
        out[cid] = {**row, "path": str(root / row["file"])}
    missing = sorted(set(carrier_ids) - set(out))
    if missing:
        raise RuntimeError(f"carriers missing: {missing}")
    return out


def _cr0_allowed_cells() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """CR-0 records restrict which (query, carrier) cells the pilot may use."""
    blob = load_json(CR0_CANDIDATES)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in blob.get("records") or []:
        qid = str(rec.get("query_id"))
        cid = str(rec.get("carrier_id"))
        if qid not in set(PILOT_QUERIES) or cid not in set(PILOT_CARRIERS):
            continue
        out.setdefault((qid, cid), rec)
    return out


def build_cells(
    query_ids: Tuple[str, ...] = PILOT_QUERIES,
    carrier_ids: Tuple[str, ...] = PILOT_CARRIERS,
) -> List[Dict[str, Any]]:
    """Enumerate pilot cells with image + item metadata attached."""
    queries = load_queries(query_ids)
    carriers = load_carriers_by_id(carrier_ids)
    allowed = _cr0_allowed_cells()
    cells: List[Dict[str, Any]] = []
    for qid in query_ids:
        if qid in QUERY_EXCLUDED:
            continue
        for cid in carrier_ids:
            if (qid, cid) not in allowed:
                raise RuntimeError(
                    f"cell {(qid, cid)} has no CR-0 record; pilot scope is the CR-0 cell set"
                )
            item = queries[qid]
            car = carriers[cid]
            cells.append(
                {
                    "query_id": qid,
                    "carrier_id": cid,
                    "item": item,
                    "carrier": car,
                    "image": open_image(car["path"]),
                }
            )
    return cells


# Queries in the pilot pool that are not usable for TraceFlip (no greedy-branch
# structure to repair, e.g. the clean decode already complies). Kept explicit so
# that an exclusion is a recorded decision rather than a silent skip.
QUERY_EXCLUDED: Tuple[str, ...] = ()
