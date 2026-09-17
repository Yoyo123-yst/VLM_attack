"""Hard-cell loader. Never reads sealed queries."""

from __future__ import annotations

from typing import Any, Dict, List

from .protocol import assert_query_unsealed, load_frozen


def build_hard_cells(frozen: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    from traceflip.datasets import build_cells

    blob = frozen or load_frozen()
    want = [tuple(s.split(":", 1)) for s in blob["scope"]["hard_cells"]]
    for q, _c in want:
        assert_query_unsealed(q)
    queries = tuple(dict.fromkeys(q for q, _ in want))
    carriers = tuple(dict.fromkeys(c for _, c in want))
    cells = [
        c
        for c in build_cells(queries, carriers)
        if (c["query_id"], c["carrier_id"]) in set(want)
    ]
    if len(cells) != len(want):
        raise RuntimeError(f"hard cells incomplete: have {len(cells)} want {len(want)}")
    return cells
