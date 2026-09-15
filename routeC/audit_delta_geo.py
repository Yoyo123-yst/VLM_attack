#!/usr/bin/env python3
"""P0.3 (CPU) — delta geometry audit.

Question: are the two deltas in a "same-cell pairing" geometrically distinct?
If cos≈0 / negative, the SAFE vs RHC deltas are NOT small perturbations around
a shared direction — they are orthogonal, so "same-cell pairing" is void.

Computes for each of the 9 CR-0 content pairs:
  - cosine similarity between flattened rhc-delta and safe-delta
  - L2 correlation distance (1 - pearson)
Plus the pairwise cosine distribution across all same-cell deltas.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from p0.datautil import load_json, save_json  # noqa: E402

OUT = ROOT / "routeC" / "out"
CR0_PAIRS = ROOT / "outputs" / "causal_reach" / "cr0" / "pairs.json"
CR0_DELTAS = ROOT / "outputs" / "causal_reach" / "cr0" / "deltas"


def _load_delta(record_id: str) -> torch.Tensor:
    p = CR0_DELTAS / f"{record_id}.delta.pt"
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d.float().reshape(-1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(torch.norm(a)) + 1e-12
    nb = float(torch.norm(b)) + 1e-12
    return float((a @ b) / (na * nb))


def _corr_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    ca = a - a.mean()
    cb = b - b.mean()
    num = float((ca @ cb))
    den = (float(torch.norm(ca)) * float(torch.norm(cb))) + 1e-12
    return 1.0 - num / den


def main() -> None:
    pairs_data = load_json(CR0_PAIRS)
    content = pairs_data.get("content_pairs", [])

    pair_rows: List[Dict[str, Any]] = []
    for p in content:
        dr = _load_delta(p["rhc_record_id"])
        ds = _load_delta(p["safe_record_id"])
        pair_rows.append(
            {
                "pair_id": p["pair_id"],
                "rhc_record_id": p["rhc_record_id"],
                "safe_record_id": p["safe_record_id"],
                "cosine": _cos(dr, ds),
                "corr_distance": _corr_dist(dr, ds),
            }
        )

    # same-cell delta pairwise cosine: group content-pair deltas by (query,carrier)
    from collections import defaultdict

    cells: Dict[str, List[str]] = defaultdict(list)
    for p in content:
        key = f"{p['query_id']}:{p['carrier_id']}"
        cells[key].append(p["rhc_record_id"])
        cells[key].append(p["safe_record_id"])

    same_cell_cos: List[float] = []
    for key, ids in cells.items():
        for a, b in combinations(sorted(set(ids)), 2):
            da = _load_delta(a)
            db = _load_delta(b)
            same_cell_cos.append(_cos(da, db))

    n = len(same_cell_cos)
    cos_mean = float(sum(same_cell_cos) / n) if n else 0.0
    cos_min = float(min(same_cell_cos)) if n else 0.0
    cos_max = float(max(same_cell_cos)) if n else 0.0

    out = {
        "n_content_pairs": len(pair_rows),
        "pairs": pair_rows,
        "same_cell_pairwise_cosine": {
            "n": n,
            "mean": cos_mean,
            "min": cos_min,
            "max": cos_max,
            "values": same_cell_cos,
        },
        "verdict": {
            "distinct_if_cos_near_zero_or_negative": (
                all(abs(r["cosine"]) < 0.3 for r in pair_rows) if pair_rows else False
            ),
            "note": "cos≈0/negative across pairs => same-cell SAFE vs RHC deltas are "
            "orthogonal, not small perturbations; 'same-cell pairing' concept void.",
        },
    }
    save_json(OUT / "audit_delta_geo.json", out)
    print(
        f"P0.3 done: n_pairs={len(pair_rows)} "
        f"cos_mean={cos_mean:.4f} cos_min={cos_min:.4f} cos_max={cos_max:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
