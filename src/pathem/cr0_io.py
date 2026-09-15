"""Read CR-0 candidates and deltas. Stage 0 does not generate new images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from p0.datautil import load_json

from .protocol import PILOT_CARRIERS, PILOT_QUERIES, SEALED, assert_query_allowed

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
CR0 = ROOT / "outputs" / "causal_reach" / "cr0"
CANDIDATES = CR0 / "candidates.json"


def load_cr0_records(
    query_ids: tuple[str, ...] = PILOT_QUERIES,
    carrier_ids: tuple[str, ...] = PILOT_CARRIERS,
) -> list[dict[str, Any]]:
    blob = load_json(CANDIDATES)
    rows = []
    for rec in blob.get("records") or []:
        qid = rec.get("query_id") or ""
        if qid in SEALED:
            raise RuntimeError(f"CR0 record leaked sealed query {qid}")
        if qid not in query_ids:
            continue
        if rec.get("carrier_id") not in carrier_ids:
            continue
        assert_query_allowed(qid, query_ids)
        rows.append(rec)
    return rows


def load_delta(path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    delta = torch.load(Path(path), map_location=device, weights_only=True)
    if not torch.is_tensor(delta):
        raise RuntimeError(f"delta is not a tensor: {path}")
    return delta


def public_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Drop raw text for logs. Keep labels and paths."""
    keep = (
        "record_id",
        "query_id",
        "carrier_id",
        "restart",
        "seed",
        "eps",
        "core_rhc",
        "core_safe_answer",
        "safety",
        "response_mode",
        "grounding",
        "quality",
        "chars",
        "delta_path",
        "legacy_label",
    )
    return {k: rec.get(k) for k in keep}
