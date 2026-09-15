"""Read-only CR-0 catalog. Does not load the VLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from p0.datautil import load_json

from .protocol import EPS, PILOT_CARRIERS, PILOT_QUERIES, SEALED, assert_query_allowed

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
CR0_CANDIDATES = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"
CR0_FROZEN = ROOT / "outputs" / "causal_reach" / "cr0" / "CR0_FROZEN.json"


def load_cr0_records() -> list[dict[str, Any]]:
    blob = load_json(CR0_CANDIDATES)
    recs = list(blob.get("records") or [])
    out = []
    for rec in recs:
        qid = rec.get("query_id")
        if qid in SEALED:
            continue
        out.append(rec)
    return out


def pilot_records() -> list[dict[str, Any]]:
    allowed_q = set(PILOT_QUERIES)
    allowed_c = set(PILOT_CARRIERS)
    rows = []
    for rec in load_cr0_records():
        if rec.get("query_id") not in allowed_q:
            continue
        if rec.get("carrier_id") not in allowed_c:
            continue
        assert_query_allowed(rec["query_id"])
        rows.append(rec)
    return rows


def prevalence(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    recs = records if records is not None else load_cr0_records()
    n = len(recs)
    n_rhc = sum(1 for r in recs if r.get("core_rhc"))
    return {
        "n": n,
        "n_core_rhc": n_rhc,
        "n_answer": sum(1 for r in recs if r.get("response_mode") == "ANSWER"),
        "n_refuse": sum(1 for r in recs if r.get("response_mode") == "REFUSE"),
        "n_deny": sum(1 for r in recs if r.get("response_mode") == "DENY"),
        "greedy_rhc_rate": (n_rhc / n) if n else None,
        "eps": EPS,
    }
