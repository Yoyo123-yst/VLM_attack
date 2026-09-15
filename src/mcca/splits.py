"""Query-ID splits. Carriers and restarts follow the query. Frozen before GPU."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

# U_refusal was fit on P0-S traces discover (h01–h46-ish). CR-0 queries are
# disjoint from that fit. Do not train a new probe on CR-0.
P0S_FIT_NOTE = "p0s_discover_U_refusal_frozen"

# Fast-Crossed calibration / test: same query groups as T1-D.
FC_CAL_QUERIES = ("h07", "h21", "h31")
FC_TEST_QUERIES = ("h01", "h11", "h41")

# CR-0: 12 queries, 2 carriers, 8 restarts. Split by query ID only.
# Val chooses τ. Test reports AUC / precision. No random row split.
CR0_VAL_QUERIES = ("h49", "h53", "h61", "h72")
CR0_TEST_QUERIES = ("h56", "h64", "h66", "h68", "h74", "h76", "h80", "h81")
CR0_ALL_QUERIES = CR0_VAL_QUERIES + CR0_TEST_QUERIES

# P1 cells are frozen here but must not be run until P0 GO.
# PASS = same cell has both core_RHC and core_safe_answer.
# FAIL = some ANSWER exists, but RHC and SAFE do not co-occur.
P1_PASS_CELLS = ("h56:c08", "h66:c07", "h72:c07", "h74:c08")
P1_FAIL_CELLS = ("h49:c07", "h53:c08", "h68:c08", "h80:c08")
P1_RESTARTS = (0, 1, 2, 3)
P1_EPS = 16 / 255
P1_STEPS = 100


def query_split(qid: str, source: str) -> str:
    if source in {"cr0", "causal_reach"}:
        if qid in CR0_VAL_QUERIES:
            return "val"
        if qid in CR0_TEST_QUERIES:
            return "test"
        return "unused"
    if source in {"fast_crossed", "fc"}:
        if qid in FC_CAL_QUERIES:
            return "val"
        if qid in FC_TEST_QUERIES:
            return "test"
        return "unused"
    return "unknown"


def assert_query_exclusive(val: Sequence[str], test: Sequence[str]) -> None:
    overlap = set(val) & set(test)
    if overlap:
        raise RuntimeError(f"query leak between val and test: {sorted(overlap)}")


def group_by_query(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r["query_id"]), []).append(r)
    return out


def freeze_record() -> Dict[str, Any]:
    assert_query_exclusive(CR0_VAL_QUERIES, CR0_TEST_QUERIES)
    assert_query_exclusive(FC_CAL_QUERIES, FC_TEST_QUERIES)
    return {
        "probe_fit": P0S_FIT_NOTE,
        "cr0_val_queries": list(CR0_VAL_QUERIES),
        "cr0_test_queries": list(CR0_TEST_QUERIES),
        "fc_val_queries": list(FC_CAL_QUERIES),
        "fc_test_queries": list(FC_TEST_QUERIES),
        "p1_pass_cells": list(P1_PASS_CELLS),
        "p1_fail_cells": list(P1_FAIL_CELLS),
        "p1_restarts": list(P1_RESTARTS),
        "p1_eps": float(P1_EPS),
        "p1_steps": int(P1_STEPS),
        "rule": "same query_id cannot appear in both val and test; carriers/restarts follow query",
    }
