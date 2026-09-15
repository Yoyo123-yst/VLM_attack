"""P0-C sample freeze: 68 static + 15 SAFE↔RHC actuator matches."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

MATCH_TIERS: Tuple[Tuple[str, Callable], ...] = (
    ("same_query_same_carrier", lambda s, r: s["query_id"] == r["query_id"] and s["carrier_id"] == r["carrier_id"]),
    ("same_query", lambda s, r: s["query_id"] == r["query_id"]),
    ("same_carrier_smode_bin", lambda s, r: s["carrier_id"] == r["carrier_id"]),
    ("smode_bin", lambda s, r: True),
)

SMODE_BIN_WIDTH = 8.0


def smode_bin(s: float, width: float = SMODE_BIN_WIDTH) -> int:
    return int(round(float(s) / float(width)))


def local_score(safe: Dict[str, Any], rhc: Dict[str, Any], width: float = SMODE_BIN_WIDTH) -> Tuple[int, int, float]:
    d_r = abs(int(safe.get("restart", 0)) - int(rhc.get("restart", 0)))
    d_bin = abs(smode_bin(safe["s_mode"], width) - smode_bin(rhc["s_mode"], width))
    d_s = abs(float(safe["s_mode"]) - float(rhc["s_mode"]))
    return (-d_r, -d_bin, -d_s)


def static_rows(scores: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in scores if r.get("core_rhc") or r.get("core_safe_answer")]


def _greedy_tier(
    safes: Sequence[Dict[str, Any]],
    rhcs: Sequence[Dict[str, Any]],
    pred: Callable,
    width: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    unused_s = list(safes)
    unused_r = list(rhcs)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    while unused_s and unused_r:
        best = None
        for s in unused_s:
            for r in unused_r:
                if not pred(s, r):
                    continue
                key = (local_score(s, r, width), s["record_id"], r["record_id"])
                if best is None or key > best[0]:
                    best = (key, s, r)
        if best is None:
            break
        _, s, r = best
        pairs.append((s, r))
        unused_s = [x for x in unused_s if x["record_id"] != s["record_id"]]
        unused_r = [x for x in unused_r if x["record_id"] != r["record_id"]]
    return pairs


def match_actuator(
    scores: Sequence[Dict[str, Any]],
    width: float = SMODE_BIN_WIDTH,
) -> List[Dict[str, Any]]:
    safe = sorted(
        [r for r in scores if r.get("core_safe_answer")],
        key=lambda x: (x["query_id"], x["carrier_id"], int(x.get("restart", 0))),
    )
    rhc = [r for r in scores if r.get("core_rhc")]
    used_s, used_r = set(), set()
    out: List[Dict[str, Any]] = []
    for name, pred in MATCH_TIERS:
        rem_s = [s for s in safe if s["record_id"] not in used_s]
        rem_r = [r for r in rhc if r["record_id"] not in used_r]
        for s, r in _greedy_tier(rem_s, rem_r, pred, width):
            used_s.add(s["record_id"])
            used_r.add(r["record_id"])
            out.append(
                {
                    "safe_id": s["record_id"],
                    "rhc_id": r["record_id"],
                    "tier": name,
                    "safe_query": s["query_id"],
                    "rhc_query": r["query_id"],
                    "safe_carrier": s["carrier_id"],
                    "rhc_carrier": r["carrier_id"],
                    "safe_restart": int(s.get("restart", 0)),
                    "rhc_restart": int(r.get("restart", 0)),
                    "safe_s_mode": float(s["s_mode"]),
                    "rhc_s_mode": float(r["s_mode"]),
                }
            )
    out.sort(key=lambda m: (m["safe_query"], m["safe_carrier"], m["safe_restart"]))
    return out
