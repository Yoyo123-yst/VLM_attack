#!/usr/bin/env python3
"""P0.2 (CPU) — re-pair content pairs after label correction + length gating.

Label correction:
  - caption-degenerate records (P0.1) lose core_safe_answer.
  - repeat-garbage records (P0.1) lose core_rhc.
Length gate: n0.axes.length_matched(rhc_chars, safe_chars, 0.33, 3.0).
Pairing: reuse fast_match_pairs cell / one-to-one logic (fast_stats_close + distance).

Output: routeC/out/audit_pairs.json with corrected n_pairs/n_queries/carriers/
        max_query_frac vs ROUTEC_FROZEN.json content gate {min_pairs:12,min_queries:4}.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from n0.axes import length_matched  # noqa: E402
from n1.pilot import fast_stats_close, fast_stats_distance, SEALED  # noqa: E402
from p0.datautil import load_json, save_json  # noqa: E402

OUT = ROOT / "routeC" / "out"
CR0 = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"
N1R = ROOT / "outputs" / "n1r_fast" / "candidates.json"
AUDIT = OUT / "audit_labels.json"
FROZEN = ROOT / "routeC" / "ROUTEC_FROZEN.json"


def _rel_diff(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / (0.5 * (float(a) + float(b)) + 1e-12)


def _records(c: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(c.get("records", []))


def load_corrections() -> Tuple[Set[str], Set[str]]:
    audit = load_json(AUDIT)
    caption_ids = {str(e["record_id"]) for e in audit["caption_degenerate"]}
    garbage_ids = {str(e["record_id"]) for e in audit["repeat_garbage"]}
    return caption_ids, garbage_ids


def corrected_pair(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    caption_ids, garbage_ids = load_corrections()
    cells: Dict[Tuple[str, str, float, int, str], List[Dict[str, Any]]] = {}
    for rec in records:
        rid = str(rec.get("record_id"))
        if rec.get("query_id") in SEALED:
            continue
        key = (
            rec["query_id"],
            rec["carrier_id"],
            float(rec["eps"]),
            int(rec.get("steps") or 40),
            str(rec.get("attack_objective") or "refusal_margin_pgd"),
        )
        cells.setdefault(key, []).append(rec)

    pairs: List[Dict[str, Any]] = []
    for (qid, cid, eps, steps, obj), rows in sorted(cells.items()):
        rhc: List[Dict[str, Any]] = []
        safe: List[Dict[str, Any]] = []
        for r in rows:
            rid = str(r.get("record_id"))
            core_rhc = bool(r.get("core_rhc")) and rid not in garbage_ids
            core_safe = bool(r.get("core_safe_answer")) and rid not in caption_ids
            if core_rhc:
                rhc.append(r)
            if core_safe:
                safe.append(r)
        used_s: Set[int] = set()
        for r in rhc:
            cand: List[Tuple[float, Dict[str, Any]]] = []
            for s in safe:
                if s["restart"] in used_s:
                    continue
                if r["restart"] == s["restart"]:
                    continue
                if not length_matched(r.get("chars") or 0, s.get("chars") or 0, 0.33, 3.0):
                    continue
                if not fast_stats_close(r["delta_stats"], s["delta_stats"], eps):
                    continue
                cand.append((fast_stats_distance(r["delta_stats"], s["delta_stats"]), s))
            if not cand:
                continue
            cand.sort(key=lambda t: t[0])
            s = cand[0][1]
            used_s.add(s["restart"])
            pairs.append(
                {
                    "pair_id": f"{qid}:{cid}:eps{eps:.6f}:{r['restart']}-{s['restart']}",
                    "query_id": qid,
                    "carrier_id": cid,
                    "eps": eps,
                    "steps": int(steps),
                    "attack_objective": obj,
                    "rhc_restart": r["restart"],
                    "safe_restart": s["restart"],
                    "rhc_record_id": r["record_id"],
                    "safe_record_id": s["record_id"],
                    "rhc_chars": r.get("chars"),
                    "safe_chars": s.get("chars"),
                    "stats_distance": cand[0][0],
                    "l2_rel_diff": _rel_diff(r["delta_stats"]["l2_rms"], s["delta_stats"]["l2_rms"]),
                    "tv_rel_diff": _rel_diff(r["delta_stats"]["tv"], s["delta_stats"]["tv"]),
                    "category": r.get("category"),
                }
            )
    return pairs


def query_share(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(pairs)
    counts = Counter(p["query_id"] for p in pairs)
    if not counts:
        return {"max_query": None, "max_n": 0, "max_frac": 0.0, "by_query": {}}
    q, k = counts.most_common(1)[0]
    return {"max_query": q, "max_n": int(k), "max_frac": float(k / n), "by_query": dict(counts)}


def gate(pairs: List[Dict[str, Any]], frozen: Dict[str, Any]) -> Dict[str, Any]:
    spec = frozen["gates"]["G0b_corrected_pairs"]
    min_pairs = int(spec["min_pairs"])
    min_queries = int(spec["min_queries"])
    qids = sorted({p["query_id"] for p in pairs})
    cids = sorted({p["carrier_id"] for p in pairs})
    share = query_share(pairs)
    reasons = []
    if len(pairs) < min_pairs:
        reasons.append(f"n_pairs {len(pairs)} < {min_pairs}")
    if len(qids) < min_queries:
        reasons.append(f"n_queries {len(qids)} < {min_queries}")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "n_pairs": len(pairs),
        "n_queries": len(qids),
        "n_carriers": len(cids),
        "query_ids": qids,
        "carrier_ids": cids,
        "query_share": share,
        "min_pairs": min_pairs,
        "min_queries": min_queries,
    }


def main() -> None:
    cr0 = load_json(CR0)
    n1r = load_json(N1R)
    frozen = load_json(FROZEN)
    caption_ids, garbage_ids = load_corrections()

    # CR-0 and n1r_fast use disjoint query sets (h49..h81 vs h01..h41).
    # They must be gated SEPARATELY — merging them inflates pair counts.
    pairs_cr0 = corrected_pair(_records(cr0))
    pairs_n1r = corrected_pair(_records(n1r))
    g_cr0 = gate(pairs_cr0, frozen)
    g_n1r = gate(pairs_n1r, frozen)

    out = {
        "n_records": {
            "cr0": len(_records(cr0)),
            "n1r_fast": len(_records(n1r)),
        },
        "n_caption_degenerate": len(caption_ids),
        "n_repeat_garbage": len(garbage_ids),
        "cr0_only": {
            "n_pairs": g_cr0["n_pairs"],
            "n_queries": g_cr0["n_queries"],
            "n_carriers": g_cr0["n_carriers"],
            "query_ids": g_cr0["query_ids"],
            "carrier_ids": g_cr0["carrier_ids"],
            "query_share": g_cr0["query_share"],
            "gate_pass": g_cr0["pass"],
            "gate_reasons": g_cr0["reasons"],
            "pairs": pairs_cr0,
        },
        "n1r_fast_only": {
            "n_pairs": g_n1r["n_pairs"],
            "n_queries": g_n1r["n_queries"],
            "n_carriers": g_n1r["n_carriers"],
            "query_ids": g_n1r["query_ids"],
            "carrier_ids": g_n1r["carrier_ids"],
            "query_share": g_n1r["query_share"],
            "gate_pass": g_n1r["pass"],
            "gate_reasons": g_n1r["reasons"],
            "pairs": pairs_n1r,
        },
        "note": "CR-0 (c07/c08, h49..h81) and n1r_fast (c05/c06, h01..h41) are "
        "disjoint experiments; gate each separately, never merged.",
    }
    save_json(OUT / "audit_pairs.json", out)
    print(
        f"P0.2 done: cr0_pairs={g_cr0['n_pairs']} cr0_queries={g_cr0['n_queries']} "
        f"cr0_pass={g_cr0['pass']} | n1r_pairs={g_n1r['n_pairs']} "
        f"n1r_queries={g_n1r['n_queries']} n1r_pass={g_n1r['pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
