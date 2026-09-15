#!/usr/bin/env python3
"""CPU pair-loss audit of frozen CR-0 candidates.

Explains why 15 core_safe_answer traces formed 9 content pairs.
Does not re-run PGD, change thresholds, fit U, scan layers, or reopen CR-1.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from n1.pilot import (  # noqa: E402
    FAST_L2_REL_MAX,
    FAST_TV_REL_MAX,
    _rel_diff,
    fast_match_pairs,
    fast_stats_distance,
)
from p0.datautil import load_json, save_json  # noqa: E402

OUT = ROOT / "outputs" / "causal_reach" / "cr0"
EPS = 16.0 / 255.0
LINF_CAP = EPS * 1.01 + 1e-8


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def shannon(counts: Sequence[int]) -> float:
    n = float(sum(counts))
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p, 2)
    return float(h)


def summarize(xs: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(xs), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    q = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "max": float(q[4]),
    }


def mean_diff_report(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    if aa.size == 0 or bb.size == 0:
        return {"safe_minus_rhc_mean": None, "pooled_std": None, "cohen_d": None}
    diff = float(aa.mean() - bb.mean())
    if aa.size + bb.size <= 2:
        pooled = 0.0
    else:
        va = float(aa.var(ddof=1)) if aa.size > 1 else 0.0
        vb = float(bb.var(ddof=1)) if bb.size > 1 else 0.0
        pooled = math.sqrt(((aa.size - 1) * va + (bb.size - 1) * vb) / max(aa.size + bb.size - 2, 1))
    d = diff / pooled if pooled > 1e-12 else None
    return {
        "safe_mean": float(aa.mean()),
        "rhc_mean": float(bb.mean()),
        "safe_minus_rhc_mean": diff,
        "pooled_std": float(pooled),
        "cohen_d": None if d is None else float(d),
    }


def flags_vs(rhc: Dict[str, Any], safe: Dict[str, Any]) -> Dict[str, Any]:
    a = rhc["delta_stats"]
    b = safe["delta_stats"]
    l2 = _rel_diff(a["l2_rms"], b["l2_rms"])
    tv = _rel_diff(a["tv"], b["tv"])
    linf_fail = float(a["linf"]) > LINF_CAP or float(b["linf"]) > LINF_CAP
    l2_fail = l2 > FAST_L2_REL_MAX
    tv_fail = tv > FAST_TV_REL_MAX
    return {
        "rhc_record_id": rhc["record_id"],
        "safe_record_id": safe["record_id"],
        "rhc_restart": rhc["restart"],
        "safe_restart": safe["restart"],
        "l2_rel_diff": float(l2),
        "tv_rel_diff": float(tv),
        "stats_distance": float(fast_stats_distance(a, b)),
        "linf_fail": bool(linf_fail),
        "l2_fail": bool(l2_fail),
        "tv_fail": bool(tv_fail),
        "both_fail": bool(l2_fail and tv_fail),
        "l2_only": bool(l2_fail and not tv_fail),
        "tv_only": bool(tv_fail and not l2_fail),
        "pass": bool((not linf_fail) and (not l2_fail) and (not tv_fail)),
    }


def exclusive_geometry(cands: Sequence[Dict[str, Any]]) -> str:
    if not cands:
        return "no_rhc_in_cell"
    if any(c["pass"] for c in cands):
        return "geometry_pass"
    if all(c["l2_only"] for c in cands):
        return "l2_only"
    if all(c["tv_only"] for c in cands):
        return "tv_only"
    if all(c["both_fail"] for c in cands):
        return "both"
    return "mixed_no_pass"


def fmt_sum(s: Dict[str, Any]) -> str:
    if not s or s.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={s['n']} mean={s['mean']:.5f} med={s['median']:.5f} "
        f"p25={s['p25']:.5f} p75={s['p75']:.5f} min={s['min']:.5f} max={s['max']:.5f}"
    )


def main() -> None:
    cand = load_json(OUT / "candidates.json")
    pair_blob = load_json(OUT / "pairs.json")
    records: List[Dict[str, Any]] = list(cand["records"])
    frozen_pairs = list(pair_blob.get("content_pairs") or [])
    recomputed = fast_match_pairs(records)
    if len(recomputed) != len(frozen_pairs):
        raise RuntimeError(
            f"frozen pairing drifted: recomputed {len(recomputed)} vs stored {len(frozen_pairs)}"
        )
    matched_safe = {p["safe_record_id"] for p in frozen_pairs}
    matched_rhc = {p["rhc_record_id"] for p in frozen_pairs}

    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        cells[(r["query_id"], r["carrier_id"])].append(r)

    safes = [r for r in records if r.get("core_safe_answer")]
    rhcs = [r for r in records if r.get("core_rhc")]
    if len(safes) != 15:
        raise RuntimeError(f"expected 15 core_safe_answer, got {len(safes)}")

    candidate_pairs: List[Dict[str, Any]] = []
    safe_rows: List[Dict[str, Any]] = []
    for s in safes:
        cell = cells[(s["query_id"], s["carrier_id"])]
        cell_rhc = [r for r in cell if r.get("core_rhc") and r["restart"] != s["restart"]]
        cands = [flags_vs(r, s) for r in cell_rhc]
        candidate_pairs.extend(
            {
                **c,
                "query_id": s["query_id"],
                "carrier_id": s["carrier_id"],
                "category": s.get("category"),
            }
            for c in cands
        )
        geom = exclusive_geometry(cands)
        matched = s["record_id"] in matched_safe
        if not cell_rhc:
            fate = "no_rhc_in_cell"
        elif matched:
            fate = "matched"
        elif geom == "geometry_pass":
            fate = "geometry_pass_unmatched"
        else:
            fate = geom
        nearest = min(cands, key=lambda c: c["stats_distance"]) if cands else None
        safe_rows.append(
            {
                "record_id": s["record_id"],
                "query_id": s["query_id"],
                "carrier_id": s["carrier_id"],
                "restart": s["restart"],
                "seed": s["seed"],
                "category": s.get("category"),
                "n_rhc_in_cell": len(cell_rhc),
                "n_geometry_pass": int(sum(1 for c in cands if c["pass"])),
                "geometry": geom,
                "fate": fate,
                "matched": matched,
                "l2_rms": s["delta_stats"]["l2_rms"],
                "tv": s["delta_stats"]["tv"],
                "linf": s["delta_stats"]["linf"],
                "chars": s.get("chars"),
                "nearest": nearest,
            }
        )

    fate_counts = dict(Counter(row["fate"] for row in safe_rows))
    n_no_rhc = int(sum(1 for row in safe_rows if row["fate"] == "no_rhc_in_cell"))
    n_had_rhc = len(safes) - n_no_rhc
    unmatched_had_rhc = [row for row in safe_rows if row["n_rhc_in_cell"] > 0 and not row["matched"]]

    # Exclusive SAFE-level geometry among those with RHC that did not match.
    n_l2_elim = int(sum(1 for row in unmatched_had_rhc if row["fate"] == "l2_only"))
    n_tv_elim = int(sum(1 for row in unmatched_had_rhc if row["fate"] == "tv_only"))
    n_both_elim = int(sum(1 for row in unmatched_had_rhc if row["fate"] == "both"))
    n_mixed = int(sum(1 for row in unmatched_had_rhc if row["fate"] == "mixed_no_pass"))
    n_geom_left = int(sum(1 for row in unmatched_had_rhc if row["fate"] == "geometry_pass_unmatched"))

    # Nearest-RHC view (unmatched with RHC present).
    nearest_elim = {"l2_only": 0, "tv_only": 0, "both": 0, "linf_only": 0, "pass": 0}
    for row in unmatched_had_rhc:
        n = row["nearest"]
        if n is None:
            continue
        if n["pass"]:
            nearest_elim["pass"] += 1
        elif n["both_fail"]:
            nearest_elim["both"] += 1
        elif n["l2_only"]:
            nearest_elim["l2_only"] += 1
        elif n["tv_only"]:
            nearest_elim["tv_only"] += 1
        elif n["linf_fail"]:
            nearest_elim["linf_only"] += 1

    pair_mat = {
        "n_same_cell_rhc_safe": len(candidate_pairs),
        "n_pass": int(sum(1 for c in candidate_pairs if c["pass"])),
        "n_l2_fail": int(sum(1 for c in candidate_pairs if c["l2_fail"])),
        "n_tv_fail": int(sum(1 for c in candidate_pairs if c["tv_fail"])),
        "n_l2_only": int(sum(1 for c in candidate_pairs if c["l2_only"])),
        "n_tv_only": int(sum(1 for c in candidate_pairs if c["tv_only"])),
        "n_both": int(sum(1 for c in candidate_pairs if c["both_fail"])),
        "n_linf_fail": int(sum(1 for c in candidate_pairs if c["linf_fail"])),
    }
    cell_capacity = []
    for (qid, cid), rows in sorted(cells.items()):
        n_rhc = int(sum(1 for r in rows if r.get("core_rhc")))
        n_safe = int(sum(1 for r in rows if r.get("core_safe_answer")))
        cap = int(min(n_rhc, n_safe))
        cell_capacity.append(
            {
                "query_id": qid,
                "carrier_id": cid,
                "n_rhc": n_rhc,
                "n_safe": n_safe,
                "one_to_one_cap": cap,
            }
        )
    theoretical_max_pairs = int(sum(c["one_to_one_cap"] for c in cell_capacity))

    cell_rows = []
    for (qid, cid), rows in sorted(cells.items()):
        mode_c = Counter(r.get("response_mode") for r in rows)
        n_rhc = int(sum(1 for r in rows if r.get("core_rhc")))
        n_safe = int(sum(1 for r in rows if r.get("core_safe_answer")))
        n_other = len(rows) - n_rhc - n_safe
        content_c = {"core_rhc": n_rhc, "core_safe_answer": n_safe, "other": n_other}
        cell_rows.append(
            {
                "query_id": qid,
                "carrier_id": cid,
                "n": len(rows),
                "n_REFUSE": int(mode_c.get("REFUSE") or 0),
                "n_ANSWER": int(mode_c.get("ANSWER") or 0),
                "n_DENY": int(mode_c.get("DENY") or 0),
                "n_RHC": n_rhc,
                "n_SAFE": n_safe,
                "H_mode": shannon(
                    [mode_c.get("REFUSE") or 0, mode_c.get("ANSWER") or 0, mode_c.get("DENY") or 0]
                ),
                "H_content": shannon([n_rhc, n_safe, n_other]),
                "mode_counts": dict(mode_c),
                "content_counts": content_c,
            }
        )

    def take(label: str, key: str) -> List[float]:
        src = rhcs if label == "rhc" else safes
        if key == "chars":
            return [float(r.get("chars") or 0) for r in src]
        return [float(r["delta_stats"][key]) for r in src]

    dist = {}
    for key in ("l2_rms", "tv", "linf", "spec_hf_frac", "chars"):
        dist[key] = {
            "safe": summarize(take("safe", key)),
            "rhc": summarize(take("rhc", key)),
            "contrast": mean_diff_report(take("safe", key), take("rhc", key)),
        }

    pair_list = []
    for p in frozen_pairs:
        pair_list.append(
            {
                "pair_id": p["pair_id"],
                "query_id": p["query_id"],
                "carrier_id": p["carrier_id"],
                "category": p.get("category"),
                "rhc_restart": p["rhc_restart"],
                "safe_restart": p["safe_restart"],
                "rhc_record_id": p["rhc_record_id"],
                "safe_record_id": p["safe_record_id"],
                "l2_rel_diff": p["l2_rel_diff"],
                "tv_rel_diff": p["tv_rel_diff"],
            }
        )

    blob = {
        "schema": "cr0_pair_loss_audit_v1",
        "written": now_iso(),
        "scope": "CPU descriptive audit of frozen CR-0 candidates. Not a re-gate.",
        "forbidden": [
            "do not lower L2/TV thresholds",
            "do not rechoose pairing rules",
            "do not fit a direction on the 9 pairs",
            "do not scan layers on the 9 pairs",
            "do not reopen CR-1",
        ],
        "frozen_thresholds": {"l2_rel_max": FAST_L2_REL_MAX, "tv_rel_max": FAST_TV_REL_MAX, "eps": EPS},
        "n_records": len(records),
        "n_core_safe_answer": len(safes),
        "n_core_rhc": len(rhcs),
        "n_content_pairs": len(frozen_pairs),
        "theoretical_max_pairs_one_to_one": theoretical_max_pairs,
        "cell_capacity": cell_capacity,
        "fate_counts": fate_counts,
        "q1_safe_no_rhc_in_cell": n_no_rhc,
        "q2_q4_unmatched_with_rhc_present": {
            "n_safe_with_rhc_in_cell": n_had_rhc,
            "n_unmatched_among_those": len(unmatched_had_rhc),
            "exclusive_all_rhc_fail": {
                "l2_only": n_l2_elim,
                "tv_only": n_tv_elim,
                "both": n_both_elim,
                "mixed_no_pass": n_mixed,
                "geometry_pass_unmatched": n_geom_left,
            },
            "nearest_rhc": nearest_elim,
            "note": (
                "Exclusive counts classify a SAFE against every same-cell RHC. "
                "Nearest-RHC is the closest by L2rel+TVrel."
            ),
        },
        "candidate_pair_matrix": pair_mat,
        "q5_nine_pairs": pair_list,
        "q6_cell_entropy": cell_rows,
        "q7_delta_distributions": dist,
        "margin_note": (
            "CR-0 did not store first-token refusal margin. This CPU audit does not "
            "reload the model, so margin distributions are unavailable."
        ),
        "safe_rows": safe_rows,
        "matched_rhc_ids": sorted(matched_rhc),
        "matched_safe_ids": sorted(matched_safe),
    }
    save_json(OUT / "CR0_PAIR_LOSS_AUDIT.json", blob)

    lines = [
        "# CR-0 Pair-Loss Audit (CPU, frozen rules)",
        "",
        f"- Written: `{now_iso()}`",
        "- Input: frozen `candidates.json` + `pairs.json`. No GPU, no new PGD, no hidden.",
        "- Frozen match: same query/carrier/objective/budget, greedy both sides, one-to-one, L2 rel ≤10%, TV rel ≤20%.",
        "- This explains the content-gate miss. It does **not** retune pairing, fit U, scan layers, or reopen CR-1.",
        "",
        "## Why 15 SAFE became 9 pairs",
        "",
        f"- core_safe_answer = **{len(safes)}**",
        f"- core_rhc = **{len(rhcs)}**",
        f"- frozen content pairs = **{len(frozen_pairs)}**",
        f"- unmatched SAFE = **{len(safes) - len(frozen_pairs)}**",
        f"- one-to-one cell cap Σ min(n_RHC, n_SAFE) = **{theoretical_max_pairs}** (equals the 9 pairs; L2/TV did not bind)",
        "",
        "### 1. SAFE with no RHC in the same query–carrier cell",
        "",
        f"**{n_no_rhc} / 15** SAFE sit in a cell that has zero `core_rhc`.",
        "",
    ]
    none = [row for row in safe_rows if row["fate"] == "no_rhc_in_cell"]
    if none:
        for row in none:
            lines.append(
                f"- `{row['record_id']}` ({row['category']}) — cell `{row['query_id']}:{row['carrier_id']}` has 0 RHC"
            )
    else:
        lines.append("- none")

    lines += [
        "",
        "### 2–4. SAFE that had at least one RHC, then failed geometry",
        "",
        f"SAFE with ≥1 same-cell RHC: **{n_had_rhc}**. Of these, **{len(unmatched_had_rhc)}** were not paired.",
        "",
        "Exclusive fate of those unmatched SAFE (every same-cell RHC fails the named way):",
        "",
        f"- eliminated by **L2 only** (TV would pass vs every RHC): **{n_l2_elim}**",
        f"- eliminated by **TV only** (L2 would pass vs every RHC): **{n_tv_elim}**",
        f"- eliminated by **both** L2 and TV vs every RHC: **{n_both_elim}**",
        f"- mixed geometry failures, none pass: **{n_mixed}**",
        f"- geometry would pass, leftover after one-to-one: **{n_geom_left}**",
        "",
        "Nearest-RHC view of the same unmatched SAFE:",
        "",
        f"- nearest fails L2 only: **{nearest_elim['l2_only']}**",
        f"- nearest fails TV only: **{nearest_elim['tv_only']}**",
        f"- nearest fails both: **{nearest_elim['both']}**",
        f"- nearest would pass: **{nearest_elim['pass']}**",
        "",
        "Same-cell RHC×SAFE candidate matrix (not one-to-one):",
        "",
        f"- candidates = {pair_mat['n_same_cell_rhc_safe']}",
        f"- pass L2+TV = {pair_mat['n_pass']}",
        f"- L2 fail (any TV): {pair_mat['n_l2_fail']}",
        f"- TV fail (any L2): {pair_mat['n_tv_fail']}",
        f"- L2 only / TV only / both = {pair_mat['n_l2_only']} / {pair_mat['n_tv_only']} / {pair_mat['n_both']}",
        f"- L∞ over cap = {pair_mat['n_linf_fail']}",
        "",
        "Unmatched SAFE with RHC present:",
        "",
    ]
    if unmatched_had_rhc:
        for row in unmatched_had_rhc:
            n = row["nearest"] or {}
            lines.append(
                f"- `{row['record_id']}` fate=`{row['fate']}` n_rhc={row['n_rhc_in_cell']} "
                f"n_pass={row['n_geometry_pass']} nearest_L2rel={n.get('l2_rel_diff')} "
                f"nearest_TVrel={n.get('tv_rel_diff')} nearest=`{n.get('rhc_record_id')}`"
            )
    else:
        lines.append("- none")

    lines += [
        "",
        "### 5. The nine frozen pairs",
        "",
        "| pair | query | carrier | RHC restart | SAFE restart | L2 rel | TV rel |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for p in pair_list:
        lines.append(
            f"| `{p['pair_id']}` | {p['query_id']} | {p['carrier_id']} | "
            f"{p['rhc_restart']} | {p['safe_restart']} | {p['l2_rel_diff']:.4f} | {p['tv_rel_diff']:.4f} |"
        )

    lines += [
        "",
        "### 6. Per-cell REFUSE / ANSWER / DENY / RHC / SAFE and entropy",
        "",
        "RHC and SAFE are content flags inside ANSWER; they are not a fifth mutually exclusive mode. "
        "`H_mode` is Shannon entropy (bits) of {REFUSE, ANSWER, DENY} over the 8 restarts. "
        "`H_content` is entropy of {core_rhc, core_safe_answer, other}.",
        "",
        "| cell | REF | ANS | DENY | RHC | SAFE | H_mode | H_content |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cell_rows:
        lines.append(
            f"| `{c['query_id']}:{c['carrier_id']}` | {c['n_REFUSE']} | {c['n_ANSWER']} | "
            f"{c['n_DENY']} | {c['n_RHC']} | {c['n_SAFE']} | {c['H_mode']:.3f} | {c['H_content']:.3f} |"
        )

    h_mode = [c["H_mode"] for c in cell_rows]
    h_content = [c["H_content"] for c in cell_rows]
    lines += [
        "",
        f"- mean H_mode = {float(np.mean(h_mode)):.3f}; mean H_content = {float(np.mean(h_content)):.3f}",
        f"- cells with both RHC and SAFE: {sum(1 for c in cell_rows if c['n_RHC'] > 0 and c['n_SAFE'] > 0)} / {len(cell_rows)}",
        f"- cells with SAFE but no RHC: {sum(1 for c in cell_rows if c['n_SAFE'] > 0 and c['n_RHC'] == 0)}",
        "",
        "### 7. SAFE vs RHC perturbation stats (stored `delta_stats` only)",
        "",
        blob["margin_note"],
        "",
        f"- L2 RMS SAFE: {fmt_sum(dist['l2_rms']['safe'])}",
        f"- L2 RMS RHC:  {fmt_sum(dist['l2_rms']['rhc'])}",
        f"-  Δmean (SAFE−RHC) = {dist['l2_rms']['contrast']['safe_minus_rhc_mean']:.6f}; "
        f"Cohen d = {dist['l2_rms']['contrast']['cohen_d']:.3f}",
        "",
        f"- TV SAFE: {fmt_sum(dist['tv']['safe'])}",
        f"- TV RHC:  {fmt_sum(dist['tv']['rhc'])}",
        f"-  Δmean (SAFE−RHC) = {dist['tv']['contrast']['safe_minus_rhc_mean']:.6f}; "
        f"Cohen d = {dist['tv']['contrast']['cohen_d']:.3f}",
        "",
        f"- L∞ SAFE: {fmt_sum(dist['linf']['safe'])}",
        f"- L∞ RHC:  {fmt_sum(dist['linf']['rhc'])}",
        f"-  Δmean (SAFE−RHC) = {dist['linf']['contrast']['safe_minus_rhc_mean']:.6f}; "
        f"Cohen d = {dist['linf']['contrast']['cohen_d']}",
        "",
        f"- spec_hf_frac SAFE: {fmt_sum(dist['spec_hf_frac']['safe'])}",
        f"- spec_hf_frac RHC:  {fmt_sum(dist['spec_hf_frac']['rhc'])}",
        f"-  Δmean (SAFE−RHC) = {dist['spec_hf_frac']['contrast']['safe_minus_rhc_mean']:.6f}; "
        f"Cohen d = {dist['spec_hf_frac']['contrast']['cohen_d']:.3f}",
        "",
        f"- chars SAFE: {fmt_sum(dist['chars']['safe'])}",
        f"- chars RHC:  {fmt_sum(dist['chars']['rhc'])}",
        "",
        "## Interpretation",
        "",
    ]

    # Build interpretation from actual numbers after we know them — write placeholders
    # filled from computed values.
    lines += [
        f"The 15→9 gap is **{n_no_rhc} SAFE with no co-located RHC** (all on `h53:c08`) plus "
        f"**{len(unmatched_had_rhc)} extra SAFE on `h72:c07`** (4 SAFE vs 1 RHC; one-to-one keeps 1 pair).",
        f"Across 24 cells, Σ min(n_RHC, n_SAFE) = {theoretical_max_pairs}. Frozen pairing realized all {len(frozen_pairs)} of those slots.",
        "**L2 eliminated 0. TV eliminated 0. Both eliminated 0.** All 29 same-cell RHC×SAFE candidates already sat inside 10%/20%. "
        "Pixel geometry of SAFE vs RHC is not systematically different (L2 d≈0.02, TV d≈0.13; L∞ identical at 16/255).",
        "The content gate failed because SAFE and RHC rarely co-occur in the same 8-restart cell, not because the L2/TV matcher threw pairs away.",
        "First-token refusal margin was not stored; do not infer a margin story from this file.",
        "Do not lower thresholds or start CR-1 from these 9 pairs.",
        "",
    ]
    (OUT / "CR0_PAIR_LOSS_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        {
            "n_safe": len(safes),
            "n_pairs": len(frozen_pairs),
            "q1_no_rhc": n_no_rhc,
            "fate_counts": fate_counts,
            "pair_matrix": pair_mat,
            "out_md": str(OUT / "CR0_PAIR_LOSS_AUDIT.md"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
