"""P0-S: decouple safety from response mode. Does not reopen Attack Gate."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from p0.catalog import attack_test_pairs, probe_pairs
from p0.judge import triple_axes
from p0.subspace import pca_basis_info, principal_angles_deg, random_basis
from p0_qwen.integrity import PIN_LAYER, PIN_REQUESTED_RANK
from p0_qwen.traces import jb_pairs, related_safe_pairs, theme_denial_pairs

MIN_N_PCA = 30
SITE = f"L{PIN_LAYER}:last_user"
CONFIRM_IDS = ("h83", "h84", "h85", "h86", "h87", "h88", "h89", "h90")


def frozen_p0s_splits(probe: Dict[str, Any], traces: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    attack_ids = {p["id"] for p in attack_test_pairs()}
    probe_ids = {p["id"] for p in probe_pairs()}
    confirm = [i for i in CONFIRM_IDS if i in probe_ids]
    discover = list(probe.get("discover") or [])
    holdout = list(probe.get("holdout") or [])
    trace_ids = {r.get("query_id") for r in (traces or {}).get("records") or []}
    if set(confirm) & set(discover) or set(confirm) & set(holdout):
        raise AssertionError("mechanism-confirm ids overlap discover/holdout")
    if set(confirm) & attack_ids:
        raise AssertionError("mechanism-confirm overlaps attack-test")
    if traces is not None and set(confirm) & trace_ids:
        raise AssertionError("mechanism-confirm already appears in traces; do not peek")
    return {
        "fit": "discover",
        "development": "holdout",
        "mechanism_confirm_ids": confirm,
        "attack_test_ids": sorted(attack_ids),
        "holdout_is_development": True,
        "confirm_not_yet_traced": True,
        "reopens_attack_gate": False,
        "n_fit": len(discover),
        "n_development": len(holdout),
        "n_confirm": len(confirm),
        "n_attack_test": len(attack_ids),
        "note": (
            "Original discover fits U_safety / U_refusal / U_denial. "
            "Current holdout is development only. h83–h90 are the unused usable "
            "probe ids reserved for one-shot mechanism-confirm. h91–h130 stay sealed."
        ),
    }


def _hidden(slot: Dict[str, Any], key: str = SITE) -> np.ndarray:
    return np.asarray(slot[key], dtype=np.float64).reshape(-1)


def _judge_slot(rec: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    if name == "clean":
        return rec.get("clean")
    blob = rec.get(name)
    if blob is None:
        return None
    return blob.get("judge") or blob


def inventory(traces: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    counts: Dict[str, Dict[str, int]] = {}
    for rec in traces.get("records") or []:
        split = rec.get("split") or "unknown"
        slot_counts = counts.setdefault(
            split,
            {"n": 0, "RHC": 0, "RELATED_SAFE": 0, "THEME_DENIAL": 0, "REF_clean": 0, "other_fail": 0},
        )
        slot_counts["n"] += 1
        if rec.get("clean", {}).get("label") == "REF":
            slot_counts["REF_clean"] += 1
        if rec.get("jb"):
            slot_counts["RHC"] += 1
        if rec.get("related_safe"):
            slot_counts["RELATED_SAFE"] += 1
        fail_j = (rec.get("fail") or {}).get("judge") or {}
        if rec.get("fail") and float(fail_j.get("theme_denial") or 0) >= 0.5:
            slot_counts["THEME_DENIAL"] += 1
        elif rec.get("fail"):
            slot_counts["other_fail"] += 1
        for name in ("clean", "jb", "related_safe", "fail"):
            j = _judge_slot(rec, name)
            if not j:
                continue
            ax = triple_axes(j)
            rows.append(
                {
                    "query_id": rec.get("query_id"),
                    "split": split,
                    "slot": name,
                    "legacy_label": ax["legacy_label"],
                    **{k: ax[k] for k in ("safety", "response_mode", "task_quality")},
                }
            )
    fit_n = counts.get("discover") or {}
    pca_ok = {
        "RHC": int(fit_n.get("RHC") or 0) >= MIN_N_PCA,
        "RELATED_SAFE": int(fit_n.get("RELATED_SAFE") or 0) >= MIN_N_PCA,
        "THEME_DENIAL": int(fit_n.get("THEME_DENIAL") or 0) >= MIN_N_PCA,
    }
    return {
        "min_n_pca": MIN_N_PCA,
        "by_split": counts,
        "pca_eligible": pca_ok,
        "need_more_related_safe": not pca_ok["RELATED_SAFE"],
        "need_more_theme_denial": not pca_ok["THEME_DENIAL"],
        "n_axis_rows": len(rows),
        "axis_rows": rows,
    }


def _stack_slot(recs: Sequence[Dict[str, Any]], slot: str) -> np.ndarray:
    rows = []
    for rec in recs:
        if slot == "clean":
            rows.append(_hidden(rec["clean_hidden"]))
        elif slot == "jb":
            rows.append(_hidden(rec["jb"]["hidden"]))
        elif slot == "related_safe":
            rows.append(_hidden(rec["related_safe"]["hidden"]))
        elif slot == "fail":
            rows.append(_hidden(rec["fail"]["hidden"]))
    if not rows:
        return np.zeros((0, 1), dtype=np.float64)
    return np.stack(rows, axis=0)


def _mean_or_pca(deltas: np.ndarray, requested: int, n_a: int, n_b: int, tag: str) -> Dict[str, Any]:
    n = int(deltas.shape[0])
    high_rank = n >= MIN_N_PCA and n_a >= MIN_N_PCA and n_b >= MIN_N_PCA
    if n == 0:
        return {
            "tag": tag,
            "U": np.zeros((0, 0)),
            "requested_rank": int(requested),
            "used_rank": 0,
            "effective_rank": 0,
            "n_deltas": 0,
            "n_a": n_a,
            "n_b": n_b,
            "mode": "empty",
            "high_rank_pca": False,
        }
    if not high_rank or n < 2:
        mu = deltas.mean(axis=0)
        nrm = float(np.linalg.norm(mu)) + 1e-12
        U = (mu / nrm).reshape(-1, 1)
        return {
            "tag": tag,
            "U": U,
            "requested_rank": int(requested),
            "used_rank": 1,
            "effective_rank": 1,
            "n_deltas": n,
            "n_a": n_a,
            "n_b": n_b,
            "mode": "mean_direction",
            "high_rank_pca": False,
            "reason": f"need>={MIN_N_PCA} per state for PCA; using mean direction",
        }
    info = pca_basis_info(deltas, rank=min(int(requested), n - 1))
    info["tag"] = tag
    info["n_deltas"] = n
    info["n_a"] = n_a
    info["n_b"] = n_b
    info["mode"] = "centered_pca"
    info["high_rank_pca"] = True
    return info


def fit_three_directions(traces: Dict[str, Any], requested_rank: int = PIN_REQUESTED_RANK) -> Dict[str, Any]:
    rhc = jb_pairs(traces, split="discover")
    rs = related_safe_pairs(traces, split="discover")
    den = theme_denial_pairs(traces, split="discover")
    H_rhc = _stack_slot(rhc, "jb")
    H_rs = _stack_slot(rs, "related_safe")
    H_ref_rs = _stack_slot(rs, "clean")
    H_den = _stack_slot(den, "fail")
    H_ref_den = _stack_slot(den, "clean")

    if H_rs.shape[0]:
        mu_rs = H_rs.mean(axis=0, keepdims=True)
        d_safety = H_rhc - mu_rs if H_rhc.shape[0] else np.zeros((0, H_rs.shape[1]))
    else:
        d_safety = np.zeros((0, H_rhc.shape[1] if H_rhc.shape[0] else 0))
    d_refusal = H_rs - H_ref_rs if H_rs.shape[0] else np.zeros((0, 1))
    d_denial = H_den - H_ref_den if H_den.shape[0] else np.zeros((0, 1))

    u_s = _mean_or_pca(d_safety, requested_rank, len(rhc), len(rs), "U_safety")
    u_r = _mean_or_pca(d_refusal, requested_rank, len(rs), len(rs), "U_refusal")
    u_d = _mean_or_pca(d_denial, requested_rank, len(den), len(den), "U_denial")
    dim = int(u_s["U"].shape[0] or u_r["U"].shape[0] or u_d["U"].shape[0])
    seeds = {"U_safety": 11, "U_refusal": 22, "U_denial": 33}
    for spec in (u_s, u_r, u_d):
        r = int(spec["U"].shape[1]) if spec["U"].size else 0
        spec["U_rand"] = (
            random_basis(dim, max(r, 1), seed=2026 + seeds[spec["tag"]])[:, : max(r, 1)] if dim else np.zeros((0, 0))
        )
        if r:
            hrefs = [_hidden(p["clean_hidden"]) for p in rhc] if rhc else []
            spec["mu_ref"] = (np.stack(hrefs).mean(axis=0) @ spec["U"]).tolist() if hrefs else []
        else:
            spec["mu_ref"] = []
    return {
        "layer": PIN_LAYER,
        "requested_rank": int(requested_rank),
        "fit_split": "discover",
        "holdout_used_for_fit": False,
        "confirm_used_for_fit": False,
        "U_safety": u_s,
        "U_refusal": u_r,
        "U_denial": u_d,
        "n_discover_rhc": len(rhc),
        "n_discover_related_safe": len(rs),
        "n_discover_theme_denial": len(den),
    }


def _overlap_frac(U: np.ndarray, V: np.ndarray) -> Optional[float]:
    if U.size == 0 or V.size == 0:
        return None
    q1, _ = np.linalg.qr(U, mode="reduced")
    q2, _ = np.linalg.qr(V, mode="reduced")
    return float(np.linalg.norm(q1.T @ q2, "fro") ** 2 / max(min(q1.shape[1], q2.shape[1]), 1))


def _angle_pack(U: np.ndarray, V: np.ndarray) -> Dict[str, Any]:
    if U.size == 0 or V.size == 0:
        return {"n_angles": 0, "mean_deg": None, "min_deg": None, "max_deg": None, "overlap_frac": None}
    ang = principal_angles_deg(U, V)
    return {
        "n_angles": int(len(ang)),
        "mean_deg": float(np.mean(ang)),
        "min_deg": float(np.min(ang)),
        "max_deg": float(np.max(ang)),
        "angles_deg": [float(x) for x in ang],
        "overlap_frac": _overlap_frac(U, V),
    }


def _split_half_stability(deltas: np.ndarray, requested: int, n_a: int, n_b: int, tag: str) -> Dict[str, Any]:
    n = int(deltas.shape[0])
    if n < 4:
        return {"n": n, "mean_min_deg": None, "note": "too few for split-half"}
    even = deltas[0::2]
    odd = deltas[1::2]
    a = _mean_or_pca(even, requested, n_a // 2, n_b // 2, tag)
    b = _mean_or_pca(odd, requested, n_a - n_a // 2, n_b - n_b // 2, tag)
    pack = _angle_pack(a["U"], b["U"])
    pack["n"] = n
    return pack


def _bootstrap_stability(deltas: np.ndarray, requested: int, n_a: int, n_b: int, tag: str, n_boot: int = 16) -> Dict[str, Any]:
    n = int(deltas.shape[0])
    if n < 4:
        return {"n_boot": 0, "mean_min_deg": None}
    base = _mean_or_pca(deltas, requested, n_a, n_b, tag)
    rng = np.random.default_rng(2026)
    mins = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = _mean_or_pca(deltas[idx], requested, n_a, n_b, tag)
        pack = _angle_pack(base["U"], boot["U"])
        if pack["min_deg"] is not None:
            mins.append(pack["min_deg"])
    return {
        "n_boot": n_boot,
        "mean_min_deg": float(np.mean(mins)) if mins else None,
        "max_min_deg": float(np.max(mins)) if mins else None,
    }


def _proj_stats(U: np.ndarray, mats: Dict[str, np.ndarray]) -> Dict[str, Any]:
    if U.size == 0:
        return {}
    u0 = U[:, 0]
    out = {}
    for name, X in mats.items():
        if X.size == 0:
            out[name] = {"n": 0}
            continue
        z = X @ u0
        out[name] = {
            "n": int(X.shape[0]),
            "mean": float(z.mean()),
            "std": float(z.std()),
            "p10": float(np.percentile(z, 10)),
            "p90": float(np.percentile(z, 90)),
        }
    return out


def diagnose_directions(traces: Dict[str, Any], fitted: Dict[str, Any]) -> Dict[str, Any]:
    rhc = jb_pairs(traces, split="discover")
    rs = related_safe_pairs(traces, split="discover")
    den = theme_denial_pairs(traces, split="discover")
    ref_clean = [r for r in traces.get("records") or [] if r.get("split") == "discover" and r.get("clean", {}).get("label") == "REF"]
    mats = {
        "RHC": _stack_slot(rhc, "jb"),
        "RELATED_SAFE": _stack_slot(rs, "related_safe"),
        "THEME_DENIAL": _stack_slot(den, "fail"),
        "REF": _stack_slot(ref_clean, "clean"),
    }
    Us = {k: np.asarray(fitted[k]["U"], dtype=np.float64) for k in ("U_safety", "U_refusal", "U_denial")}
    pairs = [("U_safety", "U_refusal"), ("U_safety", "U_denial"), ("U_refusal", "U_denial")]
    angles = {f"{a}_vs_{b}": _angle_pack(Us[a], Us[b]) for a, b in pairs}

    H_rhc, H_rs = mats["RHC"], mats["RELATED_SAFE"]
    if H_rs.shape[0] and H_rhc.shape[0]:
        d_safety = H_rhc - H_rs.mean(axis=0, keepdims=True)
    else:
        d_safety = np.zeros((0, 1))
    d_refusal = H_rs - _stack_slot(rs, "clean") if rs else np.zeros((0, 1))
    d_denial = mats["THEME_DENIAL"] - _stack_slot(den, "clean") if den else np.zeros((0, 1))
    req = int(fitted.get("requested_rank") or PIN_REQUESTED_RANK)
    stability = {
        "U_safety_split_half": _split_half_stability(d_safety, req, len(rhc), len(rs), "U_safety"),
        "U_refusal_split_half": _split_half_stability(d_refusal, req, len(rs), len(rs), "U_refusal"),
        "U_denial_split_half": _split_half_stability(d_denial, req, len(den), len(den), "U_denial"),
        "U_safety_bootstrap": _bootstrap_stability(d_safety, req, len(rhc), len(rs), "U_safety"),
        "U_refusal_bootstrap": _bootstrap_stability(d_refusal, req, len(rs), len(rs), "U_refusal"),
        "U_denial_bootstrap": _bootstrap_stability(d_denial, req, len(den), len(den), "U_denial"),
    }
    projs = {name: _proj_stats(U, mats) for name, U in Us.items()}
    entangled = False
    for pack in angles.values():
        if pack.get("mean_deg") is not None and pack["mean_deg"] < 20.0:
            entangled = True
    return {
        "angles": angles,
        "stability": stability,
        "projections": projs,
        "late_layer_entangled": entangled,
        "reading": (
            "Three directions are highly aligned at L24; safety and response mode are entangled."
            if entangled
            else "Directions are not collinear; still need factor patching to test causal specificity."
        ),
    }


def slim_fit(fitted: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(fitted)
    for key in ("U_safety", "U_refusal", "U_denial"):
        spec = dict(out[key])
        spec["U"] = np.asarray(spec["U"]).tolist()
        spec["U_rand"] = np.asarray(spec["U_rand"]).tolist()
        out[key] = spec
    return out


def looks_like_p0s_gate(row: Dict[str, Any], controls: Dict[str, Any]) -> Dict[str, Any]:
    """Descriptive death-gate check. Must not reopen Attack Gate by itself."""
    reasons = []
    st = row.get("safety_patch") or {}
    if not st.get("bidirectional"):
        reasons.append("safety_not_bidirectional")
    if st.get("ref_to_jb_dR_lo") is not None and float(st["ref_to_jb_dR_lo"]) <= 0:
        reasons.append("safety_ci_crosses_0")
    if st.get("jb_to_ref_dR_hi") is not None and float(st["jb_to_ref_dR_hi"]) >= 0:
        reasons.append("safety_ci_crosses_0")
    if float(st.get("retain") or 0) < 0.40:
        reasons.append("retain")
    if -float(st.get("benign_dU") or 0) > 0.05:
        reasons.append("benign_drop")
    if (controls.get("U_refusal") or {}).get("bidirectional"):
        reasons.append("refusal_matches_rhc")
    if (controls.get("U_denial") or {}).get("bidirectional"):
        reasons.append("denial_matches_rhc")
    if (controls.get("random") or {}).get("bidirectional"):
        reasons.append("random_matches")
    return {
        "would_pass_if_this_were_a_gate": not reasons,
        "reasons": reasons,
        "note": "development only; confirm set unused; does not reopen Attack Gate",
    }
