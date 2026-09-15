"""P0-C static AUCs, bootstrap CIs, and actuator GO metrics. CPU only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import auc_score, summarize

STATIC_GATE = 0.70
ACTUATOR_DSCR = 0.20
ACTUATOR_RHR = 0.80
EXTEND_DSCR = 0.10
MODE_HARM_DSMODE = -2.0
MODE_HARM_DMDR = 0.10
N_BOOT = 1000
BOOT_SEED = 2026


def _labeled(rows: Sequence[Dict[str, Any]], score_key: str, higher_is_rhc: bool) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
    y, s, keep = [], [], []
    for r in rows:
        if r.get(score_key) is None:
            continue
        if not (r.get("core_rhc") or r.get("core_safe_answer")):
            continue
        y.append(1 if r.get("core_rhc") else 0)
        val = float(r[score_key])
        s.append(val if higher_is_rhc else -val)
        keep.append(r)
    return y, s, keep


def within_query_macro(rows: Sequence[Dict[str, Any]], score_key: str, higher_is_rhc: bool) -> Dict[str, Any]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get(score_key) is None:
            continue
        by[str(r["query_id"])].append(r)
    per = {}
    aucs = []
    for q, v in sorted(by.items()):
        y, s, _ = _labeled(v, score_key, higher_is_rhc)
        a = auc_score(y, s) if y else None
        per[q] = {"n": len(v), "n_rhc": int(sum(y)), "auc": a}
        if a is not None:
            aucs.append(a)
    xs, ys = [], []
    for v in by.values():
        vals = [float(x[score_key]) for x in v if x.get(score_key) is not None]
        if not vals:
            continue
        mu = sum(vals) / len(vals)
        for x in v:
            if x.get(score_key) is None:
                continue
            centered = float(x[score_key]) - mu
            xs.append(centered if higher_is_rhc else -centered)
            ys.append(1 if x.get("core_rhc") else 0)
    return {
        "per_query": per,
        "n_queries_both_classes": len(aucs),
        "mean_within_query_auc": (float(sum(aucs) / len(aucs)) if aucs else None),
        "query_centered_auc": auc_score(ys, xs) if ys else None,
    }


def loqo_auc(rows: Sequence[Dict[str, Any]], score_key: str, higher_is_rhc: bool) -> Dict[str, Any]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r["query_id"])].append(r)
    qids = sorted(by)
    per = {}
    aucs = []
    for hold in qids:
        rest = [r for q, v in by.items() if q != hold for r in v]
        y, s, _ = _labeled(rest, score_key, higher_is_rhc)
        a = auc_score(y, s) if y else None
        per[hold] = {"n": len(rest), "auc": a}
        if a is not None:
            aucs.append(a)
    return {
        "per_held_out_query": per,
        "mean_loqo_auc": (float(sum(aucs) / len(aucs)) if aucs else None),
        "min_loqo_auc": (float(min(aucs)) if aucs else None),
        "max_loqo_auc": (float(max(aucs)) if aucs else None),
    }


def bootstrap_auc_ci(
    rows: Sequence[Dict[str, Any]],
    score_key: str,
    higher_is_rhc: bool,
    unit: str = "query",
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if unit == "cell":
            key = f"{r.get('query_id')}:{r.get('carrier_id')}"
        else:
            key = str(r.get("query_id"))
        groups[key].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(int(seed))
    aucs: List[float] = []
    skipped = 0
    for _ in range(int(n_boot)):
        draw = rng.choice(keys, size=len(keys), replace=True)
        subset = [r for k in draw for r in groups[str(k)]]
        y, s, _ = _labeled(subset, score_key, higher_is_rhc)
        a = auc_score(y, s) if y else None
        if a is None:
            skipped += 1
            continue
        aucs.append(float(a))
    if not aucs:
        return {"unit": unit, "n_boot": int(n_boot), "n_ok": 0, "skipped": skipped}
    arr = np.asarray(aucs, dtype=np.float64)
    return {
        "unit": unit,
        "n_boot": int(n_boot),
        "n_ok": int(arr.size),
        "skipped": int(skipped),
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
    }


def static_score_block(
    rows: Sequence[Dict[str, Any]],
    score_key: str,
    higher_is_rhc: bool,
    name: str,
) -> Dict[str, Any]:
    y, s, keep = _labeled(rows, score_key, higher_is_rhc)
    global_auc = auc_score(y, s) if y else None
    wq = within_query_macro(rows, score_key, higher_is_rhc)
    wq_mean = wq.get("mean_within_query_auc")
    return {
        "name": name,
        "score_key": score_key,
        "higher_is_rhc": bool(higher_is_rhc),
        "n": len(keep),
        "n_rhc": int(sum(y)),
        "n_safe": int(len(y) - sum(y)),
        "global_auc": global_auc,
        "within_query": wq,
        "loqo": loqo_auc(rows, score_key, higher_is_rhc),
        "bootstrap_query": bootstrap_auc_ci(rows, score_key, higher_is_rhc, unit="query"),
        "bootstrap_cell": bootstrap_auc_ci(rows, score_key, higher_is_rhc, unit="cell"),
        "score_rhc": summarize([ss for ss, yy in zip(s, y) if yy == 1]),
        "score_safe": summarize([ss for ss, yy in zip(s, y) if yy == 0]),
        "static_pass": bool((wq_mean or 0.0) >= STATIC_GATE),
        "static_gate": STATIC_GATE,
    }


def snapshot_at(traj: Dict[str, Any], t: int) -> Optional[Dict[str, Any]]:
    for s in traj.get("snapshots") or []:
        if int(s.get("t", -1)) == int(t):
            return s
    return None


def _final_snap(traj: Dict[str, Any], t: int) -> Optional[Dict[str, Any]]:
    return snapshot_at(traj, t) or (traj.get("snapshots") or [None])[-1]


def branch_metrics(
    trajs: Sequence[Dict[str, Any]],
    branch: str,
    t: int,
    safe_ids: Sequence[str],
    rhc_ids: Sequence[str],
) -> Dict[str, Any]:
    by_id = {r["start_id"]: r for r in trajs if r.get("branch") == branch}
    safe_set, rhc_set = set(safe_ids), set(rhc_ids)

    def final_flag(rid: str, key: str) -> Optional[bool]:
        tr = by_id.get(rid)
        if not tr:
            return None
        snap = _final_snap(tr, t)
        if not snap:
            return None
        return bool(snap.get(key))

    def final_mode(rid: str) -> Optional[str]:
        tr = by_id.get(rid)
        if not tr:
            return None
        snap = _final_snap(tr, t)
        if not snap:
            return None
        return snap.get("response_mode")

    def ds_mode(rid: str) -> Optional[float]:
        tr = by_id.get(rid)
        if not tr:
            return None
        s0 = snapshot_at(tr, 0)
        sT = _final_snap(tr, t)
        if not s0 or not sT or s0.get("s_mode") is None or sT.get("s_mode") is None:
            return None
        return float(sT["s_mode"]) - float(s0["s_mode"])

    def d_loss(rid: str, loss_key: str) -> Optional[float]:
        tr = by_id.get(rid)
        if not tr:
            return None
        s0 = snapshot_at(tr, 0)
        sT = _final_snap(tr, t)
        if not s0 or not sT or s0.get(loss_key) is None or sT.get(loss_key) is None:
            return None
        return float(sT[loss_key]) - float(s0[loss_key])

    safe_ok = [sid for sid in safe_ids if sid in by_id]
    rhc_ok = [rid for rid in rhc_ids if rid in by_id]
    all_ok = safe_ok + rhc_ok
    n_safe_rhc = sum(1 for sid in safe_ok if final_flag(sid, "core_rhc"))
    n_rhc_stay = sum(1 for rid in rhc_ok if final_flag(rid, "core_rhc"))
    n_refuse = sum(1 for rid in all_ok if final_mode(rid) == "REFUSE")
    per_q: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n_safe": 0, "n_safe_to_rhc": 0})
    for sid in safe_ok:
        q = by_id[sid]["query_id"]
        per_q[q]["n_safe"] += 1
        if final_flag(sid, "core_rhc"):
            per_q[q]["n_safe_to_rhc"] += 1
    scr = (n_safe_rhc / len(safe_ok)) if safe_ok else None
    rhr = (n_rhc_stay / len(rhc_ok)) if rhc_ok else None
    mdr = (n_refuse / len(all_ok)) if all_ok else None
    dsm = [ds_mode(rid) for rid in all_ok]
    dsm = [x for x in dsm if x is not None]
    dsm_safe = [ds_mode(sid) for sid in safe_ok]
    dsm_safe = [x for x in dsm_safe if x is not None]
    loss_key = {"B2": "L_c0", "B3": "L_c3", "B4": "L_c2"}.get(branch)
    dL = [d_loss(rid, loss_key) for rid in all_ok] if loss_key else []
    dL = [x for x in dL if x is not None]
    return {
        "branch": branch,
        "t": int(t),
        "n_safe": len(safe_ok),
        "n_rhc": len(rhc_ok),
        "SCR": scr,
        "n_safe_to_rhc": n_safe_rhc,
        "RHR": rhr,
        "n_rhc_retained": n_rhc_stay,
        "MDR": mdr,
        "n_refuse": n_refuse,
        "delta_s_mode": summarize(dsm) if dsm else {"n": 0},
        "delta_s_mode_safe": summarize(dsm_safe) if dsm_safe else {"n": 0},
        "delta_loss": summarize(dL) if dL else {"n": 0},
        "loss_key": loss_key,
        "per_query_safe_to_rhc": {q: dict(v) for q, v in sorted(per_q.items())},
    }


def attach_deltas(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metrics)
    if metrics.get("SCR") is not None and baseline.get("SCR") is not None:
        out["delta_SCR"] = float(metrics["SCR"]) - float(baseline["SCR"])
    else:
        out["delta_SCR"] = None
    if metrics.get("MDR") is not None and baseline.get("MDR") is not None:
        out["delta_MDR"] = float(metrics["MDR"]) - float(baseline["MDR"])
    else:
        out["delta_MDR"] = None
    bsm = (baseline.get("delta_s_mode_safe") or {}).get("mean")
    msm = (metrics.get("delta_s_mode_safe") or {}).get("mean")
    if bsm is not None and msm is not None:
        out["delta_s_mode_safe_vs_b0"] = float(msm) - float(bsm)
    else:
        out["delta_s_mode_safe_vs_b0"] = None
    return out


def queries_driving_scr(metrics: Dict[str, Any]) -> Dict[str, Any]:
    per = metrics.get("per_query_safe_to_rhc") or {}
    contrib = []
    total = int(metrics.get("n_safe_to_rhc") or 0)
    for q, v in per.items():
        n = int(v.get("n_safe_to_rhc") or 0)
        if n:
            contrib.append((q, n, n / max(total, 1)))
    contrib.sort(key=lambda x: -x[1])
    top2 = sum(x[1] for x in contrib[:2])
    return {
        "n_queries_with_conversion": len(contrib),
        "top_queries": [{"query_id": q, "n": n, "share": sh} for q, n, sh in contrib],
        "top2_share_of_conversions": (top2 / max(total, 1) if total else None),
        "not_one_or_two_queries": bool(len(contrib) >= 3 or total == 0),
    }


def actuator_pass(metrics: Dict[str, Any], t20: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dscr = metrics.get("delta_SCR")
    rhr = metrics.get("RHR")
    dL = (metrics.get("delta_loss") or {}).get("mean")
    dscr_ok = dscr is not None and dscr >= ACTUATOR_DSCR - 1e-12
    rhr_ok = rhr is not None and rhr >= ACTUATOR_RHR - 1e-12
    align_ok = True
    if dscr is not None and dscr > 0 and dL is not None:
        align_ok = dL < 0
    elif dL is None:
        align_ok = metrics.get("loss_key") is None
    spread = queries_driving_scr(metrics)
    n_conv = int(metrics.get("n_safe_to_rhc") or 0)
    if n_conv >= 3:
        spread_ok = spread["n_queries_with_conversion"] >= 3
    else:
        spread_ok = True
    t20_ok = True
    if t20 is not None:
        d20 = t20.get("delta_SCR")
        t20_ok = d20 is not None and d20 >= (dscr or 0) - 0.05
    ok = bool(dscr_ok and rhr_ok and align_ok and spread_ok and t20_ok)
    return {
        "pass": ok,
        "dscr_ok": bool(dscr_ok),
        "rhr_ok": bool(rhr_ok),
        "align_ok": bool(align_ok),
        "spread_ok": bool(spread_ok),
        "t20_ok": bool(t20_ok),
        "spread": spread,
        "thresholds": {"delta_SCR": ACTUATOR_DSCR, "RHR": ACTUATOR_RHR},
    }


def decision_cell(static_pass: bool, act_pass: bool) -> str:
    if static_pass and act_pass:
        return "PASS_PASS"
    if (not static_pass) and act_pass:
        return "FAIL_PASS"
    if static_pass and (not act_pass):
        return "PASS_FAIL"
    return "FAIL_FAIL"


def mode_harm(metrics: Dict[str, Any]) -> bool:
    dsm = metrics.get("delta_s_mode_safe_vs_b0")
    dmdr = metrics.get("delta_MDR")
    harm_s = dsm is not None and dsm <= MODE_HARM_DSMODE
    harm_m = dmdr is not None and dmdr >= MODE_HARM_DMDR
    return bool(harm_s or harm_m)


def route_decision(
    c2_static: Dict[str, Any],
    c2_act: Dict[str, Any],
    c2_gate: Dict[str, Any],
    c3_static: Dict[str, Any],
    c3_act: Dict[str, Any],
    c3_gate: Dict[str, Any],
) -> Dict[str, Any]:
    c2_ok = bool(c2_gate.get("pass"))
    c3_ok = bool(c3_gate.get("pass"))
    harm = mode_harm(c2_act)
    if c2_ok and harm:
        route, unlock, note = "A", True, "contrastive actuator PASS and harms mode; recommend unlock P1 (do not start P1 GPU here)"
    elif c2_ok and not harm:
        route, unlock, note = "B", False, "contrastive actuator PASS without mode harm; MCCA less necessary; Contrastive Content Attack story"
    elif c3_ok:
        route, unlock, note = "C", True, "write_loss actuator PASS with low static AUC; P1 only as actuator, not as a content-state sensor"
    else:
        route, unlock, note = "D", False, "nothing moves content; pause MCCA; do not start P1"
    return {
        "route": route,
        "p0c_unlock_p1": bool(unlock),
        "c2_harms_mode": bool(harm),
        "c2_static_pass": bool(c2_static.get("static_pass")),
        "c3_static_pass": bool(c3_static.get("static_pass")),
        "note": note,
    }
