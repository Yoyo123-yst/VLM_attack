#!/usr/bin/env python3
"""Rule-reclean P0-Qwen traces JB labels. CPU only unless --resample.

Does not touch calibrate.json. Hidden vectors stay in place; only jb/fail/related_safe pointers move.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p0.catalog import probe_pairs  # noqa: E402
from p0.datautil import load_json, save_json  # noqa: E402
from p0.judge import harmful_compliance, is_success_label  # noqa: E402
from p0.subspace import pca_basis  # noqa: E402
from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import extract_subspaces, maybe_load, stage_path  # noqa: E402

DEFAULT_TRACES = ROOT / "outputs/p0_qwen/full/native/traces.json"
TRIM = 400


def _trim_judge(j: dict, n: int = TRIM) -> dict:
    out = dict(j)
    if "text" in out and out["text"] is not None:
        out["text"] = out["text"][:n]
    return out


def _catalog_by_id() -> dict:
    return {p["id"]: p for p in probe_pairs()}


def principal_angles_deg(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    q1, _ = np.linalg.qr(U)
    q2, _ = np.linalg.qr(V)
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.degrees(np.arccos(s))


def restore_old_jb(traces: dict) -> int:
    """Undo pointer moves from a prior reclean so the 77 texts can be re-judged."""
    n = 0
    for rec in traces.get("records") or []:
        rc = rec.get("reclean") or {}
        if not rc:
            continue
        if rec.get("jb"):
            continue
        src = None
        if rc.get("new_label") == "RELATED_SAFE":
            src = rec.get("related_safe")
        elif rc.get("new_label") == "FAIL":
            src = rec.get("fail")
        elif rec.get("related_safe"):
            src = rec.get("related_safe")
        elif rec.get("fail"):
            src = rec.get("fail")
        if not src:
            continue
        rec["jb"] = {"judge": dict(src.get("judge") or {}), "hidden": src.get("hidden")}
        if rec.get("related_safe") is src:
            rec["related_safe"] = None
        if rec.get("fail") is src:
            rec["fail"] = None
        n += 1
    return n


def reclean_records(traces: dict) -> tuple[dict, list[dict], dict]:
    cat = _catalog_by_id()
    items = []
    counts = {
        "old_jb": 0,
        "RHC": 0,
        "RELATED_SAFE": 0,
        "FAIL": 0,
        "REF": 0,
        "discover_rhc": 0,
        "holdout_rhc": 0,
        "other_rhc": 0,
    }
    for rec in traces.get("records") or []:
        jb = rec.get("jb")
        if not jb:
            continue
        counts["old_jb"] += 1
        qid = rec.get("query_id")
        pair = cat.get(qid) or {}
        text = ((jb.get("judge") or {}).get("text")) or ""
        query = rec.get("query") or pair.get("query") or ""
        keys = pair.get("topic_keywords") or []
        old_label = (jb.get("judge") or {}).get("label")
        new_j = harmful_compliance(text, keys, query)
        new_label = new_j["label"]
        counts[new_label] = counts.get(new_label, 0) + 1
        split = rec.get("split")
        evidence = (new_j.get("text") or "")[:180]
        row = {
            "query_id": qid,
            "split": split,
            "old_label": old_label,
            "new_label": new_label,
            "R": new_j.get("R"),
            "relevance": new_j.get("relevance"),
            "theme_denial": new_j.get("theme_denial"),
            "operational": new_j.get("operational"),
            "evidence": evidence,
        }
        items.append(row)
        hidden = jb.get("hidden")
        rec["reclean"] = {"old_label": old_label, "new_label": new_label}
        if is_success_label(new_label):
            rec["jb"]["judge"] = _trim_judge(new_j)
            if split == "discover":
                counts["discover_rhc"] += 1
            elif split == "holdout":
                counts["holdout_rhc"] += 1
            else:
                counts["other_rhc"] += 1
        else:
            payload = {"judge": _trim_judge(new_j), "hidden": hidden}
            rec["jb"] = None
            if new_label == "RELATED_SAFE":
                rec["related_safe"] = payload
            elif new_label == "FAIL":
                rec["fail"] = payload
            else:
                rec["reclean_now_ref"] = True
    summary = traces.get("summary") or {}
    summary["n_jb"] = sum(1 for r in traces["records"] if r.get("jb") is not None)
    summary["n_rhc"] = summary["n_jb"]
    summary["n_fail"] = sum(1 for r in traces["records"] if r.get("fail") is not None)
    summary["n_related_safe"] = sum(1 for r in traces["records"] if r.get("related_safe") is not None)
    summary["recleaned"] = True
    traces["summary"] = summary
    counts["n_rhc"] = summary["n_rhc"]
    counts["n_rhc_discover_holdout"] = counts["discover_rhc"] + counts["holdout_rhc"]
    return traces, items, counts


def freeze_causal(out: Path, old_causal: dict | None, stable: bool, note: str) -> dict:
    best = {"layer": 24, "rank": 32, "pass_lowrank": True, "frozen": True}
    if old_causal and (old_causal.get("best") or {}).get("layer"):
        prev = dict(old_causal["best"])
        prev["layer"] = 24
        prev["rank"] = 32
        prev["frozen"] = True
        best = prev
        if not stable:
            best["pass_lowrank"] = bool(prev.get("pass_lowrank"))
            best["selection_was_pre_clean_holdout"] = True
    rec = dict(old_causal or {})
    rec["best"] = best
    rec["selection_note"] = note
    rec["frozen_layer"] = 24
    rec["frozen_rank"] = 32
    rec["u_from_recleaned_discover_rhc"] = True
    save_json(out / "causal.json", rec)
    return rec


def refresh_u(out: Path, traces: dict, counts: dict, cfg) -> dict:
    old_sub = maybe_load(stage_path(out, "subspace")) or {}
    old_causal = maybe_load(stage_path(out, "causal"))
    patch_full = maybe_load(stage_path(out, "patch_full")) or {}
    old_u = None
    try:
        old_u = np.asarray(old_sub["layers"]["24"]["ranks"]["32"]["U"], dtype=np.float64)
    except Exception:
        old_u = None

    discover = [r for r in traces["records"] if r.get("jb") is not None and r.get("split") == "discover"]
    if len(discover) < 2:
        discover = [r for r in traces["records"] if r.get("jb") is not None]
    X = []
    for p in discover:
        src = np.asarray(p["jb"]["hidden"]["L24:last_user"], dtype=np.float32).reshape(-1)
        ref = np.asarray(p["clean_hidden"]["L24:last_user"], dtype=np.float32).reshape(-1)
        X.append(src - ref)
    X = np.stack(X, axis=0)
    new_u = pca_basis(X, min(32, X.shape[0] - 1, X.shape[1]))
    angles = principal_angles_deg(old_u, new_u).tolist() if old_u is not None else []
    mean_ang = float(np.mean(angles[: min(8, len(angles))])) if angles else None
    old_n = int(old_sub.get("n_pairs") or counts["old_jb"])
    overlap = counts["discover_rhc"] / max(old_n, 1)
    dropped_frac = 1.0 - (counts["n_rhc"] / max(counts["old_jb"], 1))
    stable = bool(
        old_u is not None
        and dropped_frac <= 0.25
        and (mean_ang is not None and mean_ang <= 30.0)
    )
    gate = {
        "n_old_jb": counts["old_jb"],
        "n_rhc": counts["n_rhc"],
        "discover_rhc": counts["discover_rhc"],
        "holdout_rhc": counts["holdout_rhc"],
        "dropped_frac": dropped_frac,
        "old_discover_n_pairs": old_n,
        "discover_overlap_vs_old_n_pairs": overlap,
        "principal_angles_deg_L24_r32": angles,
        "mean_angle_first8": mean_ang,
        "stable": stable,
        "pin_layer": 24,
        "pin_rank": 32,
    }
    sub_live = stage_path(out, "subspace")
    causal_live = stage_path(out, "causal")
    if sub_live.exists():
        shutil.move(str(sub_live), str(out / "subspace_preclean.json"))
    if not stable and causal_live.exists():
        shutil.move(str(causal_live), str(out / "causal_preclean.json"))
        old_causal_for_freeze = load_json(out / "causal_preclean.json")
    else:
        old_causal_for_freeze = old_causal
    fine = list(patch_full.get("candidate_layers") or [])
    if 24 not in fine:
        fine.append(24)
    if not fine:
        fine = [20, 24, 27]
    new_sub = extract_subspaces(cfg, out, traces, fine, force=True)
    if stable:
        note = (
            "U relearned on recleaned discover RHC; principal angles small. "
            "Layer/rank pinned to L24 r=32 from pre-clean validation. Patching/causal GPU not rerun."
        )
    else:
        note = (
            "Reclean changed the RHC set substantially. subspace.json and causal.json were replaced. "
            "U relearned on recleaned discover RHC. Layer/rank still pinned to L24 r=32 "
            "(selection was on pre-clean holdout; attack test is a new catalog split)."
        )
    freeze_causal(out, old_causal_for_freeze, stable, note)
    gate["selection_note"] = note
    gate["subspace_n_pairs"] = new_sub.get("n_pairs")
    save_json(out / "u_stability.json", gate)
    return gate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=str, default=str(DEFAULT_TRACES))
    p.add_argument("--min-rhc", type=int, default=30)
    p.add_argument("--resample", action="store_true", help="GPU resample still-REF queries if RHC<min")
    args = p.parse_args()
    traces_path = Path(args.traces)
    out = traces_path.parent
    cfg = load_cfg(ROOT / "configs/p0_qwen.yaml", scale="full")
    print(f"loading {traces_path}", flush=True)
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    restored = restore_old_jb(traces)
    if restored:
        print(f"restored {restored} previously moved JB pointers for re-judge", flush=True)
    traces, items, counts = reclean_records(traces)
    reclean_blob = {
        "counts": counts,
        "items": items,
        "min_rhc": args.min_rhc,
        "calibrate_untouched": True,
        "traces_path": str(traces_path),
    }
    save_json(out / "jb_reclean.json", reclean_blob)
    tmp = traces_path.with_suffix(".json.tmp")
    print("writing recleaned traces (pointers only; hidden not recopied as new arrays)", flush=True)
    tmp.write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")
    tmp.replace(traces_path)
    meta = {
        "n_old_jb": counts["old_jb"],
        "n_rhc": counts["n_rhc"],
        "discover_rhc": counts["discover_rhc"],
        "holdout_rhc": counts["holdout_rhc"],
        "n_related_safe": counts.get("RELATED_SAFE", 0),
        "n_fail_from_old_jb": counts.get("FAIL", 0),
        "n_ref_from_old_jb": counts.get("REF", 0),
        "traces_path": str(traces_path),
        "note": "traces.json jb pointers updated in place; hidden tensors not recopied. calibrate.json not modified.",
        "gate": "pass" if counts["n_rhc_discover_holdout"] >= args.min_rhc else "stop_resample",
    }
    save_json(out / "traces_rhc.json", meta)
    print(json.dumps({k: counts[k] for k in counts if k != "items"}, indent=2), flush=True)
    n_gate = counts["n_rhc_discover_holdout"]
    if n_gate < args.min_rhc:
        print(
            f"RHC discover+holdout={n_gate} < {args.min_rhc}.",
            flush=True,
        )
        if not args.resample:
            print(
                "STOP: do not PCA or attack. Re-run with --resample on GPU after confirming free VRAM.",
                flush=True,
            )
            return
        from p0.catalog import probe_pairs
        from p0.datautil import load_carriers
        from p0_qwen.model import QwenP0
        from p0_qwen.traces import resample_missing_rhc

        print("GPU resample of still-REF records...", flush=True)
        wrapper = QwenP0(cfg, setting="native")
        train, _ = load_carriers(cfg)
        traces = resample_missing_rhc(
            wrapper,
            cfg,
            traces,
            probe_pairs(),
            train,
            cfg["layers"]["candidates"],
            delta_dir=out / "deltas",
        )
        traces, items, counts = reclean_records(traces)
        reclean_blob["counts"] = counts
        reclean_blob["items"] = items
        reclean_blob["resampled"] = True
        save_json(out / "jb_reclean.json", reclean_blob)
        tmp = traces_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")
        tmp.replace(traces_path)
        n_gate = counts["n_rhc_discover_holdout"]
        meta["n_rhc"] = counts["n_rhc"]
        meta["discover_rhc"] = counts["discover_rhc"]
        meta["holdout_rhc"] = counts["holdout_rhc"]
        meta["gate"] = "pass" if n_gate >= args.min_rhc else "stop_resample"
        save_json(out / "traces_rhc.json", meta)
        if n_gate < args.min_rhc:
            print(f"STOP after resample: RHC={n_gate} < {args.min_rhc}", flush=True)
            return
    gate = refresh_u(out, traces, counts, cfg)
    print(json.dumps({k: gate[k] for k in gate if k != "principal_angles_deg_L24_r32"}, indent=2), flush=True)
    print("PCA gate done; L24 r=32 pinned. Attack must use attack_test_pairs, not holdout.", flush=True)


if __name__ == "__main__":
    main()
