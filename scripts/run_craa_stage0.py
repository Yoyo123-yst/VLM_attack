#!/usr/bin/env python3
"""CRAA Stage 0 — pixel-controllability pre-screen (cheap, minutes).

Does NOT run P1. Does NOT refit U. Does NOT run PGD trajectories. It measures,
per frozen CR-0 attacked image x01, the first-order pixel controllability of the
frozen L24 refusal direction U_refusal and of random-direction null controls:

    R_inf(u) = ||J^T u||_1 / ||u||_2   (J = d h_24 / d x, via one JVP)

Gate (frozen): R_inf(U_refusal) significantly BELOW the random-direction
distribution. If not -> NO-GO, stop before R-1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from craa.controllability import (  # noqa: E402
    controllability,
    random_direction_set,
)
from mcca.splits import query_split  # noqa: E402
from n1.preflight import _reconstruct_x01, all_carrier_index  # noqa: E402
from otw.attack import pack  # noqa: E402
from otw.score import load_u_refusal_p0s  # noqa: E402
from p0.catalog import all_pairs  # noqa: E402
from p0.datautil import load_json, open_image, save_json  # noqa: E402

OUT = ROOT / "outputs" / "craa"
FROZEN = load_json(OUT / "CRAA_FROZEN.json")
CR0_CAND = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"

STAGE0 = FROZEN["stage0"]
SMOKE_N = int(STAGE0["smoke_n"])
FULL_N = int(STAGE0["full_n"])
N_RANDOM = int(STAGE0["n_random_per_sample"])
BASE_SEED = 2026


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_status(text: str) -> None:
    (OUT / "STATUS.txt").write_text(text + "\n", encoding="utf-8")


def load_records() -> List[Dict[str, Any]]:
    return load_json(CR0_CAND)["records"]


def pick_records(records: List[Dict[str, Any]], smoke: bool) -> List[Dict[str, Any]]:
    """Deterministic sample. Smoke: 2 query x 2 carrier x 2 restart (fixed ids)."""
    if smoke:
        # 2 queries (h49, h56) x 2 carriers x 2 restarts, in deterministic order.
        recs = [r for r in records if r["query_id"] in {"h49", "h56"}]
        recs = sorted(recs, key=lambda r: (r["query_id"], r["carrier_id"], int(r["restart"])))
        seen: Dict[str, int] = {}
        out: List[Dict[str, Any]] = []
        for r in recs:
            key = (r["query_id"], r["carrier_id"])
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= 2:
                out.append(r)
            if len(out) >= SMOKE_N:
                break
        return out
    return list(records)[:FULL_N]


def build_indexes():
    return all_carrier_index(), {p["id"]: p for p in all_pairs()}


def run(args) -> Dict[str, Any]:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from p0_qwen.config import load_cfg
    from run_p0_qwen import load_model, seed_all

    smoke = args.mode == "smoke"
    records = pick_records(load_records(), smoke)
    print({"stage0": "smoke" if smoke else "full", "n_samples": len(records)}, flush=True)

    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="full")
    seed_all(BASE_SEED)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    wrapper.model.eval()

    u_refusal = load_u_refusal_p0s()
    dim = int(u_refusal.shape[0])
    carriers, catalog = build_indexes()

    out_path = OUT / ("stage0_smoke.json" if smoke else "stage0_controllability.json")
    existing: List[Dict[str, Any]] = []
    have = set()
    if out_path.exists() and not args.force:
        blob = load_json(out_path)
        existing = blob.get("records") or []
        have = {r["record_id"] for r in existing}

    todo = [r for r in records if r["record_id"] not in have]
    records_out = list(existing)

    for i, rec in enumerate(todo, start=1):
        rid = rec["record_id"]
        qid = rec["query_id"]
        cid = rec["carrier_id"]
        img = open_image(carriers[cid]["path"])
        x01 = _reconstruct_x01(carriers[cid]["path"], rec.get("delta_path"))
        if x01 is None:
            raise RuntimeError(f"missing delta for {rid}")
        x01 = x01.to(device=wrapper.device, dtype=torch.float32)
        q = catalog[qid]["query"]
        _, ids, attn, grid = pack(wrapper, img, q)

        # Target direction (and its sign-negated twin for the free R_inf symmetry check).
        ctrl_u = controllability(wrapper, x01, ids, attn, grid, u_refusal)
        ctrl_neg = controllability(wrapper, x01, ids, attn, grid, -u_refusal)

        # Random-direction null controls (gaussian + orthogonal).
        dirs = random_direction_set(u_refusal, dim, N_RANDOM, BASE_SEED)
        rand_ctrls: List[Dict[str, Any]] = []
        for d in dirs:
            try:
                c = controllability(wrapper, x01, ids, attn, grid, d["u"])
            except RuntimeError as e:
                c = {"error": str(e)}
            rand_ctrls.append(
                {
                    "kind": d["kind"],
                    "index": d["index"],
                    "seed": d["seed"],
                    **c,
                }
            )

        row = {
            "record_id": rid,
            "query_id": qid,
            "carrier_id": cid,
            "category": rec.get("category"),
            "split": query_split(qid, "cr0"),
            "restart": rec.get("restart"),
            "response_mode": rec.get("response_mode"),
            "u_refusal": ctrl_u,
            "neg_u_refusal": ctrl_neg,
            "random": rand_ctrls,
        }
        records_out.append(row)
        save_json(
            out_path,
            {
                "schema": "craa_stage0_v1",
                "mode": "smoke" if smoke else "full",
                "n": len(records_out),
                "records": _jsonable(records_out),
            },
        )
        write_status(
            f"CRAA STAGE0 {'smoke' if smoke else 'full'} {len(records_out)}/{len(records)} "
            f"last={rid} R_inf={ctrl_u['R_inf']:.4f} {now_iso()}"
        )
        print(
            {
                "id": rid,
                "progress": f"{len(records_out)}/{len(records)}",
                "R_inf_u": ctrl_u["R_inf"],
                "R_inf_neg": ctrl_neg["R_inf"],
                "R_2_u": ctrl_u["R_2"],
            },
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"mode": "smoke" if smoke else "full", "n": len(records_out), "records": records_out}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    u_rinf = np.array([r["u_refusal"]["R_inf"] for r in records], dtype=np.float64)
    u_r2 = np.array([r["u_refusal"]["R_2"] for r in records], dtype=np.float64)
    neg_rinf = np.array([r["neg_u_refusal"]["R_inf"] for r in records], dtype=np.float64)

    rand_rinf: List[float] = []
    rand_by_kind: Dict[str, List[float]] = {"gaussian_unit": [], "random_orthogonal": []}
    for r in records:
        for c in r["random"]:
            if "R_inf" in c:
                rand_rinf.append(c["R_inf"])
                rand_by_kind[c["kind"]].append(c["R_inf"])
    rand_rinf = np.array(rand_rinf, dtype=np.float64)

    sym_err = float(np.max(np.abs(u_rinf - neg_rinf))) if len(u_rinf) else float("nan")

    def _stats(a: np.ndarray) -> Dict[str, float]:
        return {
            "n": int(a.size),
            "mean": float(a.mean()) if a.size else float("nan"),
            "std": float(a.std()) if a.size else float("nan"),
            "median": float(np.median(a)) if a.size else float("nan"),
        }

    # One-sided fraction: how many random directions have R_inf <= U's median.
    u_med = float(np.median(u_rinf)) if len(u_rinf) else float("nan")
    below_frac = (
        float((rand_rinf <= u_med).mean()) if rand_rinf.size else float("nan")
    )

    # Gate: U_refusal mean R_inf significantly below random mean (rank test via fraction).
    rand_mean = float(rand_rinf.mean()) if rand_rinf.size else float("nan")
    gate = {
        "u_median_R_inf": u_med,
        "random_mean_R_inf": rand_mean,
        "fraction_random_below_u_median": below_frac,
        "pass_candidate": bool(rand_rinf.size and len(u_rinf) and u_med < rand_mean),
    }

    return {
        "schema": "craa_stage0_summary_v1",
        "generated_at": now_iso(),
        "n_samples": len(records),
        "u_refusal_R_inf": _stats(u_rinf),
        "u_refusal_R_2": _stats(u_r2),
        "neg_u_refusal_R_inf": _stats(neg_rinf),
        "R_inf_sign_symmetry_max_abs_err": sym_err,
        "random_R_inf": _stats(rand_rinf),
        "random_R_inf_by_kind": {k: _stats(np.array(v)) for k, v in rand_by_kind.items()},
        "gate": gate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    result = run(args)
    summary = summarize(result["records"])
    summary_path = (
        OUT / "stage0_smoke_summary.json"
        if args.mode == "smoke"
        else OUT / "stage0_summary.json"
    )
    save_json(summary_path, _jsonable(summary))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
