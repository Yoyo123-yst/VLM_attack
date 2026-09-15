#!/usr/bin/env python3
"""CRAA Stage 0 robustness check: R_inf on CLEAN images (no CR-0 delta).

Answers one confound: the smoke measurement was on CR-0 attacked images x01.
If U_refusal's high R_inf is only an artifact of the CR-0 attack position, the
clean-image ratio will collapse. If it persists, U_refusal is genuinely more
pixel-controllable than random directions regardless of position.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from craa.controllability import controllability, random_direction_set  # noqa: E402
from n1.preflight import _reconstruct_x01, all_carrier_index  # noqa: E402
from otw.attack import pack  # noqa: E402
from otw.score import load_u_refusal_p0s  # noqa: E402
from p0.catalog import all_pairs  # noqa: E402
from p0.datautil import load_json, open_image, save_json  # noqa: E402

OUT = ROOT / "outputs" / "craa"
BASE_SEED = 2026
N_RANDOM = 20
# Same (query, carrier) cells as the smoke sample, without restart dimension.
CELLS = [("h49", "c07"), ("h49", "c08"), ("h56", "c07"), ("h56", "c08")]


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


def main() -> None:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from p0_qwen.config import load_cfg
    from run_p0_qwen import load_model, seed_all

    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="full")
    seed_all(BASE_SEED)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    wrapper.model.eval()

    u_refusal = load_u_refusal_p0s()
    dim = int(u_refusal.shape[0])
    carriers = all_carrier_index()
    catalog = {p["id"]: p for p in all_pairs()}

    records: List[Dict[str, Any]] = []
    for qid, cid in CELLS:
        img = open_image(carriers[cid]["path"])
        x0 = _reconstruct_x01(carriers[cid]["path"], None)  # clean image
        x0 = x0.to(device=wrapper.device, dtype=torch.float32)
        q = catalog[qid]["query"]
        _, ids, attn, grid = pack(wrapper, img, q)

        ctrl_u = controllability(wrapper, x0, ids, attn, grid, u_refusal)
        dirs = random_direction_set(u_refusal, dim, N_RANDOM, BASE_SEED)
        rand = []
        for d in dirs:
            c = controllability(wrapper, x0, ids, attn, grid, d["u"])
            rand.append({"kind": d["kind"], "index": d["index"], "R_inf": c["R_inf"], "R_2": c["R_2"]})

        records.append(
            {
                "query_id": qid,
                "carrier_id": cid,
                "u_refusal": ctrl_u,
                "random": rand,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rand_rinf = [c["R_inf"] for r in records for c in r["random"]]
    u_rinf = [r["u_refusal"]["R_inf"] for r in records]
    summary = {
        "schema": "craa_stage0_clean_check_v1",
        "n_cells": len(records),
        "u_refusal_R_inf": u_rinf,
        "random_R_inf_mean": float(np.mean(rand_rinf)),
        "random_R_inf_median": float(np.median(rand_rinf)),
        "ratios": [r["u_refusal"]["R_inf"] / float(np.median([c["R_inf"] for c in r["random"]])) for r in records],
        "records": _jsonable(records),
    }
    save_json(OUT / "stage0_clean_check.json", _jsonable(summary))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
