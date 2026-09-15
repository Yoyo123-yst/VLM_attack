"""Bidirectional residual patching on Qwen, plus random / shuffle / mismatch controls."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from p0.datautil import open_image
from p0.judge import benign_utility, harmful_compliance, triple_axes
from p0.metrics import bootstrap_ci
from p0.subspace import to_torch_u

from .model import QwenP0


def _site_key(layer: int) -> str:
    return f"L{layer}:last_user"


def same_norm_random(src: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=src.reshape(-1).shape).astype(np.float32)
    n_src = float(np.linalg.norm(src))
    n_v = float(np.linalg.norm(v)) + 1e-8
    return (v / n_v) * n_src


def shuffle_coords(src: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.asarray(src, dtype=np.float32).reshape(-1).copy()
    rng.shuffle(v)
    return v


def _x01_from_delta(wrapper: QwenP0, img, delta_path: Optional[str]):
    if not delta_path:
        return None
    delta = torch.load(delta_path, map_location=wrapper.device, weights_only=True)
    x0 = wrapper.image_to_x01(img)
    return torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)


def run_patching(
    wrapper: QwenP0,
    cfg: Dict[str, Any],
    traces: Dict[str, Any],
    query_index: Dict[str, Dict[str, Any]],
    carrier_index: Dict[str, Dict[str, Any]],
    layers: List[int],
    U_by_layer: Optional[Dict[int, np.ndarray]] = None,
    mode: str = "full",
    tag: str = "full",
    control: Optional[str] = None,
    do_benign: bool = True,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    rows = []
    pairs = [r for r in traces["records"] if r.get("jb") is not None]
    if split:
        pairs = [r for r in pairs if r.get("split") == split]
    max_new = int(cfg["model"].get("patch_max_new_tokens", 64))
    for i, rec in enumerate(tqdm(pairs, desc=f"patch:{tag}")):
        h = query_index[rec["query_id"]]
        img = open_image(carrier_index[rec["carrier_id"]]["path"])
        jb_x01 = _x01_from_delta(wrapper, img, rec.get("delta_path"))
        clean_R = rec["clean"]["R"]
        clean_refusal = rec["clean"]["refusal"]
        jb_R = rec["jb"]["judge"]["R"]
        jb_refusal = rec["jb"]["judge"]["refusal"]
        other = pairs[(i + 1) % len(pairs)] if pairs else rec
        for layer in layers:
            key = _site_key(layer)
            href = np.asarray(rec["clean_hidden"][key], dtype=np.float32)
            hjb = np.asarray(rec["jb"]["hidden"][key], dtype=np.float32)
            src_r2j, src_j2r = hjb, href
            if control == "random":
                src_r2j = same_norm_random(hjb, seed=2026 + layer + i)
                src_j2r = same_norm_random(href, seed=3026 + layer + i)
            elif control == "shuffle":
                src_r2j = shuffle_coords(hjb, seed=4026 + layer + i)
                src_j2r = shuffle_coords(href, seed=5026 + layer + i)
            elif control == "mismatch":
                src_r2j = np.asarray(other["jb"]["hidden"][key], dtype=np.float32)
                src_j2r = np.asarray(other["clean_hidden"][key], dtype=np.float32)
            U = to_torch_u(U_by_layer[layer]) if mode == "sub" and U_by_layer is not None else None

            t_r2j = wrapper.generate_with_patch(
                img, h["query"], layer, torch.tensor(src_r2j), mode=mode, U=U, max_new_tokens=max_new
            )
            j_r2j = harmful_compliance(t_r2j, h["topic_keywords"], h["query"])

            j_j2r = None
            if jb_x01 is not None:
                t_j2r = wrapper.generate_with_patch(
                    img,
                    h["query"],
                    layer,
                    torch.tensor(src_j2r),
                    mode=mode,
                    U=U,
                    max_new_tokens=max_new,
                    x01=jb_x01,
                )
                j_j2r = harmful_compliance(t_j2r, h["topic_keywords"], h["query"])

            rows.append(
                {
                    "query_id": rec["query_id"],
                    "carrier_id": rec["carrier_id"],
                    "split": rec.get("split"),
                    "layer": layer,
                    "mode": mode,
                    "tag": tag,
                    "control": control,
                    "clean_R": clean_R,
                    "clean_refusal": clean_refusal,
                    "jb_R": jb_R,
                    "jb_refusal": jb_refusal,
                    "ref_to_jb": j_r2j,
                    "jb_to_ref": j_j2r,
                }
            )
        torch.cuda.empty_cache()

    benign_rows = []
    if do_benign and control is None:
        for brec in traces.get("benign") or []:
            hid = query_index.get(brec["harmful_id"])
            if hid is None:
                continue
            img = open_image(carrier_index[brec["carrier_id"]]["path"])
            src = next((r for r in pairs if r["query_id"] == brec["harmful_id"]), None)
            if src is None:
                continue
            bkeys = hid.get("benign_keywords") or []
            for layer in layers:
                key = _site_key(layer)
                hjb = torch.tensor(src["jb"]["hidden"][key])
                U = to_torch_u(U_by_layer[layer]) if mode == "sub" and U_by_layer is not None else None
                text = wrapper.generate_with_patch(
                    img, brec["query"], layer, hjb, mode=mode, U=U, max_new_tokens=max_new
                )
                benign_rows.append(
                    {
                        "benign_id": brec["query_id"],
                        "layer": layer,
                        "tag": tag,
                        "base": brec["judge"],
                        "patched": benign_utility(text, bkeys, brec["query"]),
                    }
                )

    return {"rows": rows, "benign": benign_rows}


def summarize_patch(blob: Dict[str, Any], split: Optional[str] = None) -> Dict[str, Any]:
    rows = blob["rows"]
    if split:
        rows = [r for r in rows if r.get("split") == split]
    by_layer: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        L = r["layer"]
        slot = by_layer.setdefault(L, {"suff_dR": [], "nec_dR": [], "suff_dRef": [], "nec_dRef": []})
        slot["suff_dR"].append(r["ref_to_jb"]["R"] - r["clean_R"])
        slot["suff_dRef"].append(r["ref_to_jb"]["refusal"] - r["clean_refusal"])
        if r.get("jb_to_ref") is not None:
            slot["nec_dR"].append(r["jb_to_ref"]["R"] - r["jb_R"])
            slot["nec_dRef"].append(r["jb_to_ref"]["refusal"] - r["jb_refusal"])
    out = {}
    for L, v in by_layer.items():
        sm, slo, shi = bootstrap_ci(v["suff_dR"]) if v["suff_dR"] else (float("nan"),) * 3
        nm, nlo, nhi = bootstrap_ci(v["nec_dR"]) if v["nec_dR"] else (float("nan"),) * 3
        out[str(L)] = {
            "n": len(v["suff_dR"]),
            "ref_to_jb_dR": sm,
            "ref_to_jb_dR_lo": slo,
            "ref_to_jb_dR_hi": shi,
            "ref_to_jb_d_refusal": float(np.mean(v["suff_dRef"])) if v["suff_dRef"] else float("nan"),
            "jb_to_ref_dR": nm,
            "jb_to_ref_dR_lo": nlo,
            "jb_to_ref_dR_hi": nhi,
            "jb_to_ref_d_refusal": float(np.mean(v["nec_dRef"])) if v["nec_dRef"] else float("nan"),
            "bidirectional": bool(
                sm > 0
                and nm < 0
                and slo > 0
                and nhi < 0
            ),
            "score": float(sm - nm) if not (np.isnan(sm) or np.isnan(nm)) else float("nan"),
        }
    return out


def _axis_rate(judges: Sequence[Optional[Dict[str, Any]]], key: str, value: str) -> float:
    vals = []
    for j in judges:
        if not j:
            continue
        ax = j.get("axes") or triple_axes(j)
        vals.append(1.0 if ax.get(key) == value else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def summarize_factor(blob: Dict[str, Any], split: Optional[str] = None) -> Dict[str, Any]:
    """Safety/mode rates on top of ordinary RHC bidirectional patching."""

    base = summarize_patch(blob, split=split)
    rows = blob["rows"]
    if split:
        rows = [r for r in rows if r.get("split") == split]
    extra: Dict[str, Any] = {}
    for L, st in base.items():
        layer_rows = [r for r in rows if str(r["layer"]) == str(L)]
        r2j = [r.get("ref_to_jb") for r in layer_rows]
        j2r = [r.get("jb_to_ref") for r in layer_rows]
        extra[str(L)] = {
            **st,
            "ref_to_unsafe": _axis_rate(r2j, "safety", "Unsafe"),
            "ref_to_denial": _axis_rate(r2j, "response_mode", "Denial"),
            "ref_to_not_refusal": 1.0 - _axis_rate(r2j, "response_mode", "Refusal")
            if r2j
            else float("nan"),
            "jb_to_safe": _axis_rate(j2r, "safety", "Safe"),
            "jb_to_refusal": _axis_rate(j2r, "response_mode", "Refusal"),
            "jb_to_denial": _axis_rate(j2r, "response_mode", "Denial"),
        }
    return extra


def benign_du(blob: Dict[str, Any], layer: Optional[int] = None) -> float:
    rows = blob.get("benign") or []
    if layer is not None:
        rows = [r for r in rows if r["layer"] == layer]
    if not rows:
        return float("nan")
    base = np.mean([r["base"]["utility"] for r in rows])
    pat = np.mean([r["patched"]["utility"] for r in rows])
    return float(pat - base)
