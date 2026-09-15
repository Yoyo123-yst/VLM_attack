"""Bidirectional residual and subspace patching."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .datautil import open_image
from .judge import benign_utility, harmful_compliance
from .model import LlavaP0, image_to_clip_pixels, normalize_clip  # noqa: F401
from .subspace import to_torch_u


def _site_key(layer: int) -> str:
    return f"L{layer}:last_user"


def _pixels_from_delta(img, delta_list, device) -> torch.Tensor:
    x0 = image_to_clip_pixels(img).to(device)
    delta = torch.tensor(delta_list, device=device).unsqueeze(0)
    return normalize_clip(torch.clamp(x0 + delta, 0.0, 1.0), device, torch.float16)


def run_patching(
    wrapper: LlavaP0,
    cfg: Dict[str, Any],
    traces: Dict[str, Any],
    query_index: Dict[str, Dict[str, Any]],
    carrier_index: Dict[str, Dict[str, Any]],
    layers: List[int],
    U_by_layer: Optional[Dict[int, np.ndarray]] = None,
    mode: str = "full",
    tag: str = "full",
    do_benign: bool = True,
    recompute_clean: bool = False,
) -> Dict[str, Any]:
    rows = []
    pairs = [r for r in traces["records"] if r.get("jb") is not None]
    max_new = int(cfg["model"].get("patch_max_new_tokens", 64))
    for rec in tqdm(pairs, desc=f"patch:{tag}"):
        h = query_index[rec["query_id"]]
        img = open_image(carrier_index[rec["carrier_id"]]["path"])
        jb_pixels = None
        if rec.get("delta_path"):
            delta = torch.load(rec["delta_path"], map_location=wrapper.device, weights_only=True)
            x0 = image_to_clip_pixels(img).to(wrapper.device)
            jb_pixels = normalize_clip(torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0), wrapper.device, torch.float16)
        if recompute_clean:
            t_clean = wrapper.generate(img, h["query"], max_new_tokens=max_new)
            j_clean = harmful_compliance(t_clean, h["topic_keywords"], h["query"])
            clean_R = j_clean["R"]
            clean_refusal = j_clean["refusal"]
            if jb_pixels is not None:
                t_jb = wrapper.generate(img, h["query"], max_new_tokens=max_new, pixel_values=jb_pixels)
                j_jb_base = harmful_compliance(t_jb, h["topic_keywords"], h["query"])
            else:
                j_jb_base = rec["jb"]["judge"]
            jb_R = j_jb_base["R"]
            jb_refusal = j_jb_base["refusal"]
        else:
            clean_R = rec["clean"]["R"]
            clean_refusal = rec["clean"]["refusal"]
            jb_R = rec["jb"]["judge"]["R"]
            jb_refusal = rec["jb"]["judge"]["refusal"]
        for layer in layers:
            key = _site_key(layer)
            href = torch.tensor(rec["clean_hidden"][key])
            hjb = torch.tensor(rec["jb"]["hidden"][key])
            U = to_torch_u(U_by_layer[layer]) if mode == "sub" and U_by_layer is not None else None

            t_r2j = wrapper.generate_with_patch(
                img, h["query"], layer, hjb, mode=mode, U=U, max_new_tokens=max_new
            )
            j_r2j = harmful_compliance(t_r2j, h["topic_keywords"], h["query"])

            j_j2r = None
            if jb_pixels is not None:
                t_j2r = wrapper.generate_with_patch(
                    img,
                    h["query"],
                    layer,
                    href,
                    mode=mode,
                    U=U,
                    max_new_tokens=max_new,
                    pixel_values=jb_pixels,
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
    if do_benign:
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
        out[str(L)] = {
            "n": len(v["suff_dR"]),
            "ref_to_jb_dR": float(np.mean(v["suff_dR"])) if v["suff_dR"] else float("nan"),
            "ref_to_jb_d_refusal": float(np.mean(v["suff_dRef"])) if v["suff_dRef"] else float("nan"),
            "jb_to_ref_dR": float(np.mean(v["nec_dR"])) if v["nec_dR"] else float("nan"),
            "jb_to_ref_d_refusal": float(np.mean(v["nec_dRef"])) if v["nec_dRef"] else float("nan"),
        }
    return out
