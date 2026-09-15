"""Mode score from frozen P0-S U_refusal (feasibility only).

P0-S fits U_refusal on RELATED_SAFE − REF, so a large inner product is
answer-like, not refuse-like. Therefore s_mode = +<h, u> for this frozen
vector. Do not reuse this sign after refitting μ_REF − μ_ANSWER.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
P0S_U = ROOT / "outputs" / "p0_qwen" / "full" / "native" / "p0s_u.json"
LAYER = 24
# Frozen P0-S U_refusal: positive projection → ANSWER-like.
P0S_SIGN = 1.0


def _as_col(U: Any) -> np.ndarray:
    arr = np.asarray(U, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    u = arr[:, 0].reshape(-1)
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        raise RuntimeError("U_refusal has near-zero norm")
    return (u / n).astype(np.float64)


def load_u_refusal_p0s(path: Path | None = None) -> np.ndarray:
    blob = json.loads((path or P0S_U).read_text(encoding="utf-8"))
    spec = blob.get("U_refusal") or {}
    if int(blob.get("layer") or LAYER) != LAYER:
        raise RuntimeError(f"unexpected U layer {blob.get('layer')}")
    return _as_col(spec["U"])


def random_unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    v = rng.normal(size=int(dim)).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise RuntimeError("random unit vector has near-zero norm")
    return v / n


def s_mode_from_hidden(h: torch.Tensor, u: np.ndarray, sign: float = P0S_SIGN) -> torch.Tensor:
    vec = h.float().reshape(-1)
    uu = torch.from_numpy(np.asarray(u, dtype=np.float32)).to(device=vec.device, dtype=vec.dtype)
    if vec.numel() != uu.numel():
        raise RuntimeError(f"hidden dim {vec.numel()} != u dim {uu.numel()}")
    return float(sign) * torch.dot(vec, uu)


def s_mode(wrapper, x01: torch.Tensor, input_ids, attention_mask, grid, u: np.ndarray) -> torch.Tensor:
    pv, g = wrapper.patchify(x01)
    if grid is not None:
        g = grid
    h = wrapper.hidden_last_user(pv, input_ids, LAYER, attention_mask, g)
    return s_mode_from_hidden(h, u)
