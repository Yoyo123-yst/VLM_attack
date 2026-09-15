from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

# After centering, drop near-zero principal axes. Relative to the largest
# singular value; n=27 discover RHC typically yields r_eff <= 26.
SINGULAR_REL_THRESH = 1e-6


def _centered_svd(deltas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if deltas.ndim != 2 or deltas.shape[0] < 2:
        raise ValueError("need at least two delta vectors")
    x = deltas.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x, s, vt


def effective_rank_from_s(s: np.ndarray, rel_thresh: float = SINGULAR_REL_THRESH) -> int:
    if s.size == 0:
        return 0
    peak = float(s[0]) if float(s[0]) > 0 else 1.0
    return int(np.sum(s > peak * rel_thresh))


def pca_basis_info(
    deltas: np.ndarray,
    rank: int,
    rel_thresh: float = SINGULAR_REL_THRESH,
) -> Dict[str, Any]:
    """Centered PCA. Returns U [d, used_rank] plus requested vs effective rank."""
    x, s, vt = _centered_svd(deltas)
    r_eff = effective_rank_from_s(s, rel_thresh=rel_thresh)
    cap = max(r_eff, 1)
    used = min(int(rank), cap, int(vt.shape[0]), int(x.shape[1]))
    U = vt[:used].T.copy()
    n_sv = min(int(s.shape[0]), max(int(rank), used, 8))
    return {
        "U": U,
        "requested_rank": int(rank),
        "effective_rank": int(r_eff),
        "used_rank": int(used),
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "centered": True,
        "singular_values": [float(v) for v in s[:n_sv]],
        "rel_thresh": float(rel_thresh),
    }


def pca_basis(deltas: np.ndarray, rank: int) -> np.ndarray:
    """deltas: [n, d] -> U [d, r] with orthonormal columns (centered PCA)."""
    return pca_basis_info(deltas, rank)["U"]


def random_basis(dim: int, rank: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    return q[:, :rank].astype(np.float64)


def cov_matched_basis(deltas: np.ndarray, rank: int, seed: int = 1) -> np.ndarray:
    """Random rank-r subspace with the same per-coordinate scale as Δh."""
    rng = np.random.default_rng(seed)
    std = deltas.std(axis=0, keepdims=True) + 1e-8
    fake = rng.normal(size=deltas.shape) * std
    return pca_basis(fake, rank)


def project(h: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U.T @ h


def to_torch_u(U: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(U.astype(np.float32))


def stack_deltas(records: Sequence[Dict], key: str) -> np.ndarray:
    rows = [np.asarray(rec[key], dtype=np.float32).reshape(-1) for rec in records]
    return np.stack(rows, axis=0)


def mean_state(vectors: List[np.ndarray], U: np.ndarray) -> np.ndarray:
    zs = [project(np.asarray(v).reshape(-1), U) for v in vectors]
    return np.mean(np.stack(zs, axis=0), axis=0)


def principal_angles_deg(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    q1, _ = np.linalg.qr(np.asarray(U, dtype=np.float64), mode="reduced")
    q2, _ = np.linalg.qr(np.asarray(V, dtype=np.float64), mode="reduced")
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.degrees(np.arccos(s))


def orthogonalize_against(
    U: np.ndarray,
    U_fail: np.ndarray,
    rel_thresh: float = SINGULAR_REL_THRESH,
) -> Dict[str, Any]:
    """Orthonormal columns of U after removing the column-span of U_fail."""
    u = np.asarray(U, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("U must be [d, r]")
    f = np.asarray(U_fail, dtype=np.float64)
    if f.size == 0 or f.ndim != 2 or f.shape[1] == 0:
        q, _ = np.linalg.qr(u, mode="reduced")
        return {
            "U": q,
            "n_in": int(u.shape[1]),
            "n_fail": 0,
            "used_rank": int(q.shape[1]),
            "n_dropped": 0,
            "overlap_frac": 0.0,
            "rel_thresh": float(rel_thresh),
        }
    qf, _ = np.linalg.qr(f, mode="reduced")
    resid = u - qf @ (qf.T @ u)
    uu, s, _vt = np.linalg.svd(resid, full_matrices=False)
    peak = float(s[0]) if s.size and float(s[0]) > 0 else 0.0
    keep = s > (peak * rel_thresh if peak > 0 else rel_thresh)
    used = int(np.sum(keep))
    U_orth = uu[:, keep] if used else uu[:, :0]
    if used:
        U_orth = U_orth - qf @ (qf.T @ U_orth)
        U_orth, _ = np.linalg.qr(U_orth, mode="reduced")
        used = int(U_orth.shape[1])
    overlap_f = float(np.linalg.norm(qf.T @ u, "fro") ** 2 / max(u.shape[1], 1))
    return {
        "U": U_orth,
        "n_in": int(u.shape[1]),
        "n_fail": int(f.shape[1]),
        "used_rank": used,
        "n_dropped": int(u.shape[1]) - used,
        "overlap_frac": overlap_f,
        "rel_thresh": float(rel_thresh),
    }


def state_shift(h: np.ndarray, U: np.ndarray, mu_ref: np.ndarray, mu_jb: np.ndarray) -> float:
    z = project(np.asarray(h).reshape(-1), U)
    d_jb = float(np.linalg.norm(z - mu_jb))
    d_ref = float(np.linalg.norm(z - mu_ref))
    return d_ref - d_jb  # >0 means closer to JB than REF
