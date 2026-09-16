"""Local visual control energy. P0-B prediction only — not the P0-A optimiser."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def energy_single(
    g: torch.Tensor,
    jac: torch.Tensor,
    tau: float,
    eta: float = 1e-8,
) -> torch.Tensor:
    """E = (tau - G)_+^2 / (||J||_2^2 + eta). Zero if the constraint already holds."""
    deficit = torch.clamp(g - float(tau), min=0.0)
    denom = jac.float().reshape(-1).pow(2).sum() + float(eta)
    return deficit.pow(2) / denom


def gramian(jacobians: Sequence[torch.Tensor]) -> torch.Tensor:
    """W = J J^T for row-stacked image Jacobians."""
    if not jacobians:
        return torch.zeros(0, 0)
    rows = [j.float().reshape(1, -1) for j in jacobians]
    jmat = torch.cat(rows, dim=0)
    return jmat @ jmat.t()


def gramian_stats(w: torch.Tensor) -> dict:
    if w.numel() == 0:
        return {"rank": 0, "condition": float("inf"), "n": 0}
    w = w.float()
    eig = torch.linalg.eigvalsh((w + w.t()) * 0.5)
    pos = eig[eig > 1e-12]
    cond = float("inf")
    if pos.numel() >= 2:
        cond = float((pos.max() / pos.min()).item())
    elif pos.numel() == 1:
        cond = 1.0
    rank = int((eig > 1e-8).sum().item())
    return {"rank": rank, "condition": cond, "n": int(w.shape[0]), "eig_min": float(eig.min())}


def single_constraint_step(
    g: torch.Tensor,
    jac: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Unconstrained min ||Δ||^2 s.t. G + <J,Δ> <= tau. Closed form."""
    j = jac.float()
    deficit = float((g - float(tau)).item())
    if deficit <= 0.0:
        return torch.zeros_like(j)
    nrm = float(j.reshape(-1).pow(2).sum().item())
    if nrm <= 0.0:
        return torch.zeros_like(j)
    return ((float(tau) - float(g.item())) / nrm) * j


def project_linf_box(
    x0: torch.Tensor,
    delta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    delta = delta.clamp(-float(eps), float(eps))
    return torch.clamp(x0 + delta, 0.0, 1.0) - x0


def energy_joint(
    scores: Sequence[torch.Tensor],
    jacobians: Sequence[torch.Tensor],
    tau: float,
    eta: float = 1e-8,
) -> torch.Tensor:
    """Unconstrained quadratic energy deficit^T (JJ^T + ηI)^{-1} deficit."""
    rows = []
    defs: list[float] = []
    device = None
    dtype = torch.float32
    for g, j in zip(scores, jacobians):
        dfc = max(float(g) - float(tau), 0.0)
        if dfc <= 0.0:
            continue
        rows.append(j)
        defs.append(dfc)
        device = j.device
        dtype = j.dtype
    if not rows:
        return torch.zeros((), device=device or "cpu", dtype=dtype)
    w = gramian(rows)
    n = int(w.shape[0])
    w = w + float(eta) * torch.eye(n, device=w.device, dtype=w.dtype)
    b = torch.tensor(defs, device=w.device, dtype=w.dtype)
    try:
        sol = torch.linalg.solve(w, b)
        return (b * sol).sum()
    except Exception:
        parts = [
            energy_single(torch.as_tensor(g, device=w.device), j, tau, eta)
            for g, j in zip(scores, jacobians)
        ]
        return torch.stack(parts).sum() if parts else torch.zeros((), device=w.device)
