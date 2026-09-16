"""P0-B: measure local control energy, then a 24-step certificate intervention."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from p0.model import COMPLY_WORDS, REFUSE_WORDS, token_id_list
from traceflip.flip import generic_pgd, refusal_margin_objective

from .certificates import extract_failure_certificate, prefix_tag
from .energy import energy_joint, energy_single, gramian, gramian_stats
from .objective import accumulated_objective, certificate_jacobian, certificate_score
from .protocol import load_frozen
from .verifier_adapter import decode_prefix_text, official_decode


def _box(x0: torch.Tensor, delta: torch.Tensor, eps: float) -> torch.Tensor:
    d = delta.clamp(-float(eps), float(eps))
    return torch.clamp(x0 + d, 0.0, 1.0) - x0


def measure_state(
    wrapper,
    cell: Mapping[str, Any],
    *,
    delta: torch.Tensor,
    cert_ids: Sequence[int],
    extra_ids: Sequence[Sequence[int]] | None = None,
    frozen: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    frozen = dict(frozen or load_frozen())
    budget = frozen["budget"]
    tau = float(frozen["certificates"]["tau_nats"])
    eta = float(frozen["certificates"]["eta"])
    item = dict(cell["item"])
    item.setdefault("carrier_id", cell["carrier_id"])
    image = cell["image"]
    x0 = wrapper.image_to_x01(image)
    d0 = _box(x0, delta.to(device=x0.device, dtype=x0.dtype), float(budget["eps"]))
    base = wrapper.encode(image, item["query"])
    ids_list = [list(cert_ids)]
    for extra in extra_ids or []:
        extra = [int(x) for x in extra]
        if extra and extra not in ids_list:
            ids_list.append(extra)

    gs = []
    jacs = []
    for ids in ids_list:
        g, j = certificate_jacobian(wrapper, base, d0, x0, ids)
        gs.append(g)
        jacs.append(j)

    e_single = energy_single(gs[0], jacs[0], tau, eta)
    e_joint = energy_joint(gs, jacs, tau, eta)
    w = gramian(jacs)
    stats = gramian_stats(w)
    g0 = float(gs[0].item())
    jn = float(jacs[0].float().reshape(-1).pow(2).sum().sqrt().item())
    refuse = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply = token_id_list(wrapper.processor, COMPLY_WORDS)
    rm_val = float("nan")
    try:
        rm = refusal_margin_objective(
            wrapper, base, d0.detach().requires_grad_(True), x0, refuse, comply
        )
        rm_val = float(rm.detach().item()) if torch.is_tensor(rm) else float(rm)
    except Exception:
        rm_val = float("nan")
    return {
        "G": g0,
        "certificate_margin": g0 - tau,
        "grad_norm": jn,
        "E_single": float(e_single.item()) if torch.is_tensor(e_single) else float(e_single),
        "E_joint": float(e_joint.item()) if torch.is_tensor(e_joint) else float(e_joint),
        "diag_gramian_energy": float(sum(float(energy_single(g, j, tau, eta)) for g, j in zip(gs, jacs))),
        "gramian": stats,
        "refusal_margin": rm_val,
        "n_certs": len(ids_list),
        "delta": d0.detach(),
        "x0": x0,
        "base": base,
        "ids_list": ids_list,
        "tau": tau,
    }


def intervene_24(
    wrapper,
    cell: Mapping[str, Any],
    measured: Mapping[str, Any],
    *,
    old_hash: Optional[str] = None,
    frozen: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    frozen = dict(frozen or load_frozen())
    budget = frozen["budget"]
    cert_cfg = frozen["certificates"]
    item = dict(cell["item"])
    item.setdefault("carrier_id", cell["carrier_id"])
    image = cell["image"]
    x0 = measured["x0"]
    delta = measured["delta"].detach()
    tau = float(measured["tau"])
    target_ids = list(measured["ids_list"][0]) if measured["ids_list"] else []
    t0 = time.time()
    n_backward = 0
    g_after = None
    eps = float(budget["eps"])
    alpha = float(budget["alpha"])
    steps = int(budget["inner_steps"])
    for _ in range(steps):
        if not target_ids:
            break
        d = delta.detach().requires_grad_(True)
        g = certificate_score(wrapper, measured["base"], d, x0, target_ids)
        g_after = float(g.detach().item())
        if g_after <= tau and n_backward > 0:
            del g, d
            break
        gj = torch.autograd.grad(g, d, allow_unused=True)[0]
        del g
        if gj is None:
            break
        n_backward += 1
        delta = (d.detach() - alpha * gj.sign()).clamp(-eps, eps)
        delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0
        del gj, d
        with torch.no_grad():
            g_after = float(
                certificate_score(wrapper, measured["base"], delta, x0, target_ids).item()
            )
        if g_after <= tau:
            break
    x = torch.clamp(x0 + delta, 0.0, 1.0)
    tr = official_decode(wrapper, item, x, image, max_new_tokens=int(budget["max_new_tokens"]))
    rhc = bool(tr.get("core_rhc"))
    cert = None
    if not rhc:
        cert = extract_failure_certificate(
            tr.get("token_ids") or [],
            decode_prefix=lambda ids: decode_prefix_text(wrapper.processor, ids),
            label_fn=prefix_tag,
            max_len=int(cert_cfg["max_len"]),
            fallback_len=int(cert_cfg["fallback_len"]),
            stable_run=int(cert_cfg["stable_run"]),
        )
    new_hash = None if cert is None else cert["hash"]
    try:
        g_after = float(
            certificate_score(
                wrapper, measured["base"], delta, x0, measured["ids_list"][0]
            ).item()
        )
    except Exception:
        pass
    in_model_elim = g_after is not None and g_after <= float(measured["tau"])
    eliminated = bool(
        rhc
        or in_model_elim
        or (old_hash is not None and new_hash is not None and new_hash != old_hash)
        or (rhc is False and cert is None)
    )
    return {
        "core_rhc": rhc,
        "after_mode": str(tr.get("terminal_label") or tr.get("certificate_mode")),
        "after_cert_hash": new_hash,
        "after_cert_mode": None if cert is None else cert["mode"],
        "G_after": g_after,
        "in_model_elim": in_model_elim,
        "eliminated": eliminated,
        "steps_used": int(n_backward),
        "delta_linf": float(delta.detach().abs().max().item()),
        "wall_seconds": float(time.time() - t0),
        "n_tokens": int(tr.get("n_tokens") or 0),
    }


def collect_unique_states(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One state per (cell, certificate_hash), preferring later rounds."""
    by: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        if r.get("round") == "summary" or r.get("core_rhc"):
            continue
        h = r.get("certificate_hash")
        ids = r.get("certificate_token_ids")
        key_s = r.get("state_key")
        if not h or not ids or not key_s:
            continue
        key = (str(r.get("cell_id")), str(h))
        by[key] = {
            "cell_id": r["cell_id"],
            "query_id": r.get("query_id") or str(r["cell_id"]).split(":")[0],
            "carrier_id": r.get("carrier_id") or str(r["cell_id"]).split(":")[-1],
            "method": r.get("method"),
            "round": r.get("round"),
            "certificate_hash": h,
            "certificate_token_ids": list(ids),
            "certificate_mode": r.get("certificate_mode"),
            "state_key": key_s,
        }
    return list(by.values())
