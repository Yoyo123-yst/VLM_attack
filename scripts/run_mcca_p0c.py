#!/usr/bin/env python3
"""P0-C: static C0–C3 on 68 labeled CR-0 answers, then T=10 content actuators.

Does not unlock P1 GPU. Does not edit MCCA_FROZEN.json. No mode constraint.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mcca.content import content_losses, nll_bank, tokenize_core  # noqa: E402
from mcca.cores import core_prefixes  # noqa: E402
from mcca.contrast import c2_span_terms, contrastive_losses, pairs_for  # noqa: E402
from mcca.p0c_analyze import (  # noqa: E402
    EXTEND_DSCR,
    actuator_pass,
    attach_deltas,
    branch_metrics,
    decision_cell,
    route_decision,
    static_score_block,
)
from mcca.p0c_select import match_actuator, static_rows  # noqa: E402
from otw.attack import pack  # noqa: E402
from otw.gate import decode_axes  # noqa: E402
from otw.open_phase import clip_delta  # noqa: E402
from otw.prototypes import harmful_prototypes, safe_prototypes  # noqa: E402
from otw.score import LAYER, load_u_refusal_p0s, s_mode  # noqa: E402
from otw.write_phase import prefix_nll, tokenize_prefix, write_loss  # noqa: E402
from p0.datautil import load_json, open_image, save_json  # noqa: E402

OUT = ROOT / "outputs" / "mcca" / "p0c"
FROZEN_PATH = ROOT / "outputs" / "mcca" / "MCCA_P0C_FROZEN.json"
P0_FROZEN = ROOT / "outputs" / "mcca" / "MCCA_FROZEN.json"
CR0_CAND = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"
P0_SCORES = ROOT / "outputs" / "mcca" / "p0" / "cr0_scores.json"
STATIC_PATH = OUT / "static_scores.json"
TRAJ_DIR = OUT / "trajectories"
DELTA_DIR = OUT / "deltas"
BRANCHES = ("B0", "B1", "B2", "B3", "B4")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_status(text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.txt").write_text(text + "\n", encoding="utf-8")


def load_frozen() -> Dict[str, Any]:
    cfg = load_json(FROZEN_PATH)
    if not cfg.get("frozen_before_any_p0c_gpu"):
        raise RuntimeError("P0-C frozen flag missing; refuse GPU")
    if P0_FROZEN.exists():
        parent = load_json(P0_FROZEN)
        if parent.get("splits", {}).get("cr0_val_queries") != ["h49", "h53", "h61", "h72"]:
            raise RuntimeError("parent MCCA_FROZEN.json drifted")
    return cfg


def prepare_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_wrapper():
    from p0_qwen.config import load_cfg
    from run_p0_qwen import load_model, seed_all

    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="full")
    seed_all(2026)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    wrapper.model.eval()
    return wrapper


def cand_index() -> Dict[str, Any]:
    return {r["record_id"]: r for r in load_json(CR0_CAND)["records"]}


def p0_index() -> Dict[str, Any]:
    return {r["record_id"]: r for r in load_json(P0_SCORES)["records"]}


def verify_c0_c3(frozen: Dict[str, Any], p0: Dict[str, Any]) -> Dict[str, Any]:
    ids = list(frozen["static_ids"])
    missing_c0, missing_c3, bad = [], [], []
    for rid in ids:
        r = p0[rid]
        if r.get("L_content") is None or not np.isfinite(float(r["L_content"])):
            missing_c0.append(rid)
        if r.get("L_write") is None or not np.isfinite(float(r["L_write"])):
            missing_c3.append(rid)
        if not (r.get("core_rhc") or r.get("core_safe_answer")):
            bad.append(rid)
    ok = not missing_c0 and not missing_c3 and not bad and len(ids) == 68
    return {
        "ok": ok,
        "n": len(ids),
        "missing_c0": missing_c0,
        "missing_c3": missing_c3,
        "bad_labels": bad,
        "reuse_c0": True,
        "reuse_c3": True,
    }


def load_start_tensors(wrapper, rec: Dict[str, Any], meta: Dict[str, Any], carriers, catalog):
    qid = rec["query_id"]
    cid = rec["carrier_id"]
    img = open_image(carriers[cid]["path"])
    q = catalog[qid]["query"]
    x0, ids, attn, grid = pack(wrapper, img, q)
    delta = torch.load(meta["delta_path"], map_location="cpu", weights_only=True).float()
    delta = delta.to(device=wrapper.device, dtype=x0.dtype)
    eps = float(meta["eps"])
    delta = clip_delta(x0, delta, eps)
    return img, q, x0, delta, ids, attn, grid


def leftover_stats(delta: torch.Tensor, eps: float) -> Dict[str, float]:
    room = (float(eps) - delta.detach().abs()).clamp(min=0.0)
    return {
        "linf": float(delta.detach().abs().amax().item()),
        "mean_unused_linf": float(room.mean().item()),
        "frac_saturated": float((delta.detach().abs() >= float(eps) - 1e-6).float().mean().item()),
    }


@torch.no_grad()
def score_all_losses(wrapper, x01, ids, attn, grid, qid: str, u, frozen: Dict[str, Any]) -> Dict[str, Any]:
    k = int(frozen["contrastive"]["K"])
    m = float(frozen["contrastive"]["margin_m"])
    beta = 0.5
    cl = content_losses(wrapper, x01, ids, attn, grid, qid, beta=beta, k=k)
    c2 = contrastive_losses(wrapper, x01, ids, attn, grid, qid, k=k, m=m)
    h_ids = [tokenize_prefix(wrapper, t) for t in harmful_prototypes(qid)]
    s_ids = [tokenize_prefix(wrapper, t) for t in safe_prototypes(qid)]
    lw = float(write_loss(wrapper, x01, ids, attn, grid, h_ids, s_ids).item())
    sm = float(s_mode(wrapper, x01, ids, attn, grid, u).item())
    return {
        "s_mode": sm,
        "L_c0": cl["L_content"],
        "L_c1": c2["L_c1"],
        "L_c2": c2["L_contrast"],
        "L_c3": lw,
        "s_content": c2["s_content"],
        "S_plus": c2["S_plus"],
        "S_minus": c2["S_minus"],
        "L_opening": cl["L_opening"],
        "nll_core": cl["nll_core"],
    }


def generate_snap(wrapper, img, question, keywords, x01, max_new: int) -> Dict[str, Any]:
    ax = decode_axes(wrapper, img, question, keywords, x01=x01, max_new_tokens=max_new)
    text = ax.get("text") or ""
    return {
        "response_mode": ax.get("response_mode"),
        "core_rhc": bool(ax.get("core_rhc")),
        "core_safe_answer": bool(ax.get("core_safe_answer")),
        "safety": ax.get("safety"),
        "grounding": ax.get("grounding"),
        "quality": ax.get("quality"),
        "theme_denial": bool(ax.get("theme_denial")),
        "chars": ax.get("chars"),
        "text_head": text[:240],
    }


def snapshot_bundle(wrapper, img, q, keywords, x0, delta, ids, attn, grid, qid, u, frozen, t: int) -> Dict[str, Any]:
    x01 = torch.clamp(x0 + delta, 0.0, 1.0)
    gen = generate_snap(wrapper, img, q, keywords, x01, int(frozen["model"]["max_new_tokens"]))
    losses = score_all_losses(wrapper, x01, ids, attn, grid, qid, u, frozen)
    row = {"t": int(t), **gen, **losses, "leftover": leftover_stats(delta, float(frozen["attack"]["eps"]))}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _grad_terms(wrapper, x0, delta, ids, attn, grid, terms: Sequence[Tuple[Any, float]]) -> torch.Tensor:
    d = delta.detach().requires_grad_(True)
    any_w = False
    for nll_fn, w in terms:
        if abs(float(w)) < 1e-12:
            continue
        any_w = True
        x = torch.clamp(x0 + d, 0.0, 1.0)
        nll = nll_fn(x)
        (float(w) * nll).backward()
        del nll
    if not any_w or d.grad is None:
        raise RuntimeError("P0-C PGD got no image gradient")
    if not torch.isfinite(d.grad).all():
        raise RuntimeError("P0-C PGD got non-finite gradient")
    return d.grad.detach()


def step_b1(x0, delta, eps, alpha, seed: int) -> torch.Tensor:
    g = torch.Generator(device=delta.device)
    g.manual_seed(int(seed))
    signs = torch.randint(0, 2, delta.shape, generator=g, device=delta.device)
    signs = signs.to(dtype=delta.dtype) * 2.0 - 1.0
    return clip_delta(x0, delta.detach() - float(alpha) * signs, eps)


def step_c0(wrapper, x0, delta, ids, attn, grid, qid: str, alpha: float, eps: float) -> torch.Tensor:
    cores = core_prefixes(qid, k=3)
    x_det = torch.clamp(x0 + delta.detach(), 0.0, 1.0)
    with torch.no_grad():
        nlls = nll_bank(wrapper, x_det, ids, attn, grid, cores)
        w = torch.softmax(-nlls.float() / 0.5, dim=0)
    terms = []
    for i, text in enumerate(cores):
        pids = tokenize_core(wrapper, text)

        def _fn(x, pids=pids):
            return prefix_nll(wrapper, x, ids, attn, grid, pids)

        terms.append((_fn, float(w[i].item())))
    grad = _grad_terms(wrapper, x0, delta, ids, attn, grid, terms)
    return clip_delta(x0, delta.detach() - float(alpha) * grad.sign(), eps)


def step_c3(wrapper, x0, delta, ids, attn, grid, qid: str, alpha: float, eps: float) -> torch.Tensor:
    h_ids = [tokenize_prefix(wrapper, t) for t in harmful_prototypes(qid)]
    s_ids = [tokenize_prefix(wrapper, t) for t in safe_prototypes(qid)]
    x_det = torch.clamp(x0 + delta.detach(), 0.0, 1.0)
    with torch.no_grad():
        nll_h = torch.stack([prefix_nll(wrapper, x_det, ids, attn, grid, p) for p in h_ids])
        nll_s = torch.stack([prefix_nll(wrapper, x_det, ids, attn, grid, p) for p in s_ids])
        w_h = torch.softmax(-nll_h.float(), dim=0)
        w_s = torch.softmax(-nll_s.float(), dim=0)
    terms = []
    for i, p in enumerate(h_ids):

        def _fh(x, p=p):
            return prefix_nll(wrapper, x, ids, attn, grid, p)

        terms.append((_fh, float(w_h[i].item())))
    for j, p in enumerate(s_ids):

        def _fs(x, p=p):
            return prefix_nll(wrapper, x, ids, attn, grid, p)

        terms.append((_fs, -float(w_s[j].item())))
    grad = _grad_terms(wrapper, x0, delta, ids, attn, grid, terms)
    return clip_delta(x0, delta.detach() - float(alpha) * grad.sign(), eps)


def step_c2(wrapper, x0, delta, ids, attn, grid, qid: str, alpha: float, eps: float, m: float) -> torch.Tensor:
    pairs = pairs_for(qid, k=3)
    x_det = torch.clamp(x0 + delta.detach(), 0.0, 1.0)
    with torch.no_grad():
        pos, neg = c2_span_terms(wrapper, x_det, ids, attn, grid, qid, k=3)
        nll_p = torch.stack(pos)
        nll_n = torch.stack(neg)
        s_content = -nll_p.mean() + nll_n.mean()
        sig = torch.sigmoid(nll_p.new_tensor(float(m)) - s_content)
        k = float(nll_p.numel())
        wp = float((sig / k).item())
        wn = float(((-sig) / k).item())
    from mcca.contrast import span_nll

    terms = []
    for p in pairs:

        def _fp(x, opening=p["opening"], kt=p["k_pos"]):
            return span_nll(wrapper, x, ids, attn, grid, opening, kt)

        def _fn(x, opening=p["opening"], kt=p["k_neg"]):
            return span_nll(wrapper, x, ids, attn, grid, opening, kt)

        terms.append((_fp, wp))
        terms.append((_fn, wn))
    grad = _grad_terms(wrapper, x0, delta, ids, attn, grid, terms)
    return clip_delta(x0, delta.detach() - float(alpha) * grad.sign(), eps)


def traj_path(start_id: str, branch: str) -> Path:
    safe = start_id.replace(":", "_")
    return TRAJ_DIR / f"{safe}__{branch}.json"


def delta_path(start_id: str, branch: str) -> Path:
    safe = start_id.replace(":", "_")
    return DELTA_DIR / f"{safe}__{branch}.pt"


def save_traj(row: Dict[str, Any]) -> None:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    save_json(traj_path(row["start_id"], row["branch"]), _jsonable(row))


def load_traj(start_id: str, branch: str) -> Optional[Dict[str, Any]]:
    p = traj_path(start_id, branch)
    if not p.exists():
        return None
    return load_json(p)


def run_static(wrapper, frozen, u, carriers, catalog, meta, p0) -> List[Dict[str, Any]]:
    existing = []
    have = set()
    if STATIC_PATH.exists():
        existing = load_json(STATIC_PATH).get("records") or []
        have = {r["record_id"] for r in existing}
    todo = [rid for rid in frozen["static_ids"] if rid not in have]
    records = list(existing)
    n_total = len(frozen["static_ids"])
    ver = verify_c0_c3(frozen, p0)
    if not ver["ok"]:
        raise RuntimeError(f"C0/C3 reuse failed: {ver}")
    k = int(frozen["contrastive"]["K"])
    m = float(frozen["contrastive"]["margin_m"])
    for i, rid in enumerate(todo, start=1):
        rec = p0[rid]
        src = meta[rid]
        img, q, x0, delta, ids, attn, grid = load_start_tensors(wrapper, rec, src, carriers, catalog)
        x01 = torch.clamp(x0 + delta, 0.0, 1.0)
        with torch.no_grad():
            c2 = contrastive_losses(wrapper, x01, ids, attn, grid, rec["query_id"], k=k, m=m)
            sm = float(s_mode(wrapper, x01, ids, attn, grid, u).item())
        row = {
            "record_id": rid,
            "query_id": rec["query_id"],
            "carrier_id": rec["carrier_id"],
            "category": rec.get("category"),
            "restart": rec.get("restart"),
            "core_rhc": bool(rec.get("core_rhc")),
            "core_safe_answer": bool(rec.get("core_safe_answer")),
            "response_mode": rec.get("response_mode"),
            "s_mode_p0": rec.get("s_mode"),
            "s_mode": sm,
            "L_c0": float(rec["L_content"]),
            "L_c3": float(rec["L_write"]),
            "L_c1": c2["L_c1"],
            "L_c2": c2["L_contrast"],
            "s_content": c2["s_content"],
            "S_plus": c2["S_plus"],
            "S_minus": c2["S_minus"],
            "nll_pos": c2["nll_pos"],
            "nll_neg": c2["nll_neg"],
            "c0_reused": True,
            "c3_reused": True,
        }
        records.append(row)
        save_json(STATIC_PATH, {"n": len(records), "verify_c0_c3": ver, "records": _jsonable(records)})
        write_status(f"P0C STATIC {len(records)}/{n_total} last={rid} L_c2={row['L_c2']:.4f} {now_iso()}")
        print({"static": rid, "progress": f"{len(records)}/{n_total}", "L_c1": row["L_c1"], "L_c2": row["L_c2"]}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def actuator_starts(frozen: Dict[str, Any], p0: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for rid in frozen["actuator_safe_ids"]:
        r = dict(p0[rid])
        r["start_kind"] = "SAFE"
        out.append(r)
    for rid in frozen["actuator_rhc_ids"]:
        r = dict(p0[rid])
        r["start_kind"] = "RHC"
        out.append(r)
    return out


def run_branch(
    wrapper,
    frozen,
    u,
    carriers,
    catalog,
    meta,
    start: Dict[str, Any],
    branch: str,
    t_target: int,
    snap0: Optional[Dict[str, Any]],
    delta0: torch.Tensor,
    img,
    q,
    x0,
    ids,
    attn,
    grid,
) -> Dict[str, Any]:
    rid = start["record_id"]
    qid = start["query_id"]
    keywords = catalog[qid]["topic_keywords"]
    eps = float(frozen["attack"]["eps"])
    alpha = float(frozen["attack"]["alpha"])
    m = float(frozen["contrastive"]["margin_m"])
    gen_at = set(frozen["attack"]["generate_at_short"])
    if t_target >= 20:
        gen_at = gen_at | set(frozen["attack"]["generate_at_extend"])
    row = load_traj(rid, branch) or {
        "start_id": rid,
        "branch": branch,
        "query_id": qid,
        "carrier_id": start["carrier_id"],
        "start_kind": start["start_kind"],
        "core_rhc_start": bool(start.get("core_rhc")),
        "core_safe_start": bool(start.get("core_safe_answer")),
        "steps_done": 0,
        "T_target": int(t_target),
        "snapshots": [],
        "complete": False,
    }
    row["T_target"] = max(int(row.get("T_target") or 0), int(t_target))
    have_t = {int(s["t"]) for s in row.get("snapshots") or []}
    if 0 not in have_t:
        if snap0 is not None:
            row.setdefault("snapshots", []).insert(0, dict(snap0, t=0))
        else:
            row.setdefault("snapshots", []).append(
                snapshot_bundle(wrapper, img, q, keywords, x0, delta0, ids, attn, grid, qid, u, frozen, 0)
            )
            row["snapshots"] = sorted(row["snapshots"], key=lambda s: int(s["t"]))
            save_traj(row)
    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    dpath = delta_path(rid, branch)
    if dpath.exists() and int(row.get("steps_done") or 0) > 0:
        delta = torch.load(dpath, map_location="cpu", weights_only=True).float().to(device=x0.device, dtype=x0.dtype)
        delta = clip_delta(x0, delta, eps)
    else:
        delta = delta0.detach().clone()

    b1_seed0 = int(frozen["b1_seeds"][rid])
    while int(row["steps_done"]) < int(t_target):
        t_next = int(row["steps_done"]) + 1
        if branch == "B0":
            pass
        elif branch == "B1":
            delta = step_b1(x0, delta, eps, alpha, b1_seed0 + t_next)
        elif branch == "B2":
            delta = step_c0(wrapper, x0, delta, ids, attn, grid, qid, alpha, eps)
        elif branch == "B3":
            delta = step_c3(wrapper, x0, delta, ids, attn, grid, qid, alpha, eps)
        elif branch == "B4":
            delta = step_c2(wrapper, x0, delta, ids, attn, grid, qid, alpha, eps, m)
        else:
            raise RuntimeError(branch)
        row["steps_done"] = t_next
        torch.save(delta.detach().cpu(), dpath)
        if t_next in gen_at:
            have_t = {int(s["t"]) for s in row["snapshots"]}
            if t_next not in have_t:
                if branch == "B0":
                    s0 = [s for s in row["snapshots"] if int(s["t"]) == 0][0]
                    snap = dict(s0)
                    snap["t"] = t_next
                    snap["deterministic_copy_of_t0"] = True
                else:
                    snap = snapshot_bundle(wrapper, img, q, keywords, x0, delta, ids, attn, grid, qid, u, frozen, t_next)
                row["snapshots"].append(snap)
                row["snapshots"] = sorted(row["snapshots"], key=lambda s: int(s["t"]))
        row["complete"] = int(row["steps_done"]) >= int(row["T_target"])
        save_traj(row)
        write_status(
            f"P0C ACT {rid} {branch} t={row['steps_done']}/{row['T_target']} {now_iso()}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row["complete"] = True
    save_traj(row)
    return row


def load_all_trajs() -> List[Dict[str, Any]]:
    if not TRAJ_DIR.exists():
        return []
    return [load_json(p) for p in sorted(TRAJ_DIR.glob("*.json"))]


def pick_extend(frozen: Dict[str, Any], trajs: Sequence[Dict[str, Any]]) -> List[str]:
    safe_ids = frozen["actuator_safe_ids"]
    rhc_ids = frozen["actuator_rhc_ids"]
    b0 = branch_metrics(trajs, "B0", 10, safe_ids, rhc_ids)
    ranked = []
    for b in ("B1", "B2", "B3", "B4"):
        m = attach_deltas(branch_metrics(trajs, b, 10, safe_ids, rhc_ids), b0)
        dscr = m.get("delta_SCR") or 0.0
        ranked.append((dscr, b, m))
    ranked.sort(reverse=True)
    chosen = [b for dscr, b, _ in ranked if dscr >= EXTEND_DSCR][:2]
    return chosen


def analyze(frozen: Dict[str, Any], static_rows_: List[Dict[str, Any]], trajs: List[Dict[str, Any]], extend: List[str]) -> Dict[str, Any]:
    static = {
        "C0": static_score_block(static_rows_, "L_c0", False, "C0"),
        "C1": static_score_block(static_rows_, "L_c1", False, "C1"),
        "C2": static_score_block(static_rows_, "L_c2", False, "C2"),
        "C2_s_content": static_score_block(static_rows_, "s_content", True, "C2_s_content"),
        "C3": static_score_block(static_rows_, "L_c3", False, "C3"),
    }
    safe_ids = frozen["actuator_safe_ids"]
    rhc_ids = frozen["actuator_rhc_ids"]
    t_final = {b: (20 if b in extend else 10) for b in BRANCHES}
    raw10 = {b: branch_metrics(trajs, b, 10, safe_ids, rhc_ids) for b in BRANCHES}
    b0_10 = raw10["B0"]
    act10 = {b: attach_deltas(raw10[b], b0_10) for b in BRANCHES}
    act20 = {}
    if extend:
        raw20 = {b: branch_metrics(trajs, b, t_final[b], safe_ids, rhc_ids) for b in BRANCHES}
        b0_20 = raw20["B0"]
        act20 = {b: attach_deltas(raw20[b], b0_20) for b in BRANCHES}
    gates = {}
    cells = {}
    loss_branch = {"C0": "B2", "C3": "B3", "C2": "B4", "C1": None}
    for loss, br in loss_branch.items():
        st = static[loss if loss != "C2" else "C2"]
        if br is None:
            cells[loss] = {
                "static": "PASS" if st["static_pass"] else "FAIL",
                "actuator": "n/a",
                "cell": "PASS_FAIL" if st["static_pass"] else "FAIL_FAIL",
                "note": "C1 is a scorer ablation; no dedicated actuator branch",
            }
            continue
        m10 = act10[br]
        m20 = act20.get(br) if br in extend else None
        gate = actuator_pass(m10, m20)
        gates[loss] = gate
        cells[loss] = {
            "static": "PASS" if st["static_pass"] else "FAIL",
            "actuator": "PASS" if gate["pass"] else "FAIL",
            "cell": decision_cell(bool(st["static_pass"]), bool(gate["pass"])),
        }
    route = route_decision(static["C2"], act10["B4"], gates["C2"], static["C3"], act10["B3"], gates["C3"])
    return {
        "schema": "mcca_p0c_results_v1",
        "written_at": now_iso(),
        "frozen": str(FROZEN_PATH),
        "layer": LAYER,
        "static": static,
        "actuator_t10": act10,
        "actuator_final": act20 or act10,
        "extend_branches": list(extend),
        "gates": gates,
        "decision_cells": cells,
        "route": route,
        "p0c_unlock_p1": bool(route["p0c_unlock_p1"]),
        "p1_locked": not bool(route["p0c_unlock_p1"]),
        "n_static": len(static_rows_),
        "n_traj": len(trajs),
    }


def render_report(blob: Dict[str, Any]) -> str:
    route = blob["route"]
    st = blob["static"]
    act = blob["actuator_t10"]
    cells = blob["decision_cells"]
    lines = [
        "# MCCA P0-C — content-loss actuator test",
        "",
        f"- Written: `{blob['written_at']}`",
        f"- Frozen: `{blob['frozen']}`",
        "- P1 GPU: **not started**. `p0c_unlock_p1` is a recommendation flag only.",
        "- No mode constraint on B0–B4. U_refusal not refit. h91–h130 unread.",
        "",
        "## Decision",
        "",
        f"- Route: **{route['route']}**",
        f"- **p0c_unlock_p1: {blob['p0c_unlock_p1']}** (P1 stays locked unless this is true; this ticket still does not launch P1)",
        f"- note: {route['note']}",
        f"- C2 harms mode: {route.get('c2_harms_mode')}",
        f"- T=20 extend branches: {blob.get('extend_branches')}",
        "",
        "## Static AUC (n=68 core_RHC vs RELATED_SAFE)",
        "",
        "| loss | global | within-q | centered | LOQO | query CI95 | cell CI95 | static PASS |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for name in ("C0", "C1", "C2", "C2_s_content", "C3"):
        b = st[name]
        wq = (b.get("within_query") or {}).get("mean_within_query_auc")
        ctr = (b.get("within_query") or {}).get("query_centered_auc")
        loqo = (b.get("loqo") or {}).get("mean_loqo_auc")
        bq = b.get("bootstrap_query") or {}
        bc = b.get("bootstrap_cell") or {}
        lines.append(
            f"| {name} | {b.get('global_auc')} | {wq} | {ctr} | {loqo} | {bq.get('ci95')} | {bc.get('ci95')} | {b.get('static_pass')} |"
        )
    lines += [
        "",
        "Old static gate (within-query ≥ 0.70) is reported only. Actuators ran for C0/C2/C3 regardless.",
        "",
        "### Within-query detail",
        "",
    ]
    for name in ("C0", "C1", "C2", "C3"):
        lines.append(f"- {name}: {(st[name].get('within_query') or {}).get('per_query')}")
    lines += [
        "",
        "## Actuator T=10 (n=15 SAFE + 15 RHC, shared δ)",
        "",
        "| branch | SCR | ΔSCR | RHR | MDR | ΔMDR | mean Δs_mode (SAFE vs B0) | mean Δloss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in BRANCHES:
        m = act[b]
        dL = (m.get("delta_loss") or {}).get("mean")
        dsm = m.get("delta_s_mode_safe_vs_b0")
        lines.append(
            f"| {b} | {m.get('SCR')} | {m.get('delta_SCR')} | {m.get('RHR')} | {m.get('MDR')} | {m.get('delta_MDR')} | {dsm} | {dL} |"
        )
    lines += ["", "### Per-query SAFE→RHC (T=10)", ""]
    for b in BRANCHES:
        lines.append(f"- {b}: {(act[b].get('per_query_safe_to_rhc'))}")
    if blob.get("extend_branches") and blob.get("actuator_final") is not act:
        fin = blob["actuator_final"]
        lines += ["", "## Actuator final (T=20 on extended branches)", ""]
        for b in blob["extend_branches"]:
            m = fin.get(b) or {}
            lines.append(
                f"- {b}: SCR={m.get('SCR')} ΔSCR={m.get('delta_SCR')} RHR={m.get('RHR')} MDR={m.get('MDR')}"
            )
    lines += [
        "",
        "## Decision table (static × actuator)",
        "",
        "| loss | static | actuator | cell | meaning |",
        "|---|---|---|---|---|",
        "| C0 | {c0s} | {c0a} | {c0c} | core-prefix NLL |".format(
            c0s=cells["C0"]["static"], c0a=cells["C0"]["actuator"], c0c=cells["C0"]["cell"]
        ),
        "| C1 | {c1s} | n/a | {c1c} | Y+ span NLL scorer only |".format(
            c1s=cells["C1"]["static"], c1c=cells["C1"]["cell"]
        ),
        "| C2 | {c2s} | {c2a} | {c2c} | contrastive span loss |".format(
            c2s=cells["C2"]["static"], c2a=cells["C2"]["actuator"], c2c=cells["C2"]["cell"]
        ),
        "| C3 | {c3s} | {c3a} | {c3c} | OtW write_loss |".format(
            c3s=cells["C3"]["static"], c3a=cells["C3"]["actuator"], c3c=cells["C3"]["cell"]
        ),
        "",
        "PASS×PASS = use this loss, recommend unlock P1. FAIL×PASS = actuator not sensor. "
        "PASS×FAIL = scorer only. FAIL×FAIL = drop.",
        "",
        f"## Route {route['route']}",
        "",
        route["note"],
        "",
        "P4 patch routing and P5 multi-model stay locked. Do not start P1 from this ticket.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--t10-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen()
    p0 = p0_index()
    meta = cand_index()
    scores_rows = list(p0.values())
    matches = match_actuator(scores_rows)
    if [{"safe_id": m["safe_id"], "rhc_id": m["rhc_id"], "tier": m["tier"]} for m in matches] != frozen["actuator_matches"]:
        raise RuntimeError("frozen actuator matches drifted from matcher")
    if sorted(r["record_id"] for r in static_rows(scores_rows)) != frozen["static_ids"]:
        raise RuntimeError("frozen static ids drifted")
    write_status(f"P0C START {now_iso()} analyze_only={args.analyze_only}")

    static_recs: List[Dict[str, Any]] = []
    if STATIC_PATH.exists():
        static_recs = load_json(STATIC_PATH).get("records") or []

    wrapper = None
    u = None
    carriers = None
    catalog = None
    if not args.analyze_only:
        prepare_env()
        from n1.preflight import all_carrier_index
        from p0.catalog import all_pairs

        u = load_u_refusal_p0s()
        catalog = {p["id"]: p for p in all_pairs()}
        carriers = all_carrier_index()
        wrapper = load_wrapper()
        if not args.skip_static:
            static_recs = run_static(wrapper, frozen, u, carriers, catalog, meta, p0)
        if args.static_only:
            blob = analyze(frozen, static_recs, load_all_trajs(), [])
            save_json(OUT / "P0C_RESULTS.json", _jsonable(blob))
            (OUT / "P0C_REPORT.md").write_text(render_report(blob), encoding="utf-8")
            write_status(f"P0C STATIC-ONLY DONE {now_iso()}")
            print({"static_only": True, "n": len(static_recs)}, flush=True)
            return

        starts = actuator_starts(frozen, p0)
        snap0_cache: Dict[str, Dict[str, Any]] = {}
        delta0_cache: Dict[str, torch.Tensor] = {}
        pack_cache: Dict[str, Any] = {}
        n_jobs = len(starts) * len(BRANCHES)
        done = 0
        for start in starts:
            rid = start["record_id"]
            src = meta[rid]
            img, q, x0, delta0, ids, attn, grid = load_start_tensors(wrapper, start, src, carriers, catalog)
            pack_cache[rid] = (img, q, x0, ids, attn, grid)
            delta0_cache[rid] = delta0
            keywords = catalog[start["query_id"]]["topic_keywords"]
            if rid not in snap0_cache:
                need0 = False
                for br in BRANCHES:
                    tr = load_traj(rid, br)
                    if tr is None or not any(int(s.get("t", -1)) == 0 for s in tr.get("snapshots") or []):
                        need0 = True
                        break
                if need0:
                    snap0_cache[rid] = snapshot_bundle(
                        wrapper, img, q, keywords, x0, delta0, ids, attn, grid, start["query_id"], u, frozen, 0
                    )
                else:
                    tr0 = load_traj(rid, "B0") or next(load_traj(rid, br) for br in BRANCHES if load_traj(rid, br))
                    snap0_cache[rid] = [s for s in tr0["snapshots"] if int(s["t"]) == 0][0]
            for br in BRANCHES:
                tr = load_traj(rid, br)
                if tr and tr.get("complete") and int(tr.get("steps_done") or 0) >= 10:
                    done += 1
                    continue
                run_branch(
                    wrapper, frozen, u, carriers, catalog, meta, start, br, 10,
                    snap0_cache[rid], delta0_cache[rid], img, q, x0, ids, attn, grid,
                )
                done += 1
                print({"actuator": rid, "branch": br, "progress": f"{done}/{n_jobs}"}, flush=True)

        trajs = load_all_trajs()
        extend = [] if args.t10_only else pick_extend(frozen, trajs)
        write_status(f"P0C EXTEND {extend} {now_iso()}")
        for start in starts:
            rid = start["record_id"]
            img, q, x0, ids, attn, grid = pack_cache[rid]
            for br in extend:
                tr = load_traj(rid, br)
                if tr and tr.get("complete") and int(tr.get("steps_done") or 0) >= 20:
                    continue
                d10 = delta_path(rid, br)
                if d10.exists():
                    d0 = torch.load(d10, map_location="cpu", weights_only=True).float().to(device=x0.device, dtype=x0.dtype)
                    d0 = clip_delta(x0, d0, float(frozen["attack"]["eps"]))
                else:
                    d0 = delta0_cache[rid]
                run_branch(
                    wrapper, frozen, u, carriers, catalog, meta, start, br, 20,
                    snap0_cache[rid], d0, img, q, x0, ids, attn, grid,
                )
    else:
        if not static_recs:
            raise RuntimeError("static_scores.json missing")

    trajs = load_all_trajs()
    extend = [] if args.t10_only else pick_extend(frozen, trajs) if trajs else []
    blob = analyze(frozen, static_recs, trajs, extend)
    save_json(OUT / "P0C_RESULTS.json", _jsonable(blob))
    (OUT / "P0C_REPORT.md").write_text(render_report(blob), encoding="utf-8")
    write_status(f"P0C DONE route={blob['route']['route']} unlock={blob['p0c_unlock_p1']} {now_iso()}")
    print(
        {
            "route": blob["route"]["route"],
            "p0c_unlock_p1": blob["p0c_unlock_p1"],
            "cells": blob["decision_cells"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
