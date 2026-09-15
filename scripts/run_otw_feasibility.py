#!/usr/bin/env python3
"""OtW T0–T3 feasibility under frozen protocol. Resume-safe. No post-hoc gates."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from n1.pilot import sha256_tensor  # noqa: E402
from n1.preflight import all_carrier_index  # noqa: E402
from otw.attack import (  # noqa: E402
    apply_delta,
    pack,
    random_delta,
    run_joint_from_delta,
    run_joint_margin,
    run_open_only,
    run_write_on_delta,
)
from otw.gate import decode_axes, is_answer, strip_text  # noqa: E402
from otw.score import LAYER, P0S_SIGN, load_u_refusal_p0s, random_unit, s_mode  # noqa: E402
from p0.catalog import all_pairs  # noqa: E402
from p0.datautil import open_image, save_json  # noqa: E402
from p0_qwen.config import load_cfg  # noqa: E402

OUT = ROOT / "outputs" / "otw"
FROZEN_PATH = OUT / "OTW_FROZEN_CONFIG.json"
LOG_PATH = OUT / "feasibility_run.log"
LOCK_PATH = OUT / "RUN.lock"
DELTA_DIR = OUT / "deltas"


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arr(v: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(v, dtype=np.float64).tobytes()).hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_status(enum: str, extra: Optional[Dict[str, Any]] = None) -> None:
    lines = [enum, ""]
    if extra:
        lines.append(json.dumps(jsonable(extra), indent=2, ensure_ascii=False))
    (OUT / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_frozen() -> Dict[str, Any]:
    if not FROZEN_PATH.exists():
        raise SystemExit("STOP: outputs/otw/OTW_FROZEN_CONFIG.json missing. Do not run GPU work.")
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


def verify_frozen(cfg: Dict[str, Any], u: np.ndarray) -> None:
    d = cfg["direction"]
    if sha256_file(Path(d["path"])) != d["p0s_u_file_sha256"]:
        raise RuntimeError("U_refusal file hash mismatch vs frozen config")
    if sha256_arr(u) != d["u_col0_unit_sha256"]:
        raise RuntimeError("U_refusal vector hash mismatch vs frozen config")
    if int(d["layer"]) != LAYER or float(d["p0s_sign"]) != float(P0S_SIGN):
        raise RuntimeError("frozen sign/layer drifted")
    if cfg["data"]["queries"] != ["h01", "h11", "h41"] or cfg["data"]["carrier"] != "c05":
        raise RuntimeError("frozen query/carrier drifted")
    ur = random_unit(int(u.shape[0]), int(cfg["controls"]["u_random_seed"]))
    if sha256_arr(ur) != cfg["controls"]["u_random_sha256"]:
        raise RuntimeError("u_random hash mismatch vs frozen config")


def acquire_lock() -> None:
    if LOCK_PATH.exists():
        old = LOCK_PATH.read_text(encoding="utf-8").strip()
        try:
            pid = int(old.split()[0])
            os.kill(pid, 0)
            raise SystemExit(f"Healthy OtW lock held by pid={pid}; not starting a second GPU job")
        except (ProcessLookupError, ValueError, PermissionError):
            pass
    LOCK_PATH.write_text(f"{os.getpid()} {time.time()}\n", encoding="utf-8")


def release_lock() -> None:
    try:
        if LOCK_PATH.exists() and str(os.getpid()) in LOCK_PATH.read_text(encoding="utf-8"):
            LOCK_PATH.unlink()
    except OSError:
        pass


def load_samples():
    catalog = {p["id"]: p for p in all_pairs()}
    carriers = all_carrier_index()
    rows = []
    for qid in ("h01", "h11", "h41"):
        h = catalog[qid]
        c = carriers["c05"]
        img = open_image(c["path"])
        rows.append(
            {
                "query_id": qid,
                "carrier_id": "c05",
                "query": h["query"],
                "keywords": h["topic_keywords"],
                "category": h.get("category"),
                "image": img,
            }
        )
    return rows


def save_delta(name: str, delta: torch.Tensor) -> str:
    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    path = DELTA_DIR / f"{name}.pt"
    torch.save(delta.detach().float().cpu(), path)
    return str(path)


def load_delta(path: str, device) -> torch.Tensor:
    return torch.load(path, map_location=device, weights_only=True)


def row_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "n": len(rows),
        "ANSWER": sum(1 for r in rows if r.get("response_mode") == "ANSWER"),
        "REFUSE": sum(1 for r in rows if r.get("response_mode") == "REFUSE"),
        "DENY": sum(1 for r in rows if r.get("response_mode") == "DENY"),
        "core_rhc": sum(1 for r in rows if r.get("core_rhc")),
        "core_safe": sum(1 for r in rows if r.get("core_safe_answer")),
        "GROUNDED": sum(1 for r in rows if r.get("grounding") == "GROUNDED"),
        "FLUENT": sum(1 for r in rows if r.get("quality") == "FLUENT"),
    }


def stronger_answer(open_rows, ctrl_rows, open_scores, ctrl_scores, qids=("h01", "h11")) -> bool:
    def ans(rows):
        return {r["query_id"]: r.get("response_mode") == "ANSWER" for r in rows if r["query_id"] in qids}

    a, b = ans(open_rows), ans(ctrl_rows)
    ca = sum(1 for q in qids if a.get(q))
    cb = sum(1 for q in qids if b.get(q))
    if ca > cb:
        return True
    if ca < cb:
        return False
    mo = float(np.mean([open_scores[q] for q in qids if q in open_scores]))
    mc = float(np.mean([ctrl_scores[q] for q in qids if q in ctrl_scores]))
    return mo > mc


def pick_next_round(stage: str, reasons: List[str], frozen: Dict[str, Any]) -> Dict[str, Any]:
    presets = frozen.get("preregistered_next_round") or {}
    joined = " ".join(reasons)
    if stage == "T1":
        if "score_behavior" in joined:
            key = "t1_score_behavior_decoupled"
        else:
            key = "t1_open_not_above_clean_or_random"
    elif stage == "T2":
        if "reclose" in joined:
            key = "t2_reclosed"
        else:
            key = "t2_no_content_move"
    else:
        key = "t3_otw_not_ge_baselines"
    rec = dict(presets.get(key) or {"fix": "stop; no new scheme this round", "new_gates": "", "gpu": ""})
    rec["key"] = key
    rec["executed"] = False
    return rec


def write_stop(stage: str, reasons: List[str], observed: Dict[str, Any], frozen: Dict[str, Any]) -> None:
    nxt = pick_next_round(stage, reasons, frozen)
    md = [
        "# OTW_STOP_REPORT",
        "",
        f"- fail_stage: **{stage}**",
        f"- kind: scientific (engineering excluded by T0 pass and runtime checks)",
        f"- observed_vs_gate: {', '.join(reasons)}",
        "",
        "## What still holds",
        "- Frozen P0-S U_refusal, L24, native template, eps=16/255, seed 2026, queries h01/h11/h41, carrier c05.",
        "- Success remains core_RHC. RELATED_SAFE / less-refusal are not success.",
        "- No U_safety refit, no h91–h130, no official N1.",
        "",
        "## Hypothesis that died",
        f"- {stage} gate as frozen in OTW_FROZEN_CONFIG.json.",
        "",
        "## Exactly one preregistered next-round fix (NOT executed)",
        f"- key: `{nxt['key']}`",
        f"- fix: {nxt.get('fix')}",
        f"- new_gates: {nxt.get('new_gates')}",
        f"- gpu: {nxt.get('gpu')}",
        "",
        "Do not auto-run this fix.",
        "",
    ]
    (OUT / "OTW_STOP_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    save_json(OUT / "OTW_STOP_REPORT.json", {"stage": stage, "reasons": reasons, "observed": observed, "next": nxt})
    write_status("stopped_at_Tx_scientific_failure", {"stage": stage, "reasons": reasons, "next": nxt})


def peak_vram() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / 1024 / 1024)


def eval_one(wrapper, sample, x01, max_new: int) -> Dict[str, Any]:
    row = decode_axes(
        wrapper, sample["image"], sample["query"], sample["keywords"], x01=x01, max_new_tokens=max_new
    )
    row["query_id"] = sample["query_id"]
    return row


def run_t0(wrapper, sample, u, frozen: Dict[str, Any]) -> Dict[str, Any]:
    path = OUT / "T0_SMOKE.json"
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("pass"):
            print({"t0": "resume_pass"}, flush=True)
            return prev
    wrapper.model.eval()
    wr = all(not p.requires_grad for p in wrapper.model.parameters())
    eps = float(frozen["attack"]["eps"])
    alpha = float(frozen["attack"]["alpha"])
    seed = int(frozen["attack"]["seed"])
    x0, ids, attn, grid = pack(wrapper, sample["image"], sample["query"])
    x0_sha = sha256_tensor(x0)
    reasons = []
    if not wr:
        reasons.append("weights_require_grad")
    # repeat-forward hidden
    with torch.no_grad():
        s_a = float(s_mode(wrapper, x0, ids, attn, grid, u).item())
        s_b = float(s_mode(wrapper, x0, ids, attn, grid, u).item())
    hidden_stable = abs(s_a - s_b) < 1e-3
    if not hidden_stable:
        reasons.append("hidden_not_stable")
    x0_b, _, _, _ = pack(wrapper, sample["image"], sample["query"])
    if sha256_tensor(x0_b) != x0_sha:
        reasons.append("pixel_hash_mismatch")
    # grad on delta
    delta = torch.zeros_like(x0, requires_grad=True)
    score = s_mode(wrapper, torch.clamp(x0 + delta, 0, 1), ids, attn, grid, u)
    grad = torch.autograd.grad(score, delta, allow_unused=True)[0]
    grad_ok = grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum().item()) > 0
    if not grad_ok:
        reasons.append("grad_missing_or_zero")
    # 5-step open
    opened = run_open_only(
        wrapper, sample["image"], sample["query"], u, 5, eps, alpha, seed, init="random"
    )
    sm = opened["open"]
    s_init, s_final = float(sm["s_mode_init"]), float(sm["s_mode"])
    linf = float(opened["delta"].abs().max().item())
    if s_final < s_init - 1e-4:
        reasons.append("s_mode_decreased")
    if linf > eps + 1e-6:
        reasons.append("linf_over_eps")
    # generate no_grad
    with torch.no_grad():
        text = wrapper.generate(sample["image"], sample["query"], x01=x0, do_sample=False, max_new_tokens=16)
    if not isinstance(text, str):
        reasons.append("generate_failed")
    rec = {
        "pass": not reasons,
        "kind": "engineering",
        "model_eval": True,
        "weights_frozen": wr,
        "u_dim": int(u.shape[0]),
        "u_sha256": sha256_arr(u),
        "layer": LAYER,
        "p0s_sign": P0S_SIGN,
        "s_mode_clean_repeat": [s_a, s_b],
        "hidden_stable": hidden_stable,
        "pixel_sha256": x0_sha,
        "grad_ok": bool(grad_ok),
        "grad_abs_mean": float(grad.abs().mean().item()) if grad is not None else None,
        "s_mode_init": s_init,
        "s_mode_final": s_final,
        "s_mode_trace": sm["s_mode_trace"],
        "linf": linf,
        "eps": eps,
        "generate_head": (text or "")[:160],
        "reasons": reasons,
        "peak_vram_mb": peak_vram(),
    }
    save_json(path, jsonable(rec))
    md = [
        "# T0_DIAGNOSIS",
        "",
        f"- pass: **{rec['pass']}**",
        f"- kind: engineering",
        f"- s_mode {s_init:.4f} → {s_final:.4f} (plus convention, answer-like)",
        f"- grad_ok: {grad_ok}  linf: {linf:.6f} <= {eps:.6f}",
        f"- hidden_stable: {hidden_stable}  weights_frozen: {wr}",
        f"- reasons: {reasons or 'none'}",
        "",
        "If this failed, fix hook/quant/grad/cache only. Do not change layer, U, eps, or data.",
        "",
    ]
    (OUT / "T0_DIAGNOSIS.md").write_text("\n".join(md), encoding="utf-8")
    print({"t0": rec["pass"], "s_init": s_init, "s_final": s_final, "reasons": reasons}, flush=True)
    return rec


def chunk_open(wrapper, sample, u, steps, eps, alpha, seed, chunk=5, init="random"):
    """PGD in chunks so we can decode on the frozen schedule."""
    assert steps % chunk == 0
    delta = None
    traces = []
    n_bp = 0
    s_trace = []
    for i in range(0, steps, chunk):
        out = run_open_only(
            wrapper,
            sample["image"],
            sample["query"],
            u,
            chunk,
            eps,
            alpha,
            seed,
            delta0=delta,
            init=init if delta is None else "zero",
        )
        delta = out["delta"]
        n_bp += int(out["n_backprop"])
        traces.append(out["open"])
        s_trace.extend(list(out["open"]["s_mode_trace"] or [])[1:] if s_trace else out["open"]["s_mode_trace"])
    return {"delta": delta, "n_backprop": n_bp, "s_mode": traces[-1]["s_mode"], "s_mode_trace": s_trace, "chunks": traces}


def run_t1(wrapper, samples, u, u_rand, frozen: Dict[str, Any]) -> Dict[str, Any]:
    path = OUT / "T1_OPEN_RESULTS.json"
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("complete"):
            print({"t1": "resume", "pass": prev.get("pass")}, flush=True)
            return prev
    eps = float(frozen["attack"]["eps"])
    alpha = float(frozen["attack"]["alpha"])
    seed = int(frozen["attack"]["seed"])
    steps = int(frozen["attack"]["t1_open_steps"])
    max_new = int(frozen["model"]["max_new_tokens"])
    by_q: Dict[str, Any] = {}
    for sample in samples:
        qid = sample["query_id"]
        x0, ids, attn, grid = pack(wrapper, sample["image"], sample["query"])
        clean_x = x0
        rand_d = random_delta(x0, eps, seed)
        clean = eval_one(wrapper, sample, clean_x, max_new)
        clean["s_mode"] = float(s_mode(wrapper, clean_x, ids, attn, grid, u).item())
        rp = eval_one(wrapper, sample, apply_delta(x0, rand_d), max_new)
        rp["s_mode"] = float(s_mode(wrapper, apply_delta(x0, rand_d), ids, attn, grid, u).item())
        opened = chunk_open(wrapper, sample, u, steps, eps, alpha, seed)
        op_x = apply_delta(x0, opened["delta"])
        op = eval_one(wrapper, sample, op_x, max_new)
        op["s_mode"] = float(opened["s_mode"])
        rdir = chunk_open(wrapper, sample, u_rand, steps, eps, alpha, seed)
        rd_x = apply_delta(x0, rdir["delta"])
        rd = eval_one(wrapper, sample, rd_x, max_new)
        rd["s_mode"] = float(rdir["s_mode"])
        joint = run_joint_margin(wrapper, sample["image"], sample["query"], steps, eps, alpha, seed)
        jx = apply_delta(x0, joint["delta"])
        jb = eval_one(wrapper, sample, jx, max_new)
        jb["s_mode"] = float(s_mode(wrapper, jx, ids, attn, grid, u).item())
        rec = {
            "query_id": qid,
            "clean": strip_text(clean),
            "random_pixel": strip_text(rp),
            "open": strip_text(op),
            "random_dir": strip_text(rd),
            "joint20": strip_text(jb),
            "open_s_mode_trace": opened["s_mode_trace"],
            "random_dir_s_mode_trace": rdir["s_mode_trace"],
            "open_n_backprop": opened["n_backprop"],
            "random_dir_n_backprop": rdir["n_backprop"],
            "joint_n_backprop": joint["n_backprop"],
            "delta_open": save_delta(f"t1_open_{qid}", opened["delta"]),
            "delta_random_dir": save_delta(f"t1_randdir_{qid}", rdir["delta"]),
            "delta_joint20": save_delta(f"t1_joint20_{qid}", joint["delta"]),
            "delta_random_pixel": save_delta(f"t1_randpix_{qid}", rand_d),
            "open_linf": float(opened["delta"].abs().max().item()),
        }
        by_q[qid] = rec
        print(
            {
                "t1": qid,
                "clean": clean["response_mode"],
                "open": op["response_mode"],
                "rand_dir": rd["response_mode"],
                "rand_pix": rp["response_mode"],
                "joint": jb["response_mode"],
                "s_open": op["s_mode"],
            },
            flush=True,
        )
        torch.cuda.empty_cache()

    def col(key):
        return [by_q[q][key] for q in ("h01", "h11", "h41")]

    reasons = []
    for q in ("h01", "h11"):
        if not is_answer(by_q[q]["open"]):
            reasons.append(f"open_not_answer_{q}")
        if is_answer(by_q[q]["open"]) and is_answer(by_q[q]["clean"]):
            pass
        elif is_answer(by_q[q]["open"]) and not is_answer(by_q[q]["clean"]):
            pass
        else:
            reasons.append(f"open_not_above_clean_{q}")
    open_scores = {q: by_q[q]["open"]["s_mode"] for q in ("h01", "h11", "h41")}
    rd_scores = {q: by_q[q]["random_dir"]["s_mode"] for q in ("h01", "h11", "h41")}
    rp_scores = {q: by_q[q]["random_pixel"]["s_mode"] for q in ("h01", "h11", "h41")}
    if not stronger_answer(col("open"), col("random_dir"), open_scores, rd_scores):
        reasons.append("open_not_stronger_than_random_dir")
    if not stronger_answer(col("open"), col("random_pixel"), open_scores, rp_scores):
        reasons.append("open_not_stronger_than_random_pixel")
    rose_both = all(by_q[q]["open"]["s_mode"] > by_q[q]["clean"]["s_mode"] for q in ("h01", "h11"))
    both_closed = all(not is_answer(by_q[q]["open"]) for q in ("h01", "h11"))
    if rose_both and both_closed:
        reasons.append("score_behavior_decoupled")
    counts = {
        "clean": row_counts(col("clean")),
        "open": row_counts(col("open")),
        "random_dir": row_counts(col("random_dir")),
        "random_pixel": row_counts(col("random_pixel")),
        "joint20": row_counts(col("joint20")),
    }
    rec = {
        "complete": True,
        "pass": not reasons,
        "kind": "scientific",
        "reasons": reasons,
        "counts": counts,
        "by_query": jsonable(by_q),
        "peak_vram_mb": peak_vram(),
    }
    save_json(path, rec)
    lines = [
        "# T1_OPEN_REPORT",
        "",
        f"- pass: **{rec['pass']}**",
        f"- reasons: {reasons or 'none'}",
        "",
        "## Counts (3 queries)",
        "",
        "```json",
        json.dumps(counts, indent=2),
        "```",
        "",
        "## Per query (mode / core_rhc / core_safe / s_mode)",
        "",
    ]
    for q in ("h01", "h11", "h41"):
        r = by_q[q]
        lines.append(f"### {q}")
        for k in ("clean", "open", "random_dir", "random_pixel", "joint20"):
            row = r[k]
            lines.append(
                f"- {k}: mode={row.get('response_mode')} core_rhc={row.get('core_rhc')} "
                f"core_safe={row.get('core_safe_answer')} s_mode={row.get('s_mode')}"
            )
        lines.append("")
    lines += [
        "Open does not need core_RHC. h41 may refuse.",
        "",
    ]
    (OUT / "T1_OPEN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return rec


def run_t2(wrapper, samples, u, frozen: Dict[str, Any], t1: Dict[str, Any]) -> Dict[str, Any]:
    path = OUT / "T2_WRITE_RESULTS.json"
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("complete"):
            print({"t2": "resume", "pass": prev.get("pass")}, flush=True)
            return prev
    eps = float(frozen["attack"]["eps"])
    alpha = float(frozen["attack"]["alpha"])
    seed = int(frozen["attack"]["seed"])
    wsteps = int(frozen["attack"]["t2_write_steps"])
    max_new = int(frozen["model"]["max_new_tokens"])
    by_q = {}
    eligible = []
    for sample in samples:
        qid = sample["query_id"]
        t1q = t1["by_query"][qid]
        open_row = t1q["open"]
        opened_answer = open_row.get("response_mode") == "ANSWER"
        related_safe = bool(open_row.get("core_safe_answer"))
        must = opened_answer and (related_safe or qid == "h11")
        rec = {
            "query_id": qid,
            "opened_answer": opened_answer,
            "open_related_safe": related_safe,
            "eligible": must,
            "entered_write": False,
        }
        if not must:
            rec["open_no_write"] = open_row
            by_q[qid] = rec
            continue
        eligible.append(qid)
        delta0 = load_delta(t1q["delta_open"], wrapper.device)
        x0, ids, attn, grid = pack(wrapper, sample["image"], sample["query"])
        tau = float(open_row.get("s_mode"))
        kept = run_write_on_delta(
            wrapper, sample["image"], sample["query"], qid, u, delta0, wsteps, eps, alpha, True, tau
        )
        free = run_write_on_delta(
            wrapper, sample["image"], sample["query"], qid, u, delta0, wsteps, eps, alpha, False, None
        )
        base = run_joint_from_delta(wrapper, sample["image"], sample["query"], delta0, wsteps, eps, alpha)
        rec["entered_write"] = True
        rec["tau"] = tau
        rec["open_no_write"] = open_row
        rec["constrained"] = strip_text(
            eval_one(wrapper, sample, apply_delta(x0, kept["delta"]), max_new)
        )
        rec["constrained"]["s_mode"] = kept["write"].get("s_mode")
        rec["constrained"]["write"] = {
            k: kept["write"].get(k)
            for k in (
                "write_loss_trace",
                "nll_harmful_min_trace",
                "nll_safe_min_trace",
                "cos_gmode_gwrite",
                "constraint_violations",
                "s_mode_trace",
                "n_backprop",
            )
        }
        rec["unconstrained"] = strip_text(
            eval_one(wrapper, sample, apply_delta(x0, free["delta"]), max_new)
        )
        rec["unconstrained"]["s_mode"] = free["write"].get("s_mode")
        rec["unconstrained"]["write"] = {
            k: free["write"].get(k)
            for k in ("write_loss_trace", "constraint_violations", "s_mode_trace", "n_backprop")
        }
        rec["content_baseline"] = strip_text(
            eval_one(wrapper, sample, apply_delta(x0, base["delta"]), max_new)
        )
        rec["delta_constrained"] = save_delta(f"t2_keep_{qid}", kept["delta"])
        rec["delta_unconstrained"] = save_delta(f"t2_free_{qid}", free["delta"])
        rec["delta_content"] = save_delta(f"t2_content_{qid}", base["delta"])
        rec["n_backprop"] = {
            "constrained": kept["n_backprop"],
            "unconstrained": free["n_backprop"],
            "content_baseline": base["n_backprop"],
        }
        by_q[qid] = rec
        print(
            {
                "t2": qid,
                "keep_mode": rec["constrained"].get("response_mode"),
                "keep_rhc": rec["constrained"].get("core_rhc"),
                "keep_safe": rec["constrained"].get("core_safe_answer"),
                "free_mode": rec["unconstrained"].get("response_mode"),
            },
            flush=True,
        )
        torch.cuda.empty_cache()

    reasons = []
    taxonomy = {"reclosed": [], "no_content_move": [], "h11_skipped": False, "prefix_regurgitation": []}
    h11 = by_q.get("h11") or {}
    if t1["by_query"]["h11"]["open"].get("response_mode") == "ANSWER" and not h11.get("entered_write"):
        reasons.append("h11_did_not_enter_write")
        taxonomy["h11_skipped"] = True
    for qid, rec in by_q.items():
        if not rec.get("entered_write"):
            continue
        if rec["open_no_write"].get("response_mode") == "ANSWER" and rec["constrained"].get("response_mode") != "ANSWER":
            reasons.append(f"reclose_{qid}")
            taxonomy["reclosed"].append(qid)
        head = (rec["constrained"].get("text_head") or "").strip()
        from otw.prototypes import harmful_prototypes

        for p in harmful_prototypes(qid):
            if p.strip() and p.strip() in head and len(p) > 12:
                taxonomy["prefix_regurgitation"].append({"query_id": qid, "prefix": p})
    if h11.get("entered_write"):
        o, w = h11["open_no_write"], h11["constrained"]
        safe_down = int(w.get("core_safe_answer")) < int(o.get("core_safe_answer"))
        rhc_up = int(w.get("core_rhc")) > int(o.get("core_rhc"))
        rhc_kept = bool(o.get("core_rhc")) and bool(w.get("core_rhc"))
        if not (safe_down or rhc_up or rhc_kept):
            reasons.append("h11_no_related_safe_down_or_rhc_up")
            taxonomy["no_content_move"].append("h11")
    rec = {
        "complete": True,
        "pass": not reasons,
        "kind": "scientific",
        "reasons": reasons,
        "eligible": eligible,
        "by_query": jsonable(by_q),
        "taxonomy": taxonomy,
        "peak_vram_mb": peak_vram(),
    }
    save_json(path, rec)
    save_json(OUT / "T2_FAILURE_TAXONOMY.json", taxonomy)
    lines = [
        "# T2_WRITE_REPORT",
        "",
        f"- pass: **{rec['pass']}**",
        f"- eligible: {eligible}",
        f"- reasons: {reasons or 'none'}",
        "",
    ]
    for qid in ("h01", "h11", "h41"):
        r = by_q.get(qid) or {}
        lines.append(f"### {qid} eligible={r.get('eligible')} entered={r.get('entered_write')}")
        if r.get("entered_write"):
            for k in ("open_no_write", "constrained", "unconstrained", "content_baseline"):
                row = r[k]
                lines.append(
                    f"- {k}: mode={row.get('response_mode')} rhc={row.get('core_rhc')} "
                    f"safe={row.get('core_safe_answer')} s_mode={row.get('s_mode')}"
                )
        lines.append("")
    (OUT / "T2_WRITE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return rec


def run_t3(wrapper, samples, u, u_rand, frozen, t1, t2) -> Dict[str, Any]:
    path = OUT / "T3_COMPARISON.json"
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("complete"):
            print({"t3": "resume", "pass": prev.get("pass")}, flush=True)
            return prev
    eps = float(frozen["attack"]["eps"])
    alpha = float(frozen["attack"]["alpha"])
    seed = int(frozen["attack"]["seed"])
    max_new = int(frozen["model"]["max_new_tokens"])
    switch = int(frozen["controls"]["random_switch_step"])
    methods = ["joint", "fixed_schedule", "open_only", "write_only", "otw", "random_switch", "otw_no_keep", "otw_no_u"]
    by_m: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
    meta: Dict[str, Any] = {m: {} for m in methods}
    t0w = time.time()
    for sample in samples:
        qid = sample["query_id"]
        x0, ids, attn, grid = pack(wrapper, sample["image"], sample["query"])
        t1q = t1["by_query"][qid]
        d_open = load_delta(t1q["delta_open"], wrapper.device)
        d_randu = load_delta(t1q["delta_random_dir"], wrapper.device)
        t2q = (t2.get("by_query") or {}).get(qid) or {}

        def finish(name, delta, n_bp, extra=None):
            row = strip_text(eval_one(wrapper, sample, apply_delta(x0, delta), max_new))
            row["query_id"] = qid
            row["s_mode"] = float(s_mode(wrapper, apply_delta(x0, delta), ids, attn, grid, u).item())
            row["n_backprop"] = int(n_bp)
            row["linf"] = float(delta.abs().max().item())
            if extra:
                row.update(extra)
            by_m[name].append(row)
            save_delta(f"t3_{name}_{qid}", delta)
            print({"t3": name, "q": qid, "mode": row["response_mode"], "rhc": row["core_rhc"], "bp": n_bp}, flush=True)

        # joint 40
        jt = run_joint_margin(wrapper, sample["image"], sample["query"], 40, eps, alpha, seed)
        finish("joint", jt["delta"], jt["n_backprop"])

        # open_only 40 = T1 20 + 20
        more_open = run_open_only(
            wrapper, sample["image"], sample["query"], u, 20, eps, alpha, seed, delta0=d_open, init="zero"
        )
        finish("open_only", more_open["delta"], 20 + int(more_open["n_backprop"]))

        # write_only 40 from random init
        d0 = random_delta(x0, eps, seed)
        wo = run_write_on_delta(
            wrapper, sample["image"], sample["query"], qid, u, d0, 40, eps, alpha, False, None
        )
        finish("write_only", wo["delta"], wo["n_backprop"])

        # fixed schedule 20+20 no keep, always write
        fx = run_write_on_delta(
            wrapper, sample["image"], sample["query"], qid, u, d_open, 20, eps, alpha, False, None
        )
        finish("fixed_schedule", fx["delta"], 20 + int(fx["n_backprop"]), {"switched": True})

        # otw: reuse T2 constrained if written, else continue open 20
        if t2q.get("entered_write") and t2q.get("delta_constrained"):
            d_otw = load_delta(t2q["delta_constrained"], wrapper.device)
            finish("otw", d_otw, 40, {"switched": True, "reused_t2": True})
        else:
            finish("otw", more_open["delta"], 40, {"switched": False, "reused_t2": False})

        # otw_no_keep
        if t2q.get("entered_write") and t2q.get("delta_unconstrained"):
            finish("otw_no_keep", load_delta(t2q["delta_unconstrained"], wrapper.device), 40, {"switched": True})
        else:
            finish("otw_no_keep", more_open["delta"], 40, {"switched": False})

        # random_switch: 34 = 20 existing + 14 more open, then 6 write
        extra_open = run_open_only(
            wrapper,
            sample["image"],
            sample["query"],
            u,
            switch - 20,
            eps,
            alpha,
            seed,
            delta0=d_open,
            init="zero",
        )
        rs = run_write_on_delta(
            wrapper,
            sample["image"],
            sample["query"],
            qid,
            u,
            extra_open["delta"],
            40 - switch,
            eps,
            alpha,
            False,
            None,
        )
        finish("random_switch", rs["delta"], switch + int(rs["n_backprop"]), {"switch_step": switch})

        # otw_no_u: open on u_rand (T1 random_dir 20) + write if ANSWER else 20 more open
        rd_row = t1q["random_dir"]
        if rd_row.get("response_mode") == "ANSWER":
            nu = run_write_on_delta(
                wrapper,
                sample["image"],
                sample["query"],
                qid,
                u,
                d_randu,
                20,
                eps,
                alpha,
                True,
                float(rd_row.get("s_mode") or 0.0),
            )
            finish("otw_no_u", nu["delta"], 40, {"switched": True})
        else:
            more_r = run_open_only(
                wrapper, sample["image"], sample["query"], u_rand, 20, eps, alpha, seed, delta0=d_randu, init="zero"
            )
            finish("otw_no_u", more_r["delta"], 40, {"switched": False})
        torch.cuda.empty_cache()

    counts = {m: row_counts(by_m[m]) for m in methods}
    otw_rhc = counts["otw"]["core_rhc"]
    reasons = []
    if otw_rhc < counts["joint"]["core_rhc"]:
        reasons.append("otw_core_rhc_lt_joint")
    if otw_rhc < counts["open_only"]["core_rhc"]:
        reasons.append("otw_core_rhc_lt_open_only")
    rec = {
        "complete": True,
        "pass": not reasons,
        "kind": "scientific",
        "reasons": reasons,
        "counts": counts,
        "by_method": {m: jsonable(by_m[m]) for m in methods},
        "wall_sec": time.time() - t0w,
        "peak_vram_mb": peak_vram(),
        "random_switch_step": switch,
        "meta": meta,
    }
    save_json(path, rec)
    # csv + md
    cols = ["method", "ANSWER", "REFUSE", "DENY", "core_rhc", "core_safe", "GROUNDED", "FLUENT"]
    csv_lines = [",".join(cols)]
    for m in methods:
        c = counts[m]
        csv_lines.append(",".join(m if k == "method" else str(c[k]) for k in cols))
    (OUT / "OTW_FEASIBILITY_TABLE.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    lines = [
        "# OTW_FEASIBILITY_RESULTS",
        "",
        f"- T3 pass: **{rec['pass']}**",
        f"- reasons: {reasons or 'none'}",
        f"- wall_sec: {rec['wall_sec']:.1f}",
        f"- peak_vram_mb: {rec['peak_vram_mb']}",
        "",
        "## core_RHC / ANSWER / RELATED_SAFE (core_safe) by method (n=3)",
        "",
    ]
    for m in methods:
        c = counts[m]
        lines.append(
            f"- {m}: core_rhc={c['core_rhc']} ANSWER={c['ANSWER']} core_safe={c['core_safe']} "
            f"REFUSE={c['REFUSE']} DENY={c['DENY']}"
        )
    lines.append("")
    (OUT / "OTW_FEASIBILITY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return rec


def write_feasibility_md(t0, t1, t2, t3, frozen, integrate_note: str) -> None:
    lines = [
        "# FEASIBILITY",
        "",
        "Frozen protocol. Gates were written before any GPU numbers.",
        "",
        f"- T0: **{'PASS' if t0.get('pass') else 'FAIL'}** (engineering)",
        f"- T1: **{'PASS' if t1 and t1.get('pass') else ('FAIL' if t1 else 'not run')}** (scientific)",
        f"- T2: **{'PASS' if t2 and t2.get('pass') else ('FAIL' if t2 else 'not run')}** (scientific)",
        f"- T3: **{'PASS' if t3 and t3.get('pass') else ('FAIL' if t3 else 'not run')}** (scientific)",
        "",
        integrate_note,
        "",
        "Do not refit U_safety. Do not start official N1. Do not touch causal.json / attack.json / h91–h130.",
        "",
    ]
    (OUT / "FEASIBILITY.md").write_text("\n".join(lines), encoding="utf-8")
    blob = {
        "t0": t0.get("pass"),
        "t1": None if t1 is None else t1.get("pass"),
        "t2": None if t2 is None else t2.get("pass"),
        "t3": None if t3 is None else t3.get("pass"),
        "counts_t1": None if t1 is None else t1.get("counts"),
        "counts_t3": None if t3 is None else t3.get("counts"),
    }
    save_json(OUT / "feasibility.json", blob)


def integrate_if_passed() -> None:
    entry = ROOT / "scripts" / "run_otw.py"
    entry.write_text(
        """#!/usr/bin/env python3
\"\"\"Resumable OtW entry. Feasibility protocol only; does not start a new scientific scheme.\"\"\"
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("run_otw_feasibility.py")), run_name="__main__")
""",
        encoding="utf-8",
    )
    (OUT / "OTW_INTEGRATION_REPORT.md").write_text(
        "\n".join(
            [
                "# OTW_INTEGRATION_REPORT",
                "",
                "T0–T3 scientific gates passed under the frozen protocol.",
                "",
                "- `src/otw/` remains the integration skeleton: score, open_phase, write_phase, gate, attack, prototypes.",
                "- Unified L∞ clip, dynamic ANSWER gate, mode keep, checkpoint resume, hashes, four-axis eval.",
                "- Entry: `scripts/run_otw.py` (resumes `run_otw_feasibility.py`).",
                "- Ready to integrate further (official u_ref refit) — **not started**.",
                "- Not added: universal Open, attention loss, new model/eps/data, P1, backdoor.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "REPRODUCIBILITY.md").write_text(
        "\n".join(
            [
                "# REPRODUCIBILITY",
                "",
                "1. Confirm `outputs/otw/OTW_FROZEN_CONFIG.json` exists (do not edit gates).",
                "2. `HF_HOME=/root/autodl-tmp/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 P0_QWEN_FORCE_GPU=1 /root/autodl-tmp/conda/envs/vattack/bin/python /root/autodl-tmp/multimodal_attack_project/scripts/run_otw.py`",
                "3. Resume is automatic from `T0_SMOKE.json` / `T1_OPEN_RESULTS.json` / `T2_WRITE_RESULTS.json` / `T3_COMPARISON.json`.",
                "4. CPU tests: `/root/autodl-tmp/conda/envs/vattack/bin/python -m pytest /root/autodl-tmp/multimodal_attack_project/tests/test_otw_feasibility.py -q`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(LOG_PATH, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, logf)
    sys.stderr = Tee(sys.__stderr__, logf)
    print(f"======== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} otw feasibility ========", flush=True)
    frozen = load_frozen()
    acquire_lock()
    try:
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from run_p0_qwen import load_model, seed_all, vram_preflight

        pre = vram_preflight(10.0)
        print({"vram_preflight": pre}, flush=True)
        if not pre.get("ok"):
            write_status("blocked_by_external_condition", pre)
            write_feasibility_md({"pass": False, "reasons": ["vram"]}, None, None, None, frozen, "blocked vram")
            raise SystemExit(2)
        cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="full")
        seed_all(int(frozen["attack"]["seed"]))
        u = load_u_refusal_p0s()
        verify_frozen(frozen, u)
        u_rand = random_unit(int(u.shape[0]), int(frozen["controls"]["u_random_seed"]))
        wrapper = load_model(cfg, setting="native")
        wrapper.set_setting("native")
        wrapper.model.eval()
        samples = load_samples()
        t0 = run_t0(wrapper, samples[0], u, frozen)
        if not t0.get("pass"):
            write_status("blocked_by_external_condition", {"stage": "T0", "reasons": t0.get("reasons")})
            write_feasibility_md(t0, None, None, None, frozen, "T0 engineering fail; do not interpret as science.")
            raise SystemExit(3)
        t1 = run_t1(wrapper, samples, u, u_rand, frozen)
        if not t1.get("pass"):
            write_stop("T1", t1.get("reasons") or [], t1.get("counts") or {}, frozen)
            write_feasibility_md(t0, t1, None, None, frozen, "Stopped at T1 scientific fail. Integration not started.")
            raise SystemExit(4)
        t2 = run_t2(wrapper, samples, u, frozen, t1)
        if not t2.get("pass"):
            write_stop("T2", t2.get("reasons") or [], {"eligible": t2.get("eligible")}, frozen)
            write_feasibility_md(t0, t1, t2, None, frozen, "Stopped at T2 scientific fail. No T3. Integration not started.")
            raise SystemExit(4)
        t3 = run_t3(wrapper, samples, u, u_rand, frozen, t1, t2)
        if not t3.get("pass"):
            write_stop("T3", t3.get("reasons") or [], t3.get("counts") or {}, frozen)
            write_feasibility_md(t0, t1, t2, t3, frozen, "Stopped at T3 scientific fail. Integration not started.")
            raise SystemExit(4)
        integrate_if_passed()
        write_feasibility_md(t0, t1, t2, t3, frozen, "ready to integrate, not started (official u_ref refit deferred).")
        write_status(
            "feasibility_passed_and_integrated",
            {"t3_counts": t3.get("counts"), "md": str(OUT / "OTW_FEASIBILITY_RESULTS.md")},
        )
        print({"done": "feasibility_passed_and_integrated"}, flush=True)
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        write_status("blocked_by_external_condition", {"error": str(exc)})
        raise
    finally:
        release_lock()
        logf.close()


if __name__ == "__main__":
    main()
