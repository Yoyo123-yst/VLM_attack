#!/usr/bin/env python3
"""P0: validate s_mode and core-content loss on frozen CR-0 + independent hiddens.

Does not run P1–P5. Does not refit U. Does not start MCCA PGD.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mcca.analyze import content_block, decide_p0, mode_block, probe_hacking_block  # noqa: E402
from mcca.content import content_losses  # noqa: E402
from mcca.splits import (  # noqa: E402
    CR0_TEST_QUERIES,
    CR0_VAL_QUERIES,
    FC_CAL_QUERIES,
    FC_TEST_QUERIES,
    freeze_record,
    query_split,
)
from otw.attack import pack  # noqa: E402
from otw.prototypes import harmful_prototypes, safe_prototypes  # noqa: E402
from otw.score import LAYER, P0S_SIGN, load_u_refusal_p0s, s_mode  # noqa: E402
from otw.write_phase import tokenize_prefix, write_loss  # noqa: E402
from p0.datautil import load_json, open_image, save_json  # noqa: E402

OUT = ROOT / "outputs" / "mcca" / "p0"
FROZEN_PATH = ROOT / "outputs" / "mcca" / "MCCA_FROZEN.json"
CR0_CAND = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"
FC_CAND = ROOT / "outputs" / "n1r_fast" / "candidates.json"
FC_HIDDEN = ROOT / "outputs" / "n1r_fast" / "hidden"
N0_TRACES = ROOT / "outputs" / "p0_qwen" / "full" / "native" / "traces_n0.json"
T1_PATH = ROOT / "outputs" / "otw" / "T1_OPEN_RESULTS.json"
T3_PATH = ROOT / "outputs" / "otw" / "innovation" / "t3prime_margin_open_write" / "T3PRIME_RESULTS.json"


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
    (OUT / "STATUS.txt").write_text(text + "\n", encoding="utf-8")


def _dot(h: np.ndarray, u: np.ndarray) -> float:
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    return float(np.dot(h, u) * float(P0S_SIGN))


def _mode_from_legacy(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    lab = str(label).upper()
    if lab in {"REF", "FAIL"}:
        return "REFUSE"
    if lab in {"RHC", "RELATED_SAFE", "SAFE_ANSWER"}:
        return "ANSWER"
    if lab in {"THEME_DENIAL", "DENY"}:
        return "DENY"
    return None


def load_frozen() -> Dict[str, Any]:
    cfg = load_json(FROZEN_PATH)
    rec = freeze_record()
    if cfg["splits"]["cr0_val_queries"] != rec["cr0_val_queries"]:
        raise RuntimeError("frozen CR-0 val queries drifted from code")
    if cfg["splits"]["cr0_test_queries"] != rec["cr0_test_queries"]:
        raise RuntimeError("frozen CR-0 test queries drifted from code")
    return cfg


def load_fc_rows(u: np.ndarray) -> List[Dict[str, Any]]:
    cand = load_json(FC_CAND)
    rows: List[Dict[str, Any]] = []
    for rec in cand["records"]:
        npz_path = FC_HIDDEN / f"{rec['record_id']}.npz"
        if not npz_path.exists():
            continue
        hid = np.load(npz_path)["L24:last_user"]
        rows.append(
            {
                "source": "fast_crossed",
                "record_id": rec["record_id"],
                "query_id": rec["query_id"],
                "carrier_id": rec.get("carrier_id"),
                "category": rec.get("category"),
                "split": query_split(rec["query_id"], "fc"),
                "response_mode": rec.get("response_mode"),
                "core_rhc": bool(rec.get("core_rhc")),
                "core_safe_answer": bool(rec.get("core_safe_answer")),
                "s_mode": _dot(hid, u),
                "attack_objective": rec.get("attack_objective") or "refusal_margin_pgd",
            }
        )
    return rows


def load_n0_rows(u: np.ndarray) -> List[Dict[str, Any]]:
    blob = load_json(N0_TRACES)
    rows: List[Dict[str, Any]] = []
    for rec in blob["records"]:
        qid = rec["query_id"]
        split = rec.get("split")
        ch = rec.get("clean_hidden") or {}
        if "L24:last_user" in ch:
            lab = (rec.get("clean") or {}).get("label")
            rows.append(
                {
                    "source": "traces_n0_clean",
                    "query_id": qid,
                    "n0_split": split,
                    "response_mode": _mode_from_legacy(lab),
                    "s_mode": _dot(ch["L24:last_user"], u),
                    "core_rhc": lab == "RHC",
                    "core_safe_answer": str(lab).upper() in {"RELATED_SAFE", "SAFE_ANSWER"},
                }
            )
        for slot in ("jb", "related_safe", "fail"):
            block = rec.get(slot)
            if not isinstance(block, dict):
                continue
            hid = (block.get("hidden") or {}).get("L24:last_user")
            if hid is None:
                continue
            lab = ((block.get("judge") or {}).get("label")) or slot
            rows.append(
                {
                    "source": f"traces_n0_{slot}",
                    "query_id": qid,
                    "n0_split": split,
                    "response_mode": _mode_from_legacy(lab),
                    "s_mode": _dot(hid, u),
                    "core_rhc": str(lab).upper() == "RHC",
                    "core_safe_answer": str(lab).upper() in {"RELATED_SAFE", "SAFE_ANSWER"},
                }
            )
    return rows


def _axis_row(src: str, qid: str, block: Dict[str, Any], method: str) -> Optional[Dict[str, Any]]:
    if not isinstance(block, dict) or block.get("s_mode") is None:
        return None
    return {
        "source": src,
        "query_id": qid,
        "method": method,
        "response_mode": block.get("response_mode"),
        "core_rhc": bool(block.get("core_rhc")),
        "core_safe_answer": bool(block.get("core_safe_answer")),
        "s_mode": float(block["s_mode"]),
    }


def load_smode_opt_rows() -> List[Dict[str, Any]]:
    """Direct s_mode / Open-phase optimization, not ordinary margin-PGD."""
    rows: List[Dict[str, Any]] = []
    if T1_PATH.exists():
        t1 = load_json(T1_PATH)
        for qid, rec in (t1.get("by_query") or {}).items():
            for method in ("open",):
                row = _axis_row("t1_open", qid, rec.get(method) or {}, method)
                if row:
                    rows.append(row)
    if T3_PATH.exists():
        t3 = load_json(T3_PATH)
        for rec in t3.get("records") or []:
            qid = rec.get("query_id")
            # Open-side of OtW only. Write-phase rows are not s_mode hacking.
            row = _axis_row("t3prime_open_decode", qid, rec.get("open_decode") or {}, "open_decode")
            if row:
                rows.append(row)
    return rows


def score_cr0_gpu(frozen: Dict[str, Any], u: np.ndarray) -> List[Dict[str, Any]]:
    from n1.preflight import all_carrier_index, _reconstruct_x01
    from p0.catalog import all_pairs
    from p0_qwen.config import load_cfg
    from run_p0_qwen import load_model, seed_all

    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    score_path = OUT / "cr0_scores.json"
    existing: List[Dict[str, Any]] = []
    have = set()
    if score_path.exists():
        existing = load_json(score_path).get("records") or []
        have = {r["record_id"] for r in existing}

    cand = load_json(CR0_CAND)["records"]
    todo = [r for r in cand if r["record_id"] not in have]
    if not todo:
        print({"cr0_score": "already_complete", "n": len(existing)}, flush=True)
        return existing

    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="full")
    seed_all(2026)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    wrapper.model.eval()
    catalog = {p["id"]: p for p in all_pairs()}
    carriers = all_carrier_index()
    beta = float(frozen["content_loss"]["beta"])
    k = int(frozen["content_loss"]["K"])

    records = list(existing)
    n_total = len(cand)
    for i, rec in enumerate(todo, start=1):
        qid = rec["query_id"]
        cid = rec["carrier_id"]
        img = open_image(carriers[cid]["path"])
        x01 = _reconstruct_x01(carriers[cid]["path"], rec.get("delta_path"))
        if x01 is None:
            raise RuntimeError(f"missing delta for {rec['record_id']}")
        x01 = x01.to(device=wrapper.device, dtype=torch.float32)
        q = catalog[qid]["query"]
        x0, ids, attn, grid = pack(wrapper, img, q)
        _ = x0
        with torch.no_grad():
            score = float(s_mode(wrapper, x01, ids, attn, grid, u).item())
        row = {
            "record_id": rec["record_id"],
            "source": "cr0",
            "query_id": qid,
            "carrier_id": cid,
            "category": rec.get("category"),
            "split": query_split(qid, "cr0"),
            "restart": rec.get("restart"),
            "response_mode": rec.get("response_mode"),
            "core_rhc": bool(rec.get("core_rhc")),
            "core_safe_answer": bool(rec.get("core_safe_answer")),
            "safety": rec.get("safety"),
            "grounding": rec.get("grounding"),
            "quality": rec.get("quality"),
            "theme_denial": bool(rec.get("theme_denial")),
            "chars": rec.get("chars"),
            "s_mode": score,
            "L_content": None,
            "L_opening": None,
            "nll_core": None,
            "nll_opening": None,
            "attack_objective": rec.get("attack_objective"),
        }
        if rec.get("response_mode") == "ANSWER":
            cl = content_losses(wrapper, x01, ids, attn, grid, qid, beta=beta, k=k)
            row["L_content"] = cl["L_content"]
            row["L_opening"] = cl["L_opening"]
            row["nll_core"] = cl["nll_core"]
            row["nll_opening"] = cl["nll_opening"]
        records.append(row)
        save_json(score_path, {"n": len(records), "records": _jsonable(records)})
        write_status(
            f"P0 SCORE {len(records)}/{n_total} last={rec['record_id']} "
            f"mode={row['response_mode']} s_mode={score:.3f} "
            f"Lc={row['L_content']} {now_iso()}"
        )
        print(
            {
                "id": rec["record_id"],
                "progress": f"{len(records)}/{n_total}",
                "batch": f"{i}/{len(todo)}",
                "mode": row["response_mode"],
                "s_mode": score,
                "L_content": row["L_content"],
            },
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def score_cr0_write_loss(cr0: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """P0-B2: OtW contrastive write_loss on already-scored ANSWER rows only."""
    from n1.preflight import all_carrier_index, _reconstruct_x01
    from p0.catalog import all_pairs
    from p0_qwen.config import load_cfg
    from run_p0_qwen import load_model, seed_all

    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    todo = [
        r
        for r in cr0
        if r.get("response_mode") == "ANSWER" and r.get("L_write") is None
    ]
    if not todo:
        return cr0
    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="full")
    seed_all(2026)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    wrapper.model.eval()
    catalog = {p["id"]: p for p in all_pairs()}
    carriers = all_carrier_index()
    n_total = len(todo)
    cand = load_json(CR0_CAND)["records"]
    meta = {r["record_id"]: r for r in cand}
    for i, rec in enumerate(todo, start=1):
        rid = rec["record_id"]
        src = meta[rid]
        qid = rec["query_id"]
        cid = rec["carrier_id"]
        img = open_image(carriers[cid]["path"])
        x01 = _reconstruct_x01(carriers[cid]["path"], src.get("delta_path"))
        if x01 is None:
            raise RuntimeError(f"missing delta for {rid}")
        x01 = x01.to(device=wrapper.device, dtype=torch.float32)
        q = catalog[qid]["query"]
        _, ids, attn, grid = pack(wrapper, img, q)
        h_ids = [tokenize_prefix(wrapper, t) for t in harmful_prototypes(qid)]
        s_ids = [tokenize_prefix(wrapper, t) for t in safe_prototypes(qid)]
        with torch.no_grad():
            lw = float(write_loss(wrapper, x01, ids, attn, grid, h_ids, s_ids).item())
        rec["L_write"] = lw
        save_json(OUT / "cr0_scores.json", {"n": len(cr0), "records": _jsonable(cr0)})
        write_status(f"P0-B2 WRITE {i}/{n_total} last={rid} L_write={lw:.4f} {now_iso()}")
        print({"id": rid, "progress": f"{i}/{n_total}", "L_write": lw}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cr0


def render_report(blob: Dict[str, Any]) -> str:
    d = blob["decision"]
    p0a = blob["p0a"]["cr0_answer_vs_not"]
    p0b = blob["p0b"]["cr0_answer"]
    tau = d.get("tau")
    lines = [
        "# MCCA P0 — signal validation",
        "",
        f"- Written: `{blob['written_at']}`",
        f"- Frozen: `{FROZEN_PATH}`",
        "- GPU attack: **not run**. This ticket only scores stored CR-0 deltas and existing hiddens.",
        f"- Probe: frozen P0-S U_refusal, L{LAYER}, s_mode = +⟨h_pre, u⟩",
        "",
        "## Decision",
        "",
        f"- P0-A (mode score): **{'PASS' if d['p0a_pass'] else 'FAIL'}**",
        f"- P0-B (content loss): **{'PASS' if d['p0b_pass'] else 'FAIL'}**",
        f"- **GO P1: {d['go_p1']}**",
        f"- reasons: {d['reasons'] or 'none'}",
        "",
        "If GO is false, do not implement MCCA-Lite, do not run 32-trajectory P1 as a method paper, and do not add patch routing.",
        "",
        "## Splits (query ID only)",
        "",
        f"- U fit: P0-S discover (not CR-0)",
        f"- CR-0 val (τ only): {list(CR0_VAL_QUERIES)}",
        f"- CR-0 test: {list(CR0_TEST_QUERIES)}",
        f"- Fast-Crossed val/test: {list(FC_CAL_QUERIES)} / {list(FC_TEST_QUERIES)}",
        "",
        "## P0-A Mode score",
        "",
        f"- τ (smallest with val precision ≥ 0.80): **{tau}**  ok={p0a['tau_selection'].get('ok')} constraint_usable={p0a.get('constraint_usable')}",
        f"- Youden τ (diagnostic, val): {p0a.get('youden')}",
        f"- val at precision-τ: {p0a['tau_selection'].get('at_tau')}",
        f"- CR-0 test AUC: {p0a['test'].get('auc')}  AUPRC: {p0a['test'].get('auprc')}",
        f"- CR-0 test at frozen τ: {p0a['test'].get('at_tau')}",
        f"- CR-0 test at Youden τ: {p0a.get('test_at_youden', {}).get('at_tau')}",
        f"- CR-0 all AUC: {p0a['all'].get('auc')}",
        f"- Fast-Crossed test AUC: {(blob['p0a']['fast_crossed'].get('test') or {}).get('auc')}",
        f"- traces_n0 holdout AUC: {(blob['p0a']['n0_holdout'].get('all') or {}).get('auc')}",
        f"- ANSWER vs REFUSE-only (exclude DENY) CR-0 test AUC: {(blob['p0a']['cr0_ref_vs_ans'].get('test') or {}).get('auc')}",
        "",
        "### Reliability (CR-0 test)",
        "",
    ]
    for b in p0a["test"].get("reliability") or []:
        lines.append(
            f"- [{b['lo']:.2f}, {b['hi']:.2f}] n={b['n']} mean_s={b['mean_s']:.2f} P(ANSWER)={b['emp_rate']:.3f}"
        )
    lines += [
        "",
        "### By category / carrier (CR-0 test)",
        "",
        f"- category: {p0a['test'].get('by_category')}",
        f"- carrier: {p0a['test'].get('by_carrier')}",
        "",
        "## Probe hacking",
        "",
        f"- ordinary margin-PGD AUC (CR-0+FC): {blob['probe_hacking'].get('auc_ordinary_attack')}",
        f"- direct s_mode / Open-Write AUC: {blob['probe_hacking'].get('auc_direct_s_mode_opt')}",
        f"- drop: {blob['probe_hacking'].get('drop')}",
        f"- hacked flag: {blob['probe_hacking'].get('hacked')} (needs n_opt≥8)",
        f"- n_ordinary={blob['probe_hacking'].get('n_ordinary')} n_opt={blob['probe_hacking'].get('n_smode_opt')}",
        "",
        "## P0-B Content loss (ANSWER only)",
        "",
        f"- n_answer={p0b.get('n_answer')} n_core_rhc={p0b.get('n_core_rhc')} n_core_safe={p0b.get('n_core_safe')}",
        f"- AUC(-L_content, core_RHC): **{p0b.get('auc_core')}**",
        f"- within-query mean AUC: **{(p0b.get('within_query') or {}).get('mean_within_query_auc')}**  centered={ (p0b.get('within_query') or {}).get('query_centered_auc') }",
        f"- Spearman vs 3-level content score: **{p0b.get('spearman_content_score')}**",
        f"- opening-control AUC: {p0b.get('auc_opening_control')}",
        f"- OtW contrastive write_loss AUC: {p0b.get('auc_write_contrastive')}  within-q={ (p0b.get('within_query_write') or {}).get('mean_within_query_auc') }",
        f"- by category: {p0b.get('by_category')}",
        f"- within-query: {p0b.get('within_query')}",
        f"- L_content | RHC: {p0b.get('L_content_rhc')}",
        f"- L_content | not RHC: {p0b.get('L_content_not_rhc')}",
        "",
        "## What this allows next",
        "",
    ]
    if d["go_p1"]:
        lines += [
            "- P1 conflict diagnosis is allowed on the 8 frozen cells × 4 restarts.",
            "- Still no patch routing. Still no MCCA-Lite until P2 one-step projection works.",
        ]
    else:
        lines += [
            "- **Stop before P1.** Do not spend GPU on 32 conflict trajectories until a content loss ranks ANSWER content.",
            "- P0-A sensor is usable: CR-0 test AUC 0.88 (0.92 if DENY is dropped). Do **not** use the precision-0.80 τ as an MCCA constraint; it has recall ~0.08. If a mode constraint is used later, start from Youden τ on val (test precision 0.79 / recall 0.71).",
            "- P0-B failed as a ranker of stored CR-0 answers: core NLL, opening NLL, and OtW contrastive write_loss are all near chance within query. Query-centered AUC ≈ 0.51, so this is not a between-query scale artifact.",
            "- Separate fact from T2 (already-open Write): unconstrained contrastive write_loss ranked RHC at AUC 0.94 *after being optimized*, n=8. A loss can move content without ranking a static same-budget restart set. That does **not** license P1 on CR-0 with the current cores.",
            "- Next cheap revision is a new content target (distinctive RHC vs RELATED_SAFE spans, still no Sure/Here-is), then re-score the same 85 ANSWER deltas. Only if within-query AUC ≥ 0.70 should P1 start.",
        ]
    lines += [
        "",
        "P4 patch routing and P5 multi-model stay locked.",
        "",
    ]
    return "\n".join(lines) + "\n"


def analyze(frozen: Dict[str, Any], cr0: List[Dict[str, Any]], u: np.ndarray) -> Dict[str, Any]:
    fc = load_fc_rows(u)
    n0 = load_n0_rows(u)
    n0_hold = [r for r in n0 if r.get("n0_split") == "holdout"]
    n0_disc = [r for r in n0 if r.get("n0_split") == "discover"]
    smode_opt = load_smode_opt_rows()
    ordinary = list(cr0) + list(fc)

    p0a_cr0 = mode_block(cr0, CR0_VAL_QUERIES, CR0_TEST_QUERIES, min_precision=0.8, include_deny_as_neg=True)
    p0a_cr0_bin = mode_block(cr0, CR0_VAL_QUERIES, CR0_TEST_QUERIES, min_precision=0.8, include_deny_as_neg=False)
    p0a_fc = mode_block(fc, FC_CAL_QUERIES, FC_TEST_QUERIES, min_precision=0.8, include_deny_as_neg=True)
    # n0 holdout: tau from holdout queries not in FC test, same as T1-D spirit
    n0_val_q = sorted({r["query_id"] for r in n0_hold if r["query_id"] not in FC_TEST_QUERIES})
    n0_test_q = sorted({r["query_id"] for r in n0_hold if r["query_id"] in FC_TEST_QUERIES}) or sorted(
        {r["query_id"] for r in n0_hold}
    )
    p0a_n0 = mode_block(n0_hold, n0_val_q, n0_test_q, min_precision=0.8, include_deny_as_neg=True)
    p0b_all = content_block(cr0)
    p0b_test = content_block(cr0, CR0_TEST_QUERIES)
    p0b_val = content_block(cr0, CR0_VAL_QUERIES)
    hack = probe_hacking_block(ordinary, smode_opt)
    # P0-B gate uses all CR-0 ANSWER (cores were not fit on labels), plus test slice.
    p0b_gate = dict(p0b_all)
    if p0b_test.get("n_answer"):
        # still require the test slice to agree in sign if it has both classes
        if p0b_test.get("auc_core") is not None and p0b_test["auc_core"] < 0.55:
            p0b_gate["pass_auc"] = False
            p0b_gate["pass_spearman"] = bool((p0b_test.get("spearman_content_score") or 0) >= 0.30)
            p0b_gate["test_disagree"] = True
    decision = decide_p0(p0a_cr0, p0b_gate, hack)
    blob = {
        "schema": "mcca_p0_results_v1",
        "written_at": now_iso(),
        "frozen": str(FROZEN_PATH),
        "layer": LAYER,
        "n_cr0": len(cr0),
        "n_fc": len(fc),
        "n_n0_holdout": len(n0_hold),
        "n_n0_discover_circular": len(n0_disc),
        "cr0_mode_counts": dict(Counter(r.get("response_mode") for r in cr0)),
        "p0a": {
            "cr0_answer_vs_not": p0a_cr0,
            "cr0_ref_vs_ans": p0a_cr0_bin,
            "fast_crossed": p0a_fc,
            "n0_holdout": p0a_n0,
        },
        "p0b": {
            "cr0_answer": p0b_all,
            "cr0_val_answer": p0b_val,
            "cr0_test_answer": p0b_test,
        },
        "probe_hacking": hack,
        "decision": decision,
        "p1_locked": not bool(decision["go_p1"]),
    }
    return blob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true", help="FC/n0/T1 analysis; skip CR-0 GPU scoring")
    parser.add_argument("--analyze-only", action="store_true", help="reuse cr0_scores.json")
    parser.add_argument("--p0b2", action="store_true", help="score OtW contrastive write_loss on CR-0 ANSWER rows")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen()
    u = load_u_refusal_p0s()
    write_status(f"P0 START {now_iso()} cpu_only={args.cpu_only} p0b2={args.p0b2}")

    if args.cpu_only:
        cr0: List[Dict[str, Any]] = []
        if (OUT / "cr0_scores.json").exists():
            cr0 = load_json(OUT / "cr0_scores.json").get("records") or []
    elif args.analyze_only and not args.p0b2:
        cr0 = load_json(OUT / "cr0_scores.json")["records"]
    else:
        if args.p0b2 or args.analyze_only:
            cr0 = load_json(OUT / "cr0_scores.json")["records"]
        else:
            cr0 = score_cr0_gpu(frozen, u)
        if args.p0b2:
            cr0 = score_cr0_write_loss(cr0)

    blob = analyze(frozen, cr0, u)
    save_json(OUT / "P0_RESULTS.json", _jsonable(blob))
    md = render_report(blob)
    (OUT / "P0_REPORT.md").write_text(md, encoding="utf-8")
    write_status(f"P0 DONE go_p1={blob['decision']['go_p1']} {now_iso()}")
    print({"go_p1": blob["decision"]["go_p1"], "reasons": blob["decision"]["reasons"]}, flush=True)
    if (not args.cpu_only) and (not args.analyze_only) and cr0 and (not blob["decision"]["go_p1"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
