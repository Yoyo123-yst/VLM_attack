"""Sampled-trajectory collection on frozen CR-0 deltas. No new PGD."""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from n1.preflight import all_carrier_index
from p0.catalog import all_pairs
from p0.datautil import open_image
from p0.model import COMPLY_WORDS, REFUSE_WORDS, margin_from_logits, token_id_list

from .budget import BudgetLedger
from .cr0_io import load_cr0_records, load_delta
from .labels import label_text
from .prefixes import PREFIX_STAGES, prefix_rows_from_text
from .privacy import console_safe, text_hash
from .projection import in_box
from .protocol import (
    DECODE,
    EPS,
    EVAL_SEEDS,
    OPT_SEEDS,
    PILOT_CARRIERS,
    PILOT_QUERIES,
    STORE_RAW_OUTPUTS,
    assert_query_allowed,
    assert_seed_split,
)
from .records import TrajectoryRecord, serialize_record
from .splits import COMMITTOR_TRAIN, COMMITTOR_VAL, assert_no_leak

CATALOG = None


def catalog() -> dict[str, Any]:
    global CATALOG
    if CATALOG is None:
        CATALOG = {p["id"]: p for p in all_pairs()}
    return CATALOG


def reconstruct_x01(wrapper, carrier_id: str, delta_path: str) -> torch.Tensor:
    cidx = all_carrier_index()
    if carrier_id not in cidx:
        raise RuntimeError(f"missing carrier {carrier_id}")
    img = open_image(cidx[carrier_id]["path"])
    x0 = wrapper.image_to_x01(img)
    delta = load_delta(delta_path, device=x0.device).to(device=x0.device, dtype=x0.dtype)
    x = torch.clamp(x0 + delta, 0.0, 1.0)
    if not in_box(x0, x, EPS):
        raise RuntimeError(f"delta escapes eps box: {delta_path}")
    return x


def first_token_margin(wrapper, img, query: str, x01: torch.Tensor) -> dict[str, float]:
    packed = wrapper.encode(img, query)
    pv, grid = wrapper.patchify(x01)
    logits = wrapper.first_token_logits(
        pv, packed["input_ids"], packed.get("attention_mask"), grid
    )
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    margin = margin_from_logits(logits, refuse_ids, comply_ids)
    return {
        "refusal_margin": float(margin.detach().item()),
        "logit_max": float(logits.detach().float().max().item()),
    }


def collect_one(
    wrapper,
    cr0_rec: dict[str, Any],
    seed: int,
    *,
    do_sample: bool = True,
    ledger: Optional[BudgetLedger] = None,
    store_raw: bool = STORE_RAW_OUTPUTS,
) -> dict[str, Any]:
    qid = cr0_rec["query_id"]
    assert_query_allowed(qid, PILOT_QUERIES)
    if seed in EVAL_SEEDS:
        raise RuntimeError("evaluation seed used during collection")
    item = catalog()[qid]
    query = item["query"]
    keywords = item.get("topic_keywords") or []
    cidx = all_carrier_index()
    img = open_image(cidx[cr0_rec["carrier_id"]]["path"])
    x01 = reconstruct_x01(wrapper, cr0_rec["carrier_id"], cr0_rec["delta_path"])
    t0 = time.time()
    if do_sample:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    text = wrapper.generate(
        img,
        query,
        x01=x01,
        do_sample=bool(do_sample),
        temperature=float(DECODE["temperature"]),
        top_p=float(DECODE["top_p"]),
        seed=None,
        max_new_tokens=int(DECODE["max_new_tokens"]),
    )
    wall = time.time() - t0
    lab = label_text(text, keywords, query)
    if store_raw:
        raise RuntimeError("store_raw_outputs must stay false")
    tid = f"{cr0_rec.get('record_id')}:s{seed}:{'stoch' if do_sample else 'greedy'}"
    rec = TrajectoryRecord(
        run_id="stage1",
        method="naive_sample" if do_sample else "greedy",
        query_id=qid,
        carrier_id=cr0_rec["carrier_id"],
        delta_id=cr0_rec.get("record_id"),
        random_seed=int(seed),
        trajectory_id=tid,
        token_count=len((text or "").split()),
        terminal_label=lab["terminal_label"],
        judge_scores={
            "legacy_label": lab.get("legacy_label"),
            "relevance": lab.get("relevance"),
            "judge_id": lab.get("judge_id"),
        },
        output_hash=text_hash(text or ""),
        wall_time=wall,
        status="ok",
        core_rhc=bool(lab["core_rhc"]),
        response_mode=lab["response_mode"],
        safety=lab["safety"],
    )
    prefixes = prefix_rows_from_text(
        text, lab["terminal_label"], bool(lab["core_rhc"]), tid, qid
    )
    margin = first_token_margin(wrapper, img, query, x01)
    for row in prefixes:
        row.update(margin)
        row["token_count"] = int(rec.token_count)
        row["delta_id"] = cr0_rec.get("record_id")
        row["carrier_id"] = cr0_rec["carrier_id"]
        row["random_seed"] = int(seed)
        for stage in PREFIX_STAGES:
            row[f"stage_{stage}"] = int(row["stage"] == stage)
    if ledger is not None:
        ledger.add_generation(tokens=rec.token_count, wall_s=wall)
        ledger.add_judge(1)
        ledger.add_forward(1)
    public = serialize_record(rec)
    public.update(
        {
            "grounding": lab["grounding"],
            "quality": lab["quality"],
            "core_safe_answer": lab["core_safe_answer"],
            "cr0_greedy_mode": cr0_rec.get("response_mode"),
            "cr0_greedy_rhc": bool(cr0_rec.get("core_rhc")),
            **margin,
        }
    )
    return {
        "record": console_safe(public),
        "full": public,
        "prefixes": prefixes,
        "n_prefixes": len(prefixes),
    }


def smoke_spec() -> dict[str, Any]:
    """1 query × 1 carrier × 2 opt seeds. Not a paper protocol."""
    assert_seed_split()
    assert_no_leak()
    return {
        "query_id": "h56",
        "carrier_id": "c07",
        "prefer_greedy_rhc": True,
        "seeds": list(OPT_SEEDS[:2]),
        "committor_train": list(COMMITTOR_TRAIN),
        "committor_val": list(COMMITTOR_VAL),
        "do_sample": True,
    }


def select_stage1_cells(n_restart: int = 2) -> list[dict[str, Any]]:
    """Query-disjoint committor cells. Prefer one greedy RHC and one non-RHC."""
    assert_no_leak()
    queries = tuple(COMMITTOR_TRAIN) + tuple(COMMITTOR_VAL)
    rows = load_cr0_records(queries, PILOT_CARRIERS)
    picked: list[dict[str, Any]] = []
    for qid in queries:
        for cid in PILOT_CARRIERS:
            cell = [
                r
                for r in rows
                if r.get("query_id") == qid and r.get("carrier_id") == cid
            ]
            cell = sorted(cell, key=lambda r: int(r.get("restart") or 0))
            rhc = [r for r in cell if r.get("core_rhc")]
            neg = [r for r in cell if not r.get("core_rhc")]
            chosen = (rhc[:1] + neg[:1] + rhc[1:] + neg[1:])[: int(n_restart)]
            if not chosen:
                chosen = cell[: int(n_restart)]
            picked.extend(chosen)
    return picked


def collect_grid_spec() -> dict[str, Any]:
    """5 queries × 2 carriers × 2 restarts × 4 opt seeds. Not the 96-cell grid."""
    assert_seed_split()
    assert_no_leak()
    cells = select_stage1_cells(n_restart=2)
    seeds = list(OPT_SEEDS[:4])
    if set(seeds) & set(EVAL_SEEDS):
        raise RuntimeError("evaluation seeds leaked into collect grid")
    return {
        "query_ids": list(COMMITTOR_TRAIN) + list(COMMITTOR_VAL),
        "carrier_ids": list(PILOT_CARRIERS),
        "n_cells": len(cells),
        "cell_ids": [c.get("record_id") for c in cells],
        "seeds": seeds,
        "n_traj_planned": len(cells) * len(seeds),
        "do_sample": True,
        "exclude_h53": True,
    }
