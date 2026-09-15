"""Stage 2-Diagnostic CPU tests. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from pathem.protocol import EVAL_SEEDS, OPT_SEEDS, SEALED
from pathem.stage2_diag import (
    d1_from_trajectories,
    d2_incremental,
    d3_lineage,
    rarity_bin,
    run_diagnostic,
    verdict,
    wilson_interval,
)
from pathem.splits import COMMITTOR_TRAIN, COMMITTOR_VAL


def test_wilson_and_bins():
    p, lo, hi = wilson_interval(7, 8)
    assert 0.5 < p < 1.0
    assert 0.0 <= lo <= p <= hi <= 1.0
    assert rarity_bin(7, 8) == "common"
    assert rarity_bin(1, 16) == "rare"
    assert rarity_bin(0, 8) == "unresolved-zero"
    assert rarity_bin(3, 16) == "rare"
    assert rarity_bin(5, 16) == "intermediate"
    assert rarity_bin(1, 4) == "underpowered"
    assert rarity_bin(2, 4) == "common"


def test_d1_marks_common_h56():
    recs = []
    for i in range(16):
        recs.append({"query_id": "h56", "delta_id": "h56:c07:e16:r0", "core_rhc": i < 8, "carrier_id": "c07"})
    out = d1_from_trajectories(recs)
    assert out["by_query"][0]["n_rhc"] == 8
    assert out["by_query"][0]["bin"] in {"common", "intermediate"}


def _prefix(qid, tid, y, margin, n_words, stage, frac):
    return {
        "trajectory_id": tid,
        "query_id": qid,
        "stage": stage,
        "prefix_word_count": n_words,
        "full_word_count": 40,
        "frac": frac,
        "y_rhc": y,
        "refusal_margin": margin,
        "logit_max": 10.0,
        "token_count": 40,
        "delta_id": f"{qid}:c07:e16:r0",
        "carrier_id": "c07",
        "random_seed": 20260,
        "stage_early": int(stage == "early"),
        "stage_mid": int(stage == "mid"),
        "stage_late": int(stage == "late"),
        "stage_pre_eos": int(stage == "pre_eos"),
    }


def test_d2_no_increment_when_only_margin():
    rows = []
    stages = [("early", 0.25, 10), ("mid", 0.5, 20), ("late", 0.75, 30), ("pre_eos", 0.95, 39)]
    rng = np.random.default_rng(0)
    for i, q in enumerate(list(COMMITTOR_TRAIN) + list(COMMITTOR_VAL)):
        for t in range(8):
            y = int(rng.random() < 0.4)
            margin = 4.0 if y == 0 else -4.0
            tid = f"{q}:t{t}"
            for st, fr, nw in stages:
                rows.append(_prefix(q, tid, y, margin, nw, st, fr))
    d2 = d2_incremental(rows)
    assert d2["margin_constant_within_trajectory"] is True
    assert "delta_auroc_full_minus_margin" in d2
    assert set(EVAL_SEEDS).isdisjoint(OPT_SEEDS)


def test_d3_ranking_equals_refusal():
    runs = []
    for method, uniq, hits, clones in (
        ("ams_full", 2, 6, 4),
        ("ams_refusal", 2, 7, 5),
        ("naive", 7, 7, 0),
        ("ams_random", 0, 0, 0),
    ):
        parts = []
        for i in range(8):
            hit = i < hits
            lin = i % max(uniq, 1) if hit else i
            parts.append(
                {
                    "core_rhc": hit,
                    "lineage_id": lin,
                    "output_hash": f"h{lin}-{i}",
                    "token_count": 40 + i,
                }
            )
        runs.append(
            {
                "method": method,
                "query_id": "h56",
                "delta_id": "h56:c07:e16:r0",
                "n_particles": 8,
                "n_hit_particles": hits,
                "n_unique_lineages": uniq,
                "n_clone_hits": clones,
                "tokens": 500,
                "particles": parts if method != "naive" else [],
            }
        )
    d3 = d3_lineage(runs)
    assert d3["ranking_indistinguishable_from_refusal"] is True
    assert d3["naive_beats_ams_full_unique_roots"] is True
    assert d3["failure_is_early_resample_not_ranking"] is False


def test_verdict_terminates_without_increment():
    d1 = {
        "stage1": {"n_rare_nonzero_cells": 0},
        "vlm_naive": {"n_rare_nonzero": 1, "n_common": 2},
    }
    d2 = {"incremental_information": False}
    d3 = {
        "failure_is_early_resample_not_ranking": False,
        "resample_collapse": True,
        "ranking_indistinguishable_from_refusal": True,
    }
    d4 = {"ams_compute_underestimated_if_ignore_scoring": True}
    v = verdict(d1, d2, d3, d4, toy_gate="GO")
    assert v["decision"] == "TERMINATE_PATHEM"
    assert v["stage2b"] == "NO-GO"
    assert v["stage3_forbidden"] is True
    assert v["causes"]["committor_no_incremental_info"] is True
    assert not (set(SEALED) & {"h49", "h56", "h66"})


def test_run_diagnostic_smoke_no_eval_ids():
    traj = [{"query_id": "h49", "delta_id": "h49:c07:e16:r0", "core_rhc": True, "carrier_id": "c07"}]
    prefs = [
        _prefix("h49", "h49:t0", 1, -2.0, 10, "early", 0.2),
        _prefix("h49", "h49:t1", 0, 2.0, 10, "early", 0.2),
        _prefix("h56", "h56:t0", 1, -2.0, 10, "early", 0.2),
        _prefix("h66", "h66:t0", 0, 2.0, 10, "early", 0.2),
        _prefix("h64", "h64:t0", 0, 2.0, 10, "early", 0.2),
        _prefix("h64", "h64:t1", 1, -2.0, 10, "early", 0.2),
        _prefix("h72", "h72:t0", 1, -2.0, 10, "early", 0.2),
        _prefix("h72", "h72:t1", 0, 2.0, 10, "early", 0.2),
    ]
    runs = [
        {
            "method": "naive",
            "query_id": "h49",
            "delta_id": "h49:c07:e16:r0",
            "n_particles": 8,
            "n_hit_particles": 5,
            "n_unique_lineages": 5,
            "n_clone_hits": 0,
            "tokens": 400,
            "particles": [],
        }
    ]
    blob = run_diagnostic(traj, prefs, runs, {"budget": {}, "runs_public": runs}, toy_gate="GO")
    assert blob["verdict"]["stage3_forbidden"] is True
    assert blob["not_asr"] is True
    assert "TERMINATE_PATHEM" in blob["markdown"] or "STAGE2B" in blob["markdown"]
