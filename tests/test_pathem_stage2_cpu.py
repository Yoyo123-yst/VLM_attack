"""Stage 2 CPU tests. No GPU, no VLM generate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from pathem.collect import select_stage1_cells
from pathem.protocol import EVAL_SEEDS, OPT_SEEDS, SEALED
from pathem.scorer import LinearScorer, live_features, stage_from_frac
from pathem.splits import COMMITTOR_TRAIN
from pathem.vlm_ams import gate_from_cell_rows, stage2_cells


def test_live_features_stage_bins():
    assert stage_from_frac(0.1) == "early"
    assert stage_from_frac(0.5) == "mid"
    f = live_features(-1.0, 8.0, 12, 96)
    assert f["stage_early"] == 1.0
    assert f["prefix_word_count"] == 12
    assert "text" not in f


def test_linear_scorer_monotone_margin():
    sc = LinearScorer(
        keys=["refusal_margin"],
        coef=[-1.0],
        intercept=0.0,
        mean=[0.0],
        scale=[1.0],
        name="m",
    )
    lo = sc.predict_proba({"refusal_margin": 3.0})
    hi = sc.predict_proba({"refusal_margin": -3.0})
    assert hi > lo


def test_stage2_cells_are_train_not_eval():
    cells = select_stage1_cells(n_restart=2)
    picked = stage2_cells(cells)
    assert 1 <= len(picked) <= 3
    assert {r["query_id"] for r in picked}.issubset(set(COMMITTOR_TRAIN))
    assert {r["query_id"] for r in picked}.isdisjoint(SEALED)
    assert all(r["query_id"] != "h53" for r in picked)


def test_gate_stops_if_only_random():
    rows = [
        {"method": "ams_full", "n_unique_lineages": 3},
        {"method": "ams_random", "n_unique_lineages": 1},
        {"method": "ams_refusal", "n_unique_lineages": 4},
        {"method": "naive", "n_unique_lineages": 1},
    ]
    g = gate_from_cell_rows(rows)
    assert g["only_beats_random_not_refusal"] is True
    assert g["gate"] == "STOP"
    assert set(OPT_SEEDS).isdisjoint(EVAL_SEEDS)


def test_gate_go_when_beats_naive_and_random():
    rows = [
        {"method": "ams_full", "n_unique_lineages": 4},
        {"method": "ams_random", "n_unique_lineages": 1},
        {"method": "ams_refusal", "n_unique_lineages": 2},
        {"method": "naive", "n_unique_lineages": 1},
    ]
    g = gate_from_cell_rows(rows)
    assert g["gate"] == "GO"
    assert g["beats_refusal"] is True
    _ = np.array([1.0])
