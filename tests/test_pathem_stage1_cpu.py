"""Stage 1 CPU tests: prefixes, splits, smoke spec. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from pathem.collect import collect_grid_spec, select_stage1_cells, smoke_spec
from pathem.committor import FEATURE_KEYS, compare_margin_vs_full, row_x
from pathem.cr0_io import load_cr0_records
from pathem.prefixes import assert_group_split, prefix_cuts, prefix_rows_from_text
from pathem.privacy import console_safe
from pathem.protocol import EVAL_SEEDS, OPT_SEEDS, SEALED
from pathem.splits import COMMITTOR_TRAIN, COMMITTOR_VAL, assert_no_leak


def test_prefix_cuts_order():
    c = prefix_cuts(8)
    assert c["early"] <= c["mid"] <= c["late"] <= c["pre_eos"] <= 8
    assert prefix_cuts(0)["early"] == 0


def test_prefix_label_is_terminal_not_local():
    text = "I cannot help. " + "word " * 40
    rows = prefix_rows_from_text(text, terminal_label="RHC", core_rhc=True, trajectory_id="t1", query_id="h56")
    assert len(rows) == 4
    assert {r["stage"] for r in rows} == {"early", "mid", "late", "pre_eos"}
    assert all(r["y_rhc"] == 1 for r in rows)
    assert all(r["terminal_label"] == "RHC" for r in rows)
    assert "text" not in rows[0]
    assert rows[0]["prefix_hash"]


def test_group_split_blocks_trajectory_leak():
    rows = [
        {"trajectory_id": "t1", "query_id": "h56"},
        {"trajectory_id": "t1", "query_id": "h64"},
    ]
    try:
        assert_group_split(rows, ["h56"], ["h64"])
    except RuntimeError:
        return
    raise AssertionError("same trajectory in two queries should fail")


def test_committor_queries_disjoint():
    assert_no_leak()
    assert set(COMMITTOR_TRAIN).isdisjoint(COMMITTOR_VAL)
    assert set(COMMITTOR_TRAIN).isdisjoint(SEALED)


def test_smoke_spec_uses_opt_not_eval():
    spec = smoke_spec()
    assert spec["query_id"] == "h56"
    assert spec["query_id"] not in SEALED
    assert set(spec["seeds"]).issubset(OPT_SEEDS)
    assert set(spec["seeds"]).isdisjoint(EVAL_SEEDS)
    assert spec["do_sample"] is True


def test_smoke_cell_exists_in_cr0():
    rows = [r for r in load_cr0_records() if r["query_id"] == "h56" and r["carrier_id"] == "c07"]
    assert rows
    pub = console_safe({"query_id": "h56", "text": "secret", "status": "ok"})
    assert "secret" not in str(pub)


def test_collect_grid_is_not_96_and_skips_eval_seeds():
    spec = collect_grid_spec()
    cells = select_stage1_cells(n_restart=2)
    assert spec["n_cells"] == 20
    assert spec["n_traj_planned"] == 80
    assert "h53" not in spec["query_ids"]
    assert set(spec["seeds"]).issubset(OPT_SEEDS)
    assert set(spec["seeds"]).isdisjoint(EVAL_SEEDS)
    assert set(spec["query_ids"]) == set(COMMITTOR_TRAIN) | set(COMMITTOR_VAL)
    qids = {c["query_id"] for c in cells}
    assert qids.isdisjoint(SEALED)
    assert all(c["query_id"] != "h53" for c in cells)


def test_committor_fit_cpu_synthetic():
    rows = []
    for i, q in enumerate(list(COMMITTOR_TRAIN) + list(COMMITTOR_VAL)):
        for t in range(4):
            y = int((i + t) % 2 == 0)
            margin = -3.0 if y else 3.0
            for stage, frac in (("early", 0.25), ("mid", 0.5), ("late", 0.75), ("pre_eos", 0.99)):
                rows.append(
                    {
                        "trajectory_id": f"{q}:t{t}",
                        "query_id": q,
                        "stage": stage,
                        "y_rhc": y,
                        "refusal_margin": margin,
                        "logit_max": 10.0,
                        "frac": frac,
                        "prefix_word_count": int(40 * frac),
                        "full_word_count": 40,
                        "token_count": 40,
                        "stage_early": int(stage == "early"),
                        "stage_mid": int(stage == "mid"),
                        "stage_late": int(stage == "late"),
                        "stage_pre_eos": int(stage == "pre_eos"),
                    }
                )
    assert len(row_x(rows[0])) == len(FEATURE_KEYS)
    out = compare_margin_vs_full(rows)
    assert out["full"]["val"]["auroc"] > 0.5
    assert out["full"]["gate"] in {"GO", "STOP"}
