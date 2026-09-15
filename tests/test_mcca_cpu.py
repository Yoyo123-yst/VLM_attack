import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from mcca.analyze import content_block, decide_p0, mode_block, probe_hacking_block
from mcca.cores import CORE_BANK, OPENING_CONTROLS, core_prefixes, strip_openings
from mcca.content import soft_min_nll
from mcca.metrics import auc_score, auprc_score, smallest_tau_for_precision, spearman_rho
from mcca.project import feasible_content_step, linf_normalize, project_box
from mcca.splits import (
    CR0_TEST_QUERIES,
    CR0_VAL_QUERIES,
    FC_CAL_QUERIES,
    FC_TEST_QUERIES,
    P1_FAIL_CELLS,
    P1_PASS_CELLS,
    assert_query_exclusive,
    freeze_record,
    query_split,
)

FROZEN = ROOT / "outputs" / "mcca" / "MCCA_FROZEN.json"


def test_frozen_exists_and_matches_code_splits():
    assert FROZEN.exists(), "MCCA_FROZEN.json must exist before GPU scoring"
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert cfg["frozen_before_any_mcca_gpu"] is True
    assert cfg["stage_now"] == "P0"
    assert cfg["splits"]["cr0_val_queries"] == list(CR0_VAL_QUERIES)
    assert cfg["splits"]["cr0_test_queries"] == list(CR0_TEST_QUERIES)
    rec = freeze_record()
    assert rec["cr0_val_queries"] == cfg["splits"]["cr0_val_queries"]
    assert "P4_patch_routing" in cfg["locked_until_p3_mvp"]


def test_query_splits_are_exclusive():
    assert_query_exclusive(CR0_VAL_QUERIES, CR0_TEST_QUERIES)
    assert_query_exclusive(FC_CAL_QUERIES, FC_TEST_QUERIES)
    assert query_split("h49", "cr0") == "val"
    assert query_split("h56", "cr0") == "test"
    assert query_split("h01", "fc") == "test"
    cells = set(P1_PASS_CELLS) | set(P1_FAIL_CELLS)
    assert len(cells) == 8


def test_auc_and_tau_precision_rule():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.4, 0.55, 0.6, 0.7, 0.8, 0.9]
    assert auc_score(y, s) == 1.0
    ap = auprc_score(y, s)
    assert ap is not None and ap > 0.9
    sel = smallest_tau_for_precision(y, s, 0.8)
    assert sel["ok"] is True
    # smallest tau with precision >= 0.8: 0.55 predicts 5 positives (1 fp) -> prec=0.8
    assert abs(sel["tau"] - 0.55) < 1e-9
    assert sel["at_tau"]["precision"] >= 0.8 - 1e-12


def test_spearman_monotone():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert abs(spearman_rho(x, y) - 1.0) < 1e-6
    assert abs(spearman_rho(x, list(reversed(y))) + 1.0) < 1e-6


def test_cores_have_no_openings_and_are_short():
    banned_open = ("sure,", "here is", "here are", "of course")
    banned_recipe = ("methamphetamine", "step 2", "step 3", "heat the", "mix the ingredients")
    for qid, bank in CORE_BANK.items():
        assert len(bank) >= 3
        for t in bank:
            low = t.lower()
            assert not any(b in low[:20] for b in banned_open)
            assert not any(b in low for b in banned_recipe)
            assert 20 <= len(t) <= 140
            assert strip_openings(t) == t
        got = core_prefixes(qid, k=3)
        assert len(got) == 3
    for t in OPENING_CONTROLS:
        assert t.lower().startswith(("sure", "of course", "yes"))


def test_soft_min_recovers_hard_min():
    nll = torch.tensor([2.0, 5.0, 9.0])
    hard = float(nll.min())
    soft = float(soft_min_nll(nll, beta=1e-4).item())
    assert abs(soft - hard) < 1e-3
    # larger β underestimates the min (smoother)
    wider = float(soft_min_nll(nll, beta=2.0).item())
    assert wider < hard
    lower = float(soft_min_nll(torch.tensor([0.5, 5.0, 9.0]), beta=0.5).item())
    assert lower < float(soft_min_nll(nll, beta=0.5).item())


def test_projection_nulls_mode_decrease():
    g_c = torch.tensor([1.0, 0.0])
    a = torch.tensor([1.0, 0.0])
    # d0 = -g_c = (-1, 0); a·d0 = -1 < 0 and s already >= tau
    d, kind = feasible_content_step(g_c, a, s_mode=1.0, tau=0.0)
    assert kind == "projected"
    assert abs(float((a * d).sum().item())) < 1e-6


def test_min_correction_when_below_tau():
    g_c = torch.zeros(2)
    a = torch.tensor([2.0, 0.0])
    d, kind = feasible_content_step(g_c, a, s_mode=0.0, tau=1.0, xi=0.0)
    assert kind == "min_correction"
    # d = 0 + (1 - 0 - 0) / 4 * a = 0.25 * a = (0.5, 0)
    assert abs(float(d[0]) - 0.5) < 1e-5
    assert abs(float(d[1])) < 1e-5


def test_unconstrained_when_content_helps_mode():
    g_c = torch.tensor([-1.0, 0.0])  # d0 = (1, 0), aligned with a
    a = torch.tensor([1.0, 0.0])
    d, kind = feasible_content_step(g_c, a, s_mode=2.0, tau=0.0)
    assert kind == "unconstrained"
    assert torch.allclose(d, -g_c)


def test_linf_box():
    x0 = torch.tensor([0.5, 0.5])
    d = torch.tensor([10.0, -0.1])
    step = linf_normalize(d, eta=0.01)
    assert abs(float(step.abs().max()) - 0.01) < 1e-6
    delta = project_box(x0, torch.tensor([1.0, -1.0]), eps=0.06)
    assert float(delta.abs().max()) <= 0.06 + 1e-6
    assert torch.all((x0 + delta >= 0) & (x0 + delta <= 1))


def test_mode_and_content_blocks_cpu():
    rows = []
    for i, q in enumerate(list(CR0_VAL_QUERIES) + list(CR0_TEST_QUERIES)):
        rows.append(
            {
                "query_id": q,
                "carrier_id": "c07" if i % 2 == 0 else "c08",
                "category": "fraud",
                "response_mode": "ANSWER" if i % 2 == 0 else "REFUSE",
                "core_rhc": i % 4 == 0,
                "core_safe_answer": i % 4 == 2,
                "s_mode": 2.0 if i % 2 == 0 else -2.0,
                "L_content": 0.2 if i % 4 == 0 else 1.5,
                "L_opening": 1.0,
            }
        )
    blk = mode_block(rows, CR0_VAL_QUERIES, CR0_TEST_QUERIES, min_precision=0.8)
    assert blk["tau_selection"]["ok"] is True
    assert blk["test"]["auc"] == 1.0
    cb = content_block(rows, CR0_TEST_QUERIES)
    assert cb["n_answer"] >= 1
    hack = probe_hacking_block(rows, rows)
    dec = decide_p0(blk, {**cb, "pass_auc": True, "pass_spearman": True}, hack)
    assert "go_p1" in dec


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all", len(tests))
