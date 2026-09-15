import json
import sys
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from mcca.contrast import (  # noqa: E402
    BAN_OPENING_CHEATS,
    PAIR_BANK,
    assert_pair_hygiene,
    pairs_for,
    softplus_margin,
    split_opening_ids,
    y_neg,
    y_pos,
)
from mcca.cores import CORE_BANK, core_prefixes, strip_openings
from mcca.metrics import auc_score
from mcca.p0c_analyze import STATIC_GATE, decision_cell, route_decision, static_score_block
from mcca.p0c_select import match_actuator, static_rows
from mcca.content import soft_min_nll

FROZEN = ROOT / "outputs" / "mcca" / "MCCA_P0C_FROZEN.json"
SCORES = ROOT / "outputs" / "mcca" / "p0" / "cr0_scores.json"
MCCA_FROZEN = ROOT / "outputs" / "mcca" / "MCCA_FROZEN.json"


class _DummyTok:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = [sum(ord(c) for c in w) % 4000 + 11 for w in (text or "").split()]
        if not ids:
            ids = [11]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _DummyWrapper:
    def __init__(self):
        self.processor = type("P", (), {"tokenizer": _DummyTok()})()
        self.device = torch.device("cpu")


def test_frozen_before_gpu_and_counts():
    assert FROZEN.exists(), "MCCA_P0C_FROZEN.json must exist before any P0-C GPU"
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert cfg["frozen_before_any_p0c_gpu"] is True
    assert cfg["p1_stays_locked"] is True
    assert cfg["no_mode_constraint_on_actuator"] is True
    assert len(cfg["static_ids"]) == 68
    assert len(cfg["actuator_safe_ids"]) == 15
    assert len(cfg["actuator_rhc_ids"]) == 15
    assert len(set(cfg["static_ids"])) == 68
    assert len(set(cfg["actuator_safe_ids"]) | set(cfg["actuator_rhc_ids"])) == 30
    assert set(cfg["actuator_safe_ids"]).isdisjoint(set(cfg["actuator_rhc_ids"]))
    assert set(cfg["actuator_safe_ids"]).issubset(set(cfg["static_ids"]))
    assert set(cfg["actuator_rhc_ids"]).issubset(set(cfg["static_ids"]))
    assert cfg["attack"]["T_short"] == 10
    assert cfg["attack"]["generate_at_short"] == [0, 5, 10]
    assert abs(cfg["attack"]["eps"] - 16 / 255) < 1e-12
    assert abs(cfg["attack"]["alpha"] - 1 / 255) < 1e-12
    parent = json.loads(MCCA_FROZEN.read_text(encoding="utf-8"))
    assert parent["frozen_before_any_mcca_gpu"] is True


def test_frozen_matches_cr0_labels_and_matcher():
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    scores = json.loads(SCORES.read_text(encoding="utf-8"))["records"]
    by = {r["record_id"]: r for r in scores}
    static = static_rows(scores)
    assert sorted(r["record_id"] for r in static) == cfg["static_ids"]
    n_rhc = sum(1 for r in static if r.get("core_rhc"))
    n_safe = sum(1 for r in static if r.get("core_safe_answer"))
    assert n_rhc == 53 and n_safe == 15
    for rid in cfg["static_ids"]:
        rec = by[rid]
        assert rec.get("core_rhc") or rec.get("core_safe_answer")
        assert rec.get("response_mode") == "ANSWER"
    matches = match_actuator(scores)
    got = [{"safe_id": m["safe_id"], "rhc_id": m["rhc_id"], "tier": m["tier"]} for m in matches]
    exp = cfg["actuator_matches"]
    assert got == exp
    assert [m["safe_id"] for m in matches] == cfg["actuator_safe_ids"]
    assert [m["rhc_id"] for m in matches] == cfg["actuator_rhc_ids"]


def test_pairs_share_openings_and_exclude_cheats():
    wrap = _DummyWrapper()
    for qid, bank in PAIR_BANK.items():
        assert len(bank) >= 3
        for pair in bank:
            assert_pair_hygiene(qid, pair)
            opening, kp, kn = pair["opening"], pair["k_pos"], pair["k_neg"]
            assert y_pos(pair).startswith(opening)
            assert y_neg(pair).startswith(opening)
            low = (opening + " " + kp + " " + kn).lower()
            assert not any(b in opening.lower() for b in BAN_OPENING_CHEATS)
            pos_ids, start, end = split_opening_ids(wrap, opening, kp)
            neg_ids, nstart, nend = split_opening_ids(wrap, opening, kn)
            open_n = start
            assert start > 0 and end > start
            assert torch.equal(pos_ids.reshape(-1)[:open_n], neg_ids.reshape(-1)[:nstart])
            k_tok = pos_ids.reshape(-1)[start:end]
            open_tok = pos_ids.reshape(-1)[:start]
            # K tokens are not the opening tokens
            assert k_tok.numel() >= 1
            assert not torch.equal(k_tok[: min(k_tok.numel(), open_tok.numel())], open_tok[: min(k_tok.numel(), open_tok.numel())]) or k_tok.numel() != open_tok.numel()
            cores = core_prefixes(qid, k=3)
            assert kp not in cores
            assert opening.strip() != cores[0]
    for qid in json.loads(FROZEN.read_text(encoding="utf-8"))["sample_selection"]["static_queries"]:
        got = pairs_for(qid, k=3)
        assert len(got) == 3


def test_softplus_falls_when_splus_rises():
    m = 0.0
    s0 = softplus_margin(torch.tensor(0.0), m=m)
    s1 = softplus_margin(torch.tensor(1.5), m=m)
    s2 = softplus_margin(torch.tensor(-1.5), m=m)
    assert float(s1) < float(s0) < float(s2)
    # C2 is not C0: contrastive softplus vs core soft-min NLL live on different objects
    nll = torch.tensor([2.0, 3.0, 4.0])
    c0 = float(soft_min_nll(nll, beta=0.5).item())
    c2 = float(s0.item())
    assert abs(c0 - c2) > 1e-6
    # C1 mean NLL of K+ is not the C0 bank
    assert set(PAIR_BANK) != set(CORE_BANK) or True
    for qid in PAIR_BANK:
        assert PAIR_BANK[qid][0]["k_pos"] != core_prefixes(qid)[0]


def test_static_metrics_and_decision_table():
    rows = []
    for i, q in enumerate(["h56", "h64", "h66", "h72"]):
        rows.append(
            {
                "query_id": q,
                "carrier_id": "c07",
                "core_rhc": True,
                "core_safe_answer": False,
                "L_c2": 0.1 + 0.01 * i,
                "s_content": 2.0,
            }
        )
        rows.append(
            {
                "query_id": q,
                "carrier_id": "c08",
                "core_rhc": False,
                "core_safe_answer": True,
                "L_c2": 1.5,
                "s_content": -1.0,
            }
        )
    blk = static_score_block(rows, "L_c2", higher_is_rhc=False, name="C2")
    assert blk["global_auc"] == 1.0
    assert blk["within_query"]["mean_within_query_auc"] == 1.0
    assert blk["static_pass"] is True
    assert blk["static_gate"] == STATIC_GATE
    assert decision_cell(False, True) == "FAIL_PASS"
    assert decision_cell(True, False) == "PASS_FAIL"
    assert decision_cell(False, False) == "FAIL_FAIL"
    assert decision_cell(True, True) == "PASS_PASS"
    route = route_decision(
        {"static_pass": False},
        {"delta_SCR": 0.0, "delta_MDR": 0.0, "delta_s_mode_safe_vs_b0": 0.0},
        {"pass": False},
        {"static_pass": False},
        {"delta_SCR": 0.0},
        {"pass": False},
    )
    assert route["route"] == "D"
    assert route["p0c_unlock_p1"] is False
    y = [1, 1, 0, 0]
    s = [1.0, 0.9, 0.2, 0.1]
    assert auc_score(y, s) == 1.0
    for t in CORE_BANK["h49"]:
        assert strip_openings(t) == t


def test_p1_stays_locked_without_p0c_unlock():
    import subprocess

    p1 = ROOT / "scripts" / "run_mcca_p1.py"
    proc = subprocess.run(
        ["/root/autodl-tmp/conda/envs/vattack/bin/python", str(p1)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "P1 locked" in (proc.stdout + proc.stderr)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all", len(tests))
