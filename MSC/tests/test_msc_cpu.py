#!/usr/bin/env python3
"""CPU tests for MSC phrase banks, losses, reporting, and frozen verdicts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "MSC" / "src"))
sys.path.insert(0, str(ROOT / "CG-VSF" / "src"))

import torch

from msc.objective import static_joint_loss, switched_phrase_loss
from msc.protocol import assert_query_unsealed, load_frozen
from msc.phrases import first_token_set, jaccard, public_tokensets
from msc.reporting import BANNED_KEYS, append_jsonl, contains_banned, read_jsonl, strip_banned
from msc.states import safety_state, toward_follow
from msc.verdict import p00_verdict, p01_verdict


def _lab(**kw):
    rec = {
        "core_rhc": False,
        "core_safe_answer": False,
        "quality": "FLUENT",
        "response_mode": "ANSWER",
        "terminal_label": "OTHER",
    }
    rec.update(kw)
    return rec


HARD = [
    "h53:c07",
    "h53:c08",
    "h64:c07",
    "h64:c08",
    "h66:c07",
    "h66:c08",
    "h72:c07",
    "h72:c08",
]


def _sum(cell, method, seed, rhc, *, backward=80, n_eval_rhc=None, after=None):
    ok = bool(rhc)
    return {
        "cell_id": cell,
        "method": method,
        "round": "summary",
        "opt_seed": int(seed),
        "core_rhc": ok,
        "n_eval_rhc": int(n_eval_rhc if n_eval_rhc is not None else (5 if ok else 0)),
        "n_eval": 5,
        "backward_used": backward,
        "after_mode": after or ("FOLLOW" if ok else "RELATED_SAFE"),
    }


def _round(cell, method, seed, after, rhc=False):
    return {
        "cell_id": cell,
        "method": method,
        "round": 0,
        "opt_seed": int(seed),
        "core_rhc": bool(rhc),
        "after_mode": after,
    }


class FakeTok:
    def __init__(self, table):
        self.table = table

    def __call__(self, text, add_special_tokens=False):
        if text not in self.table:
            raise KeyError(text)
        return {"input_ids": list(self.table[text])}


class TestStates(unittest.TestCase):
    def test_refuse(self):
        self.assertEqual(safety_state(_lab(response_mode="REFUSE", terminal_label="REFUSE")), "REFUSE")

    def test_related_safe(self):
        self.assertEqual(safety_state(_lab(core_safe_answer=True, terminal_label="RELATED_SAFE")), "RELATED_SAFE")

    def test_rhc_is_follow_not_success_alias(self):
        self.assertEqual(safety_state(_lab(core_rhc=True, terminal_label="RHC")), "FOLLOW")

    def test_garbage_is_invalid(self):
        self.assertEqual(safety_state(_lab(quality="GARBAGE", terminal_label="INVALID")), "INVALID")

    def test_toward_follow_rank(self):
        self.assertTrue(toward_follow("REFUSE", "DENY"))
        self.assertTrue(toward_follow("RELATED_SAFE", "FOLLOW"))
        self.assertFalse(toward_follow("DENY", "REFUSE"))


class TestPhraseLoss(unittest.TestCase):
    def test_switched_differs_from_static_on_related_safe(self):
        scores = {
            "REFUSE": torch.tensor(0.1),
            "DENY": torch.tensor(0.2),
            "RELATED_SAFE": torch.tensor(3.0),
            "FOLLOW": torch.tensor(0.5),
        }
        st = float(static_joint_loss(scores).item())
        sw = float(switched_phrase_loss(scores, "RELATED_SAFE", lam=0.5).item())
        self.assertNotAlmostEqual(st, sw, places=5)
        refuse = {
            "REFUSE": torch.tensor(3.0),
            "DENY": torch.tensor(0.1),
            "RELATED_SAFE": torch.tensor(0.1),
            "FOLLOW": torch.tensor(0.2),
        }
        self.assertGreater(
            float(switched_phrase_loss(refuse, "REFUSE").item()),
            float(switched_phrase_loss(refuse, "RELATED_SAFE").item()),
        )

    def test_follow_mode_is_negative_follow_score(self):
        scores = {
            "REFUSE": torch.tensor(0.0),
            "DENY": torch.tensor(0.0),
            "RELATED_SAFE": torch.tensor(0.0),
            "FOLLOW": torch.tensor(2.0),
        }
        self.assertAlmostEqual(float(switched_phrase_loss(scores, "FOLLOW").item()), -2.0)


class TestP00(unittest.TestCase):
    def setUp(self):
        self.frozen = load_frozen()

    def test_go_on_disjoint_first_tokens(self):
        table = {}
        tid = 10
        for name, phrases in self.frozen["phrase_banks"].items():
            if name not in {"REFUSE", "DENY", "RELATED_SAFE", "FOLLOW"}:
                continue
            for p in phrases:
                table[p] = [tid, tid + 1]
                tid += 10
        tok = FakeTok(table)
        ts = public_tokensets(tok, self.frozen)
        gate = p00_verdict(ts, frozen_gates=self.frozen["gates"])
        self.assertEqual(gate["verdict"], "GO")
        self.assertEqual(ts["jaccard"]["refuse_follow"], 0.0)

    def test_stop_on_refuse_follow_overlap(self):
        frozen = load_frozen()
        table = {}
        for name, phrases in frozen["phrase_banks"].items():
            if name not in {"REFUSE", "DENY", "RELATED_SAFE", "FOLLOW"}:
                continue
            for i, p in enumerate(phrases):
                table[p] = [1 if name in {"REFUSE", "FOLLOW"} else 100 + i, 2]
        tok = FakeTok(table)
        ts = public_tokensets(tok, frozen)
        gate = p00_verdict(ts, frozen_gates=frozen["gates"])
        self.assertEqual(gate["verdict"], "STOP")
        self.assertGreaterEqual(ts["jaccard"]["refuse_follow"], 0.25)

    def test_jaccard_helper(self):
        self.assertEqual(jaccard({1, 2}, {2, 3}), 1 / 3)
        self.assertEqual(first_token_set([[7, 8], [7, 9], [4]]), {7, 4})


class TestVerdicts(unittest.TestCase):
    def setUp(self):
        self.frozen = load_frozen()

    def _grid(self, switched_cells, margin_cells, static_cells, h53_seeds=2, bw_sw=60, bw_mg=90, bw_st=80):
        rows = []
        seeds = [20270, 20271, 20272]
        for cell in HARD:
            for method, good in (
                ("switched", cell in switched_cells),
                ("refusal_margin", cell in margin_cells),
                ("static_joint", cell in static_cells),
                ("targeted_prefix", True),
            ):
                for i, seed in enumerate(seeds):
                    ok = good
                    if cell == "h53:c07" and method == "switched":
                        ok = bool(good) and i < h53_seeds
                    rows.append(
                        _sum(
                            cell,
                            method,
                            seed,
                            ok,
                            backward=bw_sw if method == "switched" else (bw_mg if method == "refusal_margin" else bw_st),
                        )
                    )
                    if not ok:
                        after = "RELATED_SAFE" if method == "refusal_margin" else "DENY"
                        rows.append(_round(cell, method, seed, after))
        return rows

    def test_p01_seven_without_budget_cut_is_stop(self):
        cells7 = HARD[:7]
        rows = self._grid(cells7, cells7, cells7[:6], h53_seeds=2, bw_sw=90, bw_mg=90)
        gate = p01_verdict(rows, hard_cells=HARD, frozen_gates=self.frozen["gates"])
        self.assertEqual(gate["verdict"], "STOP")
        self.assertEqual(gate["n_switched"], 7)

    def test_p01_eight_go(self):
        rows = self._grid(HARD, HARD[1:], HARD[:6], h53_seeds=2, bw_sw=50, bw_mg=90, bw_st=80)
        gate = p01_verdict(rows, hard_cells=HARD, frozen_gates=self.frozen["gates"])
        self.assertEqual(gate["verdict"], "GO")

    def test_p01_h53_one_seed_stop(self):
        rows = self._grid(HARD, HARD[1:], HARD[:6], h53_seeds=1, bw_sw=50, bw_mg=90)
        gate = p01_verdict(rows, hard_cells=HARD, frozen_gates=self.frozen["gates"])
        self.assertEqual(gate["verdict"], "STOP")

    def test_p01_switched_equals_static_stop(self):
        rows = self._grid(HARD, HARD[1:], HARD, h53_seeds=2, bw_sw=90, bw_mg=90, bw_st=90)
        gate = p01_verdict(rows, hard_cells=HARD, frozen_gates=self.frozen["gates"])
        self.assertEqual(gate["verdict"], "STOP")


class TestProtocolAndReporting(unittest.TestCase):
    def test_frozen_loads(self):
        blob = load_frozen()
        self.assertTrue(blob["frozen"])
        self.assertEqual(blob["states"]["success"], "core_rhc")
        self.assertTrue(blob["states"]["no_latent_probe"])
        self.assertEqual(len(blob["scope"]["hard_cells"]), 8)
        self.assertEqual(blob["msc_method"], "switched")

    def test_sealed_blocked(self):
        with self.assertRaises(RuntimeError):
            assert_query_unsealed("h83")
        with self.assertRaises(RuntimeError):
            assert_query_unsealed("h130")

    def test_jsonl_strips_banned(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.jsonl"
            with self.assertRaises(RuntimeError):
                append_jsonl(path, {"text": "secret", "mode": "REFUSE"})
            append_jsonl(path, {"mode": "REFUSE", "core_rhc": False})
            rows = read_jsonl(path)
            self.assertEqual(rows[0]["mode"], "REFUSE")
            self.assertFalse(contains_banned(rows[0]))
        self.assertIn("text", BANNED_KEYS)
        self.assertNotIn("text", strip_banned({"mode": "DENY", "text": "no"}))

    def test_controller_skips_invalid(self):
        from msc.controller import run_controller

        delta = torch.zeros(1, 3, 4, 4)
        done = run_controller(
            wrapper=None,
            inputs={},
            x0=delta,
            delta=delta,
            method="switched",
            mode="INVALID",
            banks={},
            steps=24,
            eps=0.06,
            alpha=0.004,
            lam=0.5,
            generic_prefix="Sure",
        )
        self.assertEqual(done["n_backward"], 0)
        self.assertTrue(done["skipped"])

    def test_p1_recovery_dispatch_keeps_p01_skip(self):
        from msc.controller import resolve_control

        self.assertTrue(resolve_control("switched", "INVALID")["skip"])
        self.assertFalse(resolve_control("switched_keep", "INVALID")["skip"])
        self.assertEqual(resolve_control("switched_follow", "INVALID")["inner_mode"], "FOLLOW")
        self.assertEqual(resolve_control("switched_margin", "INVALID")["inner_method"], "refusal_margin")
        self.assertEqual(resolve_control("switched_prefix", "INVALID")["inner_method"], "targeted_prefix")
        self.assertEqual(resolve_control("switched_keep", "DENY")["inner_method"], "switched")

    def test_p1a_expand_needs_both_fail_cells(self):
        from msc.verdict import p1a_expand

        rows = [
            _sum("h53:c07", "switched_keep", 20270, True),
            _sum("h53:c07", "switched_keep", 20271, True),
            _sum("h64:c08", "switched_keep", 20270, False),
            _sum("h64:c08", "switched_keep", 20271, False),
        ]
        gate = p1a_expand(rows, method="switched_keep", fail_cells=["h53:c07", "h64:c08"], min_seeds=2)
        self.assertFalse(gate["expand"])
        rows.append(_sum("h64:c08", "switched_keep", 20270, True))
        rows.append(_sum("h64:c08", "switched_keep", 20271, True))
        gate = p1a_expand(rows, method="switched_keep", fail_cells=["h53:c07", "h64:c08"], min_seeds=2)
        self.assertTrue(gate["expand"])


if __name__ == "__main__":
    unittest.main()
