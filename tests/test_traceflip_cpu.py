"""TraceFlip CPU tests. No GPU, no model weights, no catalog network access.

Run either way::

    python tests/test_traceflip_cpu.py           # self-contained runner
    python -m unittest tests.test_traceflip_cpu  # or via unittest

The tests pin down the parts of the method that are easy to get subtly wrong:
the reachability cost algebra, determinism/tie-breaking of branch selection, the
two-part validity test for a flip, the budget splitter, and the rule that BFR is
never reported as ASR.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from pathem.budget import BudgetLedger  # noqa: E402
from traceflip.metrics import (  # noqa: E402
    aggregate,
    budget_table,
    compare,
    diagnostics,
    pair_with,
    rates,
)
from traceflip.probe import branch_table, grad_ok, reachability_cost, select_branch  # noqa: E402
from traceflip.flip import constrained_flip, constraint_loss, flip_margin, prefix_margins  # noqa: E402
from traceflip.protocol import (  # noqa: E402
    EPS,
    EVAL_SEEDS,
    OPT_SEEDS,
    PILOT_QUERIES,
    SEALED,
    assert_query_allowed,
    assert_seed_split,
    max_new_tokens,
)
from traceflip.gate import (  # noqa: E402
    MAX_TOKEN_TRIES,
    accept_gate,
    commitment_rungs,
    gate_is_open,
    grow_budget_for,
    keep_lambda_for_prefix,
    open_token_policy,
    opening_token_ids,
    prefix_is_refusal,
)
from traceflip.repair import (  # noqa: E402
    accept_flip,
    candidate_positions,
    probe_budget_for,
    select_cost_only,
    select_earliest,
    select_random,
    select_value_only,
    split_backward_budget,
)
from traceflip.run import DEFAULT_METHODS, KNOWN_METHODS  # noqa: E402
from traceflip.report import render  # noqa: E402
from traceflip.trace import forward_tokens, margin_nats, topk_candidates  # noqa: E402


# --------------------------------------------------------------- protocol ---
class TestProtocol(unittest.TestCase):
    def test_seed_split_disjoint(self):
        assert_seed_split()
        self.assertFalse(set(OPT_SEEDS) & set(EVAL_SEEDS))
        with self.assertRaises(RuntimeError):
            assert_seed_split([1, 2], [2, 3])

    def test_sealed_queries_refused(self):
        for q in ("h83", "h130"):
            self.assertIn(q, SEALED)
            with self.assertRaises(RuntimeError):
                assert_query_allowed(q)
        for q in PILOT_QUERIES:
            assert_query_allowed(q)
        with self.assertRaises(RuntimeError):
            assert_query_allowed("h1")

    def test_max_new_tokens(self):
        self.assertEqual(max_new_tokens(False), 96)
        self.assertEqual(max_new_tokens(True), 24)
        self.assertEqual(max_new_tokens(True, 7), 7)


# ------------------------------------------------------------------ trace ---
class TestMargin(unittest.TestCase):
    def test_positive_when_target_is_argmax_by_the_win_buffer(self):
        # target 1 (logit 5.0) beats its best competitor 2 (logit 1.0) by 4 nats.
        logits = torch.tensor([0.0, 5.0, 1.0])
        self.assertAlmostEqual(float(margin_nats(logits, 1)), 4.0, places=5)

    def test_negative_when_target_is_not_argmax(self):
        logits = torch.tensor([0.0, 5.0, 1.0])
        self.assertLess(float(margin_nats(logits, 2)), 0.0)

    def test_zero_when_target_is_tied_for_argmax(self):
        # Ties leave no buffer: the target is (jointly) the argmax but wins by 0.
        logits = torch.tensor([0.0, 5.0, 5.0])
        self.assertAlmostEqual(float(margin_nats(logits, 1)), 0.0, places=6)

    def test_monotone_in_target_logit(self):
        base = torch.tensor([0.0, 2.0, 1.0])
        bumped = torch.tensor([0.0, 2.0, 3.0])
        self.assertGreater(float(margin_nats(bumped, 2)), float(margin_nats(base, 2)))

    def test_exclude_removes_competitor(self):
        logits = torch.tensor([0.0, 5.0, 1.0])
        g_all = float(margin_nats(logits, 2))
        g_excl = float(margin_nats(logits, 2, exclude=[1]))
        # Removing the strongest competitor must raise the margin.
        self.assertGreater(g_excl, g_all)
        # target 2 (1.0) now beats the only remaining competitor 0 (0.0) -> 1 nat.
        self.assertAlmostEqual(g_excl, 1.0, places=6)

    def test_margin_is_shift_invariant(self):
        a = torch.tensor([0.0, 5.0, 1.0])
        b = a + 100.0
        self.assertAlmostEqual(float(margin_nats(a, 2)), float(margin_nats(b, 2)), places=5)

    def test_topk_excludes_eos_and_orders_by_prob(self):
        logits = torch.tensor([0.0, 9.0, 8.0, 7.0, -1.0])
        out = topk_candidates(logits, 2, exclude=[1])
        self.assertEqual(out, [2, 3])

    def test_topk_with_block_target(self):
        logits = torch.tensor([5.0, 4.0, 3.0])
        self.assertEqual(topk_candidates(logits, 2, block_target=0), [1, 2])

    def test_forward_tokens_counts_prefill_plus_steps(self):
        self.assertEqual(forward_tokens(20, 10), 30)


# ---------------------------------------------------------------- probe ----
class TestReachability(unittest.TestCase):
    def test_zero_cost_when_branch_already_wins(self):
        g = torch.ones(4, 4)
        self.assertEqual(reachability_cost(g, 0.5, eps=EPS), 0.0)
        self.assertEqual(reachability_cost(g, 0.0, eps=EPS), 0.0)

    def test_cost_positive_when_branch_loses(self):
        g = torch.ones(4, 4)
        self.assertGreater(reachability_cost(g, -1.0, eps=EPS), 0.0)

    def test_cost_monotone_in_deficit(self):
        g = torch.ones(4, 4)
        c1 = reachability_cost(g, -1.0, eps=EPS)
        c2 = reachability_cost(g, -2.0, eps=EPS)
        self.assertGreater(c2, c1)

    def test_cost_monotone_decreasing_in_gradient(self):
        deficit = -1.0
        small = reachability_cost(torch.ones(4, 4), deficit, eps=EPS)
        big = reachability_cost(torch.full((4, 4), 10.0), deficit, eps=EPS)
        self.assertLess(big, small)

    def test_cost_is_dimensionless_scale(self):
        """A full-budget step with unit L1 gradient costs about 1 (minus eps0)."""
        deficit = -EPS
        g = torch.ones(4, 4) / 16.0  # ||g||_1 == 1
        c = reachability_cost(g, deficit, eps=EPS)
        # C = eps / (eps * 1 + eps0), i.e. 1 / (1 + eps0/eps) = 0.984.
        self.assertAlmostEqual(c, 1.0 / (1.0 + 1e-3 / EPS), places=6)
        self.assertLess(c, 1.0)
        self.assertGreater(c, 0.9)

    def test_grad_ok_rejects_none_nan_zero(self):
        self.assertFalse(grad_ok(None))
        self.assertFalse(grad_ok(torch.zeros(8, 8)))
        bad = torch.ones(8, 8)
        bad[0, 0] = float("nan")
        self.assertFalse(grad_ok(bad))

    def test_grad_ok_accepts_alive_gradient(self):
        self.assertTrue(grad_ok(torch.full((8, 8), 0.05)))

    def test_grad_ok_threshold_is_relative_to_eps(self):
        tiny = torch.full((8, 8), 1e-6)  # mean |g| << 0.02 * eps
        self.assertFalse(grad_ok(tiny, eps=EPS))


class TestSelection(unittest.TestCase):
    def _row(self, t, tok, value, cost, ok=True):
        return {
            "t": t,
            "token": tok,
            "value": value,
            "cost": cost,
            "grad_ok": ok,
            "incumbent": 99,
        }

    def test_value_times_reachability_prefers_cheap_high_value(self):
        rows = [
            self._row(0, 1, 1.0, 0.9),   # good value, expensive
            self._row(3, 2, 0.9, 0.0),   # slightly less value, free
        ]
        best = select_branch(rows)
        self.assertEqual((best["t"], best["token"]), (3, 2))

    def test_negative_value_is_clamped_away(self):
        rows = [
            self._row(0, 1, -5.0, 0.0),
            self._row(1, 2, 0.1, 0.0),
        ]
        best = select_branch(rows)
        self.assertEqual((best["t"], best["token"]), (1, 2))

    def test_unreachable_branches_are_filtered_out(self):
        rows = [
            self._row(0, 1, 999.0, 0.0, ok=False),
            self._row(1, 2, 0.01, 0.0, ok=True),
        ]
        self.assertEqual(select_branch(rows)["t"], 1)
        self.assertIsNone(select_branch([self._row(0, 1, 1.0, 0.0, ok=False)]))

    def test_tie_break_is_by_position_then_token(self):
        rows = [
            self._row(2, 7, 1.0, 0.0),
            self._row(1, 9, 1.0, 0.0),
            self._row(1, 3, 1.0, 0.0),
        ]
        self.assertEqual((select_branch(rows)["t"], select_branch(rows)["token"]), (1, 3))

    def test_selection_is_deterministic_regardless_of_input_order(self):
        rows = [self._row(0, 1, 1.0, 0.2), self._row(1, 5, 1.0, 0.2), self._row(2, 3, 1.0, 0.2)]
        a = select_branch(rows)
        b = select_branch(list(reversed(rows)))
        self.assertEqual((a["t"], a["token"]), (b["t"], b["token"]))

    def test_value_only_cost_only_disagree_by_construction(self):
        rows = [self._row(0, 1, 5.0, 0.9), self._row(1, 2, 0.1, 0.0)]
        self.assertEqual(select_value_only(rows)["t"], 0)
        self.assertEqual(select_cost_only(rows)["t"], 1)

    def test_earliest_selector_has_first_token_bias(self):
        rows = [self._row(5, 1, 100.0, 0.0), self._row(0, 2, 0.001, 0.0)]
        self.assertEqual(select_earliest(rows)["t"], 0)

    def test_random_selector_is_reproducible_with_seed(self):
        rows = [self._row(i, i + 1, 1.0, float(i) / 10) for i in range(8)]
        a = select_random(20260)(rows)
        b = select_random(20260)(rows)
        self.assertEqual((a["t"], a["token"]), (b["t"], b["token"]))


class TestProbeBudget(unittest.TestCase):
    class _Model:
        class _Generation:
            eos_token_id = 99

        generation_config = _Generation()

    class _Processor:
        class _Tokenizer:
            eos_token_id = 99

        tokenizer = _Tokenizer()

    class _Wrapper:
        model = None
        processor = None

        def __init__(self):
            self.model = TestProbeBudget._Model()
            self.processor = TestProbeBudget._Processor()

    def test_probe_budget_is_incremental_exact_and_values_still_run(self):
        wrapper = self._Wrapper()
        ledger = BudgetLedger(backward_passes=17)
        trace = {
            "token_ids": [10],
            "steps": [{
                "step": 0,
                "top_k": [10, 11, 12, 13],
                "margin_nats": {11: -1.0, 12: -2.0, 13: -3.0},
            }],
        }
        grads = {a: torch.ones(2, 2) for a in (11, 12)}
        rollout = {
            "prefix_ids": [],
            "forced_token": 0,
            "continuation_ids": [7],
            "n_generated": 1,
            "eos_hit": False,
        }
        with patch("traceflip.probe.branch_grads", return_value=grads), \
             patch("traceflip.probe.force_and_rollout", return_value=rollout), \
             patch("traceflip.probe.probe_value", return_value=1.5):
            rows = branch_table(
                wrapper, object(), {}, torch.zeros(1), trace, [0],
                topk=3, horizon=1, baseline_j=0.0, ledger=ledger,
                max_backward=2,
            )
        self.assertEqual(ledger.backward_passes, 19)
        live = [r for r in rows if r["grad_ok"]]
        self.assertEqual(len(live), 2)
        self.assertTrue(all(r["value"] == 1.5 for r in live))
        exhausted = [r for r in rows if r.get("budget_exhausted")]
        self.assertEqual(len(exhausted), 1)

    def test_tight_budget_is_shared_across_positions(self):
        wrapper = self._Wrapper()
        ledger = BudgetLedger()
        trace = {
            "token_ids": [10, 20, 30],
            "steps": [
                {
                    "step": t,
                    "top_k": [inc, inc + 1, inc + 2],
                    "margin_nats": {inc + 1: -1.0, inc + 2: -2.0},
                }
                for t, inc in enumerate((10, 20, 30))
            ],
        }
        calls = []

        def fake_grads(wrapper, inputs, prefix, x01, targets, incumbent, eos_exclude):
            calls.append((len(prefix), list(targets)))
            return {a: torch.ones(2, 2) for a in targets}

        rollout = {
            "prefix_ids": [],
            "forced_token": 0,
            "continuation_ids": [7],
            "n_generated": 1,
            "eos_hit": False,
        }
        with patch("traceflip.probe.branch_grads", side_effect=fake_grads), \
             patch("traceflip.probe.force_and_rollout", return_value=rollout), \
             patch("traceflip.probe.probe_value", return_value=1.0):
            branch_table(
                wrapper, object(), {}, torch.zeros(1), trace, [2, 0, 1],
                topk=2, horizon=1, baseline_j=0.0, ledger=ledger,
                max_backward=2,
            )
        self.assertEqual(ledger.backward_passes, 2)
        self.assertEqual([prefix_len for prefix_len, _ in calls], [2, 0])
        self.assertTrue(all(len(targets) == 1 for _, targets in calls))


# ----------------------------------------------------------------- flip ----
class TestBoxConstraint(unittest.TestCase):
    def test_box_respects_linf_and_pixel_range(self):
        from traceflip.flip import _box

        x0 = torch.full((1, 3, 4, 4), 0.5)
        big = torch.full((1, 3, 4, 4), 5.0)
        clipped = _box(x0, big, EPS)
        self.assertLessEqual(float(clipped.abs().max()), EPS + 1e-6)
        near_white = torch.full((1, 3, 4, 4), 0.999)
        d = _box(near_white, torch.full_like(near_white, EPS), EPS)
        self.assertLessEqual(float((near_white + d).max()), 1.0 + 1e-6)


class _LogitsOut:
    def __init__(self, logits):
        self.logits = logits


class _FlipStub:
    """Wrapper stub that returns fixed logits, ignoring pixels and tokens.

    Enough to pin the *semantics* of the constraint objectives; the real
    forward pass is exercised by the GPU preflight, not here.

    ``forward_logits`` scores through ``prepare_inputs_for_generation`` (the
    decoder's own path, needed for Qwen2-VL M-RoPE ``position_ids``), so the stub
    must expose it too. It just echoes ``input_ids`` back; ``__call__`` ignores
    the payload and returns the fixed logits.
    """

    def __init__(self, logits):
        self._logits = logits
        self.model = self

    def patchify(self, x):
        return x, None

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _update_model_kwargs_for_generation(self, output, kwargs, **_):
        out = dict(kwargs)
        if "attention_mask" in out:
            am = out["attention_mask"]
            out["attention_mask"] = torch.cat(
                [am, torch.ones(am.shape[0], 1, dtype=am.dtype, device=am.device)],
                dim=-1,
            )
        if "cache_position" in out:
            out["cache_position"] = out["cache_position"][-1:] + 1
        return out

    def __call__(self, **kwargs):
        return _LogitsOut(self._logits)


def _stub_inputs(n_base=1):
    return {
        "input_ids": torch.zeros((1, n_base), dtype=torch.long),
        "attention_mask": torch.ones((1, n_base), dtype=torch.long),
    }


class TestFlipMargin(unittest.TestCase):
    """The flip margin must be a max-over-competitors margin, not an
    incumbent delta. Grading against the incumbent lets an unreachable branch
    look feasible in-model yet never flip under re-decode."""

    def _logits(self, row):
        # shape [1, seq, vocab]; the flip margin reads position -1.
        return torch.tensor([[row]], dtype=torch.float32)

    def test_branch_is_argmax_has_positive_margin(self):
        x = torch.zeros((1, 3, 4, 4))
        stub = _FlipStub(self._logits([0.0, 5.0, 1.0]))
        g = float(
            flip_margin(stub, _stub_inputs(), x, [], branch_token=1, incumbent=0).item()
        )
        # branch 1 beats its best competitor (competitor 2 at 1.0) by 4 nats.
        self.assertAlmostEqual(g, 4.0, places=5)

    def test_margin_is_measured_against_the_best_competitor(self):
        """Regression: branch beats the incumbent but loses to a third token."""
        x = torch.zeros((1, 3, 4, 4))
        stub = _FlipStub(self._logits([0.0, 1.0, 5.0]))
        g = float(
            flip_margin(stub, _stub_inputs(), x, [], branch_token=1, incumbent=0).item()
        )
        # incumbent-delta semantics would report 1.0 > 0; max-margin reports -4.0.
        self.assertLess(g, 0.0)

    def test_eos_is_a_real_competitor_for_reproduction(self):
        """Regression: excluding EOS from the flip competitor set would let a
        branch look feasible while greedy actually emits EOS. Reproduction is the
        raw argmax, EOS included, so ``eos_exclude`` must not weaken the margin."""
        x = torch.zeros((1, 3, 4, 4))
        stub = _FlipStub(self._logits([0.0, 1.0, 9.0]))
        g_all = float(
            flip_margin(stub, _stub_inputs(), x, [], branch_token=1, incumbent=0).item()
        )
        g_excl = float(
            flip_margin(
                stub,
                _stub_inputs(),
                x,
                [],
                branch_token=1,
                incumbent=0,
                eos_exclude=[2],
            ).item()
        )
        # The strongest token (2) still competes, so both are negative and equal.
        self.assertLess(g_all, 0.0)
        self.assertEqual(g_all, g_excl)


class TestPrefixMargins(unittest.TestCase):
    def test_positions_align_with_the_prefix(self):
        x = torch.zeros((1, 3, 4, 4))
        n_base = 3
        seq = n_base + 2
        logits = torch.full((1, seq, 10), -5.0)
        # Position n_base-1 predicts the first prefix token (7); n_base predicts 8.
        logits[0, n_base - 1, 7] = 5.0
        logits[0, n_base, 8] = 5.0
        stub = _FlipStub(logits)
        g = prefix_margins(stub, _stub_inputs(n_base), x, [7, 8])
        self.assertEqual(tuple(g.shape), (2,))
        self.assertGreaterEqual(float(g.min().item()), 0.0)

    def test_broken_prefix_position_is_negative(self):
        x = torch.zeros((1, 3, 4, 4))
        n_base = 3
        logits = torch.full((1, n_base + 1, 10), -5.0)
        logits[0, n_base - 1, 7] = 5.0
        # Position for the second token is dominated by a different token.
        logits[0, n_base, 9] = 5.0
        logits[0, n_base, 8] = 1.0
        stub = _FlipStub(logits)
        g = prefix_margins(stub, _stub_inputs(n_base), x, [7, 8])
        self.assertLess(float(g[1].item()), 0.0)

    def test_empty_prefix_is_empty(self):
        x = torch.zeros((1, 3, 4, 4))
        stub = _FlipStub(torch.zeros((1, 1, 10)))
        g = prefix_margins(stub, _stub_inputs(), x, [])
        self.assertEqual(g.numel(), 0)


class TestConstraintLoss(unittest.TestCase):
    def _inputs(self, n_base=1):
        return _stub_inputs(n_base)

    def test_no_prefix_reports_inf_and_zero_positions(self):
        x0 = torch.zeros((1, 3, 4, 4))
        delta = torch.zeros_like(x0)
        stub = _FlipStub(torch.tensor([[[0.0, 5.0, 1.0]]]))
        _loss, info = constraint_loss(
            stub, self._inputs(), delta, x0, [], branch_token=1, incumbent=0
        )
        # A cell at the first position has no prefix to protect: the constraint
        # is vacuous, and that must be visible in the log, not look satisfied.
        self.assertEqual(info["n_prefix_positions"], 0)
        self.assertEqual(info["min_prefix_margin"], float("inf"))
        self.assertEqual(info["keep_violation"], 0.0)

    def test_incumbent_win_does_not_make_a_losing_branch_feasible(self):
        """End-to-end regression for the rank-vs-incumbent bug."""
        x0 = torch.zeros((1, 3, 4, 4))
        delta = torch.zeros_like(x0)
        stub = _FlipStub(torch.tensor([[[0.0, 1.0, 5.0]]]))
        _loss, info = constraint_loss(
            stub, self._inputs(), delta, x0, [], branch_token=1, incumbent=0
        )
        # branch 1 beats the incumbent 0 but token 2 is stronger.
        self.assertLess(info["g_flip"], 0.0)
        self.assertGreater(info["flip_violation"], 0.0)

    def test_winning_branch_is_feasible(self):
        x0 = torch.zeros((1, 3, 4, 4))
        delta = torch.zeros_like(x0)
        stub = _FlipStub(torch.tensor([[[0.0, 5.0, 1.0]]]))
        _loss, info = constraint_loss(
            stub, self._inputs(), delta, x0, [], branch_token=1, incumbent=0
        )
        self.assertGreaterEqual(info["g_flip"], 0.0)
        self.assertEqual(info["flip_violation"], 0.0)

    def test_no_prefix_ablation_uses_zero_lambda_during_gradient(self):
        x0 = torch.zeros((1, 1, 2, 2))
        seen_lam = []

        def fake_loss(wrapper, inputs, delta, x0_, prefix_ids, branch_token,
                      incumbent, keep_kappa, flip_kappa, lam, eos_exclude):
            seen_lam.append(float(lam))
            loss = delta.sum()
            info = {
                "g_flip": 1.0,
                "flip_violation": 0.0,
                "keep_violation": 2.0,
                "n_keep_broken": 1,
                "n_prefix_positions": 1,
                "min_prefix_margin": -1.0,
                "keep_violations": [2.0],
            }
            return loss, info

        with patch("traceflip.flip.constraint_loss", side_effect=fake_loss):
            out = constrained_flip(
                object(), {}, x0, torch.zeros_like(x0), [7], 8, 9,
                outer_steps=1, inner_steps=1, track_clip=False,
            )
        self.assertTrue(seen_lam)
        self.assertTrue(all(v == 0.0 for v in seen_lam))
        self.assertTrue(out["feasible_in_model"])


# --------------------------------------------------------------- repair ----
def _trace(ids, gaps=None):
    gaps = gaps or [1.0] * len(ids)
    return {
        "token_ids": list(ids),
        "steps": [
            {"step": i, "top_k": [int(ids[i]) + 1], "top1_gap_nats": float(gaps[i])}
            for i in range(len(ids))
        ],
    }


class TestAcceptFlip(unittest.TestCase):
    def test_accepts_only_when_prefix_and_branch_hold(self):
        old = _trace([10, 11, 12, 13])
        good = _trace([10, 11, 99, 13])
        chk = accept_flip(good, old, t=2, token=99)
        self.assertTrue(chk["valid"] and chk["prefix_kept"] and chk["branch_flipped"])

    def test_rejects_prefix_break(self):
        old = _trace([10, 11, 12, 13])
        bad = _trace([10, 88, 99, 13])
        chk = accept_flip(bad, old, t=2, token=99)
        self.assertFalse(chk["prefix_kept"])
        self.assertFalse(chk["valid"])

    def test_rejects_token_mismatch(self):
        old = _trace([10, 11, 12, 13])
        other = _trace([10, 11, 77, 13])
        chk = accept_flip(other, old, t=2, token=99)
        self.assertTrue(chk["prefix_kept"])
        self.assertFalse(chk["branch_flipped"])
        self.assertFalse(chk["valid"])

    def test_rejects_truncated_trace(self):
        old = _trace([10, 11, 12, 13])
        short = _trace([10, 11])
        self.assertFalse(accept_flip(short, old, t=2, token=99)["valid"])


class TestCandidatePositions(unittest.TestCase):
    def test_unstable_positions_are_preferred(self):
        tr = _trace([1, 2, 3, 4, 5], gaps=[5.0, 0.01, 4.0, 0.02, 3.0])
        picks = candidate_positions(tr, max_positions=2)
        self.assertEqual(picks, [1, 3])

    def test_not_biased_to_first_token(self):
        tr = _trace([1, 2, 3, 4], gaps=[9.0, 0.01, 0.5, 0.6])
        self.assertNotIn(0, candidate_positions(tr, max_positions=1))

    def test_first_token_is_excluded_when_prefix_would_be_vacuous(self):
        tr = _trace([1], gaps=[0.01])
        self.assertEqual(candidate_positions(tr), [])
        self.assertEqual(candidate_positions(tr, min_position=0), [0])

    def test_positions_preserve_instability_priority_and_are_bounded(self):
        tr = _trace(list(range(6)), gaps=[9.0, 3.0, 0.1, 2.0, 0.2, 1.0])
        picks = candidate_positions(tr, max_positions=5)
        self.assertEqual(picks, [2, 4, 5, 3, 1])
        self.assertLessEqual(len(picks), 5)

    def test_empty_trace_is_safe(self):
        self.assertEqual(candidate_positions({"steps": []}), [])


class TestBudgetSplit(unittest.TestCase):
    def test_unbounded_returns_frozen_schedule(self):
        self.assertEqual(split_backward_budget(None, 4, 24), (4, 24))

    def test_never_exceeds_remaining(self):
        for left in (1, 5, 12, 37, 96, 120, 500):
            o, i = split_backward_budget(left, 4, 24)
            self.assertGreaterEqual(o * i, 1)
            self.assertLessEqual(o * i, left)

    def test_zero_when_exhausted(self):
        self.assertEqual(split_backward_budget(0, 4, 24), (0, 0))
        self.assertEqual(split_backward_budget(-3, 4, 24), (0, 0))

    def test_probe_cap_is_a_fraction_of_the_cell_budget(self):
        self.assertEqual(probe_budget_for(None, 0.25), None)
        self.assertEqual(probe_budget_for(120, 0.25), 30)
        self.assertLess(probe_budget_for(120, 0.25), 120)

    def test_probe_and_solve_fit_inside_one_cell_budget(self):
        budget = 120
        probe = probe_budget_for(budget, 0.25)
        o, i = split_backward_budget(budget - probe, 4, 24)
        self.assertLessEqual(probe + o * i, budget)


# -------------------------------------------------------------- metrics ----
def _cell(qid, cid, rhc, method="traceflip", label="REFUSE", **kw):
    rec = {
        "method": method,
        "query_id": qid,
        "carrier_id": cid,
        "core_rhc": bool(rhc),
        "core_safe_answer": False,
        "terminal_label": label,
        "response_mode": "REFUSE" if label == "REFUSE" else "ANSWER",
        "chars": 40,
        "budget": {
            "generated_tokens": 100,
            "forward_passes": 120,
            "backward_passes": 120,
            "victim_generations": 3,
            "wall_clock_seconds": 10.0,
        },
    }
    rec.update(kw)
    return rec


class _StubTok:
    def encode(self, text, add_special_tokens=False):
        table = {
            "Sure": [101],
            "Here": [102],
            "Yes": [103],
            "Okay": [104, 99],
            "OK": [104],
            "Step": [105],
            "1": [106],
            "First": [107],
        }
        return list(table.get(str(text), [1]))

    def __call__(self, text, add_special_tokens=False):
        mapping = {
            "Sure, here": [101, 200],
            "Sure, here is a detailed step-by-step guide.": [101, 200, 201, 202],
        }
        return {"input_ids": list(mapping.get(str(text), [101]))}


class _StubProc:
    tokenizer = _StubTok()


class TestGateFlip(unittest.TestCase):
    def test_opening_tokens_are_first_ids_and_unique(self):
        ids = opening_token_ids(_StubProc())
        self.assertEqual(ids, [101, 102, 103, 104, 105, 106, 107])

    def test_keep_lambda_off_for_empty_or_refusal_prefix(self):
        refuse = [7, 8]
        self.assertEqual(keep_lambda_for_prefix([], refuse, lam=4.0), 0.0)
        self.assertEqual(keep_lambda_for_prefix([7, 9], refuse, lam=4.0), 0.0)
        self.assertTrue(prefix_is_refusal([7], refuse))
        self.assertEqual(keep_lambda_for_prefix([3, 7], refuse, lam=4.0), 4.0)
        self.assertFalse(prefix_is_refusal([3], refuse))

    def test_keep_only_after_compliance_opener(self):
        refuse = [7]
        comply = [101]
        self.assertEqual(
            keep_lambda_for_prefix([101, 9], refuse, lam=4.0, comply_ids=comply), 4.0
        )
        self.assertEqual(
            keep_lambda_for_prefix([9, 101], refuse, lam=4.0, comply_ids=comply), 0.0
        )
        self.assertTrue(gate_is_open([101, 3], comply))
        self.assertFalse(gate_is_open([7, 101], comply))

    def test_dead_token_does_not_try_farther_words(self):
        self.assertEqual(open_token_policy(-4.4), "climb_k")
        self.assertEqual(open_token_policy(0.0), "climb_k")
        self.assertEqual(open_token_policy(2.4), "climb_k")
        self.assertEqual(MAX_TOKEN_TRIES, 1)

    def test_commitment_rungs_coarse_are_shortest_first(self):
        rungs = commitment_rungs(_StubProc(), mode="coarse")
        self.assertEqual([r["name"] for r in rungs], ["k_short", "k_full"])
        self.assertLess(rungs[0]["k"], rungs[1]["k"])
        self.assertEqual(rungs[0]["ids"][0], 101)

    def test_fine_rungs_grow_one_token(self):
        rungs = commitment_rungs(_StubProc(), mode="fine")
        self.assertEqual([r["k"] for r in rungs], [2, 3, 4])
        self.assertEqual([r["name"] for r in rungs], ["k2", "k3", "k4"])
        self.assertEqual(rungs[0]["ids"], [101, 200])
        self.assertEqual(rungs[-1]["ids"], [101, 200, 201, 202])
        for prev, cur in zip(rungs, rungs[1:]):
            self.assertEqual(cur["ids"][: prev["k"]], prev["ids"])

    def test_grow_budget_does_not_starve_last_rung(self):
        self.assertEqual(grow_budget_for(96, 10, first_grow=True), 24)
        self.assertEqual(grow_budget_for(72, 9, first_grow=False), 8)
        self.assertEqual(grow_budget_for(8, 1, first_grow=False), 8)
        self.assertEqual(grow_budget_for(96, 1, first_grow=True), 96)
        self.assertEqual(grow_budget_for(0, 4, first_grow=True), 0)

    def test_accept_is_core_rhc_not_valid_flip(self):
        self.assertTrue(accept_gate({"core_rhc": True}))
        self.assertFalse(accept_gate({"core_rhc": False, "terminal_label": "REFUSE"}))
        valid_but_refuse = {"core_rhc": False, "terminal_label": "REFUSE"}
        self.assertFalse(accept_gate(valid_but_refuse))

    def test_frozen_methods_do_not_include_gateflip(self):
        self.assertNotIn("gateflip", DEFAULT_METHODS)
        self.assertIn("gateflip", KNOWN_METHODS)

    def test_traceflip_still_excludes_vacuous_t0(self):
        tr = {
            "steps": [
                {"step": 0, "top_k": [1], "top1_gap_nats": 0.01},
                {"step": 2, "top_k": [2], "top1_gap_nats": 0.50},
            ]
        }
        self.assertNotIn(0, candidate_positions(tr, max_positions=4))


class TestMetrics(unittest.TestCase):
    def test_rates_excludes_clean_and_counts_labels(self):
        rows = [
            _cell("h49", "c07", True, label="RHC"),
            _cell("h49", "c08", False, label="REFUSE"),
            _cell("h49", "c08", False, method="clean", label="REFUSE"),
        ]
        r = rates([x for x in rows if x["method"] != "clean"])
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["asr_core_rhc"], 0.5)
        self.assertAlmostEqual(r["refuse_rate"], 0.5)
        self.assertLessEqual(r["asr_wilson_lo"], r["asr_core_rhc"])
        self.assertGreaterEqual(r["asr_wilson_hi"], r["asr_core_rhc"])

    def test_empty_rates_is_safe(self):
        self.assertEqual(rates([])["n"], 0)

    def test_bfr_is_diagnostic_not_asr(self):
        rows = [
            _cell("h49", "c07", False, n_attempts=4, n_valid_flips=4, attempts=[
                {"prefix_kept": True, "branch_flipped": True, "valid": True, "feasible_in_model": True}
            ] * 4),
        ]
        r = rates(rows)
        d = diagnostics(rows)
        self.assertAlmostEqual(r["asr_core_rhc"], 0.0, places=9)
        self.assertAlmostEqual(d["bfr"], 1.0)
        self.assertNotIn("bfr", r)
        agg = aggregate({"traceflip": rows})
        blob = __import__("json").dumps(agg)
        self.assertNotIn("asr_bfr", blob)

    def test_diagnostics_separate_in_model_feasible_from_valid(self):
        rows = [
            _cell("h49", "c07", False, n_attempts=2, n_valid_flips=0, attempts=[
                {
                    "prefix_kept": False,
                    "branch_flipped": True,
                    "valid": False,
                    "feasible_in_model": True,
                },
                {
                    "prefix_kept": True,
                    "branch_flipped": False,
                    "valid": False,
                    "feasible_in_model": True,
                },
            ]),
        ]
        d = diagnostics(rows)
        self.assertAlmostEqual(d["prefix_broken_rate"], 0.5)
        self.assertAlmostEqual(d["flip_failed_after_prefix_kept_rate"], 0.5)
        self.assertAlmostEqual(d["in_model_feasible_but_invalid_rate"], 1.0)

    def test_pair_with_aligns_cells_and_drops_unmatched(self):
        a = [_cell("h49", "c07", True), _cell("h49", "c08", False)]
        b = [_cell("h49", "c07", False), _cell("h53", "c07", True)]
        xa, xb, keys = pair_with(a, b)
        self.assertEqual(keys, [("h49", "c07")])
        self.assertEqual((xa, xb), ([1], [0]))

    def test_compare_reports_paired_difference_and_mcnemar(self):
        base, ours = [], []
        for i in range(10):
            base.append(_cell("h49", f"c{i:02d}", i < 2, method="refusal_margin_pgd"))
            ours.append(_cell("h49", f"c{i:02d}", i < 7, method="traceflip"))
        cmp = compare({"refusal_margin_pgd": base, "traceflip": ours}, "refusal_margin_pgd")
        rec = cmp["paired_cells"]["traceflip"]
        self.assertEqual(rec["n_pairs"], 10)
        self.assertAlmostEqual(rec["diff"], 0.5)
        self.assertLess(rec["ci_lo"], rec["diff"])
        self.assertLess(rec["diff"], rec["ci_hi"])
        self.assertEqual(rec["mcnemar_n10"], 5)
        self.assertEqual(rec["mcnemar_n01"], 0)
        # Exact two-sided McNemar with 5 discordant pairs: 2 * P(X<=0) = 2/32.
        self.assertAlmostEqual(rec["mcnemar_p"], 2.0 / 32.0, places=9)
        self.assertEqual(cmp["paired_cells"]["refusal_margin_pgd"]["diff"], 0.0)

    def test_compare_ignores_cells_missing_from_either_side(self):
        a = [_cell("h49", "c07", True), _cell("h49", "c99", True)]
        b = [_cell("h49", "c07", False)]
        cmp = compare({"a": a, "b": b}, "b")
        self.assertEqual(cmp["paired_cells"]["a"]["n_pairs"], 1)

    def test_budget_table_reads_ledgers(self):
        rows = [_cell("h49", "c07", True), _cell("h49", "c08", True)]
        tab = budget_table({"traceflip": rows})
        self.assertAlmostEqual(tab["traceflip"]["mean_backward_passes"], 120.0)
        self.assertAlmostEqual(tab["traceflip"]["mean_generated_tokens"], 100.0)

    def test_report_renders_without_raw_text(self):
        base = [_cell("h49", "c07", False, method="refusal_margin_pgd")]
        ours = [_cell("h49", "c07", True, n_attempts=1, n_valid_flips=1, attempts=[
            {"prefix_kept": True, "branch_flipped": True, "valid": True, "feasible_in_model": True}
        ])]
        results = {"refusal_margin_pgd": base, "traceflip": ours}
        agg = aggregate(results)
        md = render(agg, compare(results, "refusal_margin_pgd"), meta={"queries": ["h49"], "carriers": ["c07"]})
        self.assertIn("TraceFlip", md)
        self.assertIn("core_rhc", md)
        self.assertIn("never be reported as ASR", md)
        self.assertIn("Solver diagnostics", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
