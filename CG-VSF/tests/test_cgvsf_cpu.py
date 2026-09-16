#!/usr/bin/env python3
"""CPU tests for certificate extraction, trim, LSE, energy, reporting, verdicts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "CG-VSF" / "src"))

import torch

from cgvsf.certificates import (
    extract_failure_certificate,
    hash_token_ids,
    update_certificate_set,
)
from cgvsf.energy import energy_joint, energy_single, gramian, gramian_stats, project_linf_box, single_constraint_step
from cgvsf.objective import logsumexp_scores
from cgvsf.reporting import BANNED_KEYS, append_jsonl, contains_banned, read_jsonl, strip_banned
from cgvsf.verdict import p0a_verdict
from cgvsf.protocol import load_frozen


def _lab(mode: str):
    def fn(text: str):
        mapping = {
            "sorry": {
                "core_rhc": False,
                "terminal_label": "REFUSE",
                "response_mode": "REFUSE",
                "quality": "FLUENT",
            },
            "no evidence": {
                "core_rhc": False,
                "terminal_label": "DENY",
                "response_mode": "DENY",
                "quality": "FLUENT",
            },
        }
        for k, v in mapping.items():
            if k in text:
                return v
        return {
            "core_rhc": False,
            "terminal_label": "OTHER",
            "response_mode": "ANSWER",
            "quality": "FLUENT",
        }

    return fn


def _decode(table):
    def fn(ids):
        return " ".join(table.get(int(i), "x") for i in ids)

    return fn


class TestCertificates(unittest.TestCase):
    def test_hash_stable(self):
        self.assertEqual(hash_token_ids([1, 2, 3]), hash_token_ids([1, 2, 3]))
        self.assertNotEqual(hash_token_ids([1, 2, 3]), hash_token_ids([1, 2, 4]))

    def test_stable_refuse_cuts_at_second(self):
        table = {1: "sorry", 2: "sorry", 3: "later"}
        cert = extract_failure_certificate(
            [1, 2, 3, 4],
            decode_prefix=_decode(table),
            label_fn=_lab("REFUSE"),
            max_len=16,
            fallback_len=8,
            stable_run=2,
        )
        self.assertEqual(cert["token_ids"], [1, 2])
        self.assertEqual(cert["mode"], "REFUSE")
        self.assertEqual(cert["length"], 2)

    def test_fluency_garbage_does_not_override_refuse(self):
        def lab(_text):
            return {
                "core_rhc": False,
                "terminal_label": "INVALID",
                "response_mode": "REFUSE",
                "quality": "GARBAGE",
                "chars": 3,
            }

        cert = extract_failure_certificate(
            [1, 2, 3],
            decode_prefix=lambda ids: "sorry sorry",
            label_fn=lab,
        )
        self.assertEqual(cert["mode"], "REFUSE")
        table = {i: "x" for i in range(10)}
        cert = extract_failure_certificate(
            list(range(10)),
            decode_prefix=_decode(table),
            label_fn=_lab("OTHER"),
            fallback_len=8,
        )
        self.assertEqual(cert["length"], 8)
        self.assertEqual(cert["mode"], "OTHER")

    def test_prefix_tag_refuse_not_garbage(self):
        from cgvsf.certificates import prefix_tag

        lab = prefix_tag("I'm sorry, I can't help with that.")
        self.assertEqual(lab["terminal_label"], "REFUSE")
        self.assertEqual(prefix_tag("").get("terminal_label"), "OTHER")
        cert = extract_failure_certificate(
            [1, 2, 3, 4],
            decode_prefix=lambda ids: "I'm sorry, I can't help with that.",
            label_fn=prefix_tag,
            stable_run=2,
        )
        self.assertEqual(cert["mode"], "REFUSE")
        self.assertEqual(cert["length"], 2)

    def test_exact_only_ignores_mode_cut(self):
        table = {1: "sorry", 2: "sorry", 3: "sorry"}
        cert = extract_failure_certificate(
            [1, 2, 3],
            decode_prefix=_decode(table),
            label_fn=_lab("REFUSE"),
            fallback_len=8,
            exact_only=True,
        )
        self.assertEqual(cert["token_ids"], [1, 2, 3])

    def test_dedupe_and_trim_mode_then_recent(self):
        a = {"hash": "h1", "mode": "REFUSE", "token_ids": [1], "length": 1}
        b = {"hash": "h2", "mode": "REFUSE", "token_ids": [2], "length": 1}
        c = {"hash": "h3", "mode": "DENY", "token_ids": [3], "length": 1}
        s = update_certificate_set([], a, method="accumulated_certificate", max_set=2)
        s = update_certificate_set(s, a, method="accumulated_certificate", max_set=2)
        self.assertEqual(len(s), 1)
        s = update_certificate_set(s, b, method="accumulated_certificate", max_set=2)
        s = update_certificate_set(s, c, method="accumulated_certificate", max_set=2)
        modes = {x["mode"] for x in s}
        self.assertIn("DENY", modes)
        last = update_certificate_set(s, c, method="last_certificate", max_set=8)
        self.assertEqual(len(last), 1)
        self.assertEqual(last[0]["hash"], "h3")


class TestObjectiveEnergy(unittest.TestCase):
    def test_logsumexp_decreases_on_descent(self):
        g = torch.tensor([0.5, -0.2], requires_grad=True)
        loss = logsumexp_scores(g)
        (grad,) = torch.autograd.grad(loss, g)
        g2 = (g - 0.1 * grad).detach()
        self.assertLess(float(logsumexp_scores(g2)), float(loss.detach()))

    def test_gramian_psd(self):
        j1 = torch.tensor([1.0, 0.0, 0.0])
        j2 = torch.tensor([0.5, 0.5, 0.0])
        w = gramian([j1, j2])
        self.assertEqual(tuple(w.shape), (2, 2))
        eig = torch.linalg.eigvalsh(w)
        self.assertTrue(bool((eig >= -1e-6).all()))
        st = gramian_stats(w)
        self.assertGreaterEqual(st["rank"], 1)

    def test_single_qp_matches_closed_form(self):
        g = torch.tensor(1.0)
        j = torch.tensor([2.0, 0.0])
        tau = 0.0
        step = single_constraint_step(g, j, tau)
        # Δ = (tau-G)/||J||^2 * J = -1/4 * [2,0] = [-0.5, 0]
        self.assertAlmostEqual(float(step[0]), -0.5, places=5)
        e = energy_single(g, j, tau, eta=0.0)
        self.assertAlmostEqual(float(e), 0.25, places=5)
        e0 = energy_single(torch.tensor(-1.0), j, tau)
        self.assertAlmostEqual(float(e0), 0.0, places=5)
        ej = energy_joint([g], [j], tau, eta=0.0)
        self.assertAlmostEqual(float(ej), 0.25, places=5)

    def test_linf_projection(self):
        x0 = torch.tensor([0.5, 0.99])
        d = torch.tensor([0.2, 0.2])
        out = project_linf_box(x0, d, 0.05)
        self.assertLessEqual(float(out.abs().max()), 0.05 + 1e-6)
        self.assertLessEqual(float((x0 + out).max()), 1.0 + 1e-6)


class TestReportingVerdict(unittest.TestCase):
    def test_strip_banned_keys(self):
        row = {"text": "secret", "core_rhc": True, "cell_id": "h64:c08"}
        self.assertTrue(contains_banned(row))
        clean = strip_banned(row)
        self.assertNotIn("text", clean)
        self.assertTrue(clean["core_rhc"])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jsonl"
            append_jsonl(p, {"cell_id": "h64:c08", "core_rhc": False, "round": 0})
            self.assertEqual(len(read_jsonl(p)), 1)

    def test_six_of_eight_is_not_go(self):
        frozen = load_frozen()
        hard = list(frozen["scope"]["hard_cells"])
        rows = []
        won = hard[:6]
        for c in hard:
            rows.append(
                {
                    "cell_id": c,
                    "method": "accumulated_certificate",
                    "round": 0,
                    "core_rhc": c in won,
                    "old_certificate_recurrence": False,
                    "after_mode": "RHC" if c in won else "DENY",
                }
            )
            rows.append(
                {
                    "cell_id": c,
                    "method": "last_certificate",
                    "round": 0,
                    "core_rhc": False,
                    "old_certificate_recurrence": True,
                    "after_mode": "DENY",
                }
            )
        v = p0a_verdict(rows, hard_cells=hard, frozen_gates=frozen["gates"])
        self.assertEqual(v["n_accumulated"], 6)
        self.assertNotEqual(v["verdict"], "GO")
        self.assertIn(v["verdict"], {"CONDITIONAL", "STOP"})

    def test_seven_of_eight_can_go(self):
        frozen = load_frozen()
        hard = list(frozen["scope"]["hard_cells"])
        rows = []
        won_acc = hard[:7]
        won_last = hard[:5]
        for c in hard:
            rows.append(
                {
                    "cell_id": c,
                    "method": "accumulated_certificate",
                    "round": 1,
                    "core_rhc": c in won_acc,
                    "old_certificate_recurrence": False,
                    "after_mode": "RHC" if c in won_acc else "REFUSE",
                }
            )
            rows.append(
                {
                    "cell_id": c,
                    "method": "last_certificate",
                    "round": 1,
                    "core_rhc": c in won_last,
                    "old_certificate_recurrence": True,
                    "after_mode": "RHC" if c in won_last else "REFUSE",
                }
            )
        v = p0a_verdict(rows, hard_cells=hard, frozen_gates=frozen["gates"])
        self.assertEqual(v["n_accumulated"], 7)
        self.assertEqual(v["verdict"], "GO")

    def test_banned_key_set_covers_raw_text(self):
        self.assertIn("text", BANNED_KEYS)
        self.assertIn("token_text", BANNED_KEYS)

    def test_p0b_wait_below_min_states(self):
        frozen = load_frozen()
        from cgvsf.verdict import auroc, p0b_verdict

        self.assertGreater(auroc([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]), 0.9)
        v = p0b_verdict(
            [{"eliminated": True, "E_joint": 0.1, "steps_used": 8}] * 4,
            frozen_gates=frozen["gates"],
        )
        self.assertEqual(v["verdict"], "WAIT")


if __name__ == "__main__":
    unittest.main()
