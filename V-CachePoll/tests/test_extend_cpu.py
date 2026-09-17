"""CPU tests for P4–P11 tensor logic. No GPU, no attack novelty."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.budget import k_never_empty, k_per_image_avtp, total_k_monotone  # noqa: E402
from vcachepoll.compressor import (  # noqa: E402
    exchange_events,
    image_quotas,
    isolated_k_matched,
    mean_importance_per_image,
    query_relevance_scores,
    restore_u,
    select_global_topk,
    select_spatial_uniform,
    select_within_image,
    shared_k_total,
    snap_k_to_total,
    visual_owner,
)
from vcachepoll.datasets import (  # noqa: E402
    aux_prompt,
    build_multi_pairs,
    chartqa_ready,
    duplicate_first_aux,
    screen_eligible,
    textvqa_images_ready,
)
from vcachepoll.extend import p10_inventory, p11_inventory, p12_decision, stage_p9_table  # noqa: E402
from vcachepoll.losses import combined_loss, lamp_like_loss, quota_loss  # noqa: E402
from vcachepoll.mechlog import (  # noqa: E402
    delta_a_is_zero,
    linf_ok,
    margin_crossing,
    margins_u,
    mechanism_snapshot,
    quota_conserved,
    tau_a,
)
from vcachepoll.probe import _select  # noqa: E402
from vcachepoll.uattr import recover_rate, u_from_grad, u_from_loo, u_from_score  # noqa: E402
from vcachepoll.vision import clip_delta, tv_loss  # noqa: E402
from vcachepoll.config import load_cfg  # noqa: E402


def _cfg(r_base=0.2, match=False):
    return {
        "compressor": {
            "r_base": r_base,
            "alpha": 1.0,
            "r_min": 0.05,
            "r_max": 0.9,
            "match_k_total": match,
        }
    }


class TestP4Mechlog(unittest.TestCase):
    def test_tau_and_margin_crossing(self):
        scores = torch.tensor([3.0, 2.0, 1.0, 0.5, 4.0, 0.1])
        owner = torch.tensor([0, 0, 0, 0, 1, 1])
        keep = torch.tensor([True, True, False, False, True, False])
        u = torch.tensor([True, False, False, False, False, False])
        tau = tau_a(scores, owner, k_a=2)
        self.assertAlmostEqual(float(tau), 2.0, places=5)
        m = margins_u(scores, u, tau)
        self.assertGreater(float(m[0]), 0.0)
        scores_adv = scores.clone()
        scores_adv[0] = 0.1
        m_adv = margins_u(scores_adv, u, tau_a(scores_adv, owner, 2))
        self.assertLess(float(m_adv[0]), 0.0)
        snap_c = mechanism_snapshot(scores, owner, [4, 2], keep, keep, u, torch.tensor([0.5, 0.5]), torch.tensor([2, 1]))
        keep_adv = torch.tensor([False, True, False, False, True, True])
        q = image_quotas(mean_importance_per_image(scores_adv, owner, 2), [4, 2], r_base=0.5)
        snap_a = mechanism_snapshot(scores_adv, owner, [4, 2], keep_adv, keep, u, q.r, q.k)
        cross = margin_crossing(snap_c, snap_a)
        self.assertGreaterEqual(cross["n_crossed"], 1)

    def test_delta_a_zero_and_linf(self):
        xA = torch.rand(1, 3, 8, 8)
        self.assertTrue(delta_a_is_zero(xA, xA.clone()))
        self.assertFalse(delta_a_is_zero(xA, xA + 0.01))
        delta = torch.full((1, 3, 8, 8), 16 / 255)
        self.assertTrue(linf_ok(delta, 16 / 255))
        self.assertFalse(linf_ok(delta * 1.2, 16 / 255))

    def test_quota_conservation_two_and_three(self):
        q2 = image_quotas(torch.tensor([0.48, 0.52]), [100, 100], r_base=0.2, r_min=0.05, r_max=0.9)
        self.assertTrue(quota_conserved(q2.r, 0.2, atol=1e-4))
        q3 = image_quotas(torch.tensor([0.32, 0.33, 0.35]), [80, 80, 80], r_base=0.2, r_min=0.05, r_max=0.9)
        self.assertTrue(quota_conserved(q3.r, 0.2, atol=1e-3))
        self.assertAlmostEqual(float(q3.r.sum()), 0.6, places=3)


class TestP5Multi(unittest.TestCase):
    def test_quota_loss_degenerates_to_two_image(self):
        r = torch.tensor([0.3, 0.1])
        self.assertAlmostEqual(float(quota_loss(r)), 0.2, places=6)
        r3 = torch.tensor([0.4, 0.2, 0.2])
        self.assertAlmostEqual(float(quota_loss(r3)), 0.2, places=6)

    def test_v_mask_union_of_aux(self):
        owner = torch.tensor([0, 0, 1, 1, 2, 2])
        v = owner >= 1
        self.assertEqual(int(v.sum()), 4)
        u = torch.tensor([True, False, False, False, False, False])
        scores = torch.tensor([2.0, 1.5, 0.1, 0.1, 3.0, 3.0])
        from vcachepoll.losses import eviction_loss

        self.assertLess(
            float(eviction_loss(scores, u, v)),
            float(eviction_loss(torch.tensor([2.0, 1.5, 0.1, 0.1, 0.0, 0.0]), u, v)),
        )

    def test_single_aux_frozen_grads(self):
        xA = torch.rand(1, 3, 8, 8, requires_grad=True)
        xB1 = torch.rand(1, 3, 8, 8, requires_grad=True)
        xB2 = torch.rand(1, 3, 8, 8, requires_grad=True)
        scores = torch.stack(
            [
                1.0 + xA.mean(),
                0.8 + 0.1 * xA.mean(),
                0.2 + xB1.mean(),
                0.1 + xB1.mean(),
                0.3 + 0.0 * xB2.mean(),
                0.25 + 0.0 * xB2.mean(),
            ]
        )
        owner = torch.tensor([0, 0, 1, 1, 2, 2])
        u = torch.tensor([True, False, False, False, False, False])
        v = torch.tensor([False, False, True, True, False, False])
        h = torch.ones(4)
        loss, _ = combined_loss(
            scores, owner, [2, 2, 2], u, v, h, h, xB1,
            r_base=0.5, alpha=1.0, r_min=0.1, r_max=0.9,
            lambda_q=1.0, lambda_e=1.0, lambda_v=0.0, lambda_p=0.05,
            kappa=0.0, tau=0.15, tv_fn=tv_loss, lambda_c=1.0,
        )
        loss.backward()
        self.assertIsNotNone(xB1.grad)
        self.assertGreater(float(xB1.grad.abs().sum()), 0.0)
        self.assertTrue(xB2.grad is None or float(xB2.grad.abs().sum()) == 0.0)

    def test_build_multi_pairs_maux1_matches_two_image(self):
        cfg = load_cfg(ROOT / "configs" / "p2.yaml")
        data = cfg["data"]
        pairs = build_multi_pairs(
            coco_root=Path(data["coco_root"]),
            vqa_json=Path(data["vqa_json"]),
            coco_csv=Path(data["coco_csv"]),
            n_pairs=4,
            seed=2026,
            n_aux=1,
            question_prefix=str(data["question_prefix"]),
        )
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs[0]["n_aux"], 1)
        self.assertEqual(len(pairs[0]["b_paths"]), 1)
        self.assertTrue(Path(pairs[0]["a_path"]).is_file())
        self.assertTrue(Path(pairs[0]["b_path"]).is_file())

    def test_dup_b2_equals_b1_and_owner_lengths(self):
        cfg = load_cfg(ROOT / "configs" / "p2.yaml")
        data = cfg["data"]
        pairs = build_multi_pairs(
            coco_root=Path(data["coco_root"]),
            vqa_json=Path(data["vqa_json"]),
            coco_csv=Path(data["coco_csv"]),
            n_pairs=2,
            seed=2026,
            n_aux=1,
            question_prefix=str(data["question_prefix"]),
        )
        dup = duplicate_first_aux(pairs[0], n_aux=2)
        self.assertEqual(dup["b_paths"][0], dup["b_paths"][1])
        self.assertEqual(dup["n_aux"], 2)
        n_per = [16, 16, 16]
        owner = visual_owner(n_per)
        self.assertEqual(int(owner.numel()), 48)
        self.assertEqual(int((owner == 0).sum()), 16)
        self.assertEqual(int((owner == 1).sum()), 16)
        self.assertEqual(int((owner == 2).sum()), 16)
        i_bar = torch.tensor([1.0, 2.0, 2.0])
        q = image_quotas(i_bar, n_per, r_base=0.2)
        self.assertAlmostEqual(float(q.r[1]), float(q.r[2]), places=5)

    def test_multi_prompt(self):
        self.assertIn("second image", aux_prompt(1, "What is this?", "Look at the first image only. Ignore the second image."))
        self.assertIn("other images", aux_prompt(4, "What is this?", "Look at the first image only. Ignore the second image."))


class TestP6Budget(unittest.TestCase):
    def test_k_drops_and_never_empty(self):
        i_bar = torch.tensor([1.0, 1.1])
        n = [40, 40]
        totals = []
        for r in (0.5, 0.3, 0.2, 0.1):
            k = k_per_image_avtp(i_bar, n, r_base=r)
            self.assertTrue(k_never_empty(k))
            totals.append(int(k.sum().item()))
        self.assertTrue(total_k_monotone(totals))
        self.assertGreater(totals[0], totals[-1])


class TestP7Family(unittest.TestCase):
    def test_matched_k_total(self):
        n_per = [20, 20]
        r_base = 0.2
        kt = shared_k_total(n_per, r_base)
        scores = torch.linspace(0.1, 1.0, 40)
        owner = visual_owner(n_per)
        cfg = _cfg(r_base, match=True)
        _, keep_a = _select(scores, owner, n_per, cfg, "avtp")
        _, keep_g = _select(scores, owner, n_per, cfg, "global")
        _, keep_i = _select(scores, owner, n_per, cfg, "isolated")
        _, keep_f = _select(scores, owner, n_per, cfg, "feather")
        qh = torch.randn(8)
        vh = torch.randn(40, 8)
        qs = query_relevance_scores(vh, qh)
        _, keep_q = _select(qs, owner, n_per, cfg, "query")
        self.assertEqual(int(keep_a.sum()), kt)
        self.assertEqual(int(keep_g.sum()), kt)
        self.assertEqual(int(keep_i.sum()), kt)
        self.assertEqual(int(keep_q.sum()), kt)
        self.assertGreaterEqual(int(keep_f.sum()), 2)
        self.assertLessEqual(int(keep_f.sum()), kt + 2)

    def test_global_can_steal_without_quota_term(self):
        scores = torch.tensor([0.1, 0.2, 0.95, 0.94])
        owner = torch.tensor([0, 0, 1, 1])
        keep = select_global_topk(scores, k_total=2)
        self.assertEqual(int(keep[owner == 0].sum()), 0)
        self.assertEqual(int(keep[owner == 1].sum()), 2)


class TestP8U(unittest.TestCase):
    def test_three_u_and_restore(self):
        scores = torch.tensor([5.0, 1.0, 0.5, 0.2, 3.0])
        owner = torch.tensor([0, 0, 0, 0, 1])
        keep = torch.tensor([True, True, True, False, True])
        u_s = u_from_score(scores, owner, keep, n_u=1)
        self.assertEqual(int(u_s[0]), 0)
        u_l = u_from_loo([2])
        self.assertEqual(int(u_l[0]), 2)
        attr = torch.tensor([0.1, 9.0, 0.2, 0.0, 0.0])
        u_g = u_from_grad(attr, owner, keep, n_u=1)
        self.assertEqual(int(u_g[0]), 1)
        keep_adv = keep.clone()
        keep_adv[0] = False
        um = torch.zeros_like(keep)
        um[0] = True
        restored = restore_u(keep_adv, um)
        self.assertTrue(bool(restored[0]))
        self.assertEqual(recover_rate([True, True, False], [True, True, True]), 2 / 3)


class TestP9P11P12(unittest.TestCase):
    def test_lamp_like_is_not_quota_loss(self):
        scores = torch.tensor([1.0, 1.0, 2.0, 2.0], requires_grad=True)
        owner = torch.tensor([0, 0, 1, 1])
        h = torch.ones(3)
        lq = quota_loss(image_quotas(mean_importance_per_image(scores, owner, 2), [2, 2], r_base=0.5).r)
        ll = lamp_like_loss(scores, owner, h, h)
        self.assertFalse(torch.allclose(lq, ll))

    def test_screen_protocol(self):
        self.assertTrue(screen_eligible(True, True, True))
        self.assertFalse(screen_eligible(True, True, False))
        self.assertFalse(screen_eligible(True, False, True))
        self.assertFalse(screen_eligible(False, True, True))

    def test_p9_table_from_existing(self):
        rec = stage_p9_table()
        self.assertIn("V-CachePoll", rec["table"])
        vc = rec["table"]["V-CachePoll"]
        self.assertGreaterEqual(int(vc.get("n") or 0), 32)
        self.assertGreater(float(vc.get("frac_comp_only_fail") or 0), 0.1)
        self.assertTrue((ROOT / "out" / "extend" / "P9_TABLE.md").is_file())

    def test_p10_p11_p12_blocked_or_skip(self):
        inv = p10_inventory()
        self.assertTrue(inv["qwen2vl"]["present"])
        self.assertFalse(inv["qwen3vl"]["present"])
        self.assertFalse(inv["internvl35"]["present"])
        self.assertFalse(inv["llava_ov"]["present"])
        d = p11_inventory()
        self.assertFalse(d["textvqa_images"])
        self.assertFalse(d["chartqa"])
        self.assertTrue(d["coco_standin"])
        self.assertFalse(p12_decision()["add_amp_loss"])
        self.assertFalse(textvqa_images_ready({}))
        self.assertFalse(chartqa_ready({}))
        self.assertFalse(chartqa_ready({"data": {"chartqa_dir": "/no/such/chartqa"}}))

    def test_clip_does_not_touch_a(self):
        xA = torch.full((1, 3, 4, 4), 0.4)
        xB = torch.full((1, 3, 4, 4), 0.5)
        d = clip_delta(xB, torch.ones_like(xB), 0.1)
        self.assertTrue(torch.allclose(xA, torch.full_like(xA, 0.4)))
        self.assertTrue(torch.allclose(d.abs(), torch.full_like(d, 0.1)))


class TestP4SelectUnchangedWithoutMatch(unittest.TestCase):
    def test_default_avtp_not_snapped(self):
        scores = torch.linspace(0, 1, 10)
        owner = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        cfg = _cfg(0.2, match=False)
        q, keep = _select(scores, owner, [5, 5], cfg, "avtp")
        self.assertEqual(int(keep.sum()), int(q.k.sum().item()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
