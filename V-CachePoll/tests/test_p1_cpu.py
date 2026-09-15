"""CPU tests for P1 quota / eviction losses and pixel helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.compressor import image_quotas, mean_importance_per_image  # noqa: E402
from vcachepoll.losses import combined_loss, eviction_loss, quota_loss, value_loss  # noqa: E402
from vcachepoll.vision import clip_delta, patchify_x01, pixel_rows_from_grid, tv_loss  # noqa: E402


class TestQuotaLoss(unittest.TestCase):
    def test_raising_b_lowers_quota_loss(self):
        owner = torch.tensor([0, 0, 1, 1])
        low_b = torch.tensor([1.0, 1.0, 0.2, 0.2])
        high_b = torch.tensor([1.0, 1.0, 2.0, 2.0])
        n = [2, 2]
        r_low = image_quotas(mean_importance_per_image(low_b, owner, 2), n, r_base=0.5, alpha=1.0).r
        r_high = image_quotas(mean_importance_per_image(high_b, owner, 2), n, r_base=0.5, alpha=1.0).r
        self.assertLess(float(quota_loss(r_high)), float(quota_loss(r_low)))
        self.assertGreater(float(r_high[1]), float(r_high[0]))

    def test_quota_has_gradient(self):
        i_bar = torch.tensor([1.0, 1.1], requires_grad=True)
        r = image_quotas(i_bar, [10, 10], r_base=0.5, alpha=1.0).r
        quota_loss(r).backward()
        self.assertIsNotNone(i_bar.grad)
        self.assertLess(float(i_bar.grad[1]), float(i_bar.grad[0]))


class TestEvictLoss(unittest.TestCase):
    def test_raising_b_lowers_evict_loss(self):
        scores_lo = torch.tensor([2.0, 2.0, 0.1, 0.1])
        scores_hi = torch.tensor([2.0, 2.0, 3.0, 3.0])
        u = torch.tensor([True, True, False, False])
        v = torch.tensor([False, False, True, True])
        self.assertLess(float(eviction_loss(scores_hi, u, v)), float(eviction_loss(scores_lo, u, v)))

    def test_combined_backward(self):
        scores = torch.tensor([1.0, 0.8, 0.7, 1.2], requires_grad=True)
        owner = torch.tensor([0, 0, 1, 1])
        u = torch.tensor([True, False, False, False])
        v = torch.tensor([False, False, True, True])
        h_adv = torch.tensor([0.2, 0.1])
        h_clean = torch.tensor([0.2, 0.1])
        xB = torch.rand(1, 3, 8, 8, requires_grad=True)
        loss, aux = combined_loss(
            scores, owner, [2, 2], u, v, h_adv, h_clean, xB,
            r_base=0.5, alpha=1.0, r_min=0.1, r_max=0.9,
            lambda_q=1.0, lambda_e=1.0, lambda_v=0.25, lambda_p=0.05,
            kappa=0.0, tau=0.15, tv_fn=tv_loss,
        )
        loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(float(aux["L_value"]) < 1e-5)


class TestVision(unittest.TestCase):
    def test_clip_delta(self):
        x0 = torch.full((1, 3, 4, 4), 0.5)
        delta = torch.ones_like(x0)
        out = clip_delta(x0, delta, eps=0.1)
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.1)))

    def test_patchify_token_rows(self):
        x = torch.rand(1, 3, 28, 56)
        pv, grid = patchify_x01(x)
        rows = pixel_rows_from_grid(grid)
        self.assertEqual(rows, [int(pv.shape[0])])
        self.assertEqual(int(pv.shape[0]), 2 * 4)  # t*h*w = 1*2*4

    def test_value_loss_identical(self):
        h = torch.randn(8)
        self.assertLess(float(value_loss(h, h)), 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
