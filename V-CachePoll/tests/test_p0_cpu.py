"""CPU tests for V-CachePoll P0: quotas, survivor exchange, COCO pair builder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.coco_pairs import build_pairs, mentions_source  # noqa: E402
from vcachepoll.compressor import (  # noqa: E402
    exchange_events,
    image_quotas,
    isolated_k,
    layer_variation_score,
    mean_importance_per_image,
    restore_victim,
    select_global_topk,
    select_within_image,
    tokens_per_image,
    visual_owner,
)
from vcachepoll.config import load_cfg  # noqa: E402


class TestQuota(unittest.TestCase):
    def test_equal_importance_equal_ratio(self):
        i_bar = torch.tensor([0.4, 0.4])
        q = image_quotas(i_bar, [100, 100], r_base=0.5, alpha=1.0)
        self.assertAlmostEqual(float(q.r[0]), 0.5, places=5)
        self.assertAlmostEqual(float(q.r[1]), 0.5, places=5)
        self.assertEqual(int(q.k[0]), 50)
        self.assertEqual(int(q.k[1]), 50)

    def test_raising_b_cuts_a(self):
        i_bar = torch.tensor([0.2, 0.8])
        q = image_quotas(i_bar, [100, 100], r_base=0.5, alpha=1.0)
        self.assertGreater(float(q.r[1]), float(q.r[0]))
        self.assertGreater(int(q.k[1]), int(q.k[0]))
        self.assertAlmostEqual(float(q.r.sum()), 1.0, places=5)

    def test_large_raw_norm_does_not_saturate(self):
        i_bar = torch.tensor([17.00, 17.92])
        q = image_quotas(i_bar, [100, 100], r_base=0.5, alpha=1.0)
        self.assertGreater(float(q.r[0]), 0.2)
        self.assertLess(float(q.r[1]), 0.8)

    def test_isolated_ignores_importance(self):
        k = isolated_k([80, 120], 0.5)
        self.assertEqual(int(k[0]), 40)
        self.assertEqual(int(k[1]), 60)


class TestSelection(unittest.TestCase):
    def test_within_image_respects_k(self):
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        owner = torch.tensor([0, 0, 0, 1, 1, 1])
        keep = select_within_image(scores, owner, torch.tensor([2, 1]))
        self.assertEqual(int(keep[owner == 0].sum()), 2)
        self.assertEqual(int(keep[owner == 1].sum()), 1)
        self.assertTrue(bool(keep[1]))  # 0.9 in A
        self.assertTrue(bool(keep[3]))  # 0.8 in B

    def test_global_can_evict_across_images(self):
        scores = torch.tensor([0.1, 0.2, 0.95, 0.94])
        owner = torch.tensor([0, 0, 1, 1])
        keep = select_global_topk(scores, k_total=2)
        self.assertFalse(bool(keep[0]))
        self.assertFalse(bool(keep[1]))
        self.assertTrue(bool(keep[2] and keep[3]))

    def test_exchange_and_restore(self):
        owner = torch.tensor([0, 0, 1, 1])
        clean = torch.tensor([True, True, False, False])
        adv = torch.tensor([True, False, True, False])
        ev = exchange_events(clean, adv, owner)
        self.assertEqual(ev["a_out"], 1)
        self.assertEqual(ev["b_in"], 1)
        self.assertEqual(ev["swap_ba"], 1)
        restored = restore_victim(adv, clean, owner)
        self.assertTrue(bool(restored[1]))

    def test_tokens_per_image_and_owner(self):
        grid = torch.tensor([[1, 4, 4], [1, 2, 4]])  # merge 2 → 4 and 2 tokens
        n = tokens_per_image(grid, merge=2)
        self.assertEqual(n, [4, 2])
        owner = visual_owner(n)
        self.assertEqual(owner.tolist(), [0, 0, 0, 0, 1, 1])

    def test_layer_variation(self):
        h0 = torch.zeros(3, 4)
        h1 = torch.ones(3, 4)
        score = layer_variation_score({0: h0, 1: h1}, score_layers=[1])
        self.assertEqual(score.shape, (3,))
        self.assertTrue(torch.allclose(score, torch.full((3,), 2.0)))


class TestJudgeAndPairs(unittest.TestCase):
    def test_mentions_source(self):
        self.assertTrue(mentions_source("A large grizzly bear sits on the grass.", "Grizzly bear"))
        self.assertTrue(mentions_source("The bus has two decks.", "Double-decker bus"))
        self.assertFalse(mentions_source("A person is skiing.", "Grizzly bear"))

    def test_answers_agree_ignores_question_echo(self):
        from vcachepoll.coco_pairs import answers_agree

        prompt = "Look at the first image only. Ignore the second image. What is nestled between the two plush figures?"
        ref = "The object nestled between the two plush figures is a blue blanket."
        pred = "A blue blanket is between the plush figures."
        self.assertTrue(answers_agree(pred, ref, prompt, source="Teddy bear"))
        self.assertFalse(answers_agree("A red truck is visible.", ref, prompt, source="Teddy bear"))

    def test_build_pairs_from_local_coco(self):
        cfg = load_cfg()
        data = cfg["data"]
        pairs = build_pairs(
            coco_root=Path(data["coco_root"]),
            vqa_json=Path(data["vqa_json"]),
            coco_csv=Path(data["coco_csv"]),
            n_pairs=8,
            seed=2026,
            question_prefix=data["question_prefix"],
        )
        self.assertEqual(len(pairs), 8)
        ids = [p["pair_id"] for p in pairs]
        self.assertEqual(len(set(ids)), 8)
        for p in pairs:
            self.assertTrue(Path(p["a_path"]).is_file(), p["a_path"])
            self.assertTrue(Path(p["b_path"]).is_file(), p["b_path"])
            self.assertNotEqual(p["a_file"], p["b_file"])
            self.assertIn("first image", p["prompt"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
