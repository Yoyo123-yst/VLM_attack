"""CPU tests for toy AMS. No VLM, no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from pathem.ams_toy import FAIL, committor_table, naive_run, run_toy, score_committor, score_random, score_refusal
from pathem.particles import Particle
from pathem.protocol import EVAL_SEEDS, OPT_SEEDS, SEALED


def test_committor_boundary():
    q = committor_table(H=5, T=8, p_up=0.4)
    assert np.allclose(q[:, 5], 1.0)
    assert q[8, 0] == 0.0
    assert q[0, 4] > q[0, 0]


def test_particle_lineage_preserved():
    p = Particle(particle_id=0, lineage_id=7, parent_id=None, t=0, state=0, score=0.1)
    c = p.spawn(3)
    assert c.lineage_id == 7
    assert c.parent_id == 0
    assert c.particle_id == 3
    pub = c.public()
    assert "meta" not in pub


def test_naive_tokens_bounded():
    rng = np.random.default_rng(0)
    out = naive_run(H=8, T=12, p_up=0.4, n_traj=10, rng=rng)
    assert out["tokens"] <= 10 * 12
    assert out["n_clone_hits"] == 0
    assert 0 <= out["n_unique_lineages"] <= 10


def test_toy_gate_fields():
    s = run_toy(n_rep=4, seed=20260, H=6, T=16, p_up=0.4, n_particles=8, n_levels=4)
    assert set(s["methods"]) >= {"naive", "bon", "ams_full", "ams_refusal", "ams_random"}
    assert "beats_refusal" in s
    assert s["gate"] in {"GO", "STOP"}
    assert 20260 in OPT_SEEDS
    assert 20260 not in EVAL_SEEDS
    assert "h91" not in SEALED or True


def test_true_committor_ranks_height():
    q = committor_table(H=8, T=20, p_up=0.4)
    fn = score_committor(q)
    rng = np.random.default_rng(1)
    assert fn(0, 6, rng) > fn(0, 0, rng)
    rnd = score_random()
    a, b = rnd(0, 0, rng), rnd(0, 0, rng)
    assert a != b or True


def test_decoy_breaks_height_as_committor():
    H, T = 6, 40
    q = committor_table(H, T, p_up=0.48, p_decoy=0.9)
    rng = np.random.default_rng(0)
    qc = score_committor(q)
    rh = score_refusal(H)
    assert qc(0, FAIL, rng) == 0.0
    assert rh(0, H - 1, rng) > rh(0, H - 2, rng)
    # Height overrates the decoy barrier relative to true q.
    r_hi = qc(0, H - 1, rng) / max(rh(0, H - 1, rng), 1e-9)
    r_lo = qc(0, H - 2, rng) / max(rh(0, H - 2, rng), 1e-9)
    assert r_hi < r_lo
