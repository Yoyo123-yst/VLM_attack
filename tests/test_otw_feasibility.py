import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from otw.prototypes import all_prototypes, harmful_prototypes, safe_prototypes
from otw.score import LAYER, P0S_SIGN, load_u_refusal_p0s, random_unit, s_mode_from_hidden
from otw.write_phase import write_loss_from_nll

FROZEN = ROOT / "outputs" / "otw" / "OTW_FROZEN_CONFIG.json"


def test_load_u_refusal_dim_positive():
    u = load_u_refusal_p0s()
    assert u.ndim == 1
    assert u.shape[0] > 0
    assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-6
    assert LAYER == 24
    assert P0S_SIGN > 0.0


def test_s_mode_plus_sign_is_answer_like():
    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    h = torch.tensor([2.0, 0.0, 0.0])
    s = s_mode_from_hidden(h, u, sign=P0S_SIGN)
    assert float(s.item()) == 2.0


def test_write_loss_formula_on_fake_nlls():
    nll_h = torch.tensor([2.0, 4.0])
    nll_s = torch.tensor([1.0, 3.0])
    got = write_loss_from_nll(nll_h, nll_s, alpha=1.0)
    expected = -torch.logsumexp(-nll_h, dim=0) + torch.logsumexp(-nll_s, dim=0)
    assert torch.allclose(got, expected)
    lower_h = write_loss_from_nll(torch.tensor([0.1, 4.0]), nll_s)
    assert float(lower_h) < float(got)
    lower_s = write_loss_from_nll(nll_h, torch.tensor([0.1, 3.0]))
    assert float(lower_s) > float(got)


def test_prototypes_are_short_not_recipes():
    banned = (
        "ingredient",
        "methamphetamine",
        "password reset token",
        "step 2",
        "step 3",
        "heat the",
        "mix the",
    )
    for qid in ("h01", "h11", "h41"):
        bank = all_prototypes(qid)
        rows = list(harmful_prototypes(qid)) + list(safe_prototypes(qid))
        assert len(bank["harmful"]) == 4
        assert len(bank["safe"]) == 4
        for t in rows:
            assert 3 <= len(t) <= 80
            assert t.count("\n") <= 1
            low = t.lower()
            assert not any(b in low for b in banned)


def test_frozen_config_exists_with_preregistered_gates():
    assert FROZEN.exists(), "OTW_FROZEN_CONFIG.json must exist before GPU work"
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert cfg["frozen_before_any_gpu_result"] is True
    assert cfg["direction"]["p0s_sign"] == 1.0
    assert cfg["direction"]["s_mode"] == "+<h,u>"
    assert cfg["data"]["queries"] == ["h01", "h11", "h41"]
    assert cfg["data"]["carrier"] == "c05"
    assert cfg["attack"]["seed"] == 2026
    assert cfg["controls"]["random_switch_step"] == 34
    gates = cfg["gates"]
    assert "Open raises ANSWER vs clean on h01" in gates["t1"]["pass"][0]
    assert "write must not re-close mouth" in gates["t2"]["pass"][0]
    assert "OtW core_RHC count >= joint" in gates["t3"]["pass"][0]


def test_random_unit_matches_frozen_hash():
    u = load_u_refusal_p0s()
    r = random_unit(int(u.shape[0]), 2026)
    import hashlib

    h = hashlib.sha256(r.astype(np.float64).tobytes()).hexdigest()
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert h == cfg["controls"]["u_random_sha256"]
