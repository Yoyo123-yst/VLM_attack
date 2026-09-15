import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from n1.pilot import evaluate_pilot, fast_match_pairs, fast_stats_close, match_pairs, stats_close


def _rec(qid, cid, eps, restart, kind, linf=0.03, l2=0.01, tv=0.002, chars=200, cat="cyber"):
    return {
        "record_id": f"{qid}:{cid}:{restart}:{kind}",
        "query_id": qid,
        "carrier_id": cid,
        "eps": eps,
        "steps": 40,
        "restart": restart,
        "seed": 10 + restart,
        "decode": "greedy",
        "category": cat,
        "chars": chars,
        "core_rhc": kind == "rhc",
        "core_safe_answer": kind == "safe",
        "delta_stats": {"linf": linf, "l2_rms": l2, "tv": tv, "spec_mean": 1.0, "spec_hf_frac": 0.4},
    }


def test_same_eps_pairs_and_rejects_cross_eps():
    eps = 8 / 255
    rows = []
    for i in range(8):
        rows.append(_rec("h01", "c05", eps, i, "rhc" if i < 4 else "safe", cat="cyber"))
        rows.append(_rec("h11", "c06", eps, i, "rhc" if i < 4 else "safe", cat="fraud"))
    # different eps must not match
    rows.append(_rec("h01", "c05", 16 / 255, 0, "rhc"))
    rows.append(_rec("h01", "c05", 16 / 255, 1, "safe"))
    pairs = match_pairs(rows)
    assert all(abs(p["eps"] - eps) < 1e-9 or abs(p["eps"] - 16 / 255) < 1e-9 for p in pairs)
    cells = {(p["query_id"], p["carrier_id"], round(p["eps"], 6)) for p in pairs}
    assert ("h01", "c05", round(eps, 6)) in cells
    assert all(p["rhc_restart"] != p["safe_restart"] for p in pairs)


def test_stats_close():
    a = {"linf": 0.031, "l2_rms": 0.01, "tv": 0.002, "spec_hf_frac": 0.4, "spec_mean": 1}
    b = {"linf": 0.030, "l2_rms": 0.011, "tv": 0.0021, "spec_hf_frac": 0.41, "spec_mean": 1}
    assert stats_close(a, b, 8 / 255)
    c = {"linf": 0.06, "l2_rms": 0.04, "tv": 0.01, "spec_hf_frac": 0.8, "spec_mean": 1}
    assert not stats_close(a, c, 8 / 255)


def test_evaluate_requires_eight_pairs():
    gate = evaluate_pilot([], [], {}, {})
    assert gate["pass"] is False
    assert any("n_pairs" in r for r in gate["reasons"])


def test_fast_stats_close_stricter_than_n1r():
    eps = 16 / 255
    a = {"linf": eps, "l2_rms": 0.020, "tv": 0.0040, "spec_hf_frac": 0.4, "spec_mean": 1}
    close = {"linf": eps, "l2_rms": 0.021, "tv": 0.0044, "spec_hf_frac": 0.4, "spec_mean": 1}
    mid_l2 = {"linf": eps, "l2_rms": 0.023, "tv": 0.0040, "spec_hf_frac": 0.4, "spec_mean": 1}
    assert fast_stats_close(a, close, eps)
    assert not fast_stats_close(a, mid_l2, eps)
    assert stats_close(a, mid_l2, eps)


def test_fast_match_same_cell_one_to_one():
    eps = 16 / 255
    rows = []
    for i in range(8):
        kind = "rhc" if i < 4 else "safe"
        rows.append(_rec("h01", "c05", eps, i, kind, linf=eps, l2=0.02, tv=0.004, cat="cyber"))
        rows.append(_rec("h11", "c06", eps, i, kind, linf=eps, l2=0.02, tv=0.004, cat="fraud"))
    pairs = fast_match_pairs(rows)
    assert len(pairs) == 8
    assert {p["carrier_id"] for p in pairs} == {"c05", "c06"}
    assert all(p["l2_rel_diff"] <= 0.10 for p in pairs)
    assert all(p["tv_rel_diff"] <= 0.20 for p in pairs)
