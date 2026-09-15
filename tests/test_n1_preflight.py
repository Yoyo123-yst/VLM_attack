import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from n1.preflight import classify_condition, verdict


def test_sampling_is_same_x01():
    rec = {
        "related_safe_from_sample": True,
        "n0_matched_attempt": {"tag": "sample_3"},
        "delta_path": "/tmp/a.pt",
        "query_id": "h01",
        "carrier_id": "c01",
    }
    cls = classify_condition(rec)
    assert cls["family"] == "same_x01_sampling"
    assert cls["difference"] == "decoding_randomness_only"
    assert cls["pixels_expected_equal"] is True


def test_scaled_delta_is_perturbation():
    rec = {
        "related_safe_from_sample": False,
        "n0_matched_attempt": {"tag": "scale_0.6"},
        "delta_path": "/tmp/a.pt",
        "related_safe_delta_path": "/tmp/b.pt",
    }
    cls = classify_condition(rec)
    assert cls["family"] == "different_perturbation"
    assert cls["pixels_expected_equal"] is False


def test_verdict_stops_when_discover_is_sampling():
    rows = []
    for i in range(30):
        rows.append(
            {
                "n0_split": "discover",
                "family": "same_x01_sampling",
                "pixels_equal": True,
                "input_ids_equal": True,
                "last_prompt_hidden_cosine": 1.0,
                "rhc_generation_condition": "greedy_decode_on_pgd_perturbed_image",
                "hidden_includes_generated_tokens": False,
                "safe_chars": 100,
                "rhc_chars": 120,
            }
        )
    gate = verdict({"rows": rows})
    assert gate["preflight_pass"] is False
    assert gate["allow_residual_scan"] is False
    assert gate["status"] == "stop_recollect"
