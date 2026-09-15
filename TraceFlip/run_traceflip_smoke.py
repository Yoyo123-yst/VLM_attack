#!/usr/bin/env python3
"""TraceFlip GPU smoke test — 1 query x 1 carrier x eps=16/255.

Purpose: prove the pipeline executes end to end on the real VLM before any
pilot run. It does not measure a method; it measures that Execute, Probe,
Select, Constrained Flip and Re-execute all run, that a delta stays inside the
L-inf box, and that a clean decode still produces a four-axis label.

    python TraceFlip/run_traceflip_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
os.environ.setdefault("P0_QWEN_KEEP_336", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from p0.datautil import save_json  # noqa: E402
from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import load_model, seed_all  # noqa: E402
from traceflip.datasets import build_cells  # noqa: E402
from traceflip.protocol import (  # noqa: E402
    EPS,
    OPT_SEEDS,
    OUTER_STEPS,
    assert_seed_split,
)
from traceflip.repair import traceflip_cell  # noqa: E402
from traceflip.run import TRACEFLIP_VARIANTS  # noqa: E402
from traceflip.trace import public_trace  # noqa: E402

OUT = HERE / "out"
QUERY = os.environ.get("TRACEFLIP_QUERY", "h49")
CARRIER = os.environ.get("TRACEFLIP_CARRIER", "c07")
MAX_BACKWARD = int(os.environ.get("TRACEFLIP_MAX_BACKWARD", "60"))
GEN_TOKENS = int(os.environ.get("TRACEFLIP_MAX_NEW_TOKENS", "48"))


def main() -> int:
    assert_seed_split()
    seed_all(OPT_SEEDS[0])
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("smoke test needs CUDA")
    print(
        {
            "gpu": torch.cuda.get_device_name(0),
            "free_mib": round(torch.cuda.mem_get_info()[0] / 1024 / 1024),
            "eps": EPS,
            "query": QUERY,
            "carrier": CARRIER,
            "max_backward": MAX_BACKWARD,
        },
        flush=True,
    )
    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale="mini")
    wrapper = load_model(cfg, setting="native")

    cells = build_cells((QUERY,), (CARRIER,))
    if not cells:
        raise RuntimeError("no cells built; is the CR-0 cell set present?")
    cell = cells[0]
    item = dict(cell["item"])
    item.setdefault("carrier_id", cell["carrier_id"])
    image = cell["image"]
    x0 = wrapper.image_to_x01(image)

    rec = traceflip_cell(
        wrapper,
        item,
        cell["carrier_id"],
        image,
        x0,
        eps=EPS,
        max_new_tokens=GEN_TOKENS,
        max_backward=MAX_BACKWARD,
        **TRACEFLIP_VARIANTS["traceflip"],
    )
    tr = rec["trace"]
    attempts = rec.get("attempts") or []
    # Any attempt that claims in-model feasibility must actually have flipped the
    # branch under re-decode. The earlier incumbent-delta margin could report
    # feasible_in_model=True while producing zero valid flips, so this is an
    # explicit, load-bearing invariant, not a nicety.
    feasible_but_not_flipped = [
        a for a in attempts if a.get("feasible_in_model") and not a.get("branch_flipped")
    ]
    checks = {
        "ran": True,
        "linf_within_box": float(rec["delta_linf"]) <= EPS + 1e-6,
        "delta_finite": bool(torch.isfinite(rec["delta"]).all()),
        "greedy_steps_recorded": len(tr["steps"]) > 0,
        "budget_recorded": (rec["budget"] or {}).get("backward_passes", 0) >= 0,
        "backward_within_budget": (rec["budget"] or {}).get("backward_passes", 10**9)
        <= MAX_BACKWARD,
        "attempts_within_retries": int(rec["n_attempts"]) <= 6 * int(OUTER_STEPS),
        "label_valid": tr["terminal_label"]
        in {"REFUSE", "RELATED_SAFE", "RHC", "DENY", "INVALID", "OTHER"},
        "inmodel_implies_flip": len(feasible_but_not_flipped) == 0,
        "no_raw_text_in_public": "text" not in public_trace(tr),
    }
    payload = {
        "written": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "cell": {"query_id": QUERY, "carrier_id": CARRIER},
        "eps": float(EPS),
        "max_backward": MAX_BACKWARD,
        "terminal_label": rec["terminal_label"],
        "core_rhc": bool(rec["core_rhc"]),
        "n_attempts": int(rec["n_attempts"]),
        "n_valid_flips": int(rec["n_valid_flips"]),
        "delta_linf": float(rec["delta_linf"]),
        "budget": rec["budget"],
        "wall_seconds": float(rec["wall_seconds"]),
        "n_trace_steps": len(tr["steps"]),
        "trace_text_hash": tr["text_hash"],
        "trace": public_trace(tr, keep_steps=3),
        "attempts": [
            {
                k: v
                for k, v in a.items()
                if k
                in {
                    "outer",
                    "t",
                    "token",
                    "score",
                    "value",
                    "cost",
                    "feasible_in_model",
                    "g_flip_final",
                    "min_prefix_margin",
                    "n_prefix_positions",
                    "n_keep_broken",
                    "prefix_kept",
                    "branch_flipped",
                    "valid",
                    "improved",
                    "accepted",
                    "n_backward",
                    "terminal_label",
                    "core_rhc",
                }
            }
            for a in (rec.get("attempts") or [])
        ],
        "checks": checks,
    }
    save_json(OUT / "smoke.json", payload)
    print(json.dumps({"checks": checks, "terminal_label": payload["terminal_label"],
                      "n_attempts": payload["n_attempts"],
                      "n_valid_flips": payload["n_valid_flips"],
                      "delta_linf": payload["delta_linf"]}, ensure_ascii=False, indent=2),
          flush=True)
    bad = [k for k, v in checks.items() if not v]
    if bad:
        print(f"SMOKE FAIL: {bad}", flush=True)
        return 1
    print("SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
