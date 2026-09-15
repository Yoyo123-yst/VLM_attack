#!/usr/bin/env python3
"""TraceFlip CPU preflight: freeze-file integrity, tests, and a fake-model dry run.

Runs with no GPU and no model weights. It checks:

1. ``TRACEFLIP_FROZEN.json`` matches the constants in ``traceflip.protocol``
   (a freeze file that drifts from the code is worse than no freeze file).
2. The CPU test-suite passes.
3. ``run_cells`` walks the full Execute -> Probe -> Select -> Flip -> Re-execute
   loop against a tiny stub wrapper, so control flow and serialisation are
   exercised without a VLM.

    python TraceFlip/check_traceflip_cpu.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

FROZEN = HERE / "TRACEFLIP_FROZEN.json"


def check_freeze_matches_code() -> list[str]:
    from traceflip import protocol as P

    blob = json.loads(FROZEN.read_text(encoding="utf-8"))
    problems: list[str] = []
    pairs = [
        (blob["budget"]["eps"], P.EPS),
        (blob["budget"]["alpha"], P.ALPHA),
        (blob["budget"]["image_size"], P.IMAGE_SIZE),
        (blob["budget"]["max_new_tokens"], P.MAX_NEW_TOKENS),
        (blob["solver"]["keep_kappa_nats"], P.KEEP_KAPPA),
        (blob["solver"]["flip_kappa_nats"], P.FLIP_KAPPA),
        (blob["solver"]["lambda_keep"], P.LAMBDA_KEEP),
        (blob["solver"]["outer_steps"], P.OUTER_STEPS),
        (blob["solver"]["inner_steps"], P.INNER_STEPS),
        (blob["solver"]["topk"], P.TOPK),
        (blob["solver"]["probe_horizon"], P.PROBE_HORIZON),
        (blob["solver"]["max_positions"], P.MAX_POSITIONS),
        (blob["solver"]["min_branch_position"], P.MIN_BRANCH_POSITION),
        (blob["solver"]["eps0"], P.EPS0),
        (blob["solver"]["grad_check_min"], P.GRAD_CHECK_MIN),
        (blob["solver"]["grad_check_frac"], P.GRAD_CHECK_FRAC),
    ]
    for frozen, live in pairs:
        if abs(float(frozen) - float(live)) > 1e-12:
            problems.append(f"frozen {frozen} != code {live}")
    if list(blob["seeds"]["opt"]) != list(P.OPT_SEEDS):
        problems.append("opt seeds drift")
    if list(blob["seeds"]["eval"]) != list(P.EVAL_SEEDS):
        problems.append("eval seeds drift")
    if tuple(blob["scope"]["queries"]) != tuple(P.PILOT_QUERIES):
        problems.append("pilot queries drift")
    if tuple(blob["scope"]["carriers"]) != tuple(P.PILOT_CARRIERS):
        problems.append("pilot carriers drift")
    if blob["scope"]["store_raw_outputs"] is not False:
        problems.append("store_raw_outputs must be false")
    if list(blob["success"]["non_success_labels"]) != list(P.TERMINAL_LABELS)[:2] + list(
        P.TERMINAL_LABELS[3:]
    ):
        problems.append("non-success label set drift")
    return problems


def run_cpu_tests() -> bool:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_traceflip_cpu.py")
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    return res.wasSuccessful()


def stub_dry_run() -> dict:
    """Exercise the full loop with a 2-token vocabulary VLM stub."""
    import torch

    from traceflip.protocol import EPS
    from traceflip.run import run_cells

    class StubWrapper:
        """Minimal stand-in: greedy picks token 0 always, so branch 1 exists."""

        image_size = 336

        def __init__(self):
            self.device = torch.device("cpu")

        def image_to_x01(self, image):
            return torch.full((1, 3, 8, 8), 0.5)

        def patchify(self, x01):
            return x01.float(), torch.tensor([[1, 8, 8]])

        def encode(self, image, question):
            ids = torch.tensor([[1, 2, 3]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "pixel_values": self.patchify(self.image_to_x01(image))[0],
                "image_grid_thw": torch.tensor([[1, 8, 8]]),
            }

        def _apply_pixels(self, inputs, x01=None, pixel_values=None, image_grid_thw=None):
            out = dict(inputs)
            if x01 is not None:
                pv, grid = self.patchify(x01)
                out["pixel_values"] = pv
                out["image_grid_thw"] = grid
            elif pixel_values is not None:
                out["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    out["image_grid_thw"] = image_grid_thw
            return out

        class _Out:
            def __init__(self, logits):
                self.logits = logits
                self.past_key_values = None
                self.attentions = None
                self.hidden_states = None

        class _M:
            def __init__(self):
                import types

                self.generation_config = types.SimpleNamespace(eos_token_id=[9])

            def __call__(self, **kw):
                n = int(kw["input_ids"].shape[-1])
                logits = torch.zeros(1, n, 8)
                # Greedy prefers token 0 (0.0) over token 1 (-0.05), but token 1's
                # logit is a steep decreasing function of the pixel mean, so a
                # tiny delta flips exactly one branch. Token 1 is second in the
                # top-k ordering, so the probe has exactly one candidate and the
                # gradient through the pixels is well-conditioned.
                logits[..., 0] = 0.0
                logits[..., 1] = -0.05
                logits[..., 2:] = -10.0
                pv = kw.get("pixel_values")
                # Keep generation position zero invariant so the dry run
                # exercises a real non-empty prefix (TraceFlip excludes t=0).
                if pv is not None and n >= 4:
                    shift = 10.0 * (0.5 - pv.float().mean())
                    logits = logits.clone()
                    logits[:, -1, 1] = logits[:, -1, 1] + shift
                return StubWrapper._Out(logits)

            def prepare_inputs_for_generation(self, ids, **kw):
                # Qwen2-VL indexes cache_position unconditionally on the first
                # call, so a caller that forgets to seed it crashes on the real
                # model. Asserting here makes the CPU preflight catch that bug.
                if "cache_position" not in kw:
                    raise TypeError("cache_position must be seeded before the first call")
                if "attention_mask" in kw and kw["attention_mask"].shape[-1] != ids.shape[-1]:
                    raise ValueError("attention_mask must cover forced prefix and generated tokens")
                # Forward the pixel tensors, exactly as the real wrapper does;
                # dropping them would make the re-decode diverge from the
                # in-model flip check.
                out = {"input_ids": ids}
                for key in ("pixel_values", "image_grid_thw", "attention_mask"):
                    if key in kw:
                        out[key] = kw[key]
                return out

            def _update_model_kwargs_for_generation(self, out, kw, **extra):
                updated = dict(kw)
                if "attention_mask" in updated:
                    am = updated["attention_mask"]
                    updated["attention_mask"] = torch.cat(
                        [am, torch.ones(am.shape[0], 1, dtype=am.dtype, device=am.device)],
                        dim=-1,
                    )
                return updated

        class _Tok:
            eos_token_id = 9

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2]}

            def encode(self, text, add_special_tokens=False):
                return [1, 2]

            def convert_ids_to_tokens(self, ids):
                return [f"t{int(i)}" for i in ids]

        class _Proc:
            def __init__(self):
                self.tokenizer = StubWrapper._Tok()

            def batch_decode(self, ids, skip_special_tokens=True):
                return ["stub response"]

        @property
        def model(self):
            if getattr(self, "_m", None) is None:
                self._m = StubWrapper._M()
            return self._m

        @property
        def processor(self):
            if getattr(self, "_p", None) is None:
                self._p = StubWrapper._Proc()
            return self._p

    cells = [
        {
            "query_id": "h49",
            "carrier_id": "c07",
            "item": {"id": "h49", "query": "q", "topic_keywords": ["x"]},
            "carrier": {"id": "c07"},
            "image": None,
        }
    ]
    wrapper = StubWrapper()
    # The label step needs a tokenizer; replace it with a fixed label so the
    # dry run stays independent of transformers.
    import traceflip.trace as T

    orig = T.label_trace
    T.label_trace = lambda trace, kw, q: {
        "terminal_label": "REFUSE",
        "core_rhc": False,
        "core_safe_answer": False,
        "safety": "SAFE",
        "response_mode": "REFUSE",
        "grounding": "UNGROUNDED",
        "quality": "FLUENT",
        "chars": len(trace.get("text") or ""),
        "relevance": 0.0,
    }
    try:
        run = run_cells(
            wrapper,
            cells,
            methods=("clean", "traceflip"),
            eps=EPS,
            max_backward=60,
            max_new_tokens=3,
            topk=2,
            log=lambda *_: None,
        )
        resumed = run_cells(
            wrapper,
            cells,
            methods=("clean", "traceflip"),
            eps=EPS,
            max_backward=60,
            max_new_tokens=3,
            topk=2,
            log=lambda *_: None,
            initial_cells=run["public_cells"],
        )
    finally:
        T.label_trace = orig
    tr = next(r for r in run["public_cells"] if r.get("method") == "traceflip")
    return {
        "methods": sorted(run["aggregate"]["methods"]),
        "n_public_cells": len(run["public_cells"]),
        "has_diagnostics": bool(run["aggregate"]["diagnostics"]),
        "attempts": int(tr.get("n_attempts") or 0),
        "valid_flips": int(tr.get("n_valid_flips") or 0),
        "backward": int((tr.get("budget") or {}).get("backward_passes") or 0),
        "delta_linf": float(tr.get("delta_linf") or 0.0),
        "resume_rows": len(resumed["public_cells"]),
    }


def main() -> int:
    problems = check_freeze_matches_code()
    if problems:
        print("FREEZE MISMATCH:")
        for p in problems:
            print(" -", p)
        return 1
    print("freeze file matches traceflip.protocol", flush=True)

    if not run_cpu_tests():
        return 1
    print("CPU tests passed", flush=True)

    info = stub_dry_run()
    print("stub dry run:", json.dumps(info, ensure_ascii=False), flush=True)
    if info["n_public_cells"] < 2:
        print("stub dry run did not produce both methods", flush=True)
        return 1
    if info["resume_rows"] != info["n_public_cells"]:
        print("resume duplicated or dropped completed jobs", flush=True)
        return 1
    if info["attempts"] < 1:
        print("stub dry run never entered the probe/select/flip loop", flush=True)
        return 1
    if info["backward"] < 1:
        print("stub dry run recorded no backward passes", flush=True)
        return 1
    if info["valid_flips"] < 1:
        print("stub dry run produced no accepted prefix-preserving flip", flush=True)
        return 1
    if info["delta_linf"] <= 0.0:
        print("stub dry run left delta at zero, so the flip cannot have been applied", flush=True)
        return 1
    print("PREFLIGHT OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
