#!/usr/bin/env python3
"""Final honesty check for the Bug-E-fixed pilot.

Compares the token ids that the *actual pilot run* stored for the ``clean``
method (produced by ``traceflip.repair.clean_cell -> execute -> greedy_trace``)
against a fresh, official ``model.generate`` decode of the same image/prompt.

If these agree token-for-token on every cell, then the pilot's labelling path
ran on the *unmodified frozen decoder*, which is the precondition for the
honesty invariant to mean anything.

No raw text is printed; only token ids, hashes and booleans.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("P0_QWEN_FORCE_GPU", "1")
os.environ.setdefault("P0_QWEN_KEEP_336", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from p0_qwen.config import load_cfg  # noqa: E402
from run_p0_qwen import load_model, seed_all  # noqa: E402
from traceflip.datasets import build_cells  # noqa: E402
from traceflip.protocol import MAX_NEW_TOKENS, OPT_SEEDS  # noqa: E402


def main() -> int:
    seed_all(OPT_SEEDS[0])
    cfg = load_cfg()
    wrapper = load_model(cfg)

    pub = json.loads((HERE / "pilot_public.json").read_text(encoding="utf-8"))
    stored = {
        (c["query_id"], c["carrier_id"]): c
        for c in pub["cells"]
        if c["method"] == "clean"
    }
    cells = build_cells(
        tuple(sorted({q for q, _ in stored})),
        tuple(sorted({ca for _, ca in stored})),
    )

    n_agree = 0
    for cell in cells:
        key = (cell["query_id"], cell["carrier_id"])
        rec = stored.get(key)
        if rec is None:
            continue
        image = cell["image"]
        base = wrapper.encode(image, cell["item"]["query"])
        gen = wrapper.model.generate(
            input_ids=base["input_ids"],
            attention_mask=base.get("attention_mask"),
            pixel_values=base.get("pixel_values"),
            image_grid_thw=base.get("image_grid_thw"),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
        gen_ids = gen[0][base["input_ids"].shape[1]:].tolist()
        pilot_ids = [int(x) for x in rec["trace"]["token_ids"]]
        agree = gen_ids == pilot_ids
        n_agree += int(agree)
        print(
            json.dumps(
                {
                    "cell": f"{key[0]}:{key[1]}",
                    "official_n": len(gen_ids),
                    "pilot_n": len(pilot_ids),
                    "first_mismatch": next(
                        (i for i, (a, b) in enumerate(zip(gen_ids, pilot_ids)) if a != b),
                        None,
                    ),
                    "agree": agree,
                }
            ),
            flush=True,
        )

    print(json.dumps({"cells_checked": len(cells), "agree": n_agree, "ALL_MATCH": n_agree == len(cells)}))
    return 0 if n_agree == len(cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
