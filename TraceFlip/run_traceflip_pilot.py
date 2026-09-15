#!/usr/bin/env python3
"""TraceFlip GPU pilot — all pilot cells, every method, equal backward budget.

    python TraceFlip/run_traceflip_pilot.py --scale mini --max-backward 120
    python TraceFlip/run_traceflip_pilot.py --queries h49 --carriers c07 \
        --methods traceflip refusal_margin_pgd clean

Writes ``out/pilot.json`` (resume-safe), ``out/pilot_report.md`` and
``out/pilot_public.json``. No raw model text is ever written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
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
from traceflip.metrics import aggregate, compare  # noqa: E402
from traceflip.protocol import (  # noqa: E402
    ALPHA,
    EPS,
    MAX_NEW_TOKENS,
    OPT_SEEDS,
    PILOT_CARRIERS,
    PILOT_QUERIES,
    TOPK,
    assert_seed_split,
)
from traceflip.report import render  # noqa: E402
from traceflip.run import (  # noqa: E402
    DEFAULT_METHODS,
    KNOWN_METHODS,
    _meta as build_run_meta,
    run_cells,
)

OUT = HERE / "out"
OUT_JSON = OUT / "pilot.json"
OUT_MD = OUT / "pilot_report.md"
OUT_STATE = OUT / "pilot_progress.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TraceFlip pilot run")
    ap.add_argument("--scale", default="mini", choices=["mini", "full"])
    ap.add_argument("--setting", default="native")
    ap.add_argument("--queries", nargs="*", default=list(PILOT_QUERIES))
    ap.add_argument("--carriers", nargs="*", default=list(PILOT_CARRIERS))
    ap.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    ap.add_argument("--max-backward", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    assert_seed_split()
    seed_all(OPT_SEEDS[0])
    OUT.mkdir(parents=True, exist_ok=True)
    if args.eps > EPS + 1e-9:
        raise RuntimeError(f"eps {args.eps} exceeds frozen 16/255 budget")
    unknown_methods = sorted(set(args.methods) - set(KNOWN_METHODS))
    if unknown_methods:
        raise RuntimeError(f"unknown methods: {unknown_methods}")
    if len(args.methods) != len(set(args.methods)):
        raise RuntimeError("duplicate methods are not allowed")
    if args.max_backward < 1 or args.max_new_tokens < 1 or args.topk < 1:
        raise RuntimeError("budgets, token count and top-k must be positive")
    for q in args.queries:
        if q not in set(PILOT_QUERIES):
            raise RuntimeError(f"query {q} outside the pilot pool")
    for c in args.carriers:
        if c not in set(PILOT_CARRIERS):
            raise RuntimeError(f"carrier {c} outside the pilot pool")

    cells = build_cells(tuple(args.queries), tuple(args.carriers))
    print(
        json.dumps(
            {
                "n_cells": len(cells),
                "methods": list(args.methods),
                "max_backward": args.max_backward,
                "eps": args.eps,
                "gpu_free_mib": round(torch.cuda.mem_get_info()[0] / 1024 / 1024)
                if torch.cuda.is_available()
                else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for c in cells:
            print(f"cell {c['query_id']}:{c['carrier_id']}", flush=True)
        return 0

    done: dict[str, list[dict]] = {}
    resumed_rows: list[dict] = []
    if args.resume and OUT_JSON.exists():
        blob = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        _validate_resume_meta(blob.get("meta") or {}, args)
        for row in blob.get("cells") or []:
            cell = row.get("cell") or {}
            key = f"{cell.get('query_id')}:{cell.get('carrier_id')}"
            done.setdefault(key, []).append(row)
            resumed_rows.append(row)
        print({"resumed_cells": {k: len(v) for k, v in done.items()}}, flush=True)

    def already(cell: dict, method: str) -> bool:
        key = f"{cell['query_id']}:{cell['carrier_id']}"
        return any(r.get("method") == method for r in done.get(key, []))

    pending_jobs = sum(
        1 for c in cells for m in args.methods if not already(c, m)
    )
    print(
        {
            "pending_jobs": pending_jobs,
            "completed_jobs": len(cells) * len(args.methods) - pending_jobs,
        },
        flush=True,
    )
    if not pending_jobs:
        print("nothing to do", flush=True)
        return 0

    if not args.resume:
        # Replace stale artefacts before the first expensive method starts.
        # Otherwise a crash in job zero leaves an old, incompatible pilot.json
        # that looks like the current run and cannot be safely resumed.
        save_json(
            OUT_JSON,
            {
                "cells": [],
                "meta": build_run_meta(
                    cells,
                    tuple(args.methods),
                    args.eps,
                    args.alpha,
                    args.max_backward,
                    args.max_new_tokens,
                    args.topk,
                ),
            },
        )

    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale=args.scale)
    wrapper = load_model(cfg, setting=args.setting)

    state = {
        "cells_done": 0,
        "n_cells": len(cells),
        "pending_jobs": pending_jobs,
        "started": _now(),
    }

    def progress(info: dict) -> None:
        state["cells_done"] = int(info.get("cell_index", 0)) + 1
        save_json(OUT_STATE, state)

    try:
        run = run_cells(
            wrapper,
            cells,
            methods=tuple(args.methods),
            eps=args.eps,
            alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
            max_backward=args.max_backward,
            topk=args.topk,
            out_path=OUT_JSON,
            progress=progress,
            initial_cells=resumed_rows,
        )
    except Exception:
        save_json(OUT_STATE, {**state, "error": traceback.format_exc()[-2000:]})
        raise

    # Rebuild the record from every cell we have (resumed + fresh) so the report
    # is over the full cell set, not just this invocation.
    rows = _dedupe(list(run["public_cells"]))
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(str(r.get("method")), []).append(r)
    agg = aggregate(by_method)
    cmp = compare(by_method, "refusal_margin_pgd") if "refusal_margin_pgd" in by_method else {}
    meta = dict(run["meta"])
    meta.update({"n_cells": len(cells), "n_rows": len(rows)})
    md = render(agg, cmp, meta=meta)
    OUT_MD.write_text(md, encoding="utf-8")
    save_json(OUT / "pilot_public.json", {"meta": meta, "cells": rows})
    save_json(
        OUT_STATE,
        {**state, "finished": _now(), "terminal_labels": _label_counts(rows)},
    )
    print(md, flush=True)
    print(f"wrote {OUT_MD}", flush=True)
    return 0


def _dedupe(rows: list[dict]) -> list[dict]:
    """Later rows win, keyed by (query, carrier, method)."""
    out: dict[tuple, dict] = {}
    for r in rows:
        cell = r.get("cell") or {}
        key = (
            r.get("query_id") or cell.get("query_id"),
            r.get("carrier_id") or cell.get("carrier_id"),
            r.get("method"),
        )
        out[key] = r
    return list(out.values())


def _validate_resume_meta(meta: dict, args: argparse.Namespace) -> None:
    """Refuse to mix checkpoints from a different experimental contract."""
    expected = {
        "implementation_revision": "budget-value-prefix-resume-v2",
        "model": "Qwen2-VL-7B-Instruct",
        "eps": float(args.eps),
        "alpha": float(args.alpha),
        "decode": "greedy",
        "max_new_tokens": int(args.max_new_tokens),
        "max_backward_per_cell": int(args.max_backward),
        "topk": int(args.topk),
        "queries": sorted(str(x) for x in args.queries),
        "carriers": sorted(str(x) for x in args.carriers),
        "store_raw_outputs": False,
    }
    mismatches = {
        key: {"saved": meta.get(key), "requested": value}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    saved_methods = list(meta.get("methods") or [])
    requested_methods = list(args.methods)
    if not set(requested_methods).issubset(set(saved_methods) or set(requested_methods)):
        mismatches["methods"] = {"saved": saved_methods, "requested": requested_methods}
    if mismatches:
        raise RuntimeError(
            "resume metadata mismatch; refusing to mix incompatible runs: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _label_counts(cells: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in cells:
        key = f"{r.get('method')}:{r.get('terminal_label')}"
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
