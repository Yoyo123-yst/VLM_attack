#!/usr/bin/env python3
"""GateFlip GPU probe — separate artefacts from the frozen TraceFlip pilot.

    python TraceFlip/run_gateflip_probe.py

Default: token-wise k* on selected cells. Writes ``TraceFlip/out/gateflip_kstar_fine/``
and does not touch ``pilot.json``.
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
from traceflip.run import KNOWN_METHODS, run_cells  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GateFlip probe (does not overwrite TraceFlip pilot)")
    ap.add_argument("--scale", default="mini", choices=["mini", "full"])
    ap.add_argument("--setting", default="native")
    ap.add_argument("--queries", nargs="*", default=["h53", "h66"])
    ap.add_argument("--carriers", nargs="*", default=list(PILOT_CARRIERS))
    ap.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="exact subset like h53:c07; overrides the query×carrier product",
    )
    ap.add_argument("--methods", nargs="*", default=["gateflip"])
    ap.add_argument("--ladder", default="fine", choices=["fine", "coarse"])
    ap.add_argument("--max-backward", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out-dir", default=str(HERE / "out" / "gateflip_kstar_fine"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    assert_seed_split()
    seed_all(OPT_SEEDS[0])
    out = Path(args.out_dir)
    out_json = out / "probe.json"
    out_md = out / "probe_report.md"
    out_state = out / "probe_progress.json"
    out.mkdir(parents=True, exist_ok=True)
    if args.eps > EPS + 1e-9:
        raise RuntimeError(f"eps {args.eps} exceeds frozen 16/255 budget")
    unknown = sorted(set(args.methods) - set(KNOWN_METHODS))
    if unknown:
        raise RuntimeError(f"unknown methods: {unknown}")
    queries = list(args.queries)
    carriers = list(args.carriers)
    cell_filter = None
    if args.cells:
        want = []
        for spec in args.cells:
            if ":" not in spec:
                raise RuntimeError(f"cell spec must be query:carrier, got {spec}")
            q, c = spec.split(":", 1)
            if q not in set(PILOT_QUERIES) or c not in set(PILOT_CARRIERS):
                raise RuntimeError(f"cell {spec} outside the pilot pool")
            want.append((q, c))
        cell_filter = {(q, c) for q, c in want}
        queries = list(dict.fromkeys(q for q, _ in want))
        carriers = list(dict.fromkeys(c for _, c in want))
    for q in queries:
        if q not in set(PILOT_QUERIES):
            raise RuntimeError(f"query {q} outside the pilot pool")
    for c in carriers:
        if c not in set(PILOT_CARRIERS):
            raise RuntimeError(f"carrier {c} outside the pilot pool")

    cells = build_cells(tuple(queries), tuple(carriers))
    if cell_filter is not None:
        cells = [x for x in cells if (x["query_id"], x["carrier_id"]) in cell_filter]
        missing = cell_filter - {(x["query_id"], x["carrier_id"]) for x in cells}
        if missing:
            raise RuntimeError(f"cells not in built grid: {sorted(missing)}")
    print(
        json.dumps(
            {
                "run": "gateflip",
                "ladder": args.ladder,
                "out_dir": str(out),
                "n_cells": len(cells),
                "methods": list(args.methods),
                "queries": queries,
                "carriers": carriers,
                "cells": [f"{c['query_id']}:{c['carrier_id']}" for c in cells],
                "max_backward": args.max_backward,
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

    resumed_rows: list[dict] = []
    if args.resume and out_json.exists():
        blob = json.loads(out_json.read_text(encoding="utf-8"))
        resumed_rows = list(blob.get("cells") or [])
        print({"resumed_rows": len(resumed_rows)}, flush=True)

    if not args.resume:
        save_json(
            out_json,
            {
                "cells": [],
                "meta": {
                    "implementation_revision": "gateflip-v2",
                    "ladder": args.ladder,
                    "queries": queries,
                    "carriers": carriers,
                    "methods": list(args.methods),
                },
            },
        )

    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale=args.scale)
    wrapper = load_model(cfg, setting=args.setting)
    state = {"started": _now(), "n_cells": len(cells)}

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
            out_path=out_json,
            initial_cells=resumed_rows,
            gateflip_ladder=args.ladder,
        )
    except Exception:
        save_json(out_state, {**state, "error": traceback.format_exc()[-2000:]})
        raise

    rows = list(run["public_cells"])
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(str(r.get("method")), []).append(r)
    agg = aggregate(by_method)
    cmp = compare(by_method, "refusal_margin_pgd") if "refusal_margin_pgd" in by_method else {}
    meta = dict(run["meta"])
    meta["implementation_revision"] = "gateflip-v2"
    meta["gateflip_ladder"] = args.ladder
    md = render(agg, cmp, meta=meta)
    out_md.write_text(md, encoding="utf-8")
    save_json(out / "probe_public.json", {"meta": meta, "cells": rows})
    save_json(
        out_state,
        {
            **state,
            "finished": _now(),
            "n_rows": len(rows),
            "win_stages": _win_stages(rows),
            "k_stars": _k_stars(rows),
            "terminal_labels": _label_counts(rows),
        },
    )
    print(md, flush=True)
    print(f"wrote {out_md}", flush=True)
    return 0


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _label_counts(cells: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in cells:
        key = f"{r.get('method')}:{r.get('terminal_label')}"
        out[key] = out.get(key, 0) + 1
    return out


def _win_stages(cells: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in cells:
        key = str(r.get("win_stage") or "none")
        out[key] = out.get(key, 0) + 1
    return out


def _k_stars(cells: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in cells:
        key = f"{r.get('k_star_name')}:{r.get('k_star')}"
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
