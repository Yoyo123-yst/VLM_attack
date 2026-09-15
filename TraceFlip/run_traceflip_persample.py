#!/usr/bin/env python3
"""TraceFlip per-sample runner with resumable per-(cell, method) jobs.

Unlike the pilot script this one is designed for long GPU runs: it keeps one
record per job, can run a subset of jobs, and resumes by job id.

    python TraceFlip/run_traceflip_persample.py --list
    python TraceFlip/run_traceflip_persample.py --job h49:c07:traceflip
    python TraceFlip/run_traceflip_persample.py --methods traceflip --max-backward 120
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
    TRACEFLIP_VARIANTS,
    _public_cell,
    _to_device,
)
from traceflip.repair import baseline_cell, clean_cell, traceflip_cell  # noqa: E402
from traceflip.run import _prefix_ids  # noqa: E402

OUT = HERE / "out"
JOBS_JSON = OUT / "persample_jobs.json"
JOBS_MD = OUT / "persample_report.md"
LOG = OUT / "persample.log"


def job_id(qid: str, cid: str, method: str) -> str:
    return f"{qid}:{cid}:{method}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TraceFlip per-sample runner")
    ap.add_argument("--scale", default="mini", choices=["mini", "full"])
    ap.add_argument("--setting", default="native")
    ap.add_argument("--queries", nargs="*", default=list(PILOT_QUERIES))
    ap.add_argument("--carriers", nargs="*", default=list(PILOT_CARRIERS))
    ap.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    ap.add_argument("--job", nargs="*", default=None, help="explicit job ids")
    ap.add_argument("--max-backward", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_jobs() -> dict[str, dict]:
    if not JOBS_JSON.exists():
        return {}
    blob = json.loads(JOBS_JSON.read_text(encoding="utf-8"))
    return {k: v for k, v in (blob.get("jobs") or {}).items()}


def main() -> int:
    args = parse_args()
    assert_seed_split()
    seed_all(OPT_SEEDS[0])
    OUT.mkdir(parents=True, exist_ok=True)
    if args.eps > EPS + 1e-9:
        raise RuntimeError(f"eps {args.eps} exceeds frozen 16/255 budget")
    unknown_methods = sorted(set(args.methods) - set(DEFAULT_METHODS))
    if unknown_methods:
        raise RuntimeError(f"unknown methods: {unknown_methods}")
    for q in args.queries:
        if q not in set(PILOT_QUERIES):
            raise RuntimeError(f"query {q} outside the pilot pool")
    for c in args.carriers:
        if c not in set(PILOT_CARRIERS):
            raise RuntimeError(f"carrier {c} outside the pilot pool")
    if JOBS_JSON.exists():
        saved = json.loads(JOBS_JSON.read_text(encoding="utf-8"))
        _validate_resume_meta(saved.get("meta") or {}, args)

    cells = build_cells(tuple(args.queries), tuple(args.carriers))
    all_jobs = [job_id(c["query_id"], c["carrier_id"], m) for c in cells for m in args.methods]
    if args.list:
        for j in all_jobs:
            print(j)
        return 0
    if args.dry_run:
        print(json.dumps({"n_jobs": len(all_jobs), "jobs": all_jobs[:10]}, ensure_ascii=False))
        return 0

    jobs: dict[str, dict] = {}
    if args.job:
        wanted = set(args.job)
        for j in wanted:
            parts = j.split(":")
            if len(parts) != 3:
                raise RuntimeError(f"bad job id {j}")
            jobs[j] = {"query_id": parts[0], "carrier_id": parts[1], "method": parts[2]}
        keep = [c for c in cells if any(c["query_id"] == v["query_id"] and c["carrier_id"] == v["carrier_id"] for v in jobs.values())]
        cells = keep
    else:
        jobs = {
            job_id(c["query_id"], c["carrier_id"], m): {
                "query_id": c["query_id"],
                "carrier_id": c["carrier_id"],
                "method": m,
            }
            for c in cells
            for m in args.methods
        }

    done = load_jobs()
    todo = {k: v for k, v in jobs.items() if k not in done or done[k].get("status") != "ok"}
    print(
        json.dumps(
            {
                "n_jobs": len(jobs),
                "todo": len(todo),
                "already_ok": len(jobs) - len(todo),
                "max_backward": args.max_backward,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not todo:
        print("nothing to do", flush=True)
        _report(done)
        return 0

    cfg = load_cfg(ROOT / "configs" / "p0_qwen.yaml", scale=args.scale)
    wrapper = load_model(cfg, setting=args.setting)
    prefix_ids = _prefix_ids(wrapper)
    cellmap = {(c["query_id"], c["carrier_id"]): c for c in cells}

    for jid, spec in todo.items():
        cell = cellmap[(spec["query_id"], spec["carrier_id"])]
        item = dict(cell["item"])
        item.setdefault("carrier_id", cell["carrier_id"])
        image = cell["image"]
        x0 = wrapper.image_to_x01(image)
        method = spec["method"]
        try:
            if method == "clean":
                rec = clean_cell(
                    wrapper, item, cell["carrier_id"], image, x0,
                    max_new_tokens=args.max_new_tokens,
                )
            elif method in ("refusal_margin_pgd", "targeted_prefix_pgd"):
                rec = baseline_cell(
                    wrapper, item, cell["carrier_id"], image, x0, method,
                    eps=args.eps, alpha=args.alpha, steps=int(args.max_backward),
                    max_new_tokens=args.max_new_tokens, prefix_ids=prefix_ids,
                )
            elif method in TRACEFLIP_VARIANTS:
                rec = traceflip_cell(
                    wrapper, item, cell["carrier_id"], image, x0,
                    eps=args.eps, alpha=args.alpha,
                    max_new_tokens=args.max_new_tokens, topk=args.topk,
                    max_backward=int(args.max_backward),
                    method_name=method,
                    **TRACEFLIP_VARIANTS[method],
                )
            else:
                raise KeyError(f"unknown method {method}")
            row = _to_device(_public_cell(rec))
            row["status"] = "ok"
            done[jid] = row
        except Exception:
            done[jid] = {"status": "error", "error": traceback.format_exc()[-1500:]}
        done[jid].setdefault("job", jid)
        done[jid]["finished"] = _now()
        save_json(JOBS_JSON, {"jobs": done, "meta": _meta(args)})
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_logline(jid, done[jid]), ensure_ascii=False) + "\n")
        print(json.dumps(_logline(jid, done[jid]), ensure_ascii=False), flush=True)

    _report(done)
    return 0


def _logline(jid: str, rec: dict) -> dict:
    return {
        "job": jid,
        "status": rec.get("status"),
        "terminal_label": rec.get("terminal_label"),
        "core_rhc": rec.get("core_rhc"),
        "n_attempts": rec.get("n_attempts"),
        "valid_flips": rec.get("n_valid_flips"),
        "backward": (rec.get("budget") or {}).get("backward_passes"),
        "seconds": rec.get("wall_seconds"),
    }


def _meta(args: argparse.Namespace) -> dict:
    return {
        "implementation_revision": "budget-value-prefix-resume-v2",
        "model": "Qwen2-VL-7B-Instruct",
        "eps": float(args.eps),
        "alpha": float(args.alpha),
        "decode": "greedy",
        "max_new_tokens": int(args.max_new_tokens),
        "max_backward_per_cell": int(args.max_backward),
        "topk": int(args.topk),
        "methods": list(args.methods),
        "queries": list(args.queries),
        "carriers": list(args.carriers),
        "success": "core_rhc",
        "store_raw_outputs": False,
    }


def _validate_resume_meta(meta: dict, args: argparse.Namespace) -> None:
    expected = _meta(args)
    mismatches = {
        key: {"saved": meta.get(key), "requested": value}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "existing per-sample jobs use a different experiment contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _report(done: dict[str, dict]) -> None:
    ok = {k: v for k, v in done.items() if v.get("status") == "ok"}
    if not ok:
        return
    results: dict[str, list[dict]] = {}
    for jid, rec in ok.items():
        results.setdefault(str(rec.get("method") or jid.split(":")[2]), []).append(rec)
    missing = [m for m in ("refusal_margin_pgd",) if m not in results]
    if missing:
        print(f"cannot compute paired comparison yet; missing baseline {missing}", flush=True)
        return
    agg = aggregate(results)
    cmp = compare(results, "refusal_margin_pgd")
    meta = {
        "queries": sorted({str(r.get("query_id")) for r in ok.values()}),
        "carriers": sorted({str(r.get("carrier_id")) for r in ok.values()}),
        "max_backward": max(
            int((r.get("budget") or {}).get("backward_passes") or 0) for r in ok.values()
        ),
    }
    JOBS_MD.write_text(render(agg, cmp, meta=meta), encoding="utf-8")
    print(f"wrote {JOBS_MD}", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
