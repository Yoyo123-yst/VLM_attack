#!/usr/bin/env python3
"""Merge GateFlip probes with the frozen TraceFlip pilot into a k*/budget table.

    python TraceFlip/render_kstar_table.py \
        --gateflip TraceFlip/out/gateflip_kstar/probe_public.json \
                   TraceFlip/out/gateflip_easy/probe_public.json \
        --pilot TraceFlip/out/pilot_public.json \
        --out TraceFlip/out/gateflip_12cell.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from p0.metrics import mean, mcnemar_test  # noqa: E402
from traceflip.metrics import cell_key, pair_with  # noqa: E402
from traceflip.protocol import PILOT_CARRIERS, PILOT_QUERIES  # noqa: E402

PILOT_METHODS = ("traceflip", "targeted_prefix_pgd", "refusal_margin_pgd")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render 12-cell k*/budget comparison")
    ap.add_argument(
        "--gateflip",
        nargs="+",
        default=[
            str(HERE / "out" / "gateflip_kstar" / "probe_public.json"),
            str(HERE / "out" / "gateflip_easy" / "probe_public.json"),
        ],
    )
    ap.add_argument("--pilot", default=str(HERE / "out" / "pilot_public.json"))
    ap.add_argument("--out", default=str(HERE / "out" / "gateflip_12cell.md"))
    ap.add_argument("--title", default="GateFlip 12-cell k* / budget table")
    ap.add_argument(
        "--ref",
        default=None,
        help="previous GateFlip json (coarse k*) for a side-by-side column",
    )
    return ap.parse_args()


def _load_cells(path: Path) -> List[dict]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, list):
        return list(blob)
    return list(blob.get("cells") or [])


def _method_map(rows: Iterable[Mapping[str, Any]], method: str) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        if str(r.get("method")) != method:
            continue
        out[cell_key(r)] = dict(r)
    return out


def _yn(flag: Any) -> str:
    return "RHC" if bool(flag) else "no"


def _k_cell(row: Optional[Mapping[str, Any]]) -> str:
    if not row:
        return "—"
    if not row.get("core_rhc"):
        return "fail"
    k = row.get("k_star")
    name = row.get("k_star_name") or row.get("win_stage") or ""
    if k is None:
        return str(name or "RHC")
    return f"{int(k)} ({name})"


def _bw(row: Optional[Mapping[str, Any]]) -> str:
    if not row:
        return "—"
    b = (row.get("budget") or {}).get("backward_passes")
    if b is None:
        return "—"
    return f"{float(b):.0f}"


def _counts(rows: Sequence[Mapping[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        if not r.get("core_rhc"):
            c["fail"] += 1
            continue
        k = r.get("k_star")
        c[str(int(k)) if k is not None else "RHC"] += 1
    return c


def render(
    gate: Mapping[Tuple[str, str], dict],
    pilot: Mapping[str, Mapping[Tuple[str, str], dict]],
    *,
    title: str,
    sources: Sequence[str],
    ref: Optional[Mapping[Tuple[str, str], dict]] = None,
) -> str:
    keys = [(q, c) for q in PILOT_QUERIES for c in PILOT_CARRIERS]
    present = [k for k in keys if k in gate]
    if present:
        keys = present
    lines = [
        f"# {title}",
        "",
        "Success is `core_rhc` only. Sealed queries h83–h130 are untouched.",
        "GateFlip artefacts were merged without rewriting `pilot.json`.",
        "",
        "Sources:",
    ]
    lines.extend(f"- `{s}`" for s in sources)
    lines += ["", "## Per-cell", ""]
    if ref:
        lines += [
            "| cell | TraceFlip | targeted-prefix | GateFlip | k* | ref k* | GF backward | ref backward | TP backward |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| cell | TraceFlip | targeted-prefix | refusal-margin | GateFlip | k* | GF backward | TP backward |",
            "|---|---|---|---|---|---|---|---|",
        ]
    missing: List[str] = []
    gf_rows: List[dict] = []
    tp_rows: List[dict] = []
    tf_rows: List[dict] = []
    for key in keys:
        tf = (pilot.get("traceflip") or {}).get(key)
        tp = (pilot.get("targeted_prefix_pgd") or {}).get(key)
        rm = (pilot.get("refusal_margin_pgd") or {}).get(key)
        gf = gate.get(key)
        if gf is None:
            missing.append(f"{key[0]}:{key[1]}")
        else:
            gf_rows.append(gf)
        if tp:
            tp_rows.append(tp)
        if tf:
            tf_rows.append(tf)
        if ref:
            old = ref.get(key)
            lines.append(
                "| {cell} | {tf} | {tp} | {gf} | {k} | {rk} | {gfb} | {rb} | {tpb} |".format(
                    cell=f"{key[0]}:{key[1]}",
                    tf=_yn(tf.get("core_rhc")) if tf else "—",
                    tp=_yn(tp.get("core_rhc")) if tp else "—",
                    gf=_yn(gf.get("core_rhc")) if gf else "—",
                    k=_k_cell(gf),
                    rk=_k_cell(old),
                    gfb=_bw(gf),
                    rb=_bw(old),
                    tpb=_bw(tp),
                )
            )
        else:
            lines.append(
                "| {cell} | {tf} | {tp} | {rm} | {gf} | {k} | {gfb} | {tpb} |".format(
                    cell=f"{key[0]}:{key[1]}",
                    tf=_yn(tf.get("core_rhc")) if tf else "—",
                    tp=_yn(tp.get("core_rhc")) if tp else "—",
                    rm=_yn(rm.get("core_rhc")) if rm else "—",
                    gf=_yn(gf.get("core_rhc")) if gf else "—",
                    k=_k_cell(gf),
                    gfb=_bw(gf),
                    tpb=_bw(tp),
                )
            )

    n_gf = len(gf_rows)
    n_rhc = sum(1 for r in gf_rows if r.get("core_rhc"))
    k_hist = _counts(gf_rows)
    mean_k = mean(
        [float(r["k_star"]) for r in gf_rows if r.get("core_rhc") and r.get("k_star") is not None]
    )
    mean_bw_gf = mean(
        [float((r.get("budget") or {}).get("backward_passes") or 0) for r in gf_rows]
    )
    mean_bw_tp = mean(
        [float((r.get("budget") or {}).get("backward_passes") or 0) for r in tp_rows]
    )
    xa, xb, paired = pair_with(gf_rows, tp_rows)
    n10 = n01 = None
    p = None
    if paired:
        n10, n01, p = mcnemar_test(xa, xb)

    lines += [
        "",
        "## Summary",
        "",
        f"- GateFlip cells: **{n_rhc}/{n_gf}** RHC"
        + (f" (missing {', '.join(missing)})" if missing else ""),
        f"- targeted-prefix: **{sum(1 for r in tp_rows if r.get('core_rhc'))}/{len(tp_rows)}** RHC",
        f"- TraceFlip: **{sum(1 for r in tf_rows if r.get('core_rhc'))}/{len(tf_rows)}** RHC",
        f"- GateFlip mean k* on wins: **{mean_k:.2f}**  (histogram: {dict(k_hist)})",
        f"- mean backward: GateFlip **{mean_bw_gf:.1f}** vs targeted-prefix **{mean_bw_tp:.1f}**",
    ]
    if paired:
        lines.append(
            f"- McNemar GateFlip vs targeted-prefix: n10={n10} n01={n01} p={p:.3f} "
            f"(pairs={len(paired)}; n10 = GF win / TP miss)"
        )
    if missing:
        lines += [
            "",
            "**Table is incomplete** until the missing cells are on disk.",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    gate: Dict[Tuple[str, str], dict] = {}
    sources: List[str] = []
    for raw in args.gateflip:
        path = Path(raw)
        if not path.exists():
            continue
        sources.append(str(path))
        for row in _load_cells(path):
            if str(row.get("method") or "gateflip") != "gateflip":
                continue
            gate[cell_key(row)] = row
    ref_map: Dict[Tuple[str, str], dict] = {}
    if args.ref:
        ref_path = Path(args.ref)
        if ref_path.exists():
            sources.append(f"ref:{ref_path}")
            for row in _load_cells(ref_path):
                if str(row.get("method") or "gateflip") != "gateflip":
                    continue
                ref_map[cell_key(row)] = row

    pilot_rows = _load_cells(Path(args.pilot)) if Path(args.pilot).exists() else []
    pilot = {m: _method_map(pilot_rows, m) for m in PILOT_METHODS}
    md = render(
        gate,
        pilot,
        title=args.title,
        sources=sources or args.gateflip,
        ref=ref_map or None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
