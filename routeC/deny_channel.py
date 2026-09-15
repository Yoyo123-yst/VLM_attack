#!/usr/bin/env python3
"""阶段① (CPU, 0 GPU) — DENY 通道刻画。

回答三个初步结论问题（纯数据审计，不重载模型）：
  Q1. DENY 是否"PGD 打不穿"——DENY cell 里 RHC 几乎为 0？
  Q2. DENY 是否 query/概念依赖——同一 carrier 不同 query，DENY 率是否翻转？
  Q3. DENY 与 REFUSE 是否互斥——一个 cell 内三通道的分布结构。

输入：outputs/causal_reach/cr0/candidates.json (192条) + n1r_fast/candidates.json (96条)
输出：routeC/out/deny_channel.json

这是纯审计脚本，不重载模型、不 fit U、不扫描 layer、不读 h83-h130。
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from p0.datautil import load_json, save_json  # noqa: E402
from p0.catalog import all_pairs  # noqa: E402

OUT = ROOT / "routeC" / "out"
CR0 = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"
N1R = ROOT / "outputs" / "n1r_fast" / "candidates.json"


def _records(c: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(c.get("records", []))


def _cell_key(r: Dict[str, Any]) -> str:
    return f"{r.get('query_id')}:{r.get('carrier_id')}"


def mode_dist(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-cell {REFUSE,ANSWER,DENY} + core_rhc/core_safe counts + entropy."""
    cells: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        cells[_cell_key(r)].append(r)

    rows: List[Dict[str, Any]] = []
    for key in sorted(cells):
        rs = cells[key]
        mode = Counter(r.get("response_mode") for r in rs)
        n = len(rs)
        # Shannon entropy over {REFUSE,ANSWER,DENY}
        import math

        h = 0.0
        for label in ("REFUSE", "ANSWER", "DENY"):
            p = mode.get(label, 0) / n if n else 0
            if p > 0:
                h -= p * math.log2(p)
        rows.append(
            {
                "cell": key,
                "n": n,
                "REFUSE": mode.get("REFUSE", 0),
                "ANSWER": mode.get("ANSWER", 0),
                "DENY": mode.get("DENY", 0),
                "core_rhc": sum(1 for r in rs if r.get("core_rhc")),
                "core_safe_answer": sum(1 for r in rs if r.get("core_safe_answer")),
                "H_mode": round(h, 3),
                "deny_rate": round(mode.get("DENY", 0) / n, 3) if n else 0.0,
                "rhc_rate": round(sum(1 for r in rs if r.get("core_rhc")) / n, 3) if n else 0.0,
            }
        )
    return {
        "cells": rows,
        "n_cells": len(rows),
        "n_total": len(records),
    }


def query_level_deny(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Across carriers, per query: does DENY rate flip (query-dependence)?"""
    by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_q[r.get("query_id")].append(r)
    cats = {p["id"]: p.get("category") for p in all_pairs()}
    rows = []
    for q in sorted(by_q):
        rs = by_q[q]
        mode = Counter(r.get("response_mode") for r in rs)
        n = len(rs)
        rows.append(
            {
                "query": q,
                "category": cats.get(q),
                "n": n,
                "DENY": mode.get("DENY", 0),
                "REFUSE": mode.get("REFUSE", 0),
                "ANSWER": mode.get("ANSWER", 0),
                "core_rhc": sum(1 for r in rs if r.get("core_rhc")),
                "deny_rate": round(mode.get("DENY", 0) / n, 3) if n else 0.0,
                "rhc_rate": round(sum(1 for r in rs if r.get("core_rhc")) / n, 3) if n else 0.0,
            }
        )
    return {"by_query": rows}


def same_carrier_cross_query(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """同一 carrier 上不同 query 的 DENY 率对比——证明是 query/概念依赖, 不是图像依赖."""
    by_c: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_c[r.get("carrier_id")].append(r)
    rows = []
    for c in sorted(by_c):
        rs = by_c[c]
        mode = Counter(r.get("response_mode") for r in rs)
        n = len(rs)
        rows.append(
            {
                "carrier": c,
                "n": n,
                "DENY": mode.get("DENY", 0),
                "REFUSE": mode.get("REFUSE", 0),
                "ANSWER": mode.get("ANSWER", 0),
                "deny_rate": round(mode.get("DENY", 0) / n, 3) if n else 0.0,
            }
        )
    return {"by_carrier": rows}


def wall_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Q1 核心: DENY cell 里 PGD 是否打不穿 (RHC≈0)?

    对每个 cell 分类:
      - pure_deny: DENY>0 且 core_rhc==0 (打不穿)
      - deny_with_rhc: DENY>0 且 core_rhc>0 (部分打穿)
      - no_deny: DENY==0
    """
    cells: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        cells[_cell_key(r)].append(r)

    pure_deny, deny_rhc, no_deny = [], [], []
    for key in sorted(cells):
        rs = cells[key]
        mode = Counter(r.get("response_mode") for r in rs)
        rhc = sum(1 for r in rs if r.get("core_rhc"))
        safe = sum(1 for r in rs if r.get("core_safe_answer"))
        rec = {
            "cell": key,
            "DENY": mode.get("DENY", 0),
            "core_rhc": rhc,
            "core_safe_answer": safe,
        }
        if mode.get("DENY", 0) > 0 and rhc == 0:
            pure_deny.append(rec)
        elif mode.get("DENY", 0) > 0 and rhc > 0:
            deny_rhc.append(rec)
        else:
            no_deny.append(rec)

    return {
        "n_pure_deny_cells": len(pure_deny),
        "n_deny_with_rhc_cells": len(deny_rhc),
        "n_no_deny_cells": len(no_deny),
        "pure_deny_cells": pure_deny,
        "deny_with_rhc_cells": deny_rhc,
        "wall_summary": (
            "DENY is a wall PGD cannot breach"
            if len(deny_rhc) == 0 and len(pure_deny) > 0
            else "DENY is partially breachable"
        ),
    }


def main() -> None:
    cr0 = _records(load_json(CR0))
    n1r = _records(load_json(N1R))
    all_rec = cr0 + n1r

    out = {
        "n_cr0": len(cr0),
        "n_n1r_fast": len(n1r),
        "n_total": len(all_rec),
        "overall_mode": dict(Counter(r.get("response_mode") for r in all_rec)),
        "overall_core": {
            "core_rhc": sum(1 for r in all_rec if r.get("core_rhc")),
            "core_safe_answer": sum(1 for r in all_rec if r.get("core_safe_answer")),
        },
        "deny_count": sum(1 for r in all_rec if r.get("response_mode") == "DENY"),
        "mode_dist": mode_dist(all_rec),
        "query_level": query_level_deny(all_rec),
        "carrier_level": same_carrier_cross_query(all_rec),
        "wall": wall_analysis(all_rec),
    }
    save_json(OUT / "deny_channel.json", out)
    print("阶段① done -> routeC/out/deny_channel.json")
    print(f"  DENY 总数={out['deny_count']}")
    print(f"  pure_deny_cells={out['wall']['n_pure_deny_cells']} "
          f"deny_with_rhc={out['wall']['n_deny_with_rhc_cells']}")
    print(f"  结论: {out['wall']['wall_summary']}")


if __name__ == "__main__":
    main()
