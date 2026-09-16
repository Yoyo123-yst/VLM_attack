#!/usr/bin/env python3
"""Rewrite NIGHT_STATUS.md from P3 progress files without touching the GPU job."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll/out")
STATUS = OUT / "NIGHT_STATUS.md"


def _load(name: str):
    p = OUT / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stats(name: str) -> str:
    blob = _load(f"p3_{name}.json")
    if not blob:
        prog = _load(f"p3_{name}_progress.json")
        if not prog:
            return f"| {name} | running or pending | — |"
        eta = prog.get("eta_s")
        eta_m = f"{eta/60:.1f}m" if isinstance(eta, (int, float)) else "—"
        return f"| {name} | {prog.get('done',0)}/{prog.get('total','?')} | eta {eta_m} last={prog.get('last')} |"
    rows = [r for r in blob.get("rows") or [] if r.get("eval")]
    n = len(rows)
    if not n:
        return f"| {name} | 0 eval rows | — |"
    cof = sum(1 for r in rows if r["eval"].get("comp_only_fail"))
    full = sum(1 for r in rows if r["eval"].get("full_ok"))
    u = sum(float(r["eval"].get("u_evict_rate") or 0) for r in rows) / n
    aout = sum(r["eval"]["avtp"]["events"]["a_out"] for r in rows) / n
    return f"| {name} | {n} done | full={full/n:.3f} cof={cof/n:.3f} ({cof}/{n}) a_out={aout:.2f} U-evict={u:.3f} |"


def main() -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# V-CachePoll night status",
        "",
        f"- updated: `{now}`",
        f"- live table from progress/result files",
        "",
        "| method | progress | metrics |",
        "|---|---|---|",
        _stats("random"),
        _stats("task"),
        _stats("caa"),
        _stats("rank"),
        "",
        "V-CachePoll numbers come from `p2_attack.json` (same 32 pairs, r_base=0.2, eps=16/255, 40 steps).",
        "",
    ]
    p2 = _load("p2_attack.json")
    if p2:
        rows = [r for r in p2.get("rows") or [] if r.get("eval")]
        n = len(rows)
        cof = sum(1 for r in rows if r["eval"].get("comp_only_fail"))
        full = sum(1 for r in rows if r["eval"].get("full_ok"))
        u = sum(float(r["eval"].get("u_evict_rate") or 0) for r in rows) / max(n, 1)
        lines.append(f"- vcache: n={n} full={full/n:.3f} cof={cof/n:.3f} ({cof}/{n}) U-evict={u:.3f}")
    p3r = _load("p3_report.json")
    if p3r:
        lines += ["", f"- decision: **{p3r.get('decision')}**", f"- reason: {p3r.get('reason')}"]
    STATUS.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
