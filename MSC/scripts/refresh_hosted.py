#!/usr/bin/env python3
"""Live HOSTED.md projection from jsonl + gpu log. Exits when P1 has a verdict."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
MS = ROOT / "MSC"
HOSTED = MS / "out" / "HOSTED.md"
P01_JSONL = MS / "out" / "p01" / "p01_results.jsonl"
P01_GPU = MS / "out" / "p01" / "p01_gpu.log"
P01_VERDICT = MS / "out" / "p01" / "p01_verdict.json"
P1_JSONL = MS / "out" / "p1" / "p1_results.jsonl"
P1_GPU = MS / "out" / "p1" / "p1_gpu.log"
P1_VERDICT = MS / "out" / "p1" / "p1_verdict.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def pgrep(name: str) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", name], text=True)
    except subprocess.CalledProcessError:
        return False
    return any("python" in line and name in line and "extglob" not in line for line in out.splitlines())


def gpu() -> str:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        return p.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def summaries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("round") == "summary":
            rows.append(rec)
    return rows


def tail_line(path: Path) -> str:
    if not path.exists():
        return "—"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[-1] if lines else "—"


def age_s(path: Path) -> int:
    if not path.exists():
        return -1
    return int(time.time() - path.stat().st_mtime)


def render() -> str:
    s = summaries(P01_JSONL)
    last = s[-1] if s else {}
    p01_alive = pgrep("MSC/scripts/run_p01.py")
    p1_alive = pgrep("MSC/scripts/run_p1.py")
    waiter = pgrep("MSC/scripts/night_watch_p1.py")
    if P01_VERDICT.exists():
        v = json.loads(P01_VERDICT.read_text(encoding="utf-8"))
        p01_line = f"P0-1 **{v.get('verdict')}** switched {v.get('n_switched')}/8"
    else:
        p01_line = f"P0-1 **RUNNING {len(s)}/96** remaining {max(0, 96-len(s))}" if p01_alive else f"P0-1 **DEAD?** {len(s)}/96"
    live_gpu = P1_GPU if p1_alive else P01_GPU
    hung = (age_s(live_gpu) > 720) and (p01_alive or p1_alive)
    hang_line = "- **HANG?** gpu log stale >12min" if hung else "- hang watch: ok"
    p1s = summaries(P1_JSONL)
    if P1_VERDICT.exists():
        g = json.loads(P1_VERDICT.read_text(encoding="utf-8"))
        p1_line = f"P1 **{g.get('verdict')}** winner `{g.get('method')}` {g.get('reasons')}"
    elif p1_alive:
        p1_line = f"P1 **RUNNING** {len(p1s)} summaries; last `{tail_line(P1_GPU)[:160]}`"
    elif waiter:
        p1_line = "P1 waiter alive; waiting for P0-1 verdict / idle GPU"
    else:
        p1_line = "P1 waiter **missing**"
    return "\n".join(
        [
            "# MSC overnight host",
            "",
            f"- updated: `{now()}`",
            f"- GPU: {gpu()}",
            f"- P0 freeze: not edited; sealed still sealed",
            "",
            "## Now",
            "",
            f"- {p01_line}",
            f"- last P0-1 cell: `{last.get('cell_id', '—')}` `{last.get('method', '—')}` seed {last.get('opt_seed', '—')} robust={last.get('robust_core_rhc')} bw={last.get('backward_used')}",
            f"- gpu log age: {age_s(live_gpu)}s; last: `{tail_line(live_gpu)[:180]}`",
            hang_line,
            f"- {p1_line}",
            "",
            "## P1 ladder",
            "",
            "1. `switched_keep` → 2. `switched_follow` → 3. `switched_margin` → 4. `switched_prefix`",
            "",
        ]
    )


def main() -> int:
    sys.path.insert(0, str(MS / "src"))
    from msc.reporting import atomic_write_text

    while True:
        atomic_write_text(HOSTED, render())
        if P1_VERDICT.exists():
            return 0
        time.sleep(45)


if __name__ == "__main__":
    raise SystemExit(main())
