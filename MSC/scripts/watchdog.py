#!/usr/bin/env python3
"""Detached watchdog: restart MSC night_watch if the P0-1 chain dies.

Does not edit frozen thresholds. Resume only. Exits when P0-1 has a verdict.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
MS = ROOT / "MSC"
PY = "/root/autodl-tmp/conda/envs/vattack/bin/python"
LOG = MS / "out" / "WATCHDOG.log"
PIDFILE = MS / "out" / "WATCHDOG.pid"
VERDICT = MS / "out" / "p01" / "p01_verdict.json"
GPU_LOG = MS / "out" / "p01" / "p01_gpu.log"
HANG_S = 900  # 15 min with no gpu-log update is a hang (phrase-set round is ~4 min)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def pgrep(needle: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", needle], text=True)
    except subprocess.CalledProcessError:
        return []
    pids = []
    me = os.getpid()
    for raw in out.split():
        if raw.isdigit() and int(raw) != me:
            pids.append(int(raw))
    return pids


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_night_watch() -> None:
    log("starting detached night_watch")
    subprocess.Popen(
        [PY, "-u", str(MS / "scripts" / "night_watch.py")],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def gpu_log_age() -> float:
    if not GPU_LOG.exists():
        return 0.0
    return time.time() - GPU_LOG.stat().st_mtime


def main() -> int:
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    log(f"watchdog start pid={os.getpid()}")
    while True:
        if VERDICT.exists():
            log("p01_verdict present; watchdog exit")
            return 0
        nw = [p for p in pgrep("MSC/scripts/night_watch.py") if alive(p)]
        p01 = [p for p in pgrep("MSC/scripts/run_p01.py") if alive(p)]
        if not nw and not p01:
            log("chain dead; resume night_watch")
            start_night_watch()
            time.sleep(30)
            continue
        age = gpu_log_age()
        if p01 and age > HANG_S:
            log(f"gpu log stale {age:.0f}s; SIGTERM run_p01 so parent night_watch can --resume")
            for pid in p01:
                try:
                    cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
                except OSError:
                    continue
                if b"python" in cmd and b"run_p01.py" in cmd:
                    try:
                        os.kill(pid, 15)
                        log(f"SIGTERM pid={pid}")
                    except OSError as exc:
                        log(f"SIGTERM {pid} failed {exc}")
            time.sleep(60)
            continue


if __name__ == "__main__":
    raise SystemExit(main())
