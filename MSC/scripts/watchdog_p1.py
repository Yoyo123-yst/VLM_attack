#!/usr/bin/env python3
"""Restart P1 night_watch after P0-1 finishes. Does not touch P0 freeze."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
MS = ROOT / "MSC"
PY = "/root/autodl-tmp/conda/envs/vattack/bin/python"
LOG = MS / "out" / "WATCHDOG_P1.log"
PIDFILE = MS / "out" / "WATCHDOG_P1.pid"
P01_VERDICT = MS / "out" / "p01" / "p01_verdict.json"
P1_VERDICT = MS / "out" / "p1" / "p1_verdict.json"
GPU_LOG = MS / "out" / "p1" / "p1_gpu.log"
HANG_S = 900


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
    me = os.getpid()
    return [int(x) for x in out.split() if x.isdigit() and int(x) != me]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_night_watch() -> None:
    log("starting detached night_watch_p1")
    subprocess.Popen(
        [PY, "-u", str(MS / "scripts" / "night_watch_p1.py")],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def main() -> int:
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    log(f"watchdog_p1 start pid={os.getpid()}")
    while True:
        if P1_VERDICT.exists():
            log("p1_verdict present; watchdog_p1 exit")
            return 0
        nw = [p for p in pgrep("MSC/scripts/night_watch_p1.py") if alive(p)]
        p1 = [p for p in pgrep("MSC/scripts/run_p1.py") if alive(p)]
        if not nw and not p1:
            log("P1 waiter/chain dead; resume night_watch_p1")
            start_night_watch()
            time.sleep(30)
            continue
        if not P01_VERDICT.exists():
            time.sleep(10)
            continue
        if p1 and GPU_LOG.exists():
            age = time.time() - GPU_LOG.stat().st_mtime
            if age > HANG_S:
                log(f"p1 gpu log stale {age:.0f}s; SIGTERM run_p1 for resume")
                for pid in p1:
                    try:
                        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
                    except OSError:
                        continue
                    if b"python" in cmd and b"run_p1.py" in cmd:
                        try:
                            os.kill(pid, 15)
                            log(f"SIGTERM pid={pid}")
                        except OSError as exc:
                            log(f"SIGTERM {pid} failed {exc}")
                time.sleep(60)
                continue
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())
