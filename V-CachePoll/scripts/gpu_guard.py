#!/usr/bin/env python3
"""While GPU_LOCK exists, keep CG-VSF from stealing the only GPU."""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

LOCK = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll/out/GPU_LOCK")
LOG = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll/out/gpu_guard.log")
PATTERNS = (
    "CG-VSF/scripts/run_p0a.py",
    "CG-VSF/scripts/run_p0",
    "CG-VSF/scripts/night_watch.py",
)


def _pids() -> list[tuple[int, str]]:
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        if any(p in cmd for p in PATTERNS):
            out.append((pid, cmd[:200]))
    return out


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while LOCK.is_file():
        for pid, cmd in _pids():
            try:
                with LOG.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        time.sleep(15)


if __name__ == "__main__":
    main()
