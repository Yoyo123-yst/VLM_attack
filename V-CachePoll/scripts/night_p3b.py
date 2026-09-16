#!/usr/bin/env python3
"""After P3: restore FastV details, then LLaVA two-image AVTP probe."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
OUT = ROOT / "out"
PY = "/root/autodl-tmp/conda/envs/vattack/bin/python"
STATUS = OUT / "NIGHT_STATUS.md"
LOCK = OUT / "GPU_LOCK"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_status(stage: str, extra: str = "") -> None:
    STATUS.write_text(
        f"# V-CachePoll night status\n\n- updated: `{_now()}`\n- stage: **{stage}**\n\n{extra}\n",
        encoding="utf-8",
    )
    LOCK.write_text(f"pid={os.getpid()}\nstage={stage}\nupdated={_now()}\n", encoding="utf-8")


def run(script: str, extra: list[str], log_name: str) -> int:
    log = OUT / log_name
    cmd = [PY, "-u", str(ROOT / "scripts" / script), *extra]
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {_now()} {' '.join(cmd)} =====\n")
        fh.flush()
        return int(subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode)


def main() -> int:
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    write_status("fastv", "Re-running FastV probe with eager attention, 8 pairs.")
    rc = run("run_fastv.py", ["--limit", "8"], "fastv.log")
    if rc != 0:
        write_status("fastv-failed", f"exit {rc}")
    write_status("llava-p0", "LLaVA-1.5 two-image AVTP + B-noise probe, 8 pairs.")
    rc2 = run("run_llava.py", ["--limit", "8"], "llava_p0.log")
    extra = ""
    for name in ("p3_report.json", "fastv_probe.json", "llava_probe.json"):
        p = OUT / name
        if p.is_file():
            extra += f"\n## {name}\n```\n{p.read_text(encoding='utf-8')[:2500]}\n```\n"
    write_status("done-p3b", extra if extra else f"fastv_rc={rc} llava_rc={rc2}")
    return 0 if rc2 == 0 else rc2


if __name__ == "__main__":
    sys.exit(main())
