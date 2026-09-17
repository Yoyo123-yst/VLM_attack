#!/usr/bin/env python3
"""Overnight MSC chain: CPU tests, G0, P0-0, then P0-1 only if GO.

Does not kill V-CachePoll. Does not open sealed queries.
Does not run HSSC P0-2. Does not revive CG-VSF or last-prompt probes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
MS = ROOT / "MSC"
VC = ROOT / "V-CachePoll"
PY = "/root/autodl-tmp/conda/envs/vattack/bin/python"
LOG = MS / "out" / "NIGHT_WATCH.log"
STATUS = MS / "out" / "STATUS.md"
LOCK = MS / "out" / "GPU_LOCK"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def parse_lock(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def gpu_mem_used_mb() -> int:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        return int(float(p.stdout.strip().splitlines()[0]))
    except Exception:
        return 10**9


def foreign_gpu_cmd() -> str:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    bits = []
    me = os.getpid()
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == me:
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except Exception:
            cmd = parts[-1] if parts else str(pid)
        if "MSC/scripts" in cmd or "night_watch.py" in cmd:
            continue
        bits.append(f"{pid}:{cmd.strip()[:80]}")
    return " | ".join(bits)


def gpu_busy() -> bool:
    foreign = foreign_gpu_cmd()
    if foreign:
        return True
    for lock_path in (
        VC / "out" / "GPU_LOCK",
        ROOT / "CG-VSF" / "out" / "GPU_LOCK",
        ROOT / "HSSC" / "out" / "GPU_LOCK",
    ):
        lock = parse_lock(lock_path)
        pid = lock.get("pid")
        if pid and pid.isdigit() and pid_alive(int(pid)):
            return True
    if gpu_mem_used_mb() > 2500:
        return True
    return False


def wait_for_gpu() -> None:
    log("waiting for GPU")
    while gpu_busy():
        used = gpu_mem_used_mb()
        log(f"GPU busy mem={used}MiB foreign={foreign_gpu_cmd() or 'none'}")
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(
            "\n".join(
                [
                    "# MSC Experiment Status",
                    "",
                    "- Stage: P0-1",
                    "- State: WAITING_FOR_GPU",
                    f"- Updated at: {_now()}",
                    "",
                    "## Latest Result",
                    f"- GPU memory used: {used} MiB",
                    f"- foreign GPU: {foreign_gpu_cmd() or 'none'}",
                    "",
                    "## Gate",
                    "- Current verdict: PENDING",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        time.sleep(45)
    for _ in range(8):
        if gpu_mem_used_mb() < 2500:
            break
        time.sleep(15)
    log(f"GPU free mem={gpu_mem_used_mb()}MiB")


def claim_lock(stage: str) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(f"pid={os.getpid()}\nstage={stage}\nupdated={_now()}\n", encoding="utf-8")


def run(cmd: list[str], log_name: str, retries: int = 2) -> int:
    path = MS / "out" / log_name
    path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("P0_QWEN_FORCE_GPU", "1")
    env.setdefault("P0_QWEN_KEEP_336", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rc = 1
    for attempt in range(1, retries + 1):
        log(f"run {' '.join(cmd)} attempt={attempt}")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== {_now()} attempt {attempt} =====\n")
            fh.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env)
            rc = int(proc.returncode)
        log(f"exit {rc} for {cmd[-1] if cmd else '?'}")
        if rc == 0:
            return 0
        time.sleep(20)
    return rc


def main() -> int:
    os.chdir(str(ROOT))
    log("MSC night_watch start")
    tests = run([PY, str(MS / "tests" / "test_msc_cpu.py")], "cpu_tests.log", retries=1)
    if tests != 0:
        log("CPU tests failed; not claiming GPU")
        return tests
    g0 = run(
        [PY, str(MS / "scripts" / "check_g0.py"), "--freeze", str(MS / "MSC_P0_FROZEN.json")],
        "g0.log",
        retries=1,
    )
    if g0 != 0:
        log("G0 failed")
        return g0
    rc0 = run([PY, "-u", str(MS / "scripts" / "run_p00.py")], "p00/p00.log", retries=2)
    run([PY, "-u", str(MS / "scripts" / "render_report.py"), "--stage", "p00"], "p00/render.log", retries=1)
    verdict0 = {}
    vp0 = MS / "out" / "p00" / "p00_verdict.json"
    if vp0.exists():
        verdict0 = json.loads(vp0.read_text(encoding="utf-8"))
        log(f"P0-0 verdict={verdict0.get('verdict')} jaccard={verdict0.get('jaccard')}")
    if verdict0.get("verdict") != "GO":
        run(
            [PY, "-u", str(MS / "scripts" / "advance_stage.py"), "--require-preregistered-gate"],
            "integration_advance.log",
            retries=1,
        )
        log("P0-0 STOP; not running P0-1")
        return 0 if rc0 == 0 else rc0
    wait_for_gpu()
    claim_lock("p01")
    rc1 = run([PY, "-u", str(MS / "scripts" / "run_p01.py"), "--resume"], "p01/p01_gpu.log", retries=3)
    run([PY, "-u", str(MS / "scripts" / "render_report.py"), "--stage", "p01"], "p01/render.log", retries=1)
    run(
        [PY, "-u", str(MS / "scripts" / "advance_stage.py"), "--require-preregistered-gate"],
        "integration_advance.log",
        retries=1,
    )
    if LOCK.exists():
        LOCK.unlink()
    log(f"night_watch done p00={rc0} p01={rc1}")
    return 0 if rc0 == 0 and rc1 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
