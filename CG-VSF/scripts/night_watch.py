#!/usr/bin/env python3
"""Overnight CG-VSF chain: wait for GPU, run P0-A, then P0-B, then stage decision.

Does not kill V-CachePoll. Does not open sealed queries. Does not put QP in the attacker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
CG = ROOT / "CG-VSF"
VC = ROOT / "V-CachePoll"
PY = "/root/autodl-tmp/conda/envs/vattack/bin/python"
LOG = CG / "out" / "NIGHT_WATCH.log"
STATUS = CG / "out" / "STATUS.md"
LOCK = CG / "out" / "GPU_LOCK"


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
        if "night_watch.py" in cmd or "run_p0a.py" in cmd or "run_p0b.py" in cmd:
            continue
        bits.append(f"{pid}:{cmd.strip()[:80]}")
    return " | ".join(bits)


def vcachepoll_busy() -> bool:
    foreign = foreign_gpu_cmd()
    if foreign:
        return True
    lock = parse_lock(VC / "out" / "GPU_LOCK")
    pid = lock.get("pid")
    if pid and pid.isdigit() and pid_alive(int(pid)):
        return True
    if gpu_mem_used_mb() > 2500:
        return True
    return False


def p3_snapshot() -> str:
    bits = []
    for name in ("p3_random_progress.json", "p3_task_progress.json", "p3_caa_progress.json", "p3_rank_progress.json"):
        p = VC / "out" / name
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        bits.append(
            f"{blob.get('stage', name)} {blob.get('done', '?')}/{blob.get('total', '?')} eta={blob.get('eta_s', '?')}"
        )
    cage = VC / "out" / "p3_cage_progress.json"
    if cage.exists():
        try:
            blob = json.loads(cage.read_text(encoding="utf-8"))
            bits.append(f"cage {blob.get('done','?')}/{blob.get('total','?')} eta={blob.get('eta_s','?')}")
        except Exception:
            pass
    return "; ".join(bits) or "no p3 progress files"


def wait_for_gpu() -> None:
    log("waiting for GPU (V-CachePoll holds the lock)")
    while vcachepoll_busy():
        used = gpu_mem_used_mb()
        lock = parse_lock(VC / "out" / "GPU_LOCK")
        log(f"GPU busy mem={used}MiB foreign={foreign_gpu_cmd() or 'none'} vcache={lock} p3={p3_snapshot()}")
        heartbeat_wait(used, lock)
        time.sleep(45)
    for _ in range(8):
        if gpu_mem_used_mb() < 2500:
            break
        time.sleep(15)
    log(f"GPU free mem={gpu_mem_used_mb()}MiB")


def heartbeat_wait(used: int, lock: dict) -> None:
    body = "\n".join(
        [
            "# CG-VSF Experiment Status",
            "",
            "- Stage: P0-A",
            "- State: WAITING_FOR_GPU",
            f"- Frozen config SHA256: pending",
            f"- Started at: {_now()}",
            f"- Updated at: {_now()}",
            "",
            "## Progress",
            "| 0 | 48 | 0 | 0 |",
            "",
            "## Latest Result",
            f"- waiting on V-CachePoll pid={lock.get('pid', '—')} stage={lock.get('stage', '—')}",
            f"- GPU memory used: {used} MiB",
            f"- foreign GPU: {foreign_gpu_cmd() or 'none'}",
            "",
            "## Gate",
            "- Current verdict: PENDING",
            "- Evidence: certificate tagger fixed; P0-A will restart on a clean JSONL",
            "- Next allowed command: wait then `python CG-VSF/scripts/run_p0a.py`",
            "",
        ]
    )
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(body, encoding="utf-8")


def claim_lock(stage: str) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(f"pid={os.getpid()}\nstage={stage}\nupdated={_now()}\n", encoding="utf-8")


def run(cmd: list[str], log_name: str, retries: int = 2) -> int:
    path = CG / "out" / log_name
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


def archive_old_p0a() -> None:
    src = CG / "out" / "p0a" / "p0a_results.jsonl"
    if not src.exists() or not src.read_text(encoding="utf-8").strip():
        return
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    modes = [str(r.get("certificate_mode")) for r in rows if r.get("round") != "summary" and r.get("certificate_mode")]
    if modes and sum(m == "GARBAGE" for m in modes) / len(modes) >= 0.8:
        dst = CG / "out" / "p0a" / f"p0a_results_garbage_tagger_{datetime.now().strftime('%H%M%S')}.jsonl"
        shutil.move(str(src), str(dst))
        log(f"archived invalid P0-A jsonl -> {dst.name}")
        return
    log(f"keeping P0-A jsonl ({len(rows)} rows, modes={sorted(set(modes))})")


def first_job_sane() -> tuple[bool, str]:
    path = CG / "out" / "p0a" / "p0a_results.jsonl"
    if not path.exists():
        return False, "no jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    certs = [r for r in rows if r.get("round") != "summary" and r.get("certificate_mode")]
    if not certs:
        return False, "no certificate rows yet"
    modes = [str(r.get("certificate_mode")) for r in certs[:8]]
    garbage_frac = sum(m == "GARBAGE" for m in modes) / len(modes)
    after = [str(r.get("after_mode")) for r in certs[:8]]
    if garbage_frac >= 0.8 and any(a in {"REFUSE", "DENY"} for a in after):
        return False, f"tagger still GARBAGE modes={modes} after={after}"
    return True, f"ok modes={modes} after={after}"


def watch_p0a_sanity(proc: subprocess.Popen, timeout_s: float = 900.0) -> None:
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < timeout_s:
        time.sleep(30)
        jsonl = CG / "out" / "p0a" / "p0a_results.jsonl"
        if not jsonl.exists():
            continue
        n = sum(1 for _ in jsonl.open())
        if n >= 2:
            ok, msg = first_job_sane()
            log(f"sanity after {n} jsonl lines: {msg}")
            if not ok:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError(f"P0-A abort: {msg}")
            return
    if proc.poll() is not None:
        return
    log("sanity window elapsed without rows; continuing to wait on P0-A")


def main() -> int:
    os.chdir(str(ROOT))
    log("night_watch start")
    tests = run(
        [PY, str(CG / "tests" / "test_cgvsf_cpu.py")],
        "cpu_tests.log",
        retries=1,
    )
    if tests != 0:
        log("CPU tests failed; not claiming GPU")
        return tests
    wait_for_gpu()
    claim_lock("p0a")
    archive_old_p0a()
    jsonl = CG / "out" / "p0a" / "p0a_results.jsonl"
    p0a_cmd = [PY, "-u", str(CG / "scripts" / "run_p0a.py")]
    if jsonl.exists() and jsonl.stat().st_size > 0:
        p0a_cmd.append("--resume")
        log("resuming P0-A from existing jsonl")
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("P0_QWEN_FORCE_GPU", "1")
    env.setdefault("P0_QWEN_KEEP_336", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p0a_log = CG / "out" / "p0a" / "p0a_gpu.log"
    p0a_log.parent.mkdir(parents=True, exist_ok=True)
    with p0a_log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== night_watch P0-A {_now()} =====\n")
        fh.flush()
        proc = subprocess.Popen(
            p0a_cmd,
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        log(f"P0-A pid={proc.pid}")
        try:
            watch_p0a_sanity(proc)
        except Exception as exc:
            log(f"sanity failed: {exc}")
            return 3
        rc = proc.wait()
    log(f"P0-A finished rc={rc}")
    if rc != 0:
        log("P0-A crashed; wait for GPU then --resume")
        wait_for_gpu()
        claim_lock("p0a-resume")
        rc = run([PY, "-u", str(CG / "scripts" / "run_p0a.py"), "--resume"], "p0a/p0a_gpu.log", retries=8)
    run([PY, "-u", str(CG / "scripts" / "render_report.py"), "--stage", "p0a"], "p0a/render.log", retries=1)
    verdict = {}
    vp = CG / "out" / "p0a" / "p0a_verdict.json"
    if vp.exists():
        verdict = json.loads(vp.read_text(encoding="utf-8"))
        log(f"P0-A verdict={verdict.get('verdict')} acc={verdict.get('n_accumulated')} last={verdict.get('n_last')}")
    claim_lock("p0b")
    rc_b = run([PY, "-u", str(CG / "scripts" / "run_p0b.py")], "p0b/p0b_gpu.log", retries=2)
    run([PY, "-u", str(CG / "scripts" / "render_report.py"), "--stage", "p0b"], "p0b/render.log", retries=1)
    run(
        [PY, "-u", str(CG / "scripts" / "advance_stage.py"), "--require-preregistered-gate"],
        "integration_advance.log",
        retries=1,
    )
    claim_lock("done")
    log(f"night_watch done p0a={rc} p0b={rc_b}")
    if LOCK.exists():
        LOCK.unlink()
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
