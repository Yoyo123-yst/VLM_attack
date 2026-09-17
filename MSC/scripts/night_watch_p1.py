#!/usr/bin/env python3
"""After P0-1 verdict: wait for GPU, run INVALID-recovery ladder, expand winner.

Does not edit P0 freeze. Does not open sealed queries. Does not kill V-CachePoll;
starts as soon as the card is empty after P0-1 (45s V-CachePoll poll leaves a window).
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
PY = "/root/autodl-tmp/conda/envs/vattack/bin/python"
LOG = MS / "out" / "NIGHT_WATCH_P1.log"
STATUS = MS / "out" / "STATUS.md"
LOCK = MS / "out" / "GPU_LOCK"
HOSTED = MS / "out" / "HOSTED.md"
P01_VERDICT = MS / "out" / "p01" / "p01_verdict.json"
P1_JSONL = MS / "out" / "p1" / "p1_results.jsonl"
P1_VERDICT = MS / "out" / "p1" / "p1_verdict.json"


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


def pgrep(needle: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", needle], text=True)
    except subprocess.CalledProcessError:
        return []
    me = os.getpid()
    return [int(x) for x in out.split() if x.isdigit() and int(x) != me]


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


def p01_still_running() -> bool:
    return bool(pgrep("MSC/scripts/run_p01.py"))


def foreign_gpu() -> str:
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
        if "night_watch_p1.py" in cmd or "run_p1.py" in cmd or "watchdog_p1.py" in cmd:
            continue
        if "MSC/scripts/run_p01.py" in cmd or "MSC/scripts/night_watch.py" in cmd:
            bits.append(f"p01:{pid}")
            continue
        bits.append(f"{pid}:{cmd.strip()[:80]}")
    return " | ".join(bits)


def wait_p01() -> None:
    log("waiting for P0-1 verdict")
    last = 0.0
    while not P01_VERDICT.exists() or p01_still_running():
        now = time.time()
        if now - last >= 30:
            log(f"p01_verdict={P01_VERDICT.exists()} run_p01={p01_still_running()} mem={gpu_mem_used_mb()}MiB")
            last = now
        time.sleep(1)
    log("P0-1 process finished")


def wait_for_gpu() -> None:
    log("waiting for GPU")
    last = 0.0
    empty_streak = 0
    while True:
        foreign = foreign_gpu()
        used = gpu_mem_used_mb()
        if not foreign:
            empty_streak += 1
        else:
            empty_streak = 0
        # V-CachePoll polls ~45s on empty compute-apps. Grab after 2 empty 1s polls,
        # even if CUDA context has not fully released below 2500MiB yet.
        if not foreign and (used < 8000 or empty_streak >= 2):
            break
        if foreign and "p01:" in foreign and not p01_still_running():
            time.sleep(1)
            continue
        now = time.time()
        if now - last >= 30:
            log(f"GPU busy mem={used}MiB foreign={foreign or 'none'}")
            last = now
        time.sleep(1)
    for _ in range(6):
        if gpu_mem_used_mb() < 2500 or not foreign_gpu():
            break
        time.sleep(1)
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


def write_hosted(body: str) -> None:
    HOSTED.parent.mkdir(parents=True, exist_ok=True)
    HOSTED.write_text(body.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    os.chdir(str(ROOT))
    sys.path.insert(0, str(MS / "src"))
    from msc.protocol import load_frozen
    from msc.reporting import read_jsonl
    from msc.verdict import p1a_expand, p1b_verdict

    log("MSC P1 night_watch start")
    wait_p01()
    p1 = json.loads((MS / "MSC_P1_FROZEN.json").read_text(encoding="utf-8"))
    parent = load_frozen()
    p01 = json.loads(P01_VERDICT.read_text(encoding="utf-8"))
    p01_switched = int(p01.get("n_switched") or 0)
    write_hosted(
        "\n".join(
            [
                "# MSC overnight host",
                "",
                f"- updated: `{_now()}`",
                f"- P0-1 verdict: **{p01.get('verdict')}** switched {p01_switched}/8",
                "- next: P1 INVALID-recovery ladder",
                "- P0 freeze is not edited",
                "",
            ]
        )
    )
    wait_for_gpu()
    claim_lock("p1")
    fail_cells = list(p1["fail_cells"])
    ladder = list(p1["ladder"])
    seeds = [str(s) for s in p1["seeds"]["opt"]]
    min_seeds = int(p1["expand_if"]["both_fail_cells_min_seeds"])
    winner = None
    for method in ladder:
        write_hosted(
            f"# MSC overnight host\n\n- updated: `{_now()}`\n- P1-A method: `{method}` on {fail_cells}\n"
        )
        rc = run(
            [
                PY,
                "-u",
                str(MS / "scripts" / "run_p1.py"),
                "--resume",
                "--methods",
                method,
                "--cells",
                *fail_cells,
                "--seeds",
                *seeds,
            ],
            "p1/p1_gpu.log",
            retries=3,
        )
        if rc != 0:
            log(f"P1-A {method} failed rc={rc}")
            continue
        rows = read_jsonl(P1_JSONL)
        gate = p1a_expand(rows, method=method, fail_cells=fail_cells, min_seeds=min_seeds)
        log(f"P1-A {method} expand={gate['expand']} seeds={gate['per_cell_seeds']}")
        (MS / "out" / "p1" / f"p1a_{method}.json").write_text(
            json.dumps(gate, indent=2), encoding="utf-8"
        )
        if gate["expand"] and winner is None:
            winner = method
            log(f"ladder winner {method}; remaining methods skipped for expansion")
            break
    hard = list(parent["scope"]["hard_cells"])
    if winner:
        rc = run(
            [
                PY,
                "-u",
                str(MS / "scripts" / "run_p1.py"),
                "--resume",
                "--methods",
                winner,
                "--cells",
                *hard,
                "--seeds",
                *seeds,
            ],
            "p1/p1_gpu.log",
            retries=3,
        )
        rows = read_jsonl(P1_JSONL)
        gate = p1b_verdict(
            rows,
            method=winner,
            hard_cells=hard,
            p01_switched_n=p01_switched,
            frozen_gates=p1["gates"],
        )
    else:
        rows = read_jsonl(P1_JSONL) if P1_JSONL.exists() else []
        gate = {
            "verdict": "STOP",
            "method": None,
            "reasons": ["no ladder method rescued both fail cells at min seeds"],
            "n_rhc": 0,
        }
        rc = 0
    P1_VERDICT.parent.mkdir(parents=True, exist_ok=True)
    P1_VERDICT.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    run([PY, "-u", str(MS / "scripts" / "render_report.py"), "--stage", "p1"], "p1/render.log", retries=1)
    write_hosted(
        "\n".join(
            [
                "# MSC overnight host",
                "",
                f"- updated: `{_now()}`",
                f"- P0-1: **{p01.get('verdict')}** switched {p01_switched}/8",
                f"- P1 winner: `{winner}`",
                f"- P1-B: **{gate.get('verdict')}** {gate.get('reasons')}",
                "- P0 freeze unchanged; sealed still sealed",
                "",
            ]
        )
    )
    from msc.reporting import write_status, now as rnow

    write_status(
        STATUS,
        stage="P1",
        state="COMPLETE",
        frozen_sha="p1",
        git_commit="local",
        started=rnow(),
        completed=1,
        total=1,
        verdict=str(gate.get("verdict")),
        evidence="; ".join(gate.get("reasons") or []) or str(winner),
        next_cmd="read MSC/out/p1/p1_verdict.json",
    )
    if LOCK.exists():
        LOCK.unlink()
    log(f"night_watch_p1 done winner={winner} verdict={gate.get('verdict')} rc={rc}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
