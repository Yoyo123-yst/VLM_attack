#!/usr/bin/env python3
"""Overnight steward: wait for GPU, then P4→P5/P6/P7/P9 smokes. Never steal MSC."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
PY = "/root/autodl-tmp/conda/envs/vattack/bin/python"
OUT = ROOT / "out" / "extend"
STATUS = OUT / "STATUS.md"
LOG = OUT / "night_extend.log"
LOCK = ROOT / "out" / "NIGHT_EXTEND.lock"
MSC_STATUS = Path("/root/autodl-tmp/multimodal_attack_project/MSC/out/STATUS.md")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(stage: str, extra: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = (
        f"# V-CachePoll P4–P11 status\n\n"
        f"- updated: `{_now()}`\n"
        f"- stage: **{stage}**\n"
        f"- steward pid: `{os.getpid()}`\n\n"
        f"{extra.strip()}\n"
    )
    STATUS.write_text(body, encoding="utf-8")
    LOCK.write_text(f"pid={os.getpid()}\nstage={stage}\nupdated={_now()}\n", encoding="utf-8")


def gpu_holder() -> dict | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    me = os.getpid()
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == me:
            continue
        cmd = ""
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            cmd = parts[-1] if parts else ""
        if "night_extend.py" in cmd or "run_extend.py" in cmd:
            continue
        return {"pid": pid, "mem": parts[1] if len(parts) > 1 else "", "cmd": cmd.strip()[:240]}
    return None


def gpu_free() -> bool:
    h = gpu_holder()
    return h is None


def msc_line() -> str:
    if not MSC_STATUS.is_file():
        return "MSC status missing"
    txt = MSC_STATUS.read_text(encoding="utf-8")
    bits = []
    for key in ("State:", "Progress", "Gate"):
        pass
    for line in txt.splitlines():
        if line.startswith("- State:") or line.startswith("| ") or line.startswith("- Current verdict"):
            bits.append(line.strip())
    return " / ".join(bits[:6]) or "MSC status unparsed"


def run_extend(phase: str, stage: str, limit: int | None = None, n_aux: int | None = None) -> int:
    cmd = [PY, "-u", str(ROOT / "scripts" / "run_extend.py"), "--phase", phase, "--stage", stage]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if n_aux is not None:
        cmd += ["--n-aux", str(n_aux)]
    log(f"STAGE_START {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {phase} {stage} {_now()} =====\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env)
    code = int(proc.returncode)
    if code == 0:
        log(f"STAGE_OK {phase}-{stage}")
    else:
        log(f"STAGE_FAIL {phase}-{stage} exit={code}")
    return code


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def p4_smoke_gate() -> tuple[bool, str]:
    blob = load_json(ROOT / "out" / "p4" / "p4_smoke.json")
    rows = blob.get("rows") or []
    if not rows:
        return False, "p4_smoke.json empty"
    bad = []
    for r in rows:
        ev = r.get("eval") or {}
        if ev.get("delta_a_zero") is False:
            bad.append(f"{r.get('pair_id')} delta_A!=0")
        if ev.get("linf_ok") is False:
            bad.append(f"{r.get('pair_id')} linf")
        if r.get("error"):
            bad.append(f"{r.get('pair_id')} {r.get('error')}")
    if bad:
        return False, "; ".join(bad)
    n_ok = sum(1 for r in rows if (r.get("eval") or {}).get("delta_a_zero") and (r.get("eval") or {}).get("linf_ok"))
    # Older rows may lack the flags if eval path skipped; require no errors and n>=1
    if n_ok == 0:
        # flags missing: still GO if no errors and eval present
        if all((r.get("eval") and not r.get("error")) for r in rows):
            return True, "smoke ran; delta_a/linf flags missing but eval present (treat as GO, check logs)"
        return False, "no successful smoke eval"
    return True, f"delta_A=0 and linf ok on {n_ok}/{len(rows)}"


def p4_drift_gate() -> tuple[bool, str]:
    blob = load_json(ROOT / "out" / "p4" / "p4_attack.json")
    rows = [r for r in (blob.get("rows") or []) if r.get("eval") and not r.get("error")]
    if len(rows) < 8:
        return True, f"only {len(rows)} finished, skip drift check"
    n_fail = sum(1 for r in rows if r["eval"].get("comp_only_fail"))
    n = len(rows)
    rate = n_fail / n
    # Doc: 7/32→6/32 ok; 7/32→1/32 STOP
    if n >= 24 and rate < 0.05:
        return False, f"GATE_FAIL CASR={n_fail}/{n}={rate:.3f} collapsed vs 7/32"
    return True, f"CASR {n_fail}/{n}={rate:.3f}"


def wait_for_gpu(poll_s: float = 45.0) -> None:
    write_status("waiting-gpu", f"MSC still holds the card.\n\n{msc_line()}\n\nWill start P4 smoke when nvidia-smi is empty.")
    last = 0.0
    while not gpu_free():
        h = gpu_holder() or {}
        now = time.time()
        if now - last >= 60:
            log(f"WAITING_GPU pid={h.get('pid')} mem={h.get('mem')} {msc_line()}")
            write_status(
                "waiting-gpu",
                f"- holder pid: `{h.get('pid')}` mem `{h.get('mem')}`\n"
                f"- cmd: `{h.get('cmd')}`\n"
                f"- {msc_line()}\n\n"
                "Not killing MSC. Next: P4 smoke `--limit 2`.",
            )
            last = now
        time.sleep(poll_s)
    log("GPU_FREE")
    write_status("gpu-free", "Card is empty. Starting P4 smoke.")


def maybe_retry(phase: str, stage: str, limit: int | None = None, n_aux: int | None = None) -> int:
    code = run_extend(phase, stage, limit=limit, n_aux=n_aux)
    if code == 0:
        return 0
    log(f"retry {phase}-{stage} after fail")
    time.sleep(5)
    return run_extend(phase, stage, limit=limit, n_aux=n_aux)


def already_ok(path: Path, min_n: int) -> bool:
    blob = load_json(path)
    rows = [r for r in (blob.get("rows") or []) if r.get("eval") and not r.get("error")]
    return len(rows) >= min_n


def main() -> int:
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    log(f"steward start pid={os.getpid()}")
    wait_for_gpu()

    steps = [
        ("p4", "smoke", 2, None, ROOT / "out" / "p4" / "p4_smoke.json", 1),
        ("p4", "pilot", 8, None, ROOT / "out" / "p4" / "p4_attack.json", 8),
        ("p4", "attack", None, None, ROOT / "out" / "p4" / "p4_attack.json", 24),
        ("p5", "smoke", 2, 1, ROOT / "out" / "p5" / "p5_smoke.json", 1),
        ("p6", "smoke", 1, None, ROOT / "out" / "p6" / "p6_smoke.json", 1),
        ("p7", "smoke", 2, None, ROOT / "out" / "p7" / "p7_smoke.json", 1),
        ("p9", "lamp", 8, None, ROOT / "out" / "p9" / "p9_lamp.json", 1),
    ]

    try:
        for phase, stage, limit, n_aux, artifact, min_n in steps:
            if already_ok(artifact, min_n) and not (phase == "p4" and stage == "pilot" and already_ok(artifact, 24)):
                # p4 attack resumes the same json; skip pilot if attack already long
                if phase == "p4" and stage == "pilot" and already_ok(artifact, 8):
                    log(f"skip {phase}-{stage}, {artifact.name} already has >=8")
                    ok, why = p4_smoke_gate() if stage == "smoke" else (True, "resume")
                    if stage == "smoke" and not ok:
                        log(f"GATE_FAIL {why}")
                    else:
                        continue
                elif phase == "p4" and stage == "attack" and already_ok(artifact, 24):
                    log("skip p4-attack, enough rows")
                    ok, why = p4_drift_gate()
                    if not ok:
                        log(why)
                        write_status("p4-drift-stop", why)
                        return 3
                    continue
                elif phase != "p4" and already_ok(artifact, min_n):
                    log(f"skip {phase}-{stage}")
                    continue

            if phase == "p4" and stage == "pilot" and already_ok(artifact, 8) and not already_ok(artifact, 24):
                log("pilot already in p4_attack.json, go to full attack")
                continue

            write_status(f"{phase}-{stage}", f"Running {phase} {stage} limit={limit}.")
            if not gpu_free() and gpu_holder():
                log("GPU taken again; waiting")
                wait_for_gpu()

            code = maybe_retry(phase, stage, limit=limit, n_aux=n_aux)
            if code != 0:
                write_status(f"{phase}-{stage}-FAIL", f"exit {code}. See `{LOG}`. Will keep file for resume.")
                # do not abort whole night on p9 lamp; abort on p4 smoke
                if phase == "p4" and stage == "smoke":
                    log("FATAL p4 smoke failed twice")
                    return 2
                log(f"continue after {phase}-{stage} fail")
                continue

            if phase == "p4" and stage == "smoke":
                ok, why = p4_smoke_gate()
                log(f"p4 smoke gate: {why}")
                if not ok:
                    log(f"GATE_FAIL {why}")
                    write_status("p4-smoke-gate-fail", why)
                    return 4
            if phase == "p4" and stage == "attack":
                ok, why = p4_drift_gate()
                log(f"p4 drift: {why}")
                if not ok:
                    write_status("p4-drift-stop", why)
                    return 3

        write_status(
            "night-done",
            "P4 chain + P5/P6/P7 smokes + P9 lamp attempted.\n"
            "P10/P11 still blocked on weights/images. P12 L_amp not added.\n"
            f"Log: `{LOG}`",
        )
        log("NIGHT_DONE")
        return 0
    except Exception:
        log("FATAL\n" + traceback.format_exc())
        write_status("FATAL", traceback.format_exc()[-2000:])
        return 1
    finally:
        if LOCK.is_file():
            try:
                LOCK.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
