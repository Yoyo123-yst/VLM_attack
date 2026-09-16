#!/usr/bin/env python3
"""Overnight chain: P3 baselines, then LLaVA capability smoke if GPU is free."""
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
SRC = ROOT / "src"
OUT = ROOT / "out"
PY = "/root/autodl-tmp/conda/envs/vattack/bin/python"
STATUS = OUT / "NIGHT_STATUS.md"
LOCK = OUT / "GPU_LOCK"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_status(stage: str, extra: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# V-CachePoll night status",
        "",
        f"- updated: `{_now()}`",
        f"- stage: **{stage}**",
        f"- pid: `{os.getpid()}`",
        "",
        extra.strip(),
        "",
    ]
    STATUS.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    LOCK.write_text(f"pid={os.getpid()}\nstage={stage}\nupdated={_now()}\n", encoding="utf-8")


def run_p3() -> int:
    write_status("p3-all", "Running Random → Task-PGD → CAA-B-only → Rank-PGD on 32 pairs.")
    cmd = [PY, "-u", str(ROOT / "scripts" / "run_p3.py"), "--stage", "all"]
    log = OUT / "p3_all.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== night_chain start {_now()} =====\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def read_p3_decision() -> dict:
    path = OUT / "p3_report.json"
    if not path.is_file():
        return {"decision": "WAIT", "reason": "no p3_report.json"}
    return json.loads(path.read_text(encoding="utf-8"))


def llava_smoke() -> dict:
    sys.path.insert(0, str(SRC))
    from vcachepoll.llava_cap import probe_llava

    return probe_llava()


def maybe_fastv_probe() -> dict:
    """Cheap Qwen FastV-style last-token attention keep, 8 pairs, no PGD."""
    sys.path.insert(0, str(SRC))
    from vcachepoll.config import load_cfg
    from vcachepoll.fastv_probe import stage_fastv_probe

    cfg = load_cfg(ROOT / "configs" / "p3.yaml")
    return stage_fastv_probe(cfg, limit=8)


def main() -> int:
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    OUT.mkdir(parents=True, exist_ok=True)
    write_status("start", "Night chain launched.")
    rc = run_p3()
    if rc != 0:
        write_status("p3-failed", f"run_p3.py exited {rc}. See out/p3_all.log.")
        return rc
    report = read_p3_decision()
    extra = json.dumps(report, ensure_ascii=False, indent=2)[:4000]
    write_status(f"p3-{report.get('decision')}", extra)
    try:
        cap = llava_smoke()
        (OUT / "llava_capability.json").write_text(json.dumps(cap, indent=2), encoding="utf-8")
        write_status("llava-smoke", json.dumps(cap, ensure_ascii=False, indent=2)[:4000])
    except Exception as exc:
        write_status("llava-smoke-failed", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
    try:
        fv = maybe_fastv_probe()
        write_status("fastv-probe", json.dumps(fv, ensure_ascii=False, indent=2, default=str)[:4000])
    except Exception as exc:
        write_status(
            "fastv-skipped-or-failed",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}",
        )
    write_status("done", extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
