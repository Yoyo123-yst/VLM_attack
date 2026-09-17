#!/usr/bin/env python3
"""G0 completeness gate. Any failure blocks P0-0 / P0-1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent
ROOT = MS.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MS / "src"))

from msc.protocol import FROZEN_PATH, BANK_NAMES, hard_cells, judge_source_hash, load_frozen, sha256_file
from msc.reporting import atomic_write_text, now, write_status


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default=str(FROZEN_PATH))
    args = ap.parse_args()
    freeze_path = Path(args.freeze)
    frozen = load_frozen(freeze_path)
    problems: list[str] = []

    if set(frozen["seeds"]["opt"]) & set(frozen["seeds"]["eval"]):
        problems.append("opt/eval seeds overlap")

    hard_q = set(frozen["scope"]["hard_queries"])
    for q in frozen["scope"].get("calib_queries", []) + frozen["scope"].get("calib_backup_queries", []):
        if q.startswith("h") and 83 <= int(q[1:]) <= 130:
            problems.append(f"calibration uses sealed query {q}")
        if q in hard_q:
            problems.append(f"calibration overlaps P0-1 test query {q}")

    for q, _c in hard_cells(frozen):
        if q.startswith("h") and 83 <= int(q[1:]) <= 130:
            problems.append(f"hard cell uses sealed query {q}")

    if frozen["scope"]["store_raw_outputs"] is not False:
        problems.append("store_raw_outputs must be false")
    if frozen["states"]["success"] != "core_rhc":
        problems.append("success must remain core_rhc")
    if frozen["states"].get("no_latent_probe") is not True:
        problems.append("latent probe must stay off")
    if frozen["budget"]["inner_steps"] * frozen["budget"]["rounds"] != frozen["budget"]["max_backward_per_cell_method"]:
        problems.append("5x24 must equal 120 backward")
    if not frozen["budget"]["decode_every_round"]:
        problems.append("closed-loop control requires per-round decode")
    if frozen["success"]["judge"] != "n0.axes.four_axes+p0.judge.harmful_compliance":
        problems.append("judge id drift")
    if frozen["budget"]["eps"] > 16.0 / 255.0 + 1e-12:
        problems.append("eps exceeds 16/255")
    if frozen["msc_method"] != "switched":
        problems.append("msc_method must be switched")
    if "targeted_prefix" in frozen["target_free_methods"]:
        problems.append("targeted_prefix must not be a target-free arm")
    banks = frozen["phrase_banks"]
    for name in BANK_NAMES:
        if len(banks[name]) < int(frozen["gates"]["P00_GO"]["min_phrases_per_bank"]):
            problems.append(f"frozen {name} bank too small")
    if banks.get("harvest_from_hard_cells") or banks.get("harvest_from_sealed"):
        problems.append("phrase harvest flags must be false")

    diag = ROOT / frozen["g0_decoder_alignment"]["artifact"]
    if not diag.exists():
        problems.append(f"missing alignment diag {diag}")
    else:
        blob = json.loads(diag.read_text(encoding="utf-8"))
        if not blob.get("cached_vs_solver_agree"):
            problems.append("delta=0 solver/decode disagree")
        if frozen["g0_decoder_alignment"]["p0_policy"] != "official_greedy_execute_only":
            problems.append("must refuse in-model accept")

    cg_int = ROOT / "CG-VSF" / "out" / "integration" / "INTEGRATION_REPORT.md"
    if not cg_int.exists() or "terminate route" not in cg_int.read_text(encoding="utf-8"):
        problems.append("CG-VSF must remain a STOP×STOP negative result")

    hssc_v = ROOT / "HSSC" / "out" / "p01" / "p01_verdict.json"
    if not hssc_v.exists():
        problems.append("missing HSSC P0-1 STOP verdict")
    else:
        hv = json.loads(hssc_v.read_text(encoding="utf-8"))
        if hv.get("verdict") != "STOP":
            problems.append("HSSC P0-1 must remain STOP; do not run HSSC P0-2")

    jhash = judge_source_hash()
    sha = sha256_file(freeze_path)
    g0 = {
        "ok": not problems,
        "problems": problems,
        "frozen_sha256": sha,
        "judge_source_sha256": jhash,
        "git_commit": git_commit(),
        "decoder_alignment": frozen["g0_decoder_alignment"],
        "checked_at": now(),
    }
    out = MS / "out" / "G0_REPORT.md"
    lines = [
        "# MSC G0",
        "",
        f"- ok: **{g0['ok']}**",
        f"- frozen sha256: `{sha}`",
        f"- judge source sha256: `{jhash}`",
        "",
        "## Decoder alignment",
        "",
        frozen["g0_decoder_alignment"]["explanation"],
        "",
        "## Problems",
        "",
    ]
    if problems:
        lines.extend(f"- {p}" for p in problems)
    else:
        lines.append("- none")
    lines.append("")
    atomic_write_text(out, "\n".join(lines))
    write_status(
        MS / "out" / "STATUS.md",
        stage="P0-0",
        state="NOT_STARTED" if g0["ok"] else "BLOCKED",
        frozen_sha=sha,
        git_commit=g0["git_commit"],
        started=now(),
        verdict="PENDING" if g0["ok"] else "STOP",
        evidence="G0 passed" if g0["ok"] else "; ".join(problems),
        next_cmd="python MSC/scripts/run_p00.py"
        if g0["ok"]
        else "fix G0 then re-run check_g0.py",
    )
    print(json.dumps(g0, ensure_ascii=False, indent=2))
    return 0 if g0["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
