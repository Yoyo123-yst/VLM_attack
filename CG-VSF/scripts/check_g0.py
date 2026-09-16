#!/usr/bin/env python3
"""G0 completeness gate. Any failure blocks P0."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CG = HERE.parent
ROOT = CG.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CG / "src"))

from cgvsf.protocol import FROZEN_PATH, hard_cells, judge_source_hash, load_frozen, sha256_file
from cgvsf.reporting import atomic_write_text, now, write_status
from traceflip.protocol import EVAL_SEEDS, OPT_SEEDS, SEALED, assert_seed_split


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

    assert_seed_split()
    if list(frozen["seeds"]["opt"]) != list(OPT_SEEDS):
        problems.append("opt seeds drift from traceflip.protocol")
    if list(frozen["seeds"]["eval"]) != list(EVAL_SEEDS):
        problems.append("eval seeds drift from traceflip.protocol")
    if set(frozen["seeds"]["opt"]) & set(frozen["seeds"]["eval"]):
        problems.append("opt/eval seeds overlap")

    for q, _c in hard_cells(frozen):
        if q in set(SEALED) or (q.startswith("h") and 83 <= int(q[1:]) <= 130):
            problems.append(f"hard cell uses sealed query {q}")

    if frozen["scope"]["store_raw_outputs"] is not False:
        problems.append("store_raw_outputs must be false")
    if frozen["certificates"]["store_raw_outputs"] is not False:
        problems.append("certificate store_raw_outputs must be false")

    methods = list(frozen["methods_p0a"])
    for req in (
        "refusal_margin",
        "last_certificate",
        "accumulated_certificate",
        "targeted_prefix_earlystop",
        "gateflip_fair",
    ):
        if req not in methods:
            problems.append(f"missing fair method {req}")
    if frozen["budget"]["inner_steps"] * frozen["budget"]["rounds"] != frozen["budget"][
        "max_backward_per_cell_method"
    ]:
        problems.append("5x24 must equal 120 backward")
    if not frozen["budget"]["decode_every_round"]:
        problems.append("targeted-prefix early-stop requires per-round decode")

    diag = ROOT / frozen["g0_decoder_alignment"]["artifact"]
    if not diag.exists():
        problems.append(f"missing alignment diag {diag}")
    else:
        blob = json.loads(diag.read_text(encoding="utf-8"))
        if not blob.get("cached_vs_solver_agree"):
            problems.append("delta=0 solver/decode disagree")
        # After-solve disagreement is documented, not a silent fail.
        expl = frozen["g0_decoder_alignment"]["explanation"]
        if "official greedy" not in expl.lower() and "official_greedy" not in frozen[
            "g0_decoder_alignment"
        ].get("p0_policy", ""):
            problems.append("decoder-alignment explanation missing P0 accept policy")
        if blob.get("inmodel_agrees_with_redecode") is True:
            pass
        elif not blob.get("inmodel_agrees_with_redecode"):
            if frozen["g0_decoder_alignment"]["p0_policy"] != "official_greedy_execute_only":
                problems.append("must explain after-solve mismatch by refusing in-model accept")

    if frozen["success"]["judge"] != "n0.axes.four_axes+p0.judge.harmful_compliance":
        problems.append("judge id drift")
    jhash = judge_source_hash()

    if frozen["budget"]["eps"] > 16.0 / 255.0 + 1e-12:
        problems.append("eps exceeds 16/255")

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
    out = CG / "out" / "G0_REPORT.md"
    lines = [
        "# CG-VSF G0",
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
        CG / "out" / "STATUS.md",
        stage="P0-A",
        state="NOT_STARTED" if g0["ok"] else "BLOCKED",
        frozen_sha=sha,
        git_commit=g0["git_commit"],
        started=now(),
        verdict="PENDING" if g0["ok"] else "STOP",
        evidence="G0 passed" if g0["ok"] else "; ".join(problems),
        next_cmd="python CG-VSF/scripts/run_p0a.py --config CG-VSF/configs/p0a.yaml --resume"
        if g0["ok"]
        else "fix G0 then re-run check_g0.py",
    )
    print(json.dumps(g0, ensure_ascii=False, indent=2))
    return 0 if g0["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
