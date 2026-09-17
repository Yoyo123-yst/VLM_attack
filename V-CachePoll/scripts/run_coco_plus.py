#!/usr/bin/env python3
"""Small COCO increment: 20 new pairs (c040–c059), then P2 + Rank only.

Does not touch the original 32-sample `out/` results.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
sys.path.insert(0, str(ROOT / "src"))

from vcachepoll.coco_pairs import build_pairs  # noqa: E402
from vcachepoll.config import load_cfg  # noqa: E402
from vcachepoll.io import load_json, save_json  # noqa: E402
from vcachepoll.p2 import stage_p2_crit, stage_p2_screen  # noqa: E402
from vcachepoll.p3 import stage_p3_all  # noqa: E402
from vcachepoll.probe import stage_screen  # noqa: E402
from vcachepoll.attack import stage_p1_attack  # noqa: E402
from vcachepoll.report import _method_stats, stage_p2_report, stage_p3_report  # noqa: E402

OUT = ROOT / "out" / "coco_plus"
OLD = ROOT / "out"
STATUS = OUT / "STATUS.md"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_status(stage: str, extra: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        f"# coco_plus status\n\n- updated: `{_now()}`\n- stage: **{stage}**\n\n{extra.strip()}\n",
        encoding="utf-8",
    )
    print(f"== {stage} ==", flush=True)


def write_plus_pairs() -> dict:
    cfg = load_cfg(ROOT / "configs" / "p0.yaml")
    data = cfg["data"]
    all_pairs = build_pairs(
        coco_root=Path(data["coco_root"]),
        vqa_json=Path(data["vqa_json"]),
        coco_csv=Path(data["coco_csv"]),
        n_pairs=60,
        seed=int(cfg["seed"]),
        question_prefix=str(data["question_prefix"]),
    )
    plus = all_pairs[40:60]
    if len(plus) != 20:
        raise SystemExit(f"expected 20 new pairs, got {len(plus)}")
    ids = [p["pair_id"] for p in plus]
    if ids[0] != "c040" or ids[-1] != "c059":
        raise SystemExit(f"unexpected ids {ids[0]}..{ids[-1]}")
    dest = OUT / "pairs.json"
    save_json(
        dest,
        {
            "dataset": "coco300_plus20",
            "n": len(plus),
            "note": "Increment only: c040–c059. Original c000–c039 stay in out/.",
            "pairs": plus,
        },
    )
    return {"n": len(plus), "ids": ids, "path": str(dest)}


def _cof_rows(path: Path) -> list:
    if not path.is_file():
        return []
    return [r for r in (load_json(path).get("rows") or []) if r.get("eval") and not r.get("error")]


def write_combined() -> dict:
    old_vc = _cof_rows(OLD / "p2_attack.json")
    new_vc = _cof_rows(OUT / "p2_attack.json")
    old_rank = _cof_rows(OLD / "p3_rank.json")
    new_rank = _cof_rows(OUT / "p3_rank.json")
    combined_vc = old_vc + new_vc
    combined_rank = old_rank + new_rank
    blob = {
        "updated_at": _now(),
        "old_n": len(old_vc),
        "new_n": len(new_vc),
        "combined_n": len(combined_vc),
        "vcache_old": _method_stats(old_vc),
        "vcache_new": _method_stats(new_vc),
        "vcache_combined": _method_stats(combined_vc),
        "rank_old": _method_stats(old_rank),
        "rank_new": _method_stats(new_rank),
        "rank_combined": _method_stats(combined_rank),
    }
    save_json(OUT / "COMBINED.json", blob)

    def line(name: str, m: dict) -> str:
        if not m.get("n"):
            return f"| {name} | 0 | — | — | — |"
        return (
            f"| {name} | {m['n']} | {m['frac_comp_only_fail']:.3f} ({m.get('n_cof', 0)}) | "
            f"{m['frac_full_hold']:.3f} | {m['mean_u_evict_rate']:.3f} |"
        )

    md = "\n".join(
        [
            "# V-CachePoll small COCO increment",
            "",
            f"- updated: `{blob['updated_at']}`",
            f"- original pool: `{blob['old_n']}` (c000–c039 kept)",
            f"- new pool: `{blob['new_n']}` (c040–c059 after screen)",
            f"- combined: `{blob['combined_n']}`",
            "",
            "| Split | n | compressed-only fail | full-token hold | U-evict |",
            "|---|---|---|---|---|",
            line("V-CachePoll old", blob["vcache_old"]),
            line("V-CachePoll new", blob["vcache_new"]),
            line("V-CachePoll combined", blob["vcache_combined"]),
            line("Rank-PGD old", blob["rank_old"]),
            line("Rank-PGD new", blob["rank_new"]),
            line("Rank-PGD combined", blob["rank_combined"]),
            "",
            "Protocol unchanged: `r_base=0.2`, `eps=16/255`, 40 steps, B-only.",
            "New batch runs Rank only; Random/Task/CAA numbers stay on the original 32.",
            "",
        ]
    )
    (OUT / "COMBINED.md").write_text(md, encoding="utf-8")
    return blob


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start_from", default="pairs", choices=["pairs", "attack", "report"])
    args = p.parse_args()

    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "out" / "GPU_LOCK"
    lock.write_text(f"coco_plus pid={os.getpid()} {_now()}\n", encoding="utf-8")
    try:
        p0 = load_cfg(ROOT / "configs" / "coco_plus_p0.yaml")
        p2 = load_cfg(ROOT / "configs" / "coco_plus_p2.yaml")
        p3 = load_cfg(ROOT / "configs" / "coco_plus_p3.yaml")

        if args.start_from == "pairs":
            write_status("pairs")
            print(write_plus_pairs(), flush=True)
            write_status("p0-screen", "20 new pairs, r_base=0.5 clean-correct filter.")
            print(stage_screen(p0), flush=True)
            write_status("p2-screen", "Keep those still correct at r_base=0.2.")
            screen = stage_p2_screen(p2)
            print(screen, flush=True)
            if int(screen.get("n_kept") or 0) < 4:
                write_combined()
                raise SystemExit(f"plus batch kept {screen.get('n_kept')}; too few to attack")
            write_status("p2-crit")
            print(stage_p2_crit(p2), flush=True)

        if args.start_from in {"pairs", "attack"}:
            write_status("p2-attack")
            print(stage_p1_attack(p2), flush=True)
            print(stage_p2_report(p2), flush=True)
            if args.start_from == "pairs":
                write_status("p3-rank", "Closest baseline only.")
                print(stage_p3_all(p3, methods=["rank"]), flush=True)

        print(stage_p3_report(p3), flush=True)
        blob = write_combined()
        write_status("done", json.dumps(blob, ensure_ascii=False, indent=2)[:2000])
    finally:
        if lock.is_file():
            lock.unlink()


if __name__ == "__main__":
    main()
