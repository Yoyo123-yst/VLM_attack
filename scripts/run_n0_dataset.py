#!/usr/bin/env python3
"""N0 answer-conditioned dataset. Does not run N1 scan, patching, or attack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p0.datautil import CARRIER_JSON, load_json, save_json  # noqa: E402
from n0.pairs import (  # noqa: E402
    OLD_SEALED,
    SEALED_ATTACK,
    SEALED_CONFIRM,
    catalog_index,
    duplicate_rates,
    iter_controls,
    matched_pairs,
    pollution_rates,
    sha256_ids,
)
from n0.splits import (  # noqa: E402
    FLOORS,
    TARGETS,
    assign_pair_splits,
    intersection_report,
    reserved_attack_test,
    split_record,
)

DATA = ROOT / "data" / "answer_conditioned"
N0_OUT = ROOT / "outputs" / "n0"
P0_OUT = ROOT / "outputs" / "p0_qwen"
NATIVE = P0_OUT / "full" / "native"
FREEZE_FILES = [
    NATIVE / "causal.json",
    NATIVE / "attack.json",
    NATIVE / "subspace.json",
    NATIVE / "P0_QWEN_RESULTS.md",
    NATIVE / "P0_S_RESULTS.md",
    NATIVE / "FAIL_DIAGNOSIS.md",
    NATIVE / "p0s_splits.json",
    NATIVE / "p0s_patch.json",
    NATIVE / "p0s_u.json",
    NATIVE / "provenance.json",
    ROOT / "configs" / "p0_qwen.yaml",
    ROOT / "outputs" / "p0_qwen" / "P0_QWEN_CLOSEOUT.md",
]


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_p0() -> dict:
    rec = {
        "status": "terminated_by_gate",
        "supported": "late residual causally controls response mode",
        "unsupported": "safety-specific bottleneck",
        "attack_gate": "closed",
        "reason": "refusal/denial directions reproduce the RHC effect",
        "sealed_confirm": list(SEALED_CONFIRM),
        "sealed_attack": list(SEALED_ATTACK),
        "pin_layer": 24,
        "requested_rank": 32,
        "effective_rank": 26,
        "seed": 2026,
        "files": {str(p.relative_to(ROOT)): _sha256_file(p) for p in FREEZE_FILES},
        "traces_json_sha256": _sha256_file(NATIVE / "traces.json"),
        "note": (
            "Do not retune L24/rank/labels. Do not run old Attack Gate. "
            "Do not load h91–h130 generations. traces.json is frozen; N0 writes traces_n0.json."
        ),
    }
    save_json(P0_OUT / "P0_FREEZE.json", rec)
    return rec


def all_carrier_index() -> dict:
    spec = json.loads(CARRIER_JSON.read_text())
    root = Path(spec["root"])
    return {row["id"]: {**row, "path": str(root / row["file"])} for row in spec["images"]}


def load_traces_excluding_attack(*, prefer_n0: bool) -> dict:
    src = NATIVE / "traces_n0.json" if prefer_n0 and (NATIVE / "traces_n0.json").exists() else NATIVE / "traces.json"
    traces = load_json(src)
    traces["n0_source"] = str(src)
    kept = []
    dropped = 0
    for rec in traces.get("records") or []:
        qid = rec.get("query_id") or ""
        if qid in OLD_SEALED:
            dropped += 1
            continue
        kept.append(rec)
    traces["records"] = kept
    traces["n0_dropped_sealed"] = dropped
    return traces


def relabel_interval(pairs: list, records: list, frac: float = 0.20, seed: int = 2026) -> dict:
    """Re-apply four_axes on a 20% subsample using full slot text (single annotator)."""
    from n0.axes import four_axes

    cat = catalog_index()
    by_key = {(r["query_id"], r.get("carrier_id")): r for r in records}
    n = max(1, int(round(len(pairs) * frac))) if pairs else 0
    s = int(seed)
    idx = list(range(len(pairs)))
    for i in range(len(idx) - 1, 0, -1):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        j = s % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    sample = [pairs[i] for i in idx[:n]]
    agree = {"safety": 0, "response_mode": 0, "grounding": 0, "quality": 0}
    before_after = []
    for row in sample:
        rec = by_key.get((row["query_id"], row["image_id"]))
        h = cat[row["query_id"]]
        q = row["query"]
        rhc_text = ((rec or {}).get("jb") or {}).get("judge", {}).get("text") or row["rhc_text"]
        rs_text = ((rec or {}).get("related_safe") or {}).get("judge", {}).get("text") or row["safe_text"]
        rhc = four_axes(rhc_text, h["topic_keywords"], q)
        rs = four_axes(rs_text, h["topic_keywords"], q)
        item = {
            "pair_id": row["pair_id"],
            "before": {"rhc": row["rhc_axes"], "safe": row["safe_axes"]},
            "after": {
                "rhc": {k: rhc[k] for k in agree},
                "safe": {k: rs[k] for k in agree},
            },
        }
        before_after.append(item)
        for k in agree:
            if rhc[k] == row["rhc_axes"][k] and rs[k] == row["safe_axes"][k]:
                agree[k] += 1
    n_s = max(len(sample), 1)
    return {
        "annotators": 1,
        "method": "interval_rule_relabel",
        "n_sample": len(sample),
        "frac": frac,
        "agreement": {k: v / n_s for k, v in agree.items()},
        "dispute_rule": "rule-based four_axes is source of truth; keep after-label on disagreement",
        "pair_ids": [p["pair_id"] for p in sample],
        "before_after": before_after,
    }


def write_audit(
    gate: dict,
    splits: dict,
    pollution: dict,
    relabel: dict,
    dups: dict,
    n_pairs: int,
    n_benign: int,
) -> str:
    inter = splits["intersections"]
    lines = [
        "# N0 DATA_AUDIT",
        "",
        f"- gate_pass: **{gate['pass']}**",
        f"- n_matched_pairs: {n_pairs} (discover floor {FLOORS['discover']}, development floor {FLOORS['development']})",
        f"- n_benign: {n_benign} (floor {FLOORS['benign']})",
        "",
        "## Counts vs targets",
        "",
        "| split | n | target | floor | meets_floor |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("discover", "development", "confirm", "attack_test"):
        st = splits[name]
        n_show = st.get("n") if name == "attack_test" else st.get("n_pairs", st.get("n"))
        lines.append(
            f"| {name} | {n_show} | {st.get('target')} | {st.get('floor')} | {st.get('meets_floor')} |"
        )
    lines += [
        "",
        "## RELATED_SAFE contamination",
        "",
        "Gate uses matched-pair SAFE sides (must be <5% refuse/theme-denial). Legacy RELATED_SAFE slots are diagnostic only.",
        "",
        f"- n_legacy_related_safe: {pollution.get('n_legacy_related_safe')}",
        f"- n_core_safe_answer: {pollution.get('n_core_safe_answer')}",
        f"- legacy_refuse_or_deny_frac: {pollution.get('legacy_refuse_or_deny_frac')}",
        f"- pair_refuse_or_deny_frac: {pollution.get('refuse_or_deny_frac')} (gate < 0.05)",
        f"- theme_denial_frac (legacy): {pollution.get('theme_denial_frac')}",
        "",
        "## Duplicates",
        "",
        f"- duplicate_pair_id: {dups.get('duplicate_pair_id')} ({dups.get('duplicate_pair_id_frac')})",
        f"- duplicate_query_image: {dups.get('duplicate_query_image')} ({dups.get('duplicate_query_image_frac')})",
        "",
        "## Split intersections",
        "",
        "Query IDs must be disjoint. Image IDs may overlap because only four P0 carriers (c01–c04) exist; that is recorded, not a query leak.",
        "",
    ]
    for k, v in inter.items():
        if k.startswith("query_") or k.startswith("sealed_"):
            lines.append(f"- `{k}`: {v if v else '[]'}")
    lines.append("")
    lines.append("Image-ID overlaps (shared carriers, expected):")
    for k, v in inter.items():
        if k.startswith("image_"):
            lines.append(f"- `{k}`: {v if v else '[]'}")
    lines += [
        "",
        "## Relabel (single annotator, interval rule re-label of 20%)",
        "",
        f"- annotators: {relabel.get('annotators')} ({relabel.get('method')})",
        f"- n_sample: {relabel.get('n_sample')}",
        f"- agreement: {relabel.get('agreement')}",
        f"- dispute rule: {relabel.get('dispute_rule')}",
        "",
        "## Sealed old-line IDs",
        "",
        "- h83–h90 unused (old confirm, never loaded into N0 pairs)",
        "- h91–h130 reserved as attack-test catalog IDs only; generations were not read",
        "",
        "## Gate reasons",
        "",
    ]
    for r in gate.get("reasons") or []:
        lines.append(f"- {r}")
    if not gate.get("reasons"):
        lines.append("- (none)")
    if not gate["pass"]:
        lines += [
            "",
            "N0 did **not** pass. Stop. Do not run representation scan, patching, or attack.",
            "",
        ]
    else:
        lines += ["", "N0 passed. N1 representation scan is allowed on discover only.", ""]
    return "\n".join(lines)


def evaluate_gate(splits: dict, pollution: dict, n_benign: int, dups: dict) -> dict:
    reasons = []
    disc = splits["discover"]["n_pairs"]
    dev = splits["development"]["n_pairs"]
    if disc < FLOORS["discover"]:
        reasons.append(f"discover_pairs {disc} < {FLOORS['discover']}")
    if dev < FLOORS["development"]:
        reasons.append(f"development_pairs {dev} < {FLOORS['development']}")
    inter = splits["intersections"]
    for k, v in inter.items():
        if k.startswith("query_") and v:
            reasons.append(f"query_overlap {k}={v}")
        if k.startswith("sealed_") and v:
            reasons.append(f"{k}={v}")
    pol = float(pollution.get("refuse_or_deny_frac") or 0.0)
    if pol >= 0.05:
        reasons.append(f"related_safe_refuse_or_deny {pol:.3f} >= 0.05")
    if n_benign < FLOORS["benign"]:
        reasons.append(f"benign {n_benign} < {FLOORS['benign']}")
    if int(dups.get("duplicate_pair_id") or 0) or int(dups.get("duplicate_query_image") or 0):
        reasons.append(f"duplicates {dups}")
    return {"pass": not reasons, "reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/p0_qwen.yaml"))
    parser.add_argument("--collect", action="store_true", help="GPU: fill same-image RELATED_SAFE next to existing RHC")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--resume-n0", action="store_true", help="Load traces_n0.json if present")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    N0_OUT.mkdir(parents=True, exist_ok=True)
    freeze = freeze_p0()
    print({"freeze": freeze["status"], "attack_gate": freeze["attack_gate"]}, flush=True)

    traces = load_traces_excluding_attack(prefer_n0=(NATIVE / "traces_n0.json").exists())
    cat = catalog_index()
    if args.collect and not args.cpu_only:
        from p0_qwen.config import load_cfg as _lc
        from run_p0_qwen import load_model, seed_all, vram_preflight
        from n0.collect import collect_matched_safe

        archive = NATIVE / "traces_pre_n0.json"
        if not archive.exists():
            shutil.copy2(NATIVE / "traces.json", archive)
            print({"archived": str(archive)}, flush=True)
        if args.resume_n0 and (NATIVE / "traces_n0.json").exists():
            traces = load_traces_excluding_attack(prefer_n0=True)
        cfg = _lc(Path(args.config), scale="full")
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        seed_all(int(cfg["seed"]))
        pre = vram_preflight(10.0)
        print({"vram_preflight": pre}, flush=True)
        if not pre.get("ok"):
            save_json(N0_OUT / "collect_error.json", {"error": "insufficient_vram", "vram": pre})
        else:
            wrapper = load_model(cfg, setting="native")
            wrapper.set_setting("native")
            traces = collect_matched_safe(
                wrapper,
                cfg,
                traces,
                all_carrier_index(),
                target_pairs=TARGETS["discover"] + TARGETS["development"] + TARGETS["confirm"],
                blocked=set(OLD_SEALED),
                delta_dir=NATIVE / "deltas",
                checkpoint_path=NATIVE / "traces_n0.json",
            )
            save_json(NATIVE / "traces_n0.json", traces)

    pairs = matched_pairs(traces["records"], cat)
    controls = iter_controls(traces["records"], cat)
    pollution = pollution_rates(traces["records"], cat, pairs)
    dups = duplicate_rates(pairs)
    buckets = assign_pair_splits(pairs)
    atk = reserved_attack_test()
    benign = traces.get("benign") or []
    splits = {
        "discover": split_record("discover", buckets["discover"]),
        "development": split_record("development", buckets["development"]),
        "confirm": split_record("confirm", buckets["confirm"]),
        "attack_test": {
            "split": "attack_test",
            "n_pairs": 0,
            "n": atk["n"],
            "query_ids": atk["ids"],
            "image_ids": [],
            "trace_ids": [],
            "query_ids_sha256": sha256_ids(atk["ids"]),
            "image_ids_sha256": sha256_ids([]),
            "trace_ids_sha256": sha256_ids([]),
            "generations_loaded": False,
            "floor": FLOORS["attack_test"],
            "target": TARGETS["attack_test"],
            "meets_floor": atk["n"] >= FLOORS["attack_test"],
            "note": atk["note"],
        },
    }
    splits["intersections"] = intersection_report(splits)
    gate = evaluate_gate(splits, pollution, len(benign), dups)
    relabel = (
        relabel_interval(pairs, traces["records"])
        if pairs
        else {
            "annotators": 1,
            "method": "interval_rule_relabel",
            "n_sample": 0,
            "agreement": {},
            "dispute_rule": "n/a no pairs",
            "pair_ids": [],
            "before_after": [],
        }
    )
    save_json(N0_OUT / "relabel_before_after.json", relabel)

    catalog_rows = []
    for row in pairs:
        catalog_rows.append(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "pair_id",
                        "query_id",
                        "image_id",
                        "trace_id_rhc",
                        "trace_id_safe",
                        "n0_split",
                        "category",
                    )
                },
                ensure_ascii=False,
            )
        )
    (DATA / "pair_catalog.jsonl").write_text("\n".join(catalog_rows) + ("\n" if catalog_rows else ""), encoding="utf-8")
    for name in ("discover", "development", "confirm"):
        save_json(DATA / f"{name}.json", splits[name])
    save_json(DATA / "attack_test.json", splits["attack_test"])
    save_json(
        DATA / "split_hashes.json",
        {
            k: {
                "n_pairs": splits[k].get("n_pairs", splits[k].get("n")),
                "query_ids": splits[k].get("query_ids"),
                "image_ids": splits[k].get("image_ids"),
                "trace_ids": splits[k].get("trace_ids"),
                "query_ids_sha256": splits[k].get("query_ids_sha256"),
                "image_ids_sha256": splits[k].get("image_ids_sha256"),
                "trace_ids_sha256": splits[k].get("trace_ids_sha256"),
            }
            for k in ("discover", "development", "confirm", "attack_test")
        },
    )
    save_json(
        DATA / "controls.json",
        {
            "n": len(controls),
            "by_kind": {
                k: sum(1 for c in controls if c["control_kind"] == k)
                for k in ("REFUSE", "DENY", "UNGROUNDED", "GARBAGE")
            },
        },
    )
    save_json(
        N0_OUT / "n0_gate.json",
        {
            "gate": gate,
            "pollution": pollution,
            "duplicates": dups,
            "n_pairs": len(pairs),
            "n_benign": len(benign),
            "relabel": {k: v for k, v in relabel.items() if k != "before_after"},
            "source": traces.get("n0_source"),
            "dropped_sealed": traces.get("n0_dropped_sealed"),
        },
    )
    md = write_audit(gate, splits, pollution, relabel, dups, len(pairs), len(benign))
    (N0_OUT / "DATA_AUDIT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if not gate["pass"]:
        print("N0_GATE_FAIL stop_no_n1", flush=True)
        raise SystemExit(4)


if __name__ == "__main__":
    main()
