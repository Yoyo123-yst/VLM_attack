#!/usr/bin/env python3
"""N1-R pilot: same-budget greedy PGD restarts. No sampling, no confirm, no N1 scan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from n0.axes import four_axes  # noqa: E402
from n1.pilot import (  # noqa: E402
    HIDDEN_KEY,
    PILOT_CARRIERS,
    PILOT_EPS,
    PILOT_LAYERS,
    PILOT_QUERIES,
    PILOT_RESTARTS,
    PILOT_STEPS,
    REPEAT_N,
    SEALED,
    delta_stats,
    evaluate_pilot,
    match_pairs,
    pgd_restart,
    sha256_ids,
    sha256_tensor,
    vec_sha256,
)
from n1.preflight import all_carrier_index  # noqa: E402
from p0.catalog import all_pairs  # noqa: E402
from p0.datautil import load_json, open_image, save_json  # noqa: E402
from p0.judge import harmful_compliance  # noqa: E402
from p0.model import COMPLY_WORDS, REFUSE_WORDS, token_id_list  # noqa: E402
from p0_qwen.config import load_cfg  # noqa: E402
from p0_qwen.traces import _trim_judge  # noqa: E402

OUT = ROOT / "outputs" / "n1r"
HIDDEN_DIR = OUT / "hidden"


def _assert_no_sealed() -> None:
    for qid in PILOT_QUERIES:
        if qid in SEALED:
            raise RuntimeError(f"pilot query sealed {qid}")


def collect(wrapper, cfg) -> Dict[str, Any]:
    catalog = {p["id"]: p for p in all_pairs()}
    carriers = all_carrier_index()
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    alpha = float(cfg["attack"]["alpha"])
    layers = list(PILOT_LAYERS)
    records = []
    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)
    for qid in PILOT_QUERIES:
        if qid in SEALED:
            raise RuntimeError(f"sealed query {qid}")
        h = catalog[qid]
        q = h["query"]
        for cid in PILOT_CARRIERS:
            c = carriers[cid]
            img = open_image(c["path"])
            x0 = wrapper.image_to_x01(img)
            x0_sha = sha256_tensor(x0)
            for eps in PILOT_EPS:
                for k in range(PILOT_RESTARTS):
                    seed = 2026 + 1009 * (int(qid[1:]) + 17 * int(cid[1:]) + 31 * k) + int(round(eps * 255))
                    rec_id = f"{qid}:{cid}:e{int(round(eps*255))}:r{k}"
                    delta = pgd_restart(
                        wrapper, img, q, PILOT_STEPS, float(eps), alpha, refuse_ids, comply_ids, seed
                    )
                    x01 = torch.clamp(x0 + delta, 0.0, 1.0)
                    pixel_sha = sha256_tensor(x01)
                    dpath = HIDDEN_DIR / f"{rec_id}.delta.pt"
                    torch.save(delta.detach().float().cpu(), dpath)
                    enc = wrapper.encode(img, q)
                    ids_sha = sha256_ids(enc["input_ids"].detach().cpu().reshape(-1).tolist())
                    hid = wrapper.collect_hidden(img, q, layers, x01=x01)
                    vecs = {kk: v.detach().float().cpu().numpy() for kk, v in hid.items()}
                    np.savez_compressed(HIDDEN_DIR / f"{rec_id}.npz", **vecs)
                    h24 = vecs[HIDDEN_KEY]
                    text = wrapper.generate(img, q, x01=x01, do_sample=False)
                    judge = harmful_compliance(text, h["topic_keywords"], q)
                    ax = four_axes(text, h["topic_keywords"], q, judge)
                    rec = {
                        "record_id": rec_id,
                        "query_id": qid,
                        "carrier_id": cid,
                        "category": h.get("category"),
                        "eps": float(eps),
                        "steps": int(PILOT_STEPS),
                        "restart": int(k),
                        "seed": int(seed),
                        "decode": "greedy",
                        "do_sample": False,
                        "attack_objective": "refusal_margin_pgd",
                        "eot": None,
                        "template": "qwen2vl_official_native",
                        "clean_image_sha256": x0_sha,
                        "pixel_sha256": pixel_sha,
                        "delta_path": str(dpath),
                        "input_ids_sha256": ids_sha,
                        "hidden_sha256": vec_sha256(h24),
                        "hidden_key": HIDDEN_KEY,
                        "delta_stats": delta_stats(delta),
                        "legacy_label": judge.get("label"),
                        "core_rhc": bool(ax["core_rhc"]),
                        "core_safe_answer": bool(ax["core_safe_answer"]),
                        "safety": ax["safety"],
                        "response_mode": ax["response_mode"],
                        "grounding": ax["grounding"],
                        "quality": ax["quality"],
                        "chars": int(ax["chars"]),
                        "judge": _trim_judge(judge),
                    }
                    records.append(rec)
                    save_json(OUT / "pilot_records.json", {"n": len(records), "records": records})
                    print(
                        {
                            "id": rec_id,
                            "label": judge.get("label"),
                            "core_rhc": rec["core_rhc"],
                            "core_safe": rec["core_safe_answer"],
                            "linf": rec["delta_stats"]["linf"],
                        },
                        flush=True,
                    )
                    torch.cuda.empty_cache()
    return {"records": records}


def load_hidden_map(records) -> Dict[str, np.ndarray]:
    out = {}
    for rec in records:
        blob = np.load(HIDDEN_DIR / f"{rec['record_id']}.npz")
        out[rec["record_id"]] = np.asarray(blob[HIDDEN_KEY], dtype=np.float64).reshape(-1)
    return out


def collect_repeats(wrapper, records, pairs) -> Dict[str, list]:
    catalog = {p["id"]: p for p in all_pairs()}
    carriers = all_carrier_index()
    need = set()
    for p in pairs:
        need.add(p["rhc_record_id"])
        need.add(p["safe_record_id"])
    by_id = {r["record_id"]: r for r in records}
    repeats = {}
    for rid in sorted(need):
        rec = by_id[rid]
        img = open_image(carriers[rec["carrier_id"]]["path"])
        q = catalog[rec["query_id"]]["query"]
        delta = torch.load(rec["delta_path"], map_location=wrapper.device, weights_only=True)
        x0 = wrapper.image_to_x01(img)
        x01 = torch.clamp(x0 + delta.to(wrapper.device), 0.0, 1.0)
        if sha256_tensor(x01) != rec["pixel_sha256"]:
            raise RuntimeError(f"stale pixel reload {rid}")
        vecs = []
        first = np.load(HIDDEN_DIR / f"{rid}.npz")[HIDDEN_KEY]
        vecs.append(np.asarray(first, dtype=np.float64).reshape(-1))
        for _ in range(REPEAT_N - 1):
            hid = wrapper.collect_hidden(img, q, list(PILOT_LAYERS), x01=x01)
            v = hid[HIDDEN_KEY].detach().float().cpu().numpy().reshape(-1)
            if vec_sha256(v) != rec["hidden_sha256"] and _ == -1:
                pass
            vecs.append(np.asarray(v, dtype=np.float64))
            torch.cuda.empty_cache()
        repeats[rid] = vecs
    return repeats


def write_md(gate: dict, label_counts: dict) -> str:
    lines = [
        "# N1-R PILOT",
        "",
        f"- gate_pass: **{gate['pass']}**",
        f"- route: {gate['route']}",
        f"- n_pairs: {gate['n_pairs']} (need ≥ 8)",
        f"- n_queries: {gate['n_queries']} {gate['query_ids']}",
        f"- n_carriers: {gate['n_carriers']} {gate['carrier_ids']}",
        f"- categories: {gate['categories']}",
        f"- eps_units: {gate['eps_units']}",
        f"- mean d_pair / sigma_repeat / ratio: {gate.get('mean_d_pair')} / {gate.get('mean_sigma')} / {gate.get('mean_ratio')}",
        f"- mean last-prompt cosine: {gate.get('mean_cosine')}",
        "",
        "## Label counts (all restarts)",
        "",
        json.dumps(label_counts, indent=2),
        "",
        "## Reasons",
        "",
    ]
    for r in gate.get("reasons") or []:
        lines.append(f"- {r}")
    if not gate.get("reasons"):
        lines.append("- (none)")
    lines += [
        "",
        "Sampling was not used. Confirm was not filled. Development/attack-test generations were not read.",
        "N1 residual scan remains blocked unless this pilot passes.",
        "",
    ]
    if not gate["pass"]:
        lines += [
            "Pilot **failed**. Do not expand recapture. Do not invent SAFE with a different PGD objective.",
            "If the same budget only yields RHC or REF/DENY, route B has no identifiable pre-generation pairs.",
            "",
        ]
    else:
        lines += ["Pilot **passed**. Formal recapture of discover/development may start next.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/p0_qwen.yaml"))
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _assert_no_sealed()
    if args.cpu_only:
        recs = load_json(OUT / "pilot_records.json")["records"] if (OUT / "pilot_records.json").exists() else []
        pairs = match_pairs(recs)
        hidden = load_hidden_map(recs) if recs else {}
        repeats = {rid: [hidden[rid]] for rid in hidden}
        gate = evaluate_pilot(recs, pairs, hidden, repeats)
        save_json(OUT / "pilot_pairs.json", {"n": len(pairs), "pairs": pairs})
        save_json(OUT / "pilot_gate.json", gate)
        md = write_md(gate, {})
        (OUT / "PILOT.md").write_text(md, encoding="utf-8")
        print(md, flush=True)
        raise SystemExit(0 if gate["pass"] else 4)

    from run_p0_qwen import load_model, seed_all, vram_preflight

    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_cfg(Path(args.config), scale="full")
    seed_all(int(cfg["seed"]))
    pre = vram_preflight(10.0)
    print({"vram_preflight": pre}, flush=True)
    if not pre.get("ok"):
        save_json(OUT / "collect_error.json", pre)
        raise SystemExit(2)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    blob = collect(wrapper, cfg)
    recs = blob["records"]
    from collections import Counter

    label_counts = dict(Counter(r["legacy_label"] for r in recs))
    pairs = match_pairs(recs)
    hidden = load_hidden_map(recs)
    repeats = collect_repeats(wrapper, recs, pairs)
    gate = evaluate_pilot(recs, pairs, hidden, repeats)
    save_json(OUT / "pilot_pairs.json", {"n": len(pairs), "pairs": pairs})
    save_json(OUT / "pilot_gate.json", gate)
    save_json(OUT / "pilot_records.json", {"n": len(recs), "records": recs, "label_counts": label_counts})
    md = write_md(gate, label_counts)
    (OUT / "PILOT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    if not gate["pass"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
