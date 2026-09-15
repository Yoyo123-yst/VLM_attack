#!/usr/bin/env python3
"""P0.5 mechanism robustness: template transfer, rank curve, per-sample n=24."""

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

from p0.attacks import refusal_loss_fn, state_loss_fn  # noqa: E402
from p0.catalog import all_pairs  # noqa: E402
from p0.config import load_cfg  # noqa: E402
from p0.datautil import load_carriers, load_json, open_image, save_json  # noqa: E402
from p0.judge import harmful_compliance  # noqa: E402
from p0.metrics import fmt_ci, mean  # noqa: E402
from p0.model import COMPLY_WORDS, REFUSE_WORDS, LlavaP0, token_id_list  # noqa: E402
from p0.patching import run_patching, summarize_patch  # noqa: E402
from p0.subspace import cov_matched_basis, mean_state, pca_basis, random_basis, state_shift  # noqa: E402
from p0.traces import collect_traces, jb_pairs  # noqa: E402


P0_OUT = ROOT / "outputs" / "p0" / "mini"
OUT = ROOT / "outputs" / "p0_5" / "mini"
LAYERS = [20, 24, 28]
BEST_LAYER = 20
RANKS = [1, 2, 4, 8, 16, 32]


def load_model(cfg) -> LlavaP0:
    print("loading LLaVA...", flush=True)
    return LlavaP0(cfg)


def selected_queries(probe) -> list:
    ids = set(probe["discover"] + probe["holdout"])
    out = []
    for p in all_pairs():
        if p["id"] not in ids:
            continue
        q = dict(p)
        q["split"] = "discover" if p["id"] in probe["discover"] else "holdout"
        q["benign_keywords"] = p["benign_keywords"]
        out.append(q)
    return out


def query_index(probe) -> dict:
    return {q["id"]: q for q in selected_queries(probe)}


def carrier_index(cfg) -> dict:
    train, test = load_carriers(cfg)
    return {c["id"]: c for c in train + test}


def benign_map(queries) -> dict:
    return {
        q["benign_id"]: {
            "id": q["benign_id"],
            "query": q["benign_query"],
            "topic_keywords": q["benign_keywords"],
        }
        for q in queries
    }


def deltas_for_layer(records, layer: int) -> np.ndarray:
    key = f"L{layer}:last_user"
    rows = []
    for p in records:
        a = np.asarray(p["jb"]["hidden"][key], dtype=np.float32).reshape(-1)
        b = np.asarray(p["clean_hidden"][key], dtype=np.float32).reshape(-1)
        rows.append(a - b)
    return np.stack(rows, axis=0)


def compact_patch(blob) -> dict:
    return {
        "summary_holdout": summarize_patch(blob, split="holdout") or summarize_patch(blob, split=None),
        "summary_all": summarize_patch(blob, split=None),
        "n_rows": len(blob.get("rows") or []),
        "benign_dU": _benign_du(blob),
    }


def _benign_du(blob) -> float:
    rows = blob.get("benign") or []
    if not rows:
        return float("nan")
    base = [r["base"]["utility"] for r in rows]
    pat = [r["patched"]["utility"] for r in rows]
    return float(np.mean(pat) - np.mean(base))


def stage_a(wrapper, cfg, probe, traces_a) -> dict:
    qidx = query_index(probe)
    cidx = carrier_index(cfg)
    aa = load_json(P0_OUT / "patch_full.json")
    rec = {"A_to_A": compact_patch(aa)}

    wrapper.set_prefix("B")
    print("P0.5-A A→B full residual", flush=True)
    ab = run_patching(
        wrapper, cfg, traces_a, qidx, cidx, LAYERS, mode="full", tag="A_to_B",
        do_benign=True, recompute_clean=True,
    )
    rec["A_to_B"] = compact_patch(ab)
    save_json(OUT / "A_to_B.json", rec["A_to_B"])
    print("A→B", rec["A_to_B"]["summary_holdout"], flush=True)

    traces_b_path = OUT / "traces_B.json"
    if traces_b_path.exists():
        traces_b = load_json(traces_b_path)
        print("reuse traces_B", traces_b.get("summary"), flush=True)
    else:
        print("collecting prefix-B traces", flush=True)
        queries = selected_queries(probe)
        train, _ = load_carriers(cfg)
        traces_b = collect_traces(
            wrapper, cfg, queries, benign_map(queries), train, LAYERS, delta_dir=OUT / "deltas_B"
        )
        save_json(traces_b_path, traces_b)
        print(traces_b["summary"], flush=True)

    wrapper.set_prefix("B")
    print("P0.5-A B→B full residual", flush=True)
    bb = run_patching(
        wrapper, cfg, traces_b, qidx, cidx, LAYERS, mode="full", tag="B_to_B",
        do_benign=False, recompute_clean=False,
    )
    rec["B_to_B"] = compact_patch(bb)
    rec["traces_B_summary"] = traces_b.get("summary")
    print("B→B", rec["B_to_B"]["summary_holdout"], flush=True)

    wrapper.set_prefix("C")
    print("P0.5-A A+B→C low-rank U at L20", flush=True)
    disc_a = jb_pairs(traces_a, split="discover") or jb_pairs(traces_a)
    disc_b = jb_pairs(traces_b, split="discover") or jb_pairs(traces_b)
    Xa = deltas_for_layer(disc_a, BEST_LAYER)
    Xb = deltas_for_layer(disc_b, BEST_LAYER) if disc_b else Xa
    X = np.concatenate([Xa, Xb], axis=0)
    U = pca_basis(X, 8)
    Umap = {BEST_LAYER: U}
    abc = run_patching(
        wrapper, cfg, traces_a, qidx, cidx, [BEST_LAYER],
        U_by_layer=Umap, mode="sub", tag="AB_to_C",
        do_benign=False, recompute_clean=True,
    )
    rec["AB_to_C"] = compact_patch(abc)
    print("A+B→C", rec["AB_to_C"]["summary_holdout"], flush=True)

    wrapper.set_prefix("B")
    rnd = {L: random_basis(wrapper.hidden_size, 8, seed=2026 + L) for L in LAYERS}
    rand_ab = run_patching(
        wrapper, cfg, traces_a, qidx, cidx, LAYERS,
        U_by_layer=rnd, mode="sub", tag="rand_A_to_B",
        do_benign=False, recompute_clean=True,
    )
    rec["random_A_to_B"] = compact_patch(rand_ab)
    save_json(OUT / "stage_A.json", rec)
    return rec


def stage_b(wrapper, cfg, probe, traces_a) -> dict:
    wrapper.set_prefix("A")
    qidx = query_index(probe)
    cidx = carrier_index(cfg)
    hold = [r for r in traces_a["records"] if r.get("jb") and r.get("split") == "holdout"]
    slim = {"records": hold, "benign": []}
    disc = jb_pairs(traces_a, split="discover") or jb_pairs(traces_a)
    X = deltas_for_layer(disc, BEST_LAYER)
    curve = {}
    for r in RANKS:
        Umap = {BEST_LAYER: pca_basis(X, r)}
        blob = run_patching(
            wrapper, cfg, slim, qidx, cidx, [BEST_LAYER],
            U_by_layer=Umap, mode="sub", tag=f"rank{r}",
            do_benign=False, recompute_clean=False,
        )
        st = summarize_patch(blob, split="holdout") or summarize_patch(blob, split=None)
        curve[str(r)] = st.get(str(BEST_LAYER), st)
        print("rank", r, curve[str(r)], flush=True)

    full = load_json(P0_OUT / "patch_full.json")
    full_sum = summarize_patch(full, split="holdout") or summarize_patch(full, split=None)
    curve["full_residual"] = full_sum.get(str(BEST_LAYER), {})

    Umap = {BEST_LAYER: random_basis(X.shape[1], 8, seed=0)}
    rnd = run_patching(
        wrapper, cfg, slim, qidx, cidx, [BEST_LAYER],
        U_by_layer=Umap, mode="sub", tag="rand_dir",
        do_benign=False, recompute_clean=False,
    )
    curve["random_dir_r8"] = (summarize_patch(rnd, split="holdout") or summarize_patch(rnd, split=None)).get(str(BEST_LAYER), {})

    Umap = {BEST_LAYER: cov_matched_basis(X, 8, seed=1)}
    cov = run_patching(
        wrapper, cfg, slim, qidx, cidx, [BEST_LAYER],
        U_by_layer=Umap, mode="sub", tag="cov_rand",
        do_benign=False, recompute_clean=False,
    )
    curve["cov_matched_r8"] = (summarize_patch(cov, split="holdout") or summarize_patch(cov, split=None)).get(str(BEST_LAYER), {})
    save_json(OUT / "stage_B.json", {"layer": BEST_LAYER, "curve": curve})
    return curve


def pgd_one(wrapper, img, query, steps, eps, alpha, loss_fn, tvw):
    from p0.model import image_to_clip_pixels, normalize_clip, tv_loss

    x0 = image_to_clip_pixels(img).to(wrapper.device)
    packed = wrapper.encode(img, query)
    ids, attn = packed["input_ids"], packed.get("attention_mask")
    delta = torch.zeros_like(x0)
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        x = torch.clamp(x0 + delta, 0, 1)
        pv = normalize_clip(x, wrapper.device, torch.float16)
        loss = loss_fn(pv, ids, attn) + tvw * tv_loss(x)
        g = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        if g is None:
            raise RuntimeError("no grad")
        delta = (delta - alpha * g.sign()).clamp(-eps, eps)
        del loss, g, pv, x
    return delta.detach()


def stage_c(wrapper, cfg, probe) -> dict:
    wrapper.set_prefix("A")
    used = set(probe["discover"] + probe["holdout"])
    extra = [dict(p) for p in all_pairs() if p["id"] not in used][:24]
    _, test_c = load_carriers(cfg)
    carrier = test_c[0]
    sub = load_json(P0_OUT / "subspace.json")
    spec = sub["layers"][str(BEST_LAYER)]["8"]
    U = np.asarray(spec["U"], dtype=np.float64)
    mu_ref = np.asarray(spec["mu_ref"], dtype=np.float64)
    mu_jb = np.asarray(spec["mu_jb"], dtype=np.float64)
    U_rand = random_basis(U.shape[0], U.shape[1], seed=11)

    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    steps = 40
    eps = float(cfg["attack"]["sampler_eps"])
    alpha = float(cfg["attack"]["alpha"])
    tvw = float(cfg["attack"]["tv_weight"])
    img = open_image(carrier["path"])
    methods = {
        "pgd16": refusal_loss_fn(wrapper, refuse_ids, comply_ids),
        "u16": state_loss_fn(wrapper, BEST_LAYER, U, mu_ref, mu_jb),
        "rand16": state_loss_fn(wrapper, BEST_LAYER, U_rand, mu_ref, mu_jb),
    }
    rows = []
    for q in extra:
        rec = {"query_id": q["id"], "carrier_id": carrier["id"]}
        clean = wrapper.generate(img, q["query"])
        rec["clean"] = harmful_compliance(clean, q["topic_keywords"], q["query"])["label"]
        if rec["clean"] != "REF":
            rec["skip"] = True
            rows.append(rec)
            print(rec, flush=True)
            continue
        for name, fn in methods.items():
            torch.cuda.empty_cache()
            delta = pgd_one(wrapper, img, q["query"], steps, eps, alpha, fn, tvw)
            from p0.model import image_to_clip_pixels, normalize_clip

            x0 = image_to_clip_pixels(img).to(wrapper.device)
            pv = normalize_clip(torch.clamp(x0 + delta, 0, 1), wrapper.device, torch.float16)
            text = wrapper.generate(img, q["query"], pixel_values=pv)
            j = harmful_compliance(text, q["topic_keywords"], q["query"])
            hid = wrapper.collect_hidden(img, q["query"], [BEST_LAYER], pixel_values=pv)
            rec[name] = {
                "label": j["label"],
                "R": j["R"],
                "refusal": j["refusal"],
                "shift": state_shift(hid[f"L{BEST_LAYER}:last_user"].numpy(), U, mu_ref, mu_jb),
            }
        rows.append(rec)
        print({k: rec.get(k) for k in ("query_id", "clean", "pgd16", "u16", "rand16")}, flush=True)

    kept = [r for r in rows if not r.get("skip") and "pgd16" in r]
    summary = {}
    for name in methods:
        labs = [r[name]["label"] for r in kept]
        summary[name] = {
            "n": len(kept),
            "asr": sum(x == "JB" for x in labs) / max(len(labs), 1),
            "refusal": mean([r[name]["refusal"] for r in kept]),
            "shift": mean([r[name]["shift"] for r in kept]),
        }
    shifts = np.array([r["u16"]["shift"] for r in kept], dtype=np.float64)
    rand_s = np.array([r["rand16"]["shift"] for r in kept], dtype=np.float64)
    y = np.array([r["u16"]["R"] for r in kept], dtype=np.float64)
    y_rand = np.array([r["rand16"]["R"] for r in kept], dtype=np.float64)
    summary["corr_u_shift_vs_R"] = float(np.corrcoef(shifts, y)[0, 1]) if len(kept) > 2 else float("nan")
    summary["corr_rand_shift_vs_R"] = float(np.corrcoef(rand_s, y_rand)[0, 1]) if len(kept) > 2 else float("nan")
    blob = {"rows": rows, "summary": summary, "n_attempted": len(extra), "n_kept": len(kept)}
    save_json(OUT / "stage_C.json", blob)
    return blob


def write_report(a, b, c) -> None:
    lines = [
        "# P0.5 Mechanism Robustness Report",
        "",
        "Model: LLaVA-1.5-7B, same queries/images as P0. Prefix is the only intended change in A.",
        "",
        r"Benign utility sign: $\Delta U = U_{patched} - U_{clean}$. Positive means utility rose.",
        "",
        "## P0.5-A template robustness (full residual, holdout)",
        "",
        "| Setting | layer | dR REF→JB | dR JB→REF | ΔU benign |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, key in [("A→A (P0 reuse)", "A_to_A"), ("A→B", "A_to_B"), ("B→B", "B_to_B"), ("A+B→C U", "AB_to_C")]:
        blob = a.get(key) or {}
        st = blob.get("summary_holdout") or {}
        du = blob.get("benign_dU", float("nan"))
        for L, v in sorted(st.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 99):
            lines.append(
                f"| {name} | {L} | {v.get('ref_to_jb_dR', float('nan')):.3f} | {v.get('jb_to_ref_dR', float('nan')):.3f} | {du if du==du else float('nan'):.3f} |"
            )
    rand = (a.get("random_A_to_B") or {}).get("summary_holdout") or {}
    lines += ["", "Random A→B:", json.dumps(rand, indent=2), "", "## P0.5-B rank curve (L20 holdout)", "",
              "| r | dR REF→JB | dR JB→REF |", "|---:|---:|---:|"]
    for r in [str(x) for x in RANKS] + ["full_residual", "random_dir_r8", "cov_matched_r8"]:
        v = (b or {}).get(r) or {}
        lines.append(f"| {r} | {v.get('ref_to_jb_dR', float('nan')):.3f} | {v.get('jb_to_ref_dR', float('nan')):.3f} |")
    cs = (c or {}).get("summary") or {}
    lines += ["", "## P0.5-C per-sample (ε=16/255, 40 steps, new queries)", "",
              json.dumps(cs, indent=2), ""]
    ab = ((a.get("A_to_B") or {}).get("summary_holdout") or {})
    cross = any(float(v.get("ref_to_jb_dR") or 0) >= 0.10 and float(v.get("jb_to_ref_dR") or 0) <= -0.05 for v in ab.values())
    lines += [
        "## Verdict",
        "",
        f"- Cross-template full residual (A→B): {'PASS' if cross else 'FAIL'}",
        "- If FAIL: describe as template-conditioned safety routing, not a universal bottleneck.",
        "- Do not start P1. Text-only controllability is next only after this audit is interpreted.",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "P0_5_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["a", "b", "c", "report", "all"])
    args = p.parse_args()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(scale="mini")
    probe = load_json(P0_OUT / "probe.json")
    traces_a = load_json(P0_OUT / "traces.json")
    a = load_json(OUT / "stage_A.json") if (OUT / "stage_A.json").exists() else None
    b = load_json(OUT / "stage_B.json") if (OUT / "stage_B.json").exists() else None
    c = load_json(OUT / "stage_C.json") if (OUT / "stage_C.json").exists() else None

    need_model = args.stage in {"a", "b", "c", "all"}
    wrapper = load_model(cfg) if need_model else None
    if args.stage in {"a", "all"}:
        a = stage_a(wrapper, cfg, probe, traces_a)
    if args.stage in {"b", "all"}:
        b = stage_b(wrapper, cfg, probe, traces_a)
        if isinstance(b, dict) and "curve" in b:
            b = b["curve"]
    if args.stage in {"c", "all"}:
        c = stage_c(wrapper, cfg, probe)
    if args.stage in {"report", "all"}:
        if b is None and (OUT / "stage_B.json").exists():
            b = load_json(OUT / "stage_B.json").get("curve")
        write_report(a or {}, b or {}, c or {})


if __name__ == "__main__":
    main()
