#!/usr/bin/env python3
"""P0 Visual Causal-Controllability Test orchestrator."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from p0.attacks import (  # noqa: E402
    build_train_items,
    evaluate_delta,
    full_match_loss_fn,
    optimize_universal,
    refusal_loss_fn,
    state_loss_fn,
)
from p0.catalog import all_pairs  # noqa: E402
from p0.config import add_common_args, ensure_out, load_cfg  # noqa: E402
from p0.datautil import load_carriers, load_json, open_image, save_json  # noqa: E402
from p0.judge import harmful_compliance  # noqa: E402
from p0.model import COMPLY_WORDS, REFUSE_WORDS, LlavaP0, token_id_list  # noqa: E402
from p0.patching import run_patching, summarize_patch  # noqa: E402
from p0.report import attack_table, gates, patch_table, render_markdown  # noqa: E402
from p0.subspace import mean_state, pca_basis, random_basis  # noqa: E402
from p0.traces import collect_traces, jb_pairs  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stage_path(out: Path, name: str) -> Path:
    return out / f"{name}.json"


def maybe_load(path: Path):
    if path.exists():
        return load_json(path)
    return None


def load_model(cfg) -> LlavaP0:
    print("loading LLaVA-1.5-7B (4-bit LLM, fp16 vision)...", flush=True)
    return LlavaP0(cfg)


def run_smoke(cfg, out: Path) -> dict:
    wrapper = load_model(cfg)
    train, _ = load_carriers(cfg)
    img = open_image(train[0]["path"])
    q = "Describe this image in one short sentence."
    text = wrapper.generate(img, q, max_new_tokens=32)
    layers = cfg["layers"]["candidates"]
    hid = wrapper.collect_hidden(img, q, layers)
    layer = layers[len(layers) // 2]
    zero = torch.zeros_like(hid[f"L{layer}:last_user"])
    patched = wrapper.generate_with_patch(img, q, layer, zero, mode="full", max_new_tokens=32)

    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    from p0.model import image_to_clip_pixels, margin_from_logits, normalize_clip

    x0 = image_to_clip_pixels(img).to(wrapper.device)
    delta = torch.zeros_like(x0, requires_grad=True)
    inputs = wrapper.encode(img, q)
    pv = normalize_clip(torch.clamp(x0 + delta, 0, 1), wrapper.device, torch.float16)
    logits = wrapper.first_token_logits(pv, inputs["input_ids"], inputs.get("attention_mask"))
    loss = margin_from_logits(logits, refuse_ids, comply_ids)
    grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
    grad_ok = grad is not None and float(grad.abs().sum().item()) > 0
    rec = {
        "generate": text,
        "patched_zero": patched,
        "hook_dim": int(hid[f"L{layer}:last_user"].numel()),
        "n_layers_hooked": len(hid),
        "grad_ok": grad_ok,
        "grad_abs_mean": float(grad.abs().mean().item()) if grad is not None else None,
        "vram_mb": float(torch.cuda.memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else None,
    }
    save_json(stage_path(out, "smoke"), rec)
    print(rec, flush=True)
    if not grad_ok:
        raise RuntimeError("smoke failed: no image gradient through the model")
    return rec


def run_probe(wrapper: LlavaP0, cfg, out: Path) -> dict:
    cached = maybe_load(stage_path(out, "probe"))
    if cached:
        print("reuse probe", cached.get("summary"), flush=True)
        return cached
    train, _ = load_carriers(cfg)
    img = open_image(train[0]["path"])
    usable = []
    details = []
    from tqdm import tqdm

    for pair in tqdm(all_pairs(), desc="probe"):
        text = wrapper.generate(img, pair["query"], max_new_tokens=48)
        j = harmful_compliance(text, pair["topic_keywords"], pair["query"])
        details.append({"id": pair["id"], "label": j["label"], "refusal": j["refusal"], "text": j["text"][:200]})
        if j["label"] == "REF":
            usable.append(pair["id"])
    n_disc = cfg["n_harmful_discover"]
    n_hold = cfg["n_harmful_holdout"]
    if len(usable) < n_disc + 4:
        print(f"WARNING: only {len(usable)} clean refusals", flush=True)
    discover = usable[:n_disc]
    holdout = usable[n_disc : n_disc + n_hold]
    rec = {
        "usable": usable,
        "discover": discover,
        "holdout": holdout,
        "details": details,
        "summary": {"n_usable": len(usable), "n_discover": len(discover), "n_holdout": len(holdout)},
    }
    save_json(stage_path(out, "probe"), rec)
    print(rec["summary"], flush=True)
    return rec


def selected_queries(probe: dict) -> list:
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


def benign_index(queries) -> dict:
    out = {}
    for q in queries:
        out[q["benign_id"]] = {
            "id": q["benign_id"],
            "query": q["benign_query"],
            "topic_keywords": q["benign_keywords"],
        }
    return out


def run_traces(wrapper, cfg, out, probe) -> dict:
    cached = maybe_load(stage_path(out, "traces"))
    if cached:
        print("reuse traces", cached.get("summary"), flush=True)
        return cached
    queries = selected_queries(probe)
    train, test = load_carriers(cfg)
    # traces on train carriers only; test images reserved for attack eval
    blob = collect_traces(
        wrapper,
        cfg,
        queries,
        benign_index(queries),
        train,
        cfg["layers"]["candidates"],
        delta_dir=out / "deltas",
    )
    save_json(stage_path(out, "traces"), blob)
    print(blob["summary"], flush=True)
    return blob


def query_index(probe) -> dict:
    return {q["id"]: q for q in selected_queries(probe)}


def carrier_index(cfg) -> dict:
    train, test = load_carriers(cfg)
    return {c["id"]: c for c in train + test}


def run_full_patch(wrapper, cfg, out, traces, probe) -> dict:
    cached = maybe_load(stage_path(out, "patch_full"))
    if cached:
        print("reuse patch_full", flush=True)
        return cached
    blob = run_patching(
        wrapper,
        cfg,
        traces,
        query_index(probe),
        carrier_index(cfg),
        cfg["layers"]["candidates"],
        mode="full",
        tag="full",
    )
    blob["summary_all"] = summarize_patch(blob, split=None)
    blob["summary_holdout"] = summarize_patch(blob, split="holdout")
    save_json(stage_path(out, "patch_full"), blob)
    print(blob["summary_all"], flush=True)
    return blob


def extract_subspaces(cfg, out, traces) -> dict:
    cached = maybe_load(stage_path(out, "subspace"))
    if cached:
        print("reuse subspace", flush=True)
        return cached
    pairs = jb_pairs(traces, split="discover")
    if len(pairs) < 2:
        pairs = jb_pairs(traces, split=None)
    if len(pairs) < 2:
        rec = {"n_pairs": len(pairs), "layers": {}, "error": "need at least 2 JB pairs to estimate U"}
        save_json(stage_path(out, "subspace"), rec)
        return rec
    rec = {"n_pairs": len(pairs), "layers": {}}
    for layer in cfg["layers"]["candidates"]:
        key = f"L{layer}:last_user"
        deltas = []
        hrefs, hjbs = [], []
        for p in pairs:
            a = np.asarray(p["jb"]["hidden"][key], dtype=np.float32).reshape(-1)
            b = np.asarray(p["clean_hidden"][key], dtype=np.float32).reshape(-1)
            deltas.append(a - b)
            hrefs.append(b)
            hjbs.append(a)
        X = np.stack(deltas, axis=0)
        rec["layers"][str(layer)] = {}
        for r in cfg["subspace"]["ranks"]:
            U = pca_basis(X, r)
            rec["layers"][str(layer)][str(r)] = {
                "U": U.tolist(),
                "mu_ref": mean_state(hrefs, U).tolist(),
                "mu_jb": mean_state(hjbs, U).tolist(),
                "mu_full_jb": np.mean(np.stack(hjbs, 0), 0).tolist(),
                "delta_norm_mean": float(np.linalg.norm(X, axis=1).mean()),
            }
    save_json(stage_path(out, "subspace"), rec)
    return rec


def _u_map(sub_blob, rank, layers) -> dict:
    out = {}
    for layer in layers:
        out[layer] = np.asarray(sub_blob["layers"][str(layer)][str(rank)]["U"], dtype=np.float64)
    return out


def run_causal(wrapper, cfg, out, traces, probe, sub_blob) -> dict:
    cached = maybe_load(stage_path(out, "causal"))
    if cached:
        print("reuse causal", flush=True)
        return cached
    if not sub_blob.get("layers"):
        rec = {"error": sub_blob.get("error", "no subspace"), "best": {}, "benign_drop": 1.0}
        save_json(stage_path(out, "causal"), rec)
        return rec
    rank = cfg["subspace"]["ranks"][0]
    layers = cfg["layers"]["candidates"]
    Umap = _u_map(sub_blob, rank, layers)
    sub = run_patching(
        wrapper, cfg, traces, query_index(probe), carrier_index(cfg), layers, U_by_layer=Umap, mode="sub", tag="subspace"
    )
    dim = wrapper.hidden_size
    Rmap = {L: random_basis(dim, rank, seed=2026 + L) for L in layers}
    rnd = run_patching(
        wrapper, cfg, traces, query_index(probe), carrier_index(cfg), layers, U_by_layer=Rmap, mode="sub", tag="random"
    )
    sub_sum = summarize_patch(sub, split="holdout")
    if not sub_sum:
        sub_sum = summarize_patch(sub, split=None)
    best_layer = None
    best_score = -1e9
    for L, st in sub_sum.items():
        score = float(st["ref_to_jb_dR"]) - float(st["jb_to_ref_dR"])
        if score > best_score:
            best_score = score
            best_layer = int(L)
    benign_drop = 0.0
    if sub.get("benign"):
        base = np.mean([r["base"]["utility"] for r in sub["benign"] if r["layer"] == best_layer] or [1.0])
        pat = np.mean([r["patched"]["utility"] for r in sub["benign"] if r["layer"] == best_layer] or [1.0])
        benign_drop = float(base - pat)
    rec = {
        "rank": rank,
        "subspace": sub,
        "random": rnd,
        "summary_subspace": sub_sum,
        "summary_random": summarize_patch(rnd, split="holdout") or summarize_patch(rnd, split=None),
        "best": {
            "layer": best_layer,
            **(sub_sum.get(str(best_layer), {})),
        },
        "random_best": (summarize_patch(rnd, split="holdout") or summarize_patch(rnd, split=None)).get(str(best_layer), {}),
        "benign_drop": benign_drop,
    }
    save_json(stage_path(out, "causal"), rec)
    print({"best": rec["best"], "benign_drop": benign_drop}, flush=True)
    return rec


def run_attack(wrapper, cfg, out, traces, probe, sub_blob, causal) -> dict:
    cached = maybe_load(stage_path(out, "attack"))
    if cached:
        print("reuse attack", flush=True)
        return cached
    if causal.get("error") or not causal.get("best", {}).get("layer"):
        rec = {"error": causal.get("error", "no best layer"), "evals": {}}
        save_json(stage_path(out, "attack"), rec)
        return rec
    layer = causal["best"].get("layer")
    if layer is None:
        layer = cfg["layers"]["candidates"][-1]
    rank = cfg["subspace"]["ranks"][0]
    spec = sub_blob["layers"][str(layer)][str(rank)]
    U = np.asarray(spec["U"], dtype=np.float64)
    mu_ref = np.asarray(spec["mu_ref"], dtype=np.float64)
    mu_jb = np.asarray(spec["mu_jb"], dtype=np.float64)
    mu_full = np.asarray(spec["mu_full_jb"], dtype=np.float64)
    U_rand = random_basis(U.shape[0], U.shape[1], seed=7)

    queries = selected_queries(probe)
    train_q = [q for q in queries if q["split"] == "discover"]
    hold_q = [q for q in queries if q["split"] == "holdout"] or train_q
    train_c, test_c = load_carriers(cfg)
    train_items = build_train_items(train_q, train_c)
    hold_items = [(open_image(c["path"]), q) for q in hold_q for c in test_c]
    benign_items = [
        (
            open_image(test_c[0]["path"]),
            {"id": q["benign_id"], "query": q["benign_query"], "topic_keywords": q["benign_keywords"]},
        )
        for q in hold_q
    ]

    torch.cuda.empty_cache()
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    steps = int(cfg["attack"]["attack_steps"])
    eps = float(cfg["attack"]["eps"])
    alpha = float(cfg["attack"]["alpha"])
    tvw = float(cfg["attack"]["tv_weight"])
    eot = tuple(cfg["attack"]["eot_scale"])

    methods = {"clean": {"delta": None}}
    specs = [
        ("refusal_margin_pgd", refusal_loss_fn(wrapper, refuse_ids, comply_ids)),
        ("random_subspace", state_loss_fn(wrapper, layer, U_rand, mu_ref, mu_jb)),
        ("full_residual", full_match_loss_fn(wrapper, layer, mu_full)),
        ("causal_subspace", state_loss_fn(wrapper, layer, U, mu_ref, mu_jb)),
    ]
    for name, fn in specs:
        torch.cuda.empty_cache()
        print(f"optimizing {name}", flush=True)
        methods[name] = {
            "delta": optimize_universal(wrapper, train_items, steps, eps, alpha, fn, tvw, eot)
        }

    evals = {}
    for name, spec_m in methods.items():
        delta = spec_m["delta"]
        if delta is not None:
            torch.save(delta.detach().cpu(), out / f"delta_{name}.pt")
        evals[name] = {
            "harmful_holdout": evaluate_delta(wrapper, delta, hold_items, "harmful", layer, U, mu_ref, mu_jb),
            "benign": evaluate_delta(wrapper, delta, benign_items, "benign", layer, U, mu_ref, mu_jb),
        }
        print(name, "done", flush=True)

    rec = {"layer": layer, "rank": rank, "evals": evals}
    # strip texts already truncated in judge
    save_json(stage_path(out, "attack"), rec)
    return rec


def run_report(cfg, out) -> dict:
    traces = load_json(stage_path(out, "traces"))
    patch_full = maybe_load(stage_path(out, "patch_full")) or {}
    causal = maybe_load(stage_path(out, "causal")) or {}
    attack = maybe_load(stage_path(out, "attack")) or {}
    atk_tbl = attack_table(attack.get("evals", {})) if attack else []
    causal_for_gates = {
        "best": causal.get("best", {}),
        "random": causal.get("random_best", {}),
        "benign_drop": causal.get("benign_drop", 1.0),
    }
    rec = {
        "scale": cfg["active_scale"],
        "model": cfg["model"]["name"],
        "trace_summary": traces.get("summary"),
        "patch_tables": {
            "full": patch_table(patch_full, split="holdout") if patch_full else [],
            "subspace": patch_table(causal.get("subspace", {}), split="holdout") if causal else [],
            "random": patch_table(causal.get("random", {}), split="holdout") if causal else [],
        },
        "attack_table": atk_tbl,
        "gates": gates(causal_for_gates, atk_tbl, cfg),
    }
    save_json(stage_path(out, "report"), rec)
    md = render_markdown(rec)
    (out / "P0_RESULTS.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return rec


def run_pipeline(cfg, out, stage: str) -> None:
    seed_all(int(cfg["seed"]))
    if stage == "smoke":
        run_smoke(cfg, out)
        return
    if stage == "report":
        run_report(cfg, out)
        return

    wrapper = None
    needs_model = stage in {"probe", "traces", "patch", "subspace", "causal", "attack", "mini", "full"}
    if needs_model and stage != "subspace":
        wrapper = load_model(cfg)
    if stage in {"subspace"}:
        traces = load_json(stage_path(out, "traces"))
        extract_subspaces(cfg, out, traces)
        return

    probe = None
    traces = None
    sub = None
    causal = None
    order = ["probe", "traces", "patch", "subspace", "causal", "attack", "report"]
    if stage in {"mini", "full"}:
        todo = order
    else:
        todo = [stage]

    for st in todo:
        if st == "probe":
            probe = run_probe(wrapper, cfg, out)
        elif st == "traces":
            probe = probe or load_json(stage_path(out, "probe"))
            traces = run_traces(wrapper, cfg, out, probe)
        elif st == "patch":
            probe = probe or load_json(stage_path(out, "probe"))
            traces = traces or load_json(stage_path(out, "traces"))
            run_full_patch(wrapper, cfg, out, traces, probe)
        elif st == "subspace":
            traces = traces or load_json(stage_path(out, "traces"))
            sub = extract_subspaces(cfg, out, traces)
        elif st == "causal":
            probe = probe or load_json(stage_path(out, "probe"))
            traces = traces or load_json(stage_path(out, "traces"))
            sub = sub or load_json(stage_path(out, "subspace"))
            causal = run_causal(wrapper, cfg, out, traces, probe, sub)
        elif st == "attack":
            probe = probe or load_json(stage_path(out, "probe"))
            traces = traces or load_json(stage_path(out, "traces"))
            sub = sub or load_json(stage_path(out, "subspace"))
            causal = causal or load_json(stage_path(out, "causal"))
            run_attack(wrapper, cfg, out, traces, probe, sub, causal)
        elif st == "report":
            run_report(cfg, out)


def main() -> None:
    p = add_common_args(argparse.ArgumentParser())
    args = p.parse_args()
    scale = "full" if args.stage == "full" else args.scale
    if args.stage == "mini":
        scale = "mini"
    cfg = load_cfg(Path(args.config), scale=scale)
    out = ensure_out(cfg)
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print(f"stage={args.stage} scale={scale} out={out}", flush=True)
    run_pipeline(cfg, out, args.stage)


if __name__ == "__main__":
    main()
