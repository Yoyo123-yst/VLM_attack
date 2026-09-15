#!/usr/bin/env python3
"""P0-S: safety vs response-mode decoupling. Development; does not reopen Attack Gate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p0.catalog import probe_pairs  # noqa: E402
from p0.datautil import CARRIER_JSON, load_json, save_json  # noqa: E402
from p0_qwen.config import ensure_out, load_cfg, setting_out  # noqa: E402
from p0_qwen.fail_diag import l24_rhc_fail_u  # noqa: E402
from p0_qwen.integrity import PIN_LAYER, assert_split_isolation, build_provenance  # noqa: E402
from p0_qwen.patching import benign_du, run_patching, summarize_factor  # noqa: E402
from p0_qwen.safety_axes import (  # noqa: E402
    CONFIRM_IDS,
    MIN_N_PCA,
    diagnose_directions,
    fit_three_directions,
    frozen_p0s_splits,
    inventory,
    looks_like_p0s_gate,
    slim_fit,
)
from p0_qwen.traces import collect_related_safe_gap  # noqa: E402
from run_p0_qwen import (  # noqa: E402
    carrier_index,
    load_model,
    maybe_load,
    query_index,
    seed_all,
    stage_path,
    vram_preflight,
)


def _fmt(x, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "nan"
    if v != v:
        return "nan"
    return f"{v:.{nd}f}"


def _cpu(cfg, out: Path) -> dict:
    probe = maybe_load(stage_path(out, "probe")) or {}
    traces = load_json(stage_path(out, "traces"))
    assert_split_isolation(probe, traces)
    splits = frozen_p0s_splits(probe, traces)
    inv = inventory(traces)
    fitted = fit_three_directions(traces)
    diag = diagnose_directions(traces, fitted)
    prov = build_provenance(cfg, out)
    rec = {
        "role": "p0s_dev",
        "reopens_attack_gate": False,
        "official_next": "p0s_decouple",
        "layer": PIN_LAYER,
        "min_n_pca": MIN_N_PCA,
        "splits": splits,
        "inventory": {k: v for k, v in inv.items() if k != "axis_rows"},
        "inventory_rows": inv.get("axis_rows"),
        "fit": slim_fit(fitted),
        "diagnosis": diag,
        "provenance": prov,
        "note": (
            "Theme denial is safety-success / task-failure. Do not subtract FAIL from RHC. "
            "Holdout is development. Confirm ids h83–h90 and attack h91–h130 are unused."
        ),
    }
    save_json(out / "p0s_splits.json", splits)
    save_json(out / "p0s_diag.json", rec)
    save_json(out / "p0s_u.json", rec["fit"])
    print(
        {
            "discover_RELATED_SAFE": (inv.get("by_split") or {}).get("discover", {}).get("RELATED_SAFE"),
            "discover_THEME_DENIAL": (inv.get("by_split") or {}).get("discover", {}).get("THEME_DENIAL"),
            "discover_RHC": (inv.get("by_split") or {}).get("discover", {}).get("RHC"),
            "pca_eligible": inv.get("pca_eligible"),
            "U_safety_mode": fitted["U_safety"].get("mode"),
            "U_refusal_mode": fitted["U_refusal"].get("mode"),
            "U_denial_mode": fitted["U_denial"].get("mode"),
            "entangled": diag.get("late_layer_entangled"),
            "reopens_attack_gate": False,
        },
        flush=True,
    )
    return rec


def _collect(cfg, out: Path, cpu_rec: dict) -> dict:
    probe = load_json(stage_path(out, "probe"))
    traces = load_json(stage_path(out, "traces"))
    inv = cpu_rec.get("inventory") or {}
    if not inv.get("need_more_related_safe"):
        print("RELATED_SAFE already >= 30 on discover; skip collect", flush=True)
        return traces
    pre = out / "traces_pre_p0s.json"
    if not pre.exists():
        import shutil

        shutil.copy2(stage_path(out, "traces"), pre)
        print(f"archived traces -> {pre}", flush=True)
    blocked = set(CONFIRM_IDS)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    qidx = query_index(probe)
    harmful = []
    for p in probe_pairs():
        if p["id"] not in (probe.get("discover") or []):
            continue
        q = dict(p)
        q["split"] = "discover"
        harmful.append(q)
    cidx = carrier_index(cfg)
    used = {r["carrier_id"] for r in traces["records"]}
    spec = load_json(CARRIER_JSON)
    root = Path(spec["root"])
    all_train = [{**row, "path": str(root / row["file"])} for row in spec["images"] if row.get("split") == "train"]
    extra = [c for c in all_train if c["id"] not in used]
    retry = list(cidx.values())
    layers = cfg["layers"]["candidates"]
    delta_dir = out / "deltas"
    print({"extra_carriers": [c["id"] for c in extra], "blocked": sorted(blocked)}, flush=True)
    traces = collect_related_safe_gap(
        wrapper,
        cfg,
        traces,
        harmful,
        extra,
        layers,
        target_n=MIN_N_PCA,
        split="discover",
        blocked_ids=blocked,
        delta_dir=delta_dir,
        retry_carriers=retry,
    )
    save_json(stage_path(out, "traces"), traces)
    return traces


def _eval_factor(wrapper, cfg, traces, qidx, cidx, U, tag, do_benign: bool) -> tuple:
    if U is None or np.asarray(U).size == 0:
        return {"n": 0, "bidirectional": False}, float("nan")
    blob = run_patching(
        wrapper,
        cfg,
        traces,
        qidx,
        cidx,
        [PIN_LAYER],
        U_by_layer={PIN_LAYER: np.asarray(U, dtype=np.float64)},
        mode="sub",
        tag=tag,
        do_benign=do_benign,
        split="holdout",
    )
    st = summarize_factor(blob, split="holdout").get(str(PIN_LAYER), {})
    dU = benign_du(blob, PIN_LAYER) if do_benign else float("nan")
    return st, dU


def _gpu_patch(cfg, out: Path, cpu_rec: dict) -> dict:
    causal_before = maybe_load(stage_path(out, "causal")) or {}
    attack_before = maybe_load(stage_path(out, "attack")) or {}
    probe = load_json(stage_path(out, "probe"))
    traces = load_json(stage_path(out, "traces"))
    sub = load_json(stage_path(out, "subspace"))
    packed = l24_rhc_fail_u(sub)
    fit = cpu_rec.get("fit") or load_json(out / "p0s_u.json")
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    qidx, cidx = query_index(probe), carrier_index(cfg)
    full_st = ((maybe_load(stage_path(out, "patch_full")) or {}).get("summary_holdout") or {}).get(str(PIN_LAYER)) or {}
    full_score = float(full_st.get("score") or 0.0)

    def pack(name, U, do_benign):
        st, dU = _eval_factor(wrapper, cfg, traces, qidx, cidx, U, f"p0s_{name}_L24", do_benign)
        retain = float(st.get("score") or 0.0) / full_score if abs(full_score) > 1e-8 else float("nan")
        return {**st, "benign_dU": dU, "retain": retain, "tag": name}

    rows = {
        "U_safety": pack("U_safety", fit["U_safety"]["U"], True),
        "U_refusal": pack("U_refusal", fit["U_refusal"]["U"], False),
        "U_denial": pack("U_denial", fit["U_denial"]["U"], False),
        "U_rhc": pack("U_rhc", packed["U"], True),
        "random": pack("random", fit["U_safety"]["U_rand"], False),
    }
    rec = {
        "role": "p0s_dev",
        "holdout_is_development": True,
        "reopens_attack_gate": False,
        "layer": PIN_LAYER,
        "n_holdout_rhc": rows["U_rhc"].get("n"),
        "full_score": full_score,
        "table": rows,
        "descriptive_gate": looks_like_p0s_gate(
            {"safety_patch": rows["U_safety"]},
            {"U_refusal": rows["U_refusal"], "U_denial": rows["U_denial"], "random": rows["random"]},
        ),
        "confirm_used": False,
        "attack_used": False,
        "note": "Development holdout only. Confirm and attack-test unused.",
    }
    save_json(out / "p0s_patch.json", rec)
    causal = maybe_load(stage_path(out, "causal")) or {}
    attack = maybe_load(stage_path(out, "attack")) or {}
    if causal.get("attack_blocked") is not True or attack.get("error") != attack_before.get("error"):
        raise RuntimeError("P0-S must not clear official Attack Gate artifacts")
    if (causal.get("best") or {}).get("layer") != (causal_before.get("best") or {}).get("layer"):
        raise RuntimeError("P0-S mutated causal.best")
    print(
        {
            "U_safety_bidir": rows["U_safety"].get("bidirectional"),
            "descriptive_would_pass": rec["descriptive_gate"]["would_pass_if_this_were_a_gate"],
            "reopens_attack_gate": False,
        },
        flush=True,
    )
    return rec


def render_markdown(cpu_rec: dict, patch: dict | None) -> str:
    inv = cpu_rec.get("inventory") or {}
    by = inv.get("by_split") or {}
    fit = cpu_rec.get("fit") or {}
    diag = cpu_rec.get("diagnosis") or {}
    splits = cpu_rec.get("splits") or {}
    ang = diag.get("angles") or {}
    lines = [
        "# P0-S Safety vs response-mode decoupling",
        "",
        "This is the last mechanism-repair experiment on the P0-Qwen line. "
        "It does **not** reopen Attack Gate. Theme denial is **Safe** on the safety axis "
        "and **Grounding failure** on task quality — not an attack-failure negative.",
        "",
        f"- Frozen layer: L{cpu_rec.get('layer')}",
        f"- Fit split: `{splits.get('fit')}`; development: `{splits.get('development')}`",
        f"- Mechanism-confirm (unused): {splits.get('mechanism_confirm_ids')}",
        f"- Attack-test h91–h130: sealed",
        f"- PCA floor: {cpu_rec.get('min_n_pca')} per major state on discover",
        "",
        "## S1 inventory (historical four-way unchanged)",
        "",
        "| split | n | RHC | RELATED_SAFE | THEME_DENIAL | REF_clean | other_fail |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, st in by.items():
        lines.append(
            f"| {split} | {st.get('n')} | {st.get('RHC')} | {st.get('RELATED_SAFE')} | "
            f"{st.get('THEME_DENIAL')} | {st.get('REF_clean')} | {st.get('other_fail')} |"
        )
    lines += [
        "",
        f"PCA-eligible: `{inv.get('pca_eligible')}`.",
        "",
        "## S2 directions (discover only)",
        "",
        "| U | n_a / n_b | mode | used_rank |",
        "|---|---|---|---:|",
    ]
    for key, lab in (("U_safety", "RHC vs RELATED_SAFE"), ("U_refusal", "RELATED_SAFE vs REF"), ("U_denial", "THEME_DENIAL vs REF")):
        spec = fit.get(key) or {}
        lines.append(
            f"| {key} ({lab}) | {spec.get('n_a')} / {spec.get('n_b')} | {spec.get('mode')} | {spec.get('used_rank')} |"
        )
    lines += ["", "## S3 representation diagnosis", ""]
    for k, pack in ang.items():
        lines.append(
            f"- {k}: min/mean/max deg = {_fmt(pack.get('min_deg'), 1)} / "
            f"{_fmt(pack.get('mean_deg'), 1)} / {_fmt(pack.get('max_deg'), 1)}; "
            f"overlap={_fmt(pack.get('overlap_frac'), 3)}"
        )
    lines += [
        "",
        f"- late-layer entangled: **{diag.get('late_layer_entangled')}**",
        f"- reading: {diag.get('reading')}",
        "",
    ]
    if not patch:
        lines += ["## S4 factor patching", "", "Not run.", ""]
    else:
        table = patch.get("table") or {}

        def _row(name):
            p = table.get(name) or {}
            return (
                f"| {name} | {p.get('n')} | {p.get('bidirectional')} | "
                f"{_fmt(p.get('ref_to_jb_dR'))} [{_fmt(p.get('ref_to_jb_dR_lo'), 2)}, {_fmt(p.get('ref_to_jb_dR_hi'), 2)}] | "
                f"{_fmt(p.get('jb_to_ref_dR'))} [{_fmt(p.get('jb_to_ref_dR_lo'), 2)}, {_fmt(p.get('jb_to_ref_dR_hi'), 2)}] | "
                f"{_fmt(p.get('ref_to_not_refusal'))} | {_fmt(p.get('ref_to_denial'))} | "
                f"{_fmt(p.get('retain'))} | {_fmt(p.get('benign_dU'))} |"
            )

        lines += [
            "## S4 factor patching (development holdout only)",
            "",
            "| basis | n | bidir | REF→RHC (95% CI) | RHC→Safe (95% CI) | Δ not-refusal | Δ denial | retain | benign dU |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            _row("U_safety"),
            _row("U_refusal"),
            _row("U_denial"),
            _row("U_rhc"),
            _row("random"),
            "",
            f"- descriptive death-gate would_pass: {(patch.get('descriptive_gate') or {}).get('would_pass_if_this_were_a_gate')}",
            f"- reasons: {(patch.get('descriptive_gate') or {}).get('reasons')}",
            "",
            "Even a development 'pass' does **not** reopen Attack Gate. Confirm is one-shot and unused.",
            "",
            "### Reading (development only)",
            "",
            "RELATED_SAFE on discover is still below 30, so \(U_{\\mathrm{safety}}\) and "
            "\(U_{\\mathrm{refusal}}\) are **mean directions**, not high-rank PCA. "
            "\(U_{\\mathrm{denial}}\) used centered PCA because theme-denial traces reached 37.",
            "",
            "On the exposed holdout, \(U_{\\mathrm{safety}}\) (RHC vs RELATED_SAFE) does **not** "
            "open or close RHC. \(U_{\\mathrm{refusal}}\) and \(U_{\\mathrm{denial}}\) **do**, "
            "and \(U_{\\mathrm{refusal}}\) is stronger than the original \(U_{\\mathrm{RHC}}\). "
            "That matches the label diagnosis: the residual switch is closer to "
            "*whether the model answers vs refuses/denies* than to *whether an answer is safe*. "
            "It is not a unique safety bottleneck. Do not burn mechanism-confirm or h91–h130.",
            "",
        ]
    lines += [
        "## Official status",
        "",
        "- Attack Gate: **closed**",
        "- Next: freeze P0-S config; do not subtract FAIL from RHC; do not use h91–h130",
        "- Not claimed: unique safety bottleneck, attack eligibility, P1-M",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs/p0_qwen.yaml"))
    p.add_argument("--scale", type=str, default="full")
    p.add_argument("--setting", type=str, default="native")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--collect-related-safe", action="store_true")
    p.add_argument("--gpu-patch", action="store_true")
    args = p.parse_args()
    cfg = load_cfg(Path(args.config), scale=args.scale)
    out = ensure_out(setting_out(cfg, args.setting))
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    seed_all(int(cfg["seed"]))
    print(
        f"p0s cpu_only={args.cpu_only} collect={args.collect_related_safe} gpu_patch={args.gpu_patch} out={out}",
        flush=True,
    )
    cpu_rec = _cpu(cfg, out)
    patch = None
    run_gpu = args.gpu_patch or args.collect_related_safe
    if run_gpu and not args.cpu_only:
        pre = vram_preflight(10.0)
        print({"vram_preflight": pre}, flush=True)
        if not pre.get("ok"):
            save_json(out / "p0s_patch.json", {"error": "insufficient_vram", "vram": pre, "reopens_attack_gate": False})
            (out / "P0_S_RESULTS.md").write_text(render_markdown(cpu_rec, None), encoding="utf-8")
            return
        if args.collect_related_safe:
            _collect(cfg, out, cpu_rec)
            cpu_rec = _cpu(cfg, out)
        if args.gpu_patch:
            patch = _gpu_patch(cfg, out, cpu_rec)
    md = render_markdown(cpu_rec, patch)
    (out / "P0_S_RESULTS.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
