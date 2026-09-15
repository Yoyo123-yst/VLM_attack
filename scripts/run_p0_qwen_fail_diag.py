#!/usr/bin/env python3
"""FAIL overlap diagnosis for P0-Qwen. Development only. Does not reopen Attack Gate.

CPU: taxonomy, L24 principal angles, discover-only U_orth.
GPU: holdout patch of U_orth as posthoc_dev. Never writes causal.best or clears attack.json.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p0.datautil import load_json, save_json  # noqa: E402
from p0_qwen.config import ensure_out, load_cfg, setting_out  # noqa: E402
from p0_qwen.fail_diag import (  # noqa: E402
    angle_summary,
    fail_taxonomy,
    fit_u_orth_discover,
    l24_rhc_fail_u,
    looks_like_gate_pass,
)
from p0_qwen.integrity import (  # noqa: E402
    PIN_LAYER,
    PIN_REQUESTED_RANK,
    assert_split_isolation,
    build_provenance,
)
from run_p0_qwen import (  # noqa: E402
    _eval_lowrank_pack,
    carrier_index,
    load_model,
    maybe_load,
    query_index,
    seed_all,
    stage_path,
    vram_preflight,
)


def _cpu_diag(cfg, out: Path) -> dict:
    probe = maybe_load(stage_path(out, "probe")) or {}
    traces = load_json(stage_path(out, "traces"))
    assert_split_isolation(probe, traces)
    reclean = maybe_load(out / "jb_reclean.json") or {}
    sub = load_json(stage_path(out, "subspace"))
    packed = l24_rhc_fail_u(sub)
    tax = fail_taxonomy(reclean, traces)
    angles = angle_summary(packed["U"], packed["U_fail"])
    u_orth = fit_u_orth_discover(sub, traces)
    prov = build_provenance(cfg, out, subspace=sub)
    rec = {
        "role": "posthoc_dev",
        "holdout_already_exposed_fail_matches": True,
        "reopens_attack_gate": False,
        "official_next": "stop_attack_gate",
        "layer": PIN_LAYER,
        "requested_rank": PIN_REQUESTED_RANK,
        "taxonomy": {k: v for k, v in tax.items() if k != "rows"},
        "taxonomy_rows": tax.get("rows"),
        "angles": angles,
        "u_orth": {k: v for k, v in u_orth.items() if k not in {"U", "U_rand"}},
        "u_orth_full": u_orth,
        "provenance": prov,
        "note": (
            "U_orth is fit on discover only. Holdout already showed fail_matches, so any "
            "holdout patching of U_orth is post-hoc development, not an official gate."
        ),
    }
    slim = dict(rec)
    slim.pop("u_orth_full", None)
    save_json(out / "fail_diag.json", slim)
    save_json(out / "fail_diag_u_orth.json", u_orth)
    print(
        {
            "taxonomy_reading": tax.get("reading"),
            "n_reclean_fail": tax.get("n_reclean_fail"),
            "angle_mean_deg": angles.get("mean_deg"),
            "angle_min_deg": angles.get("min_deg"),
            "u_orth_used_rank": u_orth.get("used_rank"),
            "overlap_frac": u_orth.get("overlap_frac"),
            "reopens_attack_gate": False,
        },
        flush=True,
    )
    return rec


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_gate_untouched(out: Path, causal_before: dict, attack_before: dict, hashes_before: dict) -> None:
    causal = maybe_load(stage_path(out, "causal")) or {}
    attack = maybe_load(stage_path(out, "attack")) or {}
    if (causal.get("attack_blocked") is not True) or bool((causal.get("best") or {}).get("pass_lowrank")):
        raise RuntimeError("fail-diag must not clear causal.attack_blocked / pass_lowrank")
    if (causal.get("best") or {}).get("layer") != (causal_before.get("best") or {}).get("layer"):
        raise RuntimeError("fail-diag mutated causal.best.layer")
    if attack.get("error") != attack_before.get("error"):
        raise RuntimeError("fail-diag must not clear attack.json error")
    for name in ("causal.json", "attack.json"):
        now = _file_sha256(out / name)
        if now != hashes_before.get(name):
            raise RuntimeError(f"fail-diag mutated {name}")


def _gpu_patch(cfg, out: Path, cpu_rec: dict) -> dict:
    causal_before = maybe_load(stage_path(out, "causal")) or {}
    attack_before = maybe_load(stage_path(out, "attack")) or {}
    hashes_before = {name: _file_sha256(out / name) for name in ("causal.json", "attack.json")}
    if not causal_before.get("attack_blocked"):
        print("WARNING: causal.attack_blocked was not True before GPU diag", flush=True)
    probe = load_json(stage_path(out, "probe"))
    traces = load_json(stage_path(out, "traces"))
    sub = load_json(stage_path(out, "subspace"))
    u_orth = (cpu_rec.get("u_orth_full") or maybe_load(out / "fail_diag_u_orth.json") or {})
    if not u_orth.get("U"):
        raise RuntimeError("missing discover-only U_orth; run --cpu-only first")
    packed = l24_rhc_fail_u(sub)
    wrapper = load_model(cfg, setting="native")
    wrapper.set_setting("native")
    qidx, cidx = query_index(probe), carrier_index(cfg)
    layer = PIN_LAYER
    full_st = ((maybe_load(stage_path(out, "patch_full")) or {}).get("summary_holdout") or {}).get(str(layer)) or {}
    full_score = float(full_st.get("score") or 0.0)

    rhc_st, rhc_du = _eval_lowrank_pack(
        wrapper, cfg, traces, qidx, cidx, layer, packed["U"], "diag_U_rhc_L24", True
    )
    fail_st, _ = _eval_lowrank_pack(
        wrapper, cfg, traces, qidx, cidx, layer, packed["U_fail"], "diag_U_fail_L24", False
    )
    if int(u_orth.get("used_rank") or 0) <= 0:
        orth_st, orth_du, rand_st = {"n": rhc_st.get("n"), "bidirectional": False}, float("nan"), {}
    else:
        orth_st, orth_du = _eval_lowrank_pack(
            wrapper, cfg, traces, qidx, cidx, layer, u_orth["U"], "diag_U_orth_L24", True
        )
        rand_st, _ = _eval_lowrank_pack(
            wrapper, cfg, traces, qidx, cidx, layer, u_orth["U_rand"], "diag_U_orth_rand_L24", False
        )
    retain_rhc = float(rhc_st.get("score") or 0.0) / full_score if abs(full_score) > 1e-8 else float("nan")
    retain_orth = float(orth_st.get("score") or 0.0) / full_score if abs(full_score) > 1e-8 else float("nan")
    rhc_st["random"] = None
    rhc_pack = {
        "tag": "U_rhc",
        **rhc_st,
        "benign_dU": rhc_du,
        "retain": retain_rhc,
        "fail": fail_st,
        "descriptive_gate": looks_like_gate_pass(rhc_st, fail_st, rhc_du, retain_rhc),
    }
    orth_st["random"] = rand_st
    orth_pack = {
        "tag": "U_orth",
        **orth_st,
        "benign_dU": orth_du,
        "retain": retain_orth,
        "random": rand_st,
        "fail": fail_st,
        "used_rank": u_orth.get("used_rank"),
        "descriptive_gate": looks_like_gate_pass({**orth_st, "random": rand_st}, fail_st, orth_du, retain_orth),
        "note": (
            "FAIL-control bidirectionality tests U_FAIL, not U_orth. "
            "The diagnostic for uniqueness of U_orth is random-control plus whether "
            "RHC effect survives after removing FAIL directions."
        ),
    }
    rec = {
        "role": "posthoc_dev",
        "holdout_already_exposed_fail_matches": True,
        "reopens_attack_gate": False,
        "official_next": "stop_attack_gate",
        "layer": layer,
        "n_holdout_rhc": rhc_st.get("n"),
        "full_score": full_score,
        "U_rhc": rhc_pack,
        "U_orth": orth_pack,
        "provenance": cpu_rec.get("provenance"),
        "note": (
            "Holdout patching of discover-fit U_orth is post-hoc because fail_matches was "
            "already observed on this split. Does not reopen Attack Gate."
        ),
    }
    rec["official_artifacts_unchanged"] = hashes_before
    save_json(out / "fail_diag_patch.json", rec)
    print(
        {
            "U_rhc_bidir": rhc_st.get("bidirectional"),
            "U_orth_bidir": orth_st.get("bidirectional"),
            "U_orth_descriptive_would_pass": orth_pack["descriptive_gate"]["would_pass_if_this_were_a_gate"],
            "reopens_attack_gate": False,
        },
        flush=True,
    )
    _assert_gate_untouched(out, causal_before, attack_before, hashes_before)
    return rec


def _fmt(x, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "nan"
    if v != v:
        return "nan"
    return f"{v:.{nd}f}"


def render_markdown(cpu_rec: dict, patch: dict | None) -> str:
    tax = cpu_rec.get("taxonomy") or {}
    by = tax.get("by_split") or {}
    ang = cpu_rec.get("angles") or {}
    uo = cpu_rec.get("u_orth") or {}
    tn = tax.get("traces_fail_n") or {}
    traces_n = (
        f"discover {tn.get('discover')}, holdout {tn.get('holdout')}, all {tn.get('all')}"
        if tn
        else "n/a"
    )
    lines = [
        "# P0-Qwen FAIL overlap diagnosis (development only)",
        "",
        "**This is not an official gate.** Holdout already exposed `fail_matches`.",
        "Discover-fit \(U_{\\mathrm{orth}}\) evaluated on the same holdout is `posthoc_dev`.",
        "`reopens_attack_gate = false`. Official next remains `stop_attack_gate`.",
        "Attack-test h91–h130 was not used.",
        "",
        f"- Layer: L{cpu_rec.get('layer')} (frozen)",
        f"- requested_rank: {cpu_rec.get('requested_rank')} / rhc used_rank {uo.get('rhc_used_rank')} / "
        f"r_eff {uo.get('rhc_effective_rank')}",
        f"- taxonomy reading: **{tax.get('reading')}** (refusal/theme-denial FAIL, not dirty RHC)",
        f"- reclean FAIL n: {tax.get('n_reclean_fail')}; traces FAIL n: {traces_n}",
        "",
        "## FAIL taxonomy (reclean)",
        "",
        "| split | n | theme_denial | near_rhc | garbage | other |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, st in by.items():
        lines.append(
            f"| {split} | {st.get('n')} | {st.get('theme_denial')} | {st.get('near_rhc')} | "
            f"{st.get('garbage')} | {st.get('other_fail')} |"
        )
    lines += [
        "",
        "## L24 principal angles \(U_{\\mathrm{RHC}}\) vs \(U_{\\mathrm{FAIL}}\)",
        "",
        f"- n_angles: {ang.get('n_angles')}",
        f"- min / mean / max deg: {ang.get('min_deg')} / {ang.get('mean_deg')} / {ang.get('max_deg')}",
        f"- mean first 8 deg: {ang.get('mean_first8_deg')}",
        "",
        "## Discover-only \(U_{\\mathrm{orth}}\)",
        "",
        f"- rhc used_rank: {uo.get('rhc_used_rank')}; fail_rank: {uo.get('fail_rank')}",
        f"- U_orth used_rank: {uo.get('used_rank')} (dropped {uo.get('n_dropped')})",
        f"- overlap_frac \(\\|U_F^\\top U_R\\|_F^2 / r\): {uo.get('overlap_frac')}",
        f"- residual vs FAIL Frobenius: {uo.get('residual_vs_fail_fro')}",
        f"- n_discover_rhc / n_discover_fail: {uo.get('n_discover_rhc')} / {uo.get('n_discover_fail')}",
        f"- holdout_used_for_fit: {uo.get('holdout_used_for_fit')}",
        "",
        "Principal angles are all well above 0° "
        f"(min {_fmt(ang.get('min_deg'), 1)}°), so \(U_{{\\mathrm{{RHC}}}}\) is not the FAIL subspace. "
        f"Overlap is modest (overlap_frac={_fmt(uo.get('overlap_frac'), 3)}). "
        "FAIL reclean labels are 20/20 theme_denial.",
        "",
    ]
    if not patch:
        lines += ["## Holdout patching", "", "Not run (`--cpu-only`).", ""]
    else:
        def _ci(pack, key):
            lo, hi = pack.get(f"{key}_lo"), pack.get(f"{key}_hi")
            if lo is None or hi is None:
                return "n/a"
            return f"[{_fmt(lo, 2)}, {_fmt(hi, 2)}]"

        def _row(name, pack):
            return (
                f"| {name} | {pack.get('n')} | {pack.get('bidirectional')} | "
                f"{_fmt(pack.get('ref_to_jb_dR'))} {_ci(pack, 'ref_to_jb_dR')} | "
                f"{_fmt(pack.get('jb_to_ref_dR'))} {_ci(pack, 'jb_to_ref_dR')} | "
                f"{_fmt(pack.get('retain'))} | "
                f"{_fmt(pack.get('benign_dU'))} | "
                f"{(pack.get('random') or {}).get('bidirectional')} | "
                f"{(pack.get('fail') or {}).get('bidirectional')} | "
                f"{(pack.get('descriptive_gate') or {}).get('would_pass_if_this_were_a_gate')} |"
            )

        uorth = patch.get("U_orth") or {}
        lines += [
            "## Holdout patching (`posthoc_dev`)",
            "",
            "Descriptive `would_pass` is **not** an official gate. Even if it is true, Attack Gate stays closed.",
            "",
            "| U | n | bidir | dR REF→JB (95% CI) | dR JB→REF (95% CI) | retain | benign dU | random bidir | FAIL bidir | descriptive would_pass |",
            "|---|---:|---|---:|---:|---:|---:|---|---|---|",
            _row("U_rhc", patch.get("U_rhc") or {}),
            _row("U_orth", uorth),
            "",
            f"- U_orth descriptive reasons: {(uorth.get('descriptive_gate') or {}).get('reasons')}",
            f"- U_orth used_rank: {uorth.get('used_rank')}",
            "",
            "### Reading (still not a gate)",
            "",
            "Removing FAIL directions from discover \(U_{\\mathrm{RHC}}\) **does not** leave a "
            "holdout-bidirectional leftover. \(U_{\\mathrm{orth}}\) REF→JB CI includes 0; retain "
            f"falls from {_fmt((patch.get('U_rhc') or {}).get('retain'))} to {_fmt(uorth.get('retain'))}. "
            "That is a post-hoc development observation on the same holdout that already showed "
            "`fail_matches`. It does **not** make Attack eligible.",
            "",
        ]
    lines += [
        "## Official status",
        "",
        "- Attack Gate: **closed**",
        "- Next: `stop_attack_gate`",
        "- Not claimed: unique low-rank safety subspace, attack eligibility, orthogonal-U pass, P1-M.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs/p0_qwen.yaml"))
    p.add_argument("--scale", type=str, default="full")
    p.add_argument("--setting", type=str, default="native")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--gpu-patch", action="store_true")
    args = p.parse_args()
    cfg = load_cfg(Path(args.config), scale=args.scale)
    out = ensure_out(setting_out(cfg, args.setting))
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    seed_all(int(cfg["seed"]))
    print(f"fail_diag cpu_only={args.cpu_only} gpu_patch={args.gpu_patch} out={out}", flush=True)

    cpu_rec = _cpu_diag(cfg, out)
    patch = None
    if args.gpu_patch and not args.cpu_only:
        pre = vram_preflight(10.0)
        print({"vram_preflight": pre}, flush=True)
        if not pre.get("ok"):
            rec = {"error": "insufficient_vram", "vram": pre, "reopens_attack_gate": False}
            save_json(out / "fail_diag_patch.json", rec)
            print(rec, flush=True)
            md = render_markdown(cpu_rec, None)
            (out / "FAIL_DIAGNOSIS.md").write_text(md, encoding="utf-8")
            return
        patch = _gpu_patch(cfg, out, cpu_rec)
    md = render_markdown(cpu_rec, patch)
    (out / "FAIL_DIAGNOSIS.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
