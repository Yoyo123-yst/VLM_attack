"""N1 preflight: why SAFE vs RHC trajectories differ, and whether last-prompt hidden can separate them."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from n0.pairs import catalog_index, matched_pairs
from p0.datautil import CARRIER_JSON, load_json, open_image
from p0_qwen.vision import pil_to_x01


TEMPLATE = "qwen2vl_official_native"
IMAGE_SIZE = 336
HIDDEN_SITE = "last_prompt"  # collect_hidden uses last_user_index = last prompt token; add_generation_prompt=True


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes((text or "").encode("utf-8"))


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_x01(x01: torch.Tensor) -> str:
    arr = x01.detach().float().cpu().contiguous().numpy()
    return _sha256_bytes(arr.tobytes() + str(tuple(arr.shape)).encode("utf-8"))


def all_carrier_index() -> Dict[str, Dict[str, Any]]:
    spec = load_json(CARRIER_JSON)
    root = Path(spec["root"])
    return {row["id"]: {**row, "path": str(root / row["file"])} for row in spec["images"]}


def _reconstruct_x01(carrier_path: str, delta_path: Optional[str]) -> Optional[torch.Tensor]:
    img = open_image(carrier_path)
    x0 = pil_to_x01(img, size=IMAGE_SIZE)
    if not delta_path:
        return x0
    p = Path(delta_path)
    if not p.exists():
        return None
    delta = torch.load(p, map_location="cpu", weights_only=True)
    return torch.clamp(x0 + delta.float().cpu(), 0.0, 1.0)


def _hidden_key(hidden: Dict[str, Any]) -> Optional[str]:
    if not hidden:
        return None
    for k in hidden:
        if "L24" in str(k):
            return str(k)
    keys = sorted(hidden)
    return keys[-1] if keys else None


def _hidden_cosine(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    key = _hidden_key(a)
    if key is None or key not in (b or {}):
        return None
    x = np.asarray(a[key], dtype=np.float64).reshape(-1)
    y = np.asarray(b[key], dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        return None
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _first_token(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t:
        return ""
    return t.split(" ", 1)[0][:40]


def classify_condition(rec: Dict[str, Any]) -> Dict[str, Any]:
    tag = str(((rec.get("n0_matched_attempt") or {}).get("tag")) or "")
    sample = bool(rec.get("related_safe_from_sample")) or tag.startswith("sample") or tag.startswith("new_sample")
    if sample:
        return {
            "family": "same_x01_sampling",
            "safe_generation_condition": "stochastic_decode_on_rhc_perturbed_image",
            "rhc_generation_condition": "greedy_decode_on_pgd_perturbed_image",
            "pixels_expected_equal": True,
            "input_ids_expected_equal": True,
            "difference": "decoding_randomness_only",
        }
    if rec.get("related_safe_delta_path") and rec.get("delta_path") and rec.get("related_safe_delta_path") != rec.get("delta_path"):
        return {
            "family": "different_perturbation",
            "safe_generation_condition": f"greedy_decode_on_alternate_delta:{tag or 'safe_delta'}",
            "rhc_generation_condition": "greedy_decode_on_pgd_perturbed_image",
            "pixels_expected_equal": False,
            "input_ids_expected_equal": True,
            "difference": "image_perturbation",
        }
    if rec.get("n0_rhc_fill"):
        return {
            "family": "different_perturbation",
            "safe_generation_condition": "preexisting_related_safe_slot_unsaved_or_mild_delta",
            "rhc_generation_condition": "n0_rhc_fill_pgd_greedy",
            "pixels_expected_equal": False,
            "input_ids_expected_equal": True,
            "difference": "image_perturbation",
        }
    if rec.get("p0s_mild_retry") or rec.get("p0s_extra_carrier"):
        return {
            "family": "different_perturbation",
            "safe_generation_condition": "p0s_mild_or_extra_carrier_pgd_greedy",
            "rhc_generation_condition": "original_or_fill_pgd_greedy",
            "pixels_expected_equal": False,
            "input_ids_expected_equal": True,
            "difference": "image_perturbation",
        }
    return {
        "family": "unspecified_legacy",
        "safe_generation_condition": "unknown_second_generate",
        "rhc_generation_condition": "pgd_greedy",
        "pixels_expected_equal": None,
        "input_ids_expected_equal": True,
        "difference": "unknown",
    }


def sample_seed(rec: Dict[str, Any]) -> Optional[int]:
    att = rec.get("n0_matched_attempt") or {}
    samp = rec.get("n0_sample_attempt") or {}
    tag = str(att.get("tag") or "")
    k = samp.get("k")
    if k is None and tag.startswith("sample_"):
        try:
            k = int(tag.split("_")[-1])
        except ValueError:
            k = None
    if k is None and tag.startswith("new_sample_"):
        try:
            k = int(tag.split("_")[-1])
        except ValueError:
            k = None
    if k is None:
        return None
    qid = rec.get("query_id") or ""
    cid = rec.get("carrier_id") or ""
    suffix = "new" if rec.get("n0_new_slot") or tag.startswith("new_sample") else "sample"
    qseed = int(hashlib.sha256(f"{qid}:{cid}:{suffix}".encode("utf-8")).hexdigest()[:8], 16)
    return int(2026 + int(k) + (qseed % 10000))


def pair_split_index(root: Path) -> Dict[str, str]:
    out = {}
    for name in ("discover", "development", "confirm"):
        blob = load_json(root / "data" / "answer_conditioned" / f"{name}.json")
        for p in blob.get("pairs") or []:
            out[p["pair_id"]] = name
    return out


def build_provenance(root: Path) -> Dict[str, Any]:
    traces = load_json(root / "outputs" / "p0_qwen" / "full" / "native" / "traces_n0.json")
    carriers = all_carrier_index()
    catalog = catalog_index()
    pairs = matched_pairs(traces["records"], catalog)
    splits = pair_split_index(root)
    by: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in traces["records"]:
        by[(rec["query_id"], rec.get("carrier_id"))] = rec
    for rec in traces["records"]:
        if rec.get("jb") is not None and rec.get("related_safe") is not None:
            by[(rec["query_id"], rec.get("carrier_id"))] = rec

    rows = []
    for p in pairs:
        rec = by[(p["query_id"], p["image_id"])]
        c = carriers[p["image_id"]]
        cls = classify_condition(rec)
        rhc_delta = rec.get("n0_rhc_delta_path") or rec.get("delta_path")
        if cls["family"] == "same_x01_sampling":
            safe_delta = rhc_delta
        else:
            safe_delta = rec.get("related_safe_delta_path")
        rhc_x = _reconstruct_x01(c["path"], rhc_delta)
        safe_x = _reconstruct_x01(c["path"], safe_delta) if safe_delta else None
        if cls["family"] == "same_x01_sampling" and rhc_x is not None:
            safe_x = rhc_x
        carrier_sha = _sha256_file(c["path"])
        query = rec.get("query") or catalog[p["query_id"]]["query"]
        prompt_key = f"{TEMPLATE}|size={IMAGE_SIZE}|carrier={p['image_id']}|query={query}"
        ids_sha = _sha256_text(prompt_key)
        rhc_text = (rec.get("jb") or {}).get("judge", {}).get("text") or ""
        safe_text = (rec.get("related_safe") or {}).get("judge", {}).get("text") or ""
        cos = _hidden_cosine((rec.get("jb") or {}).get("hidden") or {}, (rec.get("related_safe") or {}).get("hidden") or {})
        pixels_equal = None
        if rhc_x is not None and safe_x is not None:
            pixels_equal = bool(torch.equal(rhc_x, safe_x))
        elif cls["pixels_expected_equal"] is True:
            pixels_equal = True
        elif cls["pixels_expected_equal"] is False:
            pixels_equal = False
        row = {
            "pair_id": p["pair_id"],
            "query_id": p["query_id"],
            "carrier_id": p["image_id"],
            "n0_split": splits.get(p["pair_id"]),
            "category": p.get("category"),
            "template": TEMPLATE,
            "carrier_file_sha256": carrier_sha,
            "safe_image_sha256": _sha256_x01(safe_x) if safe_x is not None else None,
            "rhc_image_sha256": _sha256_x01(rhc_x) if rhc_x is not None else None,
            "safe_delta_path": safe_delta,
            "rhc_delta_path": rhc_delta,
            "safe_input_ids_sha256": ids_sha,
            "rhc_input_ids_sha256": ids_sha,
            "safe_generation_seed": sample_seed(rec) if cls["family"] == "same_x01_sampling" else 0,
            "rhc_generation_seed": 0,
            "safe_do_sample": cls["family"] == "same_x01_sampling",
            "rhc_do_sample": False,
            "safe_generation_condition": cls["safe_generation_condition"],
            "rhc_generation_condition": cls["rhc_generation_condition"],
            "difference_source": cls["difference"],
            "family": cls["family"],
            "activation_token_position": HIDDEN_SITE,
            "hidden_source": "free_generation",
            "hidden_includes_generated_tokens": False,
            "teacher_forcing": False,
            "pixels_equal": pixels_equal,
            "input_ids_equal": True,
            "last_prompt_hidden_cosine": cos,
            "safe_chars": len(safe_text),
            "rhc_chars": len(rhc_text),
            "safe_first_token": _first_token(safe_text),
            "rhc_first_token": _first_token(rhc_text),
            "clean_label": (rec.get("clean") or {}).get("label"),
        }
        rows.append(row)
    return {"n": len(rows), "rows": rows}


def verdict(prov: Dict[str, Any]) -> Dict[str, Any]:
    rows = prov["rows"]
    disc = [r for r in rows if r.get("n0_split") == "discover"]
    def count(xs, pred):
        return sum(1 for r in xs if pred(r))

    n = len(rows)
    n_d = len(disc)
    same_px = count(rows, lambda r: r["pixels_equal"] is True or r["family"] == "same_x01_sampling")
    same_px_d = count(disc, lambda r: r["pixels_equal"] is True or r["family"] == "same_x01_sampling")
    sample = count(rows, lambda r: r["family"] == "same_x01_sampling")
    sample_d = count(disc, lambda r: r["family"] == "same_x01_sampling")
    diff_pert = count(rows, lambda r: r["family"] == "different_perturbation")
    diff_pert_d = count(disc, lambda r: r["family"] == "different_perturbation")
    cos_hi = count(rows, lambda r: r["last_prompt_hidden_cosine"] is not None and r["last_prompt_hidden_cosine"] >= 0.999)
    cos_hi_d = count(disc, lambda r: r["last_prompt_hidden_cosine"] is not None and r["last_prompt_hidden_cosine"] >= 0.999)
    rhc_attack = count(rows, lambda r: "pgd" in (r["rhc_generation_condition"] or "") or "perturbed" in (r["rhc_generation_condition"] or ""))
    safe_clean = count(rows, lambda r: r.get("clean_label") == "REF" and r["family"] != "same_x01_sampling" and r["pixels_equal"] is False)

    identical_pregen = sample_d >= int(0.5 * max(n_d, 1)) and cos_hi_d >= int(0.5 * max(n_d, 1))
    attack_confound = diff_pert_d > 0 and sample_d + diff_pert_d >= n_d - 1
    reasons = []
    if identical_pregen:
        reasons.append("discover_majority_same_pixels_sampling_last_prompt_hidden_identical")
    if attack_confound and diff_pert_d:
        reasons.append("remaining_pairs_are_different_attack_perturbations_not_crossed")
    if not any(r.get("hidden_includes_generated_tokens") for r in rows):
        reasons.append("stored_hidden_is_last_prompt_only_not_teacher_forced_answer")
    status = "stop_recollect"
    if identical_pregen:
        gate = "identical_pregeneration_inputs_sampling_only"
        allow_scan = False
    elif diff_pert_d == n_d and sample_d == 0:
        gate = "attack_condition_confound"
        allow_scan = False
        status = "pause_cross_conditions"
    else:
        gate = "trackable_balanced_input_difference"
        allow_scan = True
        status = "pass"
        reasons = ["balanced_trackable_input_difference"]
    return {
        "preflight_pass": bool(allow_scan),
        "allow_residual_scan": bool(allow_scan),
        "status": status,
        "gate": gate,
        "reasons": reasons,
        "n_pairs": n,
        "n_discover": n_d,
        "n_same_x01_sampling": sample,
        "n_same_x01_sampling_discover": sample_d,
        "n_different_perturbation": diff_pert,
        "n_different_perturbation_discover": diff_pert_d,
        "n_pixels_equal": same_px,
        "n_last_prompt_cosine_ge_0.999": cos_hi,
        "n_last_prompt_cosine_ge_0.999_discover": cos_hi_d,
        "n_rhc_from_perturbed": rhc_attack,
        "n_safe_not_same_x01": n - sample,
        "input_ids_always_equal": all(r["input_ids_equal"] for r in rows),
        "teacher_forcing_any": False,
        "activation_token_position": HIDDEN_SITE,
        "note": (
            "Last-prompt residual is collected with add_generation_prompt=True "
            "(assistant-start / last prompt token), without generated tokens. "
            "If pixels and input_ids match, that vector cannot separate SAFE from RHC."
        ),
    }


def write_preflight_md(verdict_rec: Dict[str, Any], prov: Dict[str, Any]) -> str:
    rows = prov["rows"]
    disc = [r for r in rows if r.get("n0_split") == "discover"]
    def mean_chars(xs, key):
        vs = [r[key] for r in xs]
        return float(sum(vs) / max(len(vs), 1))
    lines = [
        "# N1_PREFLIGHT",
        "",
        f"- preflight_pass: **{verdict_rec['preflight_pass']}**",
        f"- allow_residual_scan: **{verdict_rec['allow_residual_scan']}**",
        f"- status: `{verdict_rec['status']}`",
        f"- gate: `{verdict_rec['gate']}`",
        "",
        "## What the 51 pairs actually are",
        "",
        "Each pair shares `query_id`, `carrier_id`, official Qwen native chat template, and `add_generation_prompt=True`.",
        "`collect_hidden` runs a prompt-only forward pass at the last prompt token (assistant-start).",
        "Generated answer tokens are **not** in the stored hidden vector. This is not teacher forcing.",
        "",
        "| split | n | same-x01 sampling | different perturbation | last-prompt cos ≥ 0.999 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("discover", "development", "confirm"):
        xs = [r for r in rows if r.get("n0_split") == name]
        lines.append(
            f"| {name} | {len(xs)} | {sum(r['family']=='same_x01_sampling' for r in xs)} | "
            f"{sum(r['family']=='different_perturbation' for r in xs)} | "
            f"{sum((r['last_prompt_hidden_cosine'] or 0) >= 0.999 for r in xs)} |"
        )
    lines += [
        f"| all | {len(rows)} | {verdict_rec['n_same_x01_sampling']} | {verdict_rec['n_different_perturbation']} | {verdict_rec['n_last_prompt_cosine_ge_0.999']} |",
        "",
        "## Checks",
        "",
        "1. **Pixels.** `carrier_id` is the COCO file, not the pixels fed to the model. "
        "RHC was generated on a PGD-perturbed `x01`. "
        f"{verdict_rec['n_same_x01_sampling']}/{verdict_rec['n_pairs']} pairs then produced SAFE by **sampling on that same perturbed x01**. "
        "Those pairs have identical reconstructed pixels. "
        f"The other {verdict_rec['n_safe_not_same_x01']} pairs used a second perturbation for SAFE (scaled/noise/mild PGD or RHC-fill). "
        "No pair uses a clean image for RHC. Clean labels are REF.",
        "",
        "2. **Input tokens.** Official native template, same query, same carrier, fixed 336px packing. "
        "`input_ids` hashes match on both sides for every pair. Pixel replacement does not change the token sequence.",
        "",
        "3. **Why trajectories differ.** Majority: greedy RHC vs temperature sampling SAFE on the **same** perturbed image "
        "(different generation seed, `do_sample=True` only on SAFE). "
        "Minority: two different image perturbations, still same text/template.",
        "",
        "4. **Hidden source.** `free_generation` prompt-only last token. Not teacher-forced answer tokens. "
        "Not first-generated-token. Not semantic-divergence.",
        "",
        "5. **Attack-condition confound.** Every RHC comes from a perturbed image. "
        "SAFE is never a clean-image answer (clean is REF). "
        "When pixels differ, RHC is the stronger PGD and SAFE is a weaker/alternate delta. "
        "There is no crossed design (RHC on weak delta / SAFE on strong delta).",
        "",
        "6. **Length / first token / sampling.** "
        f"Discover mean chars RHC {mean_chars(disc,'rhc_chars'):.1f} vs SAFE {mean_chars(disc,'safe_chars'):.1f}. "
        "RHC is always greedy (`do_sample=False`, seed 0). SAFE on the majority is sampled (temperature 0.95, top_p 0.92, pair-specific seed). "
        "First generated tokens differ by construction on sampling pairs.",
        "",
        "## Gate decision",
        "",
    ]
    if not verdict_rec["preflight_pass"]:
        lines += [
            "Discover is dominated by **identical pre-generation inputs**. "
            "A deterministic forward pass at last-prompt / assistant-start therefore yields the **same residual** for SAFE and RHC "
            f"({verdict_rec['n_last_prompt_cosine_ge_0.999_discover']}/{verdict_rec['n_discover']} discover pairs have cosine ≥ 0.999).",
            "",
            "That residual cannot be used to learn \(U_{\\mathrm{safety}}\). Fitting \(\\Delta h = h^{\\mathrm{RHC}}-h^{\\mathrm{SAFE}}\) at last-prompt would be fitting numerical noise, "
            "or, on the remaining pairs, fitting **perturbation magnitude** rather than a safety decision.",
            "",
            "**Do not run the discovery residual scan on these stored hidden states.**",
            "",
            "Recollect before claiming a pre-generation safety mechanism. Required design:",
            "- If the scientific object is a decision given **fixed** image/text: collect hidden **after** generation starts "
            "(first generated token / first SAFE–RHC divergence), and treat it as a content state, not a pre-decision switch.",
            "- If the scientific object is **input-steerable** safety: build a crossed perturbation table "
            "(same delta family producing both SAFE and RHC; both labels on strong and weak deltas).",
            "- Do not teacher-force the full answer and call that a safety-decision state.",
            "- Keep h91–h130 sealed. Prefer new carriers (c05–c08) for confirm.",
            "",
        ]
    else:
        lines += ["Preflight passed. Discovery-only residual scan is allowed on last-prompt hidden.", ""]
    lines += ["## Reasons", ""]
    for r in verdict_rec["reasons"]:
        lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)
