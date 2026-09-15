# Route C — Causal Cross-Modal (DENY) Representation-Guided Attack

## Problem statement

A jailbreak attack against Qwen2-VL-7B-Instruct whose success signal is the
**multimodal-specific DENY surface** (visual-grounding denial), guided by a
causal cross-modal representation claim:

> If we can steer the visual representation along a direction that flips the
> model from a safe *denial* into an unsafe *answer*, then the attack exploits a
> decision boundary that is unique to multimodal models (unavailable to
> text-only LLMs).

## Why the simple form died

Three findings killed the naive "label + residual patch" version:

1. **Label poisoning** — `core_safe_answer` is *absence-defined* (no positive
   condition), so caption-degenerate outputs get mislabeled as the SAFE pole.
2. **Label instability** — `fluency.repeat_ratio <= 0.45` splits near-identical
   behaviors into two labels (RHC vs GARBAGE).
3. **Tooling gap** — `generate_with_patch` early-returns when `seq == 1`, so
   mid-generation patching is unsupported (needs `src/routec/patch_mid.py`).

## Frozen gates (write-before-run, see `ROUTEC_FROZEN.json`)

- **G0a** — degenerate content pairs `<= 1/9` → continue; `>= 2/9` → rebuild dataset.
- **G0b** — corrected `n_pairs >= 8` AND `n_queries >= 4` → continue; else downgrade.
- G0c / G0d belong to later GPU phases (P0.4 / P0.5).

## Phase 0 — A段 (pure CPU)

| Script | Output | Question |
| --- | --- | --- |
| `audit_labels.py` (P0.1) | `out/audit_labels.json` | which records are caption-degenerate / repeat-garbage |
| `audit_pairs.py` (P0.2) | `out/audit_pairs.json` | how many corrected+length-gated content pairs survive |
| `audit_delta_geo.py` (P0.3) | `out/audit_delta_geo.json` | are same-cell deltas geometrically distinct (cosine/corr) |

## Phase 0 — B段 (GPU standby, NOT run in A段)

- `recollect_hidden.py` (P0.4) — recollect `L12/L16/L20/L24:last_user` hidden
  states for the 42 DENY records + same-cell REFUSE/ANSWER controls.
- `src/routec/patch_mid.py` (P0.5) — mid-generation patching tool (future).

## Ground rules

- Do **not** modify existing files outside `routeC/` + `src/routec/`.
- Sealed untouched: h83–h90, h91–h130.
