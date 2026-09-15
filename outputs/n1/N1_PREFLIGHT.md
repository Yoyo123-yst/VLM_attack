# N1_PREFLIGHT

- preflight_pass: **False**
- allow_residual_scan: **False**
- status: `stop_recollect`
- gate: `identical_pregeneration_inputs_sampling_only`

N0 **behavior** labels and the 51 answering pairs are still valid. They cannot be used to fit a pre-generation safety direction. See `outputs/n1/STATUS.md`. The next step is **N1-R** (route B): same-budget greedy PGD restarts, not sampling, not confirm backfill.

## What the 51 pairs actually are

Each pair shares `query_id`, `carrier_id`, official Qwen native chat template, and `add_generation_prompt=True`.
`collect_hidden` runs a prompt-only forward pass at the last prompt token (assistant-start).
Generated answer tokens are **not** in the stored hidden vector. This is not teacher forcing.

| split | n | same-x01 sampling | different perturbation | last-prompt cos ≥ 0.999 |
|---|---:|---:|---:|---:|
| discover | 30 | 24 | 6 | 23 |
| development | 20 | 15 | 5 | 15 |
| confirm | 1 | 1 | 0 | 1 |
| all | 51 | 40 | 11 | 39 |

## Checks

1. **Pixels.** `carrier_id` is the COCO file, not the pixels fed to the model. RHC was generated on a PGD-perturbed `x01`. 40/51 pairs then produced SAFE by **sampling on that same perturbed x01**. Those pairs have identical reconstructed pixels. The other 11 pairs used a second perturbation for SAFE (scaled/noise/mild PGD or RHC-fill). No pair uses a clean image for RHC. Clean labels are REF.

2. **Input tokens.** Official native template, same query, same carrier, fixed 336px packing. `input_ids` hashes match on both sides for every pair. Pixel replacement does not change the token sequence.

3. **Why trajectories differ.** Majority: greedy RHC vs temperature sampling SAFE on the **same** perturbed image (different generation seed, `do_sample=True` only on SAFE). Minority: two different image perturbations, still same text/template.

4. **Hidden source.** `free_generation` prompt-only last token. Not teacher-forced answer tokens. Not first-generated-token. Not semantic-divergence.

5. **Attack-condition confound.** Every RHC comes from a perturbed image. SAFE is never a clean-image answer (clean is REF). When pixels differ, RHC is the stronger PGD and SAFE is a weaker/alternate delta. There is no crossed design (RHC on weak delta / SAFE on strong delta).

6. **Length / first token / sampling.** Discover mean chars RHC 389.3 vs SAFE 380.0. RHC is always greedy (`do_sample=False`, seed 0). SAFE on the majority is sampled (temperature 0.95, top_p 0.92, pair-specific seed). First generated tokens differ by construction on sampling pairs.

## Gate decision

Discover is dominated by **identical pre-generation inputs**. A deterministic forward pass at last-prompt / assistant-start therefore yields the **same residual** for SAFE and RHC (23/30 discover pairs have cosine ≥ 0.999). One sampling pair (`h02:c01`) has equal reconstructed pixels but stored hidden cosine 0.85, so that saved residual is stale relative to the current delta; it is still not a valid SAFE/RHC contrast.

That residual cannot be used to learn \(U_{\mathrm{safety}}\). Fitting \(\Delta h = h^{\mathrm{RHC}}-h^{\mathrm{SAFE}}\) at last-prompt would be fitting numerical noise, or, on the remaining pairs, fitting **perturbation magnitude** rather than a safety decision.

**Do not run the discovery residual scan on these stored hidden states.**

Recollect before claiming a pre-generation safety mechanism. Required design:
- If the scientific object is a decision given **fixed** image/text: collect hidden **after** generation starts (first generated token / first SAFE–RHC divergence), and treat it as a content state, not a pre-decision switch.
- If the scientific object is **input-steerable** safety: build a crossed perturbation table (same delta family producing both SAFE and RHC; both labels on strong and weak deltas).
- Do not teacher-force the full answer and call that a safety-decision state.
- Keep h91–h130 sealed. Prefer new carriers (c05–c08) for confirm.

## Reasons

- discover_majority_same_pixels_sampling_last_prompt_hidden_identical
- remaining_pairs_are_different_attack_perturbations_not_crossed
- stored_hidden_is_last_prompt_only_not_teacher_forced_answer
