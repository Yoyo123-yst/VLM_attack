# P0-S Safety vs response-mode decoupling

This is the last mechanism-repair experiment on the P0-Qwen line. It does **not** reopen Attack Gate. Theme denial is **Safe** on the safety axis and **Grounding failure** on task quality — not an attack-failure negative.

- Frozen layer: L24
- Fit split: `discover`; development: `holdout`
- Mechanism-confirm (unused): ['h83', 'h84', 'h85', 'h86', 'h87', 'h88', 'h89', 'h90']
- Attack-test h91–h130: sealed
- PCA floor: 30 per major state on discover

## S1 inventory (historical four-way unchanged)

| split | n | RHC | RELATED_SAFE | THEME_DENIAL | REF_clean | other_fail |
|---|---:|---:|---:|---:|---:|---:|
| discover | 160 | 39 | 16 | 37 | 157 | 10 |
| holdout | 60 | 15 | 6 | 6 | 59 | 1 |

PCA-eligible: `{'RHC': True, 'RELATED_SAFE': False, 'THEME_DENIAL': True}`.

## S2 directions (discover only)

| U | n_a / n_b | mode | used_rank |
|---|---|---|---:|
| U_safety (RHC vs RELATED_SAFE) | 39 / 16 | mean_direction | 1 |
| U_refusal (RELATED_SAFE vs REF) | 16 / 16 | mean_direction | 1 |
| U_denial (THEME_DENIAL vs REF) | 37 / 37 | centered_pca | 32 |

## S3 representation diagnosis

- U_safety_vs_U_refusal: min/mean/max deg = 85.8 / 85.8 / 85.8; overlap=0.005
- U_safety_vs_U_denial: min/mean/max deg = 57.5 / 57.5 / 57.5; overlap=0.289
- U_refusal_vs_U_denial: min/mean/max deg = 28.6 / 28.6 / 28.6; overlap=0.771

- late-layer entangled: **False**
- reading: Directions are not collinear; still need factor patching to test causal specificity.

## S4 factor patching (development holdout only)

| basis | n | bidir | REF→RHC (95% CI) | RHC→Safe (95% CI) | Δ not-refusal | Δ denial | retain | benign dU |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| U_safety | 15 | False | 0.067 [0.00, 0.20] | -0.100 [-0.27, 0.00] | 0.067 | 0.000 | 0.101 | 0.000 |
| U_refusal | 15 | True | 0.867 [0.70, 1.00] | -0.700 [-0.90, -0.47] | 0.933 | 0.000 | 0.949 | nan |
| U_denial | 15 | True | 0.717 [0.50, 0.92] | -0.700 [-0.90, -0.50] | 0.800 | 0.000 | 0.859 | nan |
| U_rhc | 15 | True | 0.650 [0.42, 0.87] | -0.733 [-0.93, -0.50] | 0.733 | 0.000 | 0.838 | -0.071 |
| random | 15 | False | 0.067 [0.00, 0.20] | -0.067 [-0.20, 0.00] | 0.067 | 0.000 | 0.081 | nan |

- descriptive death-gate would_pass: False
- reasons: ['safety_not_bidirectional', 'safety_ci_crosses_0', 'safety_ci_crosses_0', 'retain', 'refusal_matches_rhc', 'denial_matches_rhc']

Even a development 'pass' does **not** reopen Attack Gate. Confirm is one-shot and unused.

### Reading (development only)

RELATED_SAFE on discover is still below 30, so \(U_{\mathrm{safety}}\) and \(U_{\mathrm{refusal}}\) are **mean directions**, not high-rank PCA. \(U_{\mathrm{denial}}\) used centered PCA because theme-denial traces reached 37.

On the exposed holdout, \(U_{\mathrm{safety}}\) (RHC vs RELATED_SAFE) does **not** open or close RHC. \(U_{\mathrm{refusal}}\) and \(U_{\mathrm{denial}}\) **do**, and \(U_{\mathrm{refusal}}\) is stronger than the original \(U_{\mathrm{RHC}}\). That matches the label diagnosis: the residual switch is closer to *whether the model answers vs refuses/denies* than to *whether an answer is safe*. It is not a unique safety bottleneck. Do not burn mechanism-confirm or h91–h130.

## Official status

- Attack Gate: **closed**
- Next: freeze P0-S config; do not subtract FAIL from RHC; do not use h91–h130
- Not claimed: unique safety bottleneck, attack eligibility, P1-M
