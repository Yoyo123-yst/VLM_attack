# P0-Qwen FAIL overlap diagnosis (development only)

**This is not an official gate.** Holdout already exposed `fail_matches`.
Discover-fit \(U_{\mathrm{orth}}\) evaluated on the same holdout is `posthoc_dev`.
`reopens_attack_gate = false`. Official next remains `stop_attack_gate`.
Attack-test h91–h130 was not used.

- Layer: L24 (frozen)
- requested_rank: 32 / rhc used_rank 26 / r_eff 26
- taxonomy reading: **theme_denial_dominant** (refusal/theme-denial FAIL, not dirty RHC)
- reclean FAIL n: 20; traces FAIL n: discover 21, holdout 7, all 28

## FAIL taxonomy (reclean)

| split | n | theme_denial | near_rhc | garbage | other |
|---|---:|---:|---:|---:|---:|
| discover | 14 | 14 | 0 | 0 | 0 |
| holdout | 6 | 6 | 0 | 0 | 0 |

## L24 principal angles \(U_{\mathrm{RHC}}\) vs \(U_{\mathrm{FAIL}}\)

- n_angles: 8
- min / mean / max deg: 36.31513602229452 / 58.77371299637723 / 75.88109822030482
- mean first 8 deg: 58.77371299637723

## Discover-only \(U_{\mathrm{orth}}\)

- rhc used_rank: 26; fail_rank: 8
- U_orth used_rank: 26 (dropped 0)
- overlap_frac \(\|U_F^\top U_R\|_F^2 / r\): 0.08879064427526306
- residual vs FAIL Frobenius: 2.604371459775329e-16
- n_discover_rhc / n_discover_fail: 27 / 21
- holdout_used_for_fit: False

Principal angles are all well above 0° (min 36.3°), so \(U_{\mathrm{RHC}}\) is not the FAIL subspace. Overlap is modest (overlap_frac=0.089). FAIL reclean labels are 20/20 theme_denial.

## Holdout patching (`posthoc_dev`)

Descriptive `would_pass` is **not** an official gate. Even if it is true, Attack Gate stays closed.

| U | n | bidir | dR REF→JB (95% CI) | dR JB→REF (95% CI) | retain | benign dU | random bidir | FAIL bidir | descriptive would_pass |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| U_rhc | 15 | True | 0.650 [0.42, 0.87] | -0.733 [-0.93, -0.50] | 0.838 | -0.071 | None | True | False |
| U_orth | 15 | False | 0.100 [0.00, 0.27] | -0.367 [-0.60, -0.13] | 0.283 | 0.000 | False | True | False |

- U_orth descriptive reasons: ['not_bidirectional', 'retain', 'fail_matches']
- U_orth used_rank: 26

### Reading (post-hoc, not a gate)

The overlap number 0.089 is **not** evidence that “there is no safety subspace after removing FAIL.”
All 20 reclean FAIL items are theme denial: task failure, **safety success**. Orthogonalizing
\(U_{\mathrm{RHC}}\) against that direction likely deleted part of the safety decision itself.

Geometrically the subspaces are not globally aligned (min angle 36.3°), but the small intersection
carries most of the REF↔RHC causal effect (0.650→0.100, retain 0.838→0.283; benign −0.071→0).
Leftover \(U_{\mathrm{orth}}\) still moves JB→REF somewhat, but cannot open REF→JB. That is a
jailbreak-state / recovery feature, not a sufficient safety switch.

Do **not** keep subtracting FAIL from RHC. Next is P0-S (safety vs response-mode axes). Attack Gate stays closed.

## Official status

- Attack Gate: **closed**
- Next: `stop_attack_gate`
- Not claimed: unique low-rank safety subspace, attack eligibility, orthogonal-U pass, P1-M.
