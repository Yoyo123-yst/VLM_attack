# P0-Qwen closeout (terminated by gate)

```text
P0-Qwen-L24 status: terminated_by_gate
Supported: late residual causally controls response mode
Unsupported: safety-specific bottleneck
Attack gate: closed
Reason: refusal/denial directions reproduce the RHC effect
```

This file freezes the old P0-Qwen / P0-S line. Do not modify L24, requested rank 32, \(r_{\mathrm{eff}}=26\), historical four-way labels, or the old control directions to reopen that gate.

## What was supported

Late residual patching on recleaned holdout RHC is bidirectional at L16–L27. Frozen L24 full residual: REF→JB **0.883** \([0.75, 1.00]\), JB→REF **−0.767** \([−0.93, −0.60]\). The model has a causal residual state that changes whether it refuses / denies / answers with RHC.

## What was not supported

A unique low-rank **safety** subspace. Official L24 \(U_{\mathrm{RHC}}\) (requested \(r=32\), \(r_{\mathrm{eff}}=26\), \(n_{\mathrm{discover\_RHC}}=27\)) failed the cleaned-holdout gate (`benign_drop`, `fail_matches`). FAIL reclean items were 20/20 theme denial: safety success, task failure. Orthogonalizing FAIL out of \(U_{\mathrm{RHC}}\) collapsed REF→JB from 0.650 to 0.100.

P0-S (development holdout only): \(U_{\mathrm{safety}}=\mathrm{mean}(h^{\mathrm{RHC}}-h^{\mathrm{RELATED\_SAFE}})\) did **not** switch RHC (0.067, CI includes 0). \(U_{\mathrm{refusal}}\) (0.867) and \(U_{\mathrm{denial}}\) (0.717) reproduced the RHC effect more strongly than \(U_{\mathrm{RHC}}\) (0.650).

Lowering refusal is not “breaking a safety mechanism.”

## Frozen artifacts (do not retune)

| item | value |
|---|---|
| model | Qwen2-VL-7B-Instruct, native template, seed 2026 |
| pin layer | L24 |
| requested / effective rank | 32 / 26 |
| official next | terminated_by_gate |
| attack.json error | `l24_cleaned_holdout_gate_failed` |
| causal.attack_blocked | true |
| causal.best.pass_lowrank | false |

## Sealed IDs (old line)

- **h83–h90**: old P0-S mechanism-confirm; never traced; do not use for N0 development fitting
- **h91–h130**: old attack-test catalog; do not load or use their generation results for N0/N1/N2

## Forbidden on this line

- Old `--stage attack` / Attack Gate
- FAIL-orthogonalization of \(U_{\mathrm{RHC}}\)
- Reselecting L24 rank
- Using h91–h130 as a development set
- Calling “less refusal” a safety-mechanism break
- P0-T, P1-M, P1-A as continuations of this \(U\)

## Successor question

When the model **already answers**, is there a causal mechanism that chooses RELATED_SAFE vs RHC, and can image/text inputs steer it as a target-free multimodal attack? That work starts at **N0** (answer-conditioned dataset). It does not reuse the terminated L24 safety-bottleneck claim.
