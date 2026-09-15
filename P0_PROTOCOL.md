# P0: Visual Causal-Controllability Test

Single-model gate. No text suffix, no cross-model alignment, no target-token CE as the main loss.

## Claim under test

On LLaVA-1.5-7B, does there exist a mid-to-late residual subspace \(U\) that is:

1. **Causal** — REF↔JB subspace patching changes relevant harmful compliance
2. **Controllable** — a shared \(\ell_\infty\) visual perturbation moves \(U^\top h\) toward \(\mu^{JB}\)
3. **Selective** — the same intervention drops matched benign utility by at most 5 points

## What P0 is not allowed to claim

- Cross-architecture transfer
- Text–image synergy
- Universal prompt jailbreak
- Black-box API attack

## Adaptations from the written protocol

- Model is LLaVA-1.5-7B, not Qwen2-VL-7B (only local white-box weights).
- Weights are 4-bit NF4; vision tower and projector stay in fp16 so pixel gradients exist.
- JB traces are sampled by a *per-sample* visual refusal-margin PGD with a relaxed budget. That sampler is not the method.
- Mini scale runs first. Full scale (60/60/8) is the same code with a larger config.

## Continue-to-P1 gates

All must hold on the holdout query split:

1. At least one residual \(U\) is found
2. Bidirectional patching works on holdout
3. Same-norm random directions do not reproduce the effect
4. Visual perturbation stably shifts \(U^\top h\) toward JB
5. Benign utility drop ≤ 5 points
6. Causal-subspace visual attack beats ordinary visual refusal-margin PGD on cross-query ASR
