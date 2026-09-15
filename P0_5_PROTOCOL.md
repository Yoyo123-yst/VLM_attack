# P0.5 Mechanism Robustness Audit

Audit the P0 causal finding. Not a new attack paper, not P1.

## Goal

Show the residual safety mediator is not an artifact of one prefix, one rank, or n=6.

## Model

Same LLaVA-1.5-7B used in P0. Qwen2-VL-7B is downloaded in parallel for later replication, not for this audit.

## A — template robustness

Keep model, images, queries. Only change the safety prefix.

| Setting | Discover | Test | Question |
|---|---|---|---|
| A→A | Prefix A | A | reproduce P0 |
| A→B | Prefix A | B | cross-template |
| B→B | Prefix B | B | independent under B |
| A+B→C | A and B jointly | C | more general state |

Pass if A→B or A+B→C still has stable bidirectional full-residual patching, random ≈ 0, benign not collapsed.

If that fails, the claim becomes template-conditioned safety routing, not a universal bottleneck.

## B — rank audit

On P0 holdout, layer 20 (P0 best), \(r\in\{1,2,4,8,16,32\}\) plus:

- same-norm random direction
- covariance-scale random low-rank subspace

Look for a knee. If only r≈d works, drop “bottleneck / low-dimensional”.

## C — per-sample visual entrance

n=24 queries that were **not** in the P0 n=6 signal set. Same ε=16/255, 40 steps, no EOT, zero init.

Compare refusal-margin PGD, U-guided PGD, random-subspace PGD.

Primary: whether \(\Delta U^\top h\) predicts relevant harmful compliance better than a random direction.
