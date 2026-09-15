# PathEM Stage 2 toy AMS — GO (no VLM)

Date: 2026-09-14. CPU only. Unique lineages, not clones. Not ASR.

## Process

Reflecting walk on `{0,...,H}` plus absorbing `FAIL`. From `H-1` a decoy leak
goes to FAIL so height `x/H` (refusal analog) is not the true committor.
True `q(t,x)` is exact DP. Token budget of naive/BoN = AMS-full tokens.

| | |
|---|---|
| H, T, p_up, p_decoy | 8, 40, 0.48, 0.75 |
| particles, levels, reps | 24, 5, 80 |
| seed | 20260 (OPT) |

## Unique-lineage means (token-matched)

| Method | unique_mean | tokens_mean | clone_hits_mean |
|---|---|---|---|
| naive | 1.55 | 1072.15 | 0 |
| BoN | 1.54 | 1072.15 | 0 |
| AMS-full (true q) | **2.90** | 1072.15 | 0.56 |
| AMS-refusal (height) | 1.34 | 830.43 | 0 |
| AMS-random | 2.13 | 1349.51 | 7.08 |

## Gates

| Check | Result |
|---|---|
| AMS-full > naive | yes |
| AMS-full > random | yes |
| AMS-full > refusal-height | yes |
| only beats random, not refusal | **false** |
| **Toy gate** | **GO** |

Toy GO = machinery works when the score is the true committor.
This does **not** authorize Stage 3 projection.

## VLM stop rule (not yet run)

If VLM particles beat random but **not** the Stage-1 `refusal_margin` scorer:
STOP PathEM (committor learned non-refusal). Do not claim ASR from SMC hits.

## Next

Token-budget-matched **VLM** particles on a few Stage-1 cells, M=8.
Not outer EM. Not eval seeds. Not sealed queries.
