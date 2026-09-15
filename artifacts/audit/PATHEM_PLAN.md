# PathEM plan (relaxed gates)

Attack paper, not a mechanism paper. SMC is an optimizer. Final metric is
standard-decoder core_RHC / ASR on frozen judge.

## Mapping

```
E-step  Q(τ) ≈ P_δ(τ | τ ∈ B_H)   →  prefix committor + AMS splitting
M-step  max_δ E_Q log P_δ(τ)      →  reuse write_phase.prefix_token_logp on sampled RHC τ
Eval    y ~ p_standard(y | q, x+δ) with fresh seeds
```

Reuse: QwenP0, four_axes, pgd_restart (warm start), clip_delta, CR-0 deltas.
Do not reuse: OtW harmful-prefix banks, U_refusal as objective, MCCA content losses.

## Relaxed vs original prompt

| Original | Relaxed |
|---|---|
| Stage 0 requires GPU multi-seed generate | CPU infra + CR-0 delta box; GPU generate is Stage 1 collect |
| AUROC ≥ 0.70 hard | better than random + prevalence Brier; 0.70 advisory |
| 3 VLMs / 2 benches in the first loop | locked until Stage 3 |
| held-out eval populated now | empty until Stage 3 |
| pytest required | stdlib test loader |
| LPIPS/SSIM | skip (not in repo) |
| large particle M | M=8, 1 outer iter |

Not relaxed: sealed IDs, no raw logs, disjoint seeds, standard decoder for success.

## Stages

### Stage 0 — infra (this ticket)
CPU tests + CR-0 index. **GO** on 10/10 tests.

### Stage 1 — committor feasibility (needs GPU auth)
Collect stochastic trajectories on CR-0 deltas for train queries h49/h56/h66
and val h64/h72. N=4 opt seeds per cell is the floor; raise if RHC count < ~20
positives after collect.

Prefix labels = **full-trajectory terminal core_rhc**, not prefix judge.

Features, in order: last-layer prefix hidden if cheap; else first-token logits
+ refusal margin + length. Logistic / tiny MLP only.

Hard GO: AUROC > 0.5 on query-disjoint val **and** Brier < prevalence baseline
**and** enrichment not only at EOS.
If hidden-state committor ≤ refusal-margin scorer: try calibration, then STOP PathEM.

### Stage 2 — splitting (token-budget matched)
Toy AMS first (no VLM). Then VLM particles vs naive / BoN / random-score /
refusal-score. Success = unique lineages, not clones. If only beats random but
not refusal-score: STOP (committor learned non-refusal).

### Stage 3 — visual projection (decisive)
M-step on weighted sampled RHC τ. Then **drop SMC**, restore greedy+standard
sample, **evaluation seeds only**.
GO iff standard-decoder core_RHC rises vs warm-start and vs budget-matched BoN,
and the rise is not only refusal-rate drop.
If SMC-internal success ↑ but standard decoder flat: STOP (“paths found, image
does not control them”). That is not an attack paper.

### Stage 4 — outer EM
Only if 1–3 GO. One then few outer iterations. Early stop on val seeds.

### Stage 5
Locked.

## Files to add later (not now)

- `src/pathem/committor.py`, `scripts/run_pathem_stage1.py`
- `src/pathem/ams_toy.py`, `src/pathem/particles.py`
- `src/pathem/mstep.py` wrapping `prefix_token_logp`
- `src/pathem/eval_standard.py`
- `tests/test_pathem_ams_toy.py`

## Compute (pilot)

| Stage | GPU | Notes |
|---|---|---|
| 0 | 0 | done |
| 1 collect | ~4–8 h | 5 queries × 2 carriers × ~4 deltas × 4 seeds × 96 tok |
| 1 fit | CPU minutes | logistic |
| 2 | ~4 h | M=8, few cells |
| 3 | ~4–6 h | 8 projection steps + fresh-seed eval |
| 4 | do not estimate until 3 GO | |

## Risks

1. CR-0 labels are greedy; stochastic RHC rate may collapse (h53 already 0/16 greedy).
2. Prefix hidden during generate is not implemented; last-user hidden is the wrong object.
3. 4-bit + teacher-forced long τ may OOM; cap projection length (e.g. 32–48 tokens).
4. Recent LM rare-event SMC papers: novelty must be **image projection + standard decoder**, not “we used SMC”.
5. No git: config hash will be dirty-workspace metadata only.

## Stage 0 / 1 next

Stage 0 is GO for engineering. Do **not** auto-start Stage 1 GPU collect.
Next authorized action: Stage 1 data-collection script + tiny generate smoke
on 1 query × 1 carrier × 2 seeds, `store_raw_outputs: false`.
