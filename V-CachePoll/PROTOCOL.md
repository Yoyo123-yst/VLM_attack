# V-CachePoll P0 — COCO stand-in

Frozen 2026-09-15. This is a mechanism probe, not a paper result.

## Claim under test

On Qwen2-VL-7B with an AVTP-style shared visual budget, changing only auxiliary image B can change the keep quota and survivor set of unmodified main image A.

## Current data

TextVQA is not on this machine. P0 uses COCO val2017:

- A: coco300 images with existing VQA questions
- B: a different val2017 image, unused as A
- Question refers to the first image only

Later datasets (TextVQA, POPE, ChartQA, MuirBench) plug into the same pair schema.

## Model / compressor

- Model: local `Qwen2-VL-7B-Instruct` (AVTP paper used Qwen3-VL-8B; substitution is recorded)
- Importance: hidden-state variation at LLM layers 1, 14, 19 (AVTP scoring, no attention)
- `avtp`: \(r_i=r_{base}+\alpha(\bar I_i-\bar I_{avg})\), then within-image Top-K
- `global_topk`: one shared Top-K over all visual tokens (cross-image ranking)
- Pruning **drops** unselected visual tokens from the prefix, then calls official `generate`. Compacted images get a rectangular `image_grid_thw` that matches the kept count.

## Stages

1. `pairs` — write A/B JSON (CPU)
2. `smoke` — load model, 2 samples, dump \(\bar I_i, r_i\), survivors, answers
3. `screen` — keep clean-correct: A-only has content, and both full-token A+B and AVTP A+B agree with that A-only answer (not csv `source`, which is often the wrong noun)
4. `probe` — \(\ell_\infty\) noise \(\epsilon=8/255\) on B only; measure \(\Delta r\), A-out/B-in, full-token vs compressed
5. `report` — continue / stop

## Continue

- B changes A's quota or survivor set on a stable fraction of samples
- legal noise produces B-in / A-out
- full-token answers stay mostly correct

## P1 — quota poisoning + eviction set

On the 35 clean-correct pairs, optimize only B.

- Losses: \(L_{quota}=r_A-r_B\), token-level \(L_{evict}\), full-token \(L_{value}\), TV
- \(\epsilon=16/255\), \(\alpha=1/255\), 40 signed PGD steps, finite-scale exchange search every 5 steps
- Success vs P0 random: larger quota shift or more A-out, with full-token still mostly correct

```bash
python tests/test_p1_cpu.py
python scripts/run_p1.py --stage smoke
python scripts/run_p1.py --stage attack
python scripts/run_p1.py --stage report
```

## P2 — tight budget + critical tokens

Same 35 pairs. \(r_{base}=0.2\). Identify A tokens whose removal changes the answer, then evict those.

```bash
python tests/test_p1_cpu.py
python scripts/run_p2.py --stage screen
python scripts/run_p2.py --stage crit
python scripts/run_p2.py --stage smoke
python scripts/run_p2.py --stage attack
python scripts/run_p2.py --stage report
```

Live status: `out/p2_progress.json`.

## P3 — same-protocol baselines

Same 32 P2 pairs, \(r_{base}=0.2\), \(\epsilon=16/255\), 40 steps.

Order: Random → Task-PGD → CAA-B-only → Rank-PGD. V-CachePoll is the existing `out/p2_attack.json`.

```bash
python tests/test_p1_cpu.py
python scripts/run_p3.py --stage all
python scripts/run_p3.py --stage report
```

Live status: `out/p3_random_progress.json`, `out/p3_task_progress.json`, `out/p3_caa_progress.json`, `out/p3_rank_progress.json`, `out/NIGHT_STATUS.md`.


