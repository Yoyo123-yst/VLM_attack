# V-CachePoll night status

Updated: 2026-09-16 03:12 +08. Overnight jobs finished. Decision: **CONTINUE**.

Primary report: `out/P3_REPORT.md`

## P3 (main)

Same 32 P2 pairs, Qwen2-VL-7B, \(r_{base}=0.2\), \(\epsilon=16/255\), 40 steps.

| Method | compressed-only fail | full-token hold | mean A-out | U-evict |
|---|---|---|---|---|
| Random | 2/32 | 1.000 | 0.91 | 0.138 |
| Task-PGD | 0/32 | 0.969 | 0.47 | 0.069 |
| CAA-B-only | 2/32 | 1.000 | 3.09 | 0.430 |
| CAGE-B-only | 4/32 | 0.969 | 3.69 | 0.518 |
| Rank-PGD | 5/32 | 0.969 | 6.69 | 0.833 |
| Armijo (search every step) | 5/32 | 0.969 | 5.94 | 0.733 |
| V-CachePoll | **7/32** | 0.969 | 7.41 | 0.826 |

Pass condition met: V-CachePoll compressed-only ASR is highest, full-token stays correct.

Paper-facing reading:

- Do not add another jailbreak loss.
- Do not sell finite-scale search as novelty. Armijo/search-every-step does not beat Rank. The extra 2 fails vs Rank come from threshold targeting (`L_crit`), not step-size search.
- Rank already matches U-evict. The method claim is quota poisoning + directed eviction, not adaptive steps.

## FastV

8 pairs. Last-token attention Top-K at the same K keeps ~48 A / ~4 B tokens. Answers stayed correct. Query-attention is already source-biased; AVTP shared quotas are the pollution surface.

## LLaVA-1.5-7B

Two-image pack works. 6/8 clean-correct under AVTP. 40-step quota+evict: A-out 13, U-evict 0.48, full-token 6/6, compressed-only fail 0/6. Slot theft transfers; task-fail does not on the 576-token CLIP grid. Next transfer target should be LLaVA-OV / InternVL, not more LLaVA-1.5 PGD.

## Not done (by design)

TextVQA, ChartQA, InternVL, MuirBench, LAMP. No new dataset until disk is added.

## Files

- `out/P3_REPORT.md`
- `out/p3_{random,task,caa,cage,rank,armijo}.json`
- `out/p2_attack.json` (V-CachePoll)
- `out/fastv_probe.json`
- `out/llava_capability.json`, `out/llava_probe.json`, `out/llava_attack.json`
