# V-CachePoll P0 report

- Dataset: `coco300_standin` (COCO stand-in)
- Model: `Qwen2-VL-7B-Instruct`
- Decision: **CONTINUE**
- Reason: B changes A's quota or survivor set while full-token mostly holds

| Metric | Value |
|---|---|
| screen kept / rejected | 35 / 5 |
| probe n | 35 |
| mean \|Δr_A\| | 0.0035 |
| frac quota shift ≥ threshold | 0.000 |
| frac A-out (AVTP) | 0.371 |
| frac B-in/A-out swap (AVTP) | 0.371 |
| frac B-in/A-out swap (global Top-K) | 0.514 |
| frac full-token still correct on B_adv | 1.000 |
| frac AVTP compressed-only fail | 0.000 |
| causal n (A-out > 0) | 13 |
| isolated-quota recover | 1.0 |
| restore-A recover | 1.0 |

CONTINUE means: change B (noise only) moves A's shared-budget survivors, and the full-token model usually still answers A correctly.

This is a mechanism probe, not a paper table.
