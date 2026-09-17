# V-CachePoll P3 report

- Dataset: `coco300_standin`
- Model: `Qwen2-VL-7B-Instruct`
- \(r_{base}=0.2\), \(\epsilon=0.0627\), steps=40
- Decision: **CONTINUE**
- Reason: V-CachePoll compressed-only fail beats same-budget baselines; full-token holds

| Method | n | full-token hold | compressed-only fail | mean A-out | mean U-out | U-evict rate |
|---|---|---|---|---|---|---|
| random | 0 | — | — | — | — | — |
| task | 0 | — | — | — | — | — |
| caa | 0 | — | — | — | — | — |
| rank | 17 | 1.000 | 0.235 | 5.76 | 4.59 | 0.672 |
| vcache | 17 | 1.000 | 0.294 | 5.88 | 4.88 | 0.731 |

Same 32 P2 pairs, same \(\epsilon\) and step budget. `vcache` is the P2 V-CachePoll run.
Random uses 40 random restarts scored by exact compressor J. Rank-PGD is quota+evict PGD without exchange search.
