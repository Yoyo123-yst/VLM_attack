# V-CachePoll P2 report

- Dataset: `coco300_standin` (COCO stand-in)
- Model: `Qwen2-VL-7B-Instruct`
- \(r_{base}=0.2\), \(\epsilon=0.0627\), steps=40
- Decision: **CONTINUE**
- Reason: compressed-only task failure with full-token holding

| Metric | Value |
|---|---|
| finished / errors | 17 / 0 |
| localized critical tokens | 5 (frac=0.29411764705882354) |
| frac A-out | 1.000 |
| mean A-out / U-out | 5.88 / 4.88 |
| mean U eviction rate | 0.731 |
| frac full-token still correct | 1.000 |
| frac compressed-only fail | 0.294 |
| isolated still ok | 0.941 |
| restore-A still ok | 1.000 |
| restore recovers fail | 1.0 |

P2 asks whether a tighter shared budget plus targeting high-value A tokens produces compression-only task failure.
