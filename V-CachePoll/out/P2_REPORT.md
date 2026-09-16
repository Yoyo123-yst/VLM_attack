# V-CachePoll P2 report

- Dataset: `coco300_standin` (COCO stand-in)
- Model: `Qwen2-VL-7B-Instruct`
- \(r_{base}=0.2\), \(\epsilon=0.0627\), steps=40
- Decision: **CONTINUE**
- Reason: compressed-only task failure with full-token holding

| Metric | Value |
|---|---|
| finished / errors | 32 / 0 |
| localized critical tokens | 7 (frac=0.21875) |
| frac A-out | 1.000 |
| mean A-out / U-out | 7.41 / 5.59 |
| mean U eviction rate | 0.826 |
| frac full-token still correct | 0.969 |
| frac compressed-only fail | 0.219 |
| isolated still ok | 0.969 |
| restore-A still ok | 1.000 |
| restore recovers fail | 1.0 |

P2 asks whether a tighter shared budget plus targeting high-value A tokens produces compression-only task failure.

**What broke through:** 7/32 samples are wrong only under shared AVTP. Full-token stays correct (31/32). Restoring A's clean survivors recovers all 7 failures. Isolated per-image quotas also keep those 7 correct. That is Claim 3 on this stand-in: the error tracks shared-slot eviction, not LAMP-style semantic bleed.

**How U was chosen:** dropping low-score A tokens changed answers more often than dropping high-score ones (7 drop-bot vs 1 drop-top). 4 of the 7 compressed-only fails used drop-bot U.

**Next (paper-scale, not another method rewrite):** Random / Task-PGD / CAA-B-only / Rank-PGD at the same \(r_{base}=0.2\) budget; then TextVQA when disk allows. Do not start a new loss until those baselines exist.

