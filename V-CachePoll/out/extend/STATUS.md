# V-CachePoll P4–P11 status

- updated: `2026-09-17T05:21:18+08:00`
- stage: **night-done**
- steward exit: 0

MSC finished overnight. GPU chain ran 05:00–05:21. Original `out/p2_*` not overwritten.

## GPU results

| Stage | n | Acc (no prune) | ASR (comp-only) | restore on ASR | δ_A=0 |
|---|---:|---:|---:|---:|---:|
| P4 freeze (full) | 32 | 100% | **25.0%** (was 21.9%) | 87.5% (7/8) | 100% |
| P5 smoke M_aux=1 | 2 | 100% | 0% | — | 100% |
| P6 smoke | 1 | 100% | 0% | — | 100% |
| P7 smoke family | 2 | 100% | 0% | — | 100% |
| P9 LAMP-like | 8 | 100% | 25.0% | 100% | 100% |

P4 8-sample pilot was 2/8 ASR before the 32-sample rerun. Drift gate passed (not a collapse vs 7/32).

## Still blocked

- P10 new models: weights missing
- P11 TextVQA / ChartQA: images missing
- P12 `L_amp`: skipped per doc
- P5/P6/P7 **full** attacks: only smokes ran (doc gate: 2-sample then scale)

## Next when you are back

P5 multi-aux (`n_aux` 2/4) beyond smoke; P6 budget sweep; P7 N>2; P9 table refresh vs LAMP. Need disk for TextVQA / Qwen3.
