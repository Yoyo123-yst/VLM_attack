# V-CachePoll P1 report

- Dataset: `coco300_standin` (COCO stand-in)
- Model: `Qwen2-VL-7B-Instruct`
- Attack: quota + eviction PGD on B only, $\epsilon=0.0627$, steps=40
- Decision: **CONTINUE**
- Reason: quota/eviction beat P0 random while full-token holds

| Metric | Value |
|---|---|
| finished / errors | 35 / 0 |
| mean \|Δr_A\| | 0.0561 |
| frac quota shift ≥ 0.02 | 0.886 |
| frac A-out (AVTP) | 1.000 |
| mean A-out count | 7.34 |
| P0 random mean A-out | 0.40 |
| frac B-in/A-out (AVTP) | 1.000 |
| frac B-in/A-out (global Top-K) | 1.000 |
| frac full-token still correct | 1.000 |
| frac compressed-only fail | 0.000 |
| isolated-quota still ok | 1.000 |
| restore-A still ok | 1.000 |

P1 is the first optimized attack on the 35 clean-correct pairs. It is not a paper table.

**What this means:** changing only B now stably steals A's quota and survivor slots. Full-token answers stay correct, so this is not LAMP-style semantic pollution.

**What it is not yet:** compressed-only task failure is still 0%. COCO VQA survives losing ~7 A tokens. Isolated/restore being 1.0 is expected while the compressed answer never breaks.

**Next:** keep the same 35 pairs, attack for task-critical eviction (`r_base=0.2` and/or leave-one-out U), not a new method. TextVQA waits for disk.
