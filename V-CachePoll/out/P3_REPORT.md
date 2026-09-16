# V-CachePoll P3 report

- Dataset: `coco300_standin`
- Model: `Qwen2-VL-7B-Instruct`
- \(r_{base}=0.2\), \(\epsilon=0.0627\), steps=40
- Decision: **CONTINUE**
- Reason: V-CachePoll compressed-only fail beats same-budget baselines; full-token holds

| Method | n | full-token hold | compressed-only fail | mean A-out | mean U-out | U-evict rate |
|---|---|---|---|---|---|---|
| random | 32 | 1.000 | 0.062 | 0.91 | 0.91 | 0.138 |
| task | 32 | 0.969 | 0.000 | 0.47 | 0.47 | 0.069 |
| caa | 32 | 1.000 | 0.062 | 3.09 | 2.91 | 0.430 |
| cage | 32 | 0.969 | 0.125 | 3.69 | 3.47 | 0.518 |
| rank | 32 | 0.969 | 0.156 | 6.69 | 5.66 | 0.833 |
| armijo | 32 | 0.969 | 0.156 | 5.94 | 4.97 | 0.733 |
| vcache | 32 | 0.969 | 0.219 | 7.41 | 5.59 | 0.826 |

Same 32 P2 pairs, same \(\epsilon\) and step budget. `vcache` is the P2 V-CachePoll run.
Random uses 40 random restarts scored by exact compressor J. Rank-PGD is quota+evict PGD without exchange search. `armijo` is Rank-PGD with exchange search every step.

## Reading

- Task-PGD never produced a compression-only fail (0/32). Ordinary output attack is not this failure mode.
- Random and CAA-B-only each have 2/32 compressed-only fails. CAGE-B-only (push dropped B tokens into the survivor set) gets 4/32 and U-evict 0.52 — above CAA, well below V-CachePoll. Survivor disruption is not victim eviction.
- Rank-PGD already matches V-CachePoll on U-evict (0.833 vs 0.826). Finite-scale search plus the threshold term only adds two compressed-only fails (5 to 7). Search is an implementation detail, not the claim.
- Armijo-style Rank-PGD (exchange search every step) does **not** beat Rank: same 5/32 compressed-only fails, slightly *lower* A-out/U-evict. Extra search frequency is not the gain.
- V-CachePoll still has the highest compressed-only ASR (7/32 = 21.9%) with full-token hold 96.9%.

## FastV / LLaVA (overnight add-on)

- FastV last-token attention at the same K keeps ~48 A vs ~4 B tokens on this prompt; both FastV and AVTP answers stayed correct on 8/8. Query attention is already source-biased; AVTP shared quotas are the more natural pollution surface.
- LLaVA-1.5 two-image pack works. 6/8 pairs are clean-correct under AVTP at \(r_{base}=0.2\). Quota+evict PGD steals slots (20 steps: A-out 9.7 / U-evict 0.36; 40 steps: A-out 13.0 / U-evict 0.48) with full-token hold 1.0, but 0/6 compressed-only fails. Mechanism transfers; task-fail does not on this 576-token CLIP grid.


