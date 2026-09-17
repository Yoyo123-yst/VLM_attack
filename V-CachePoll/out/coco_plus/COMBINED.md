# V-CachePoll small COCO increment

- updated: `2026-09-16T16:04:37+08:00`
- original pool: `32` (c000–c039 kept)
- new pool: `17` (c040–c059 after screen)
- combined: `49`

| Split | n | compressed-only fail | full-token hold | U-evict |
|---|---|---|---|---|
| V-CachePoll old | 32 | 0.219 (7) | 0.969 | 0.826 |
| V-CachePoll new | 17 | 0.294 (5) | 1.000 | 0.731 |
| V-CachePoll combined | 49 | 0.245 (12) | 0.980 | 0.793 |
| Rank-PGD old | 32 | 0.156 (5) | 0.969 | 0.833 |
| Rank-PGD new | 17 | 0.235 (4) | 1.000 | 0.672 |
| Rank-PGD combined | 49 | 0.184 (9) | 0.980 | 0.777 |

Protocol unchanged: `r_base=0.2`, `eps=16/255`, 40 steps, B-only.
New batch runs Rank only; Random/Task/CAA numbers stay on the original 32.
