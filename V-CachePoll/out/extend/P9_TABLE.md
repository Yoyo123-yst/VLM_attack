# P9 ordinary multi-image vs compression-resource attack

| Method | n | Full Acc | Shared CASR | Isolated still-ok | Restore-A on COF |
|---|---:|---:|---:|---:|---:|
| V-CachePoll | 32 | 0.969 | 0.219 | 0.969 | 1.000 |
| Task-PGD | 32 | 0.969 | 0.000 | 0.938 | None |
| CAGE-B | 32 | 0.969 | 0.125 | 0.969 | 1.000 |
| CAA-B | 32 | 1.000 | 0.062 | 1.000 | 1.000 |
| Rank-PGD | 32 | 0.969 | 0.156 | 0.969 | 1.000 |
| Random | 32 | 1.000 | 0.062 | 0.969 | 0.000 |
| LAMP-like | 0 | — | — | — | — |

V-CachePoll should show shared CASR ≫ isolated failure (isolated still-ok high) and restore-A recovery.
Ordinary Task-PGD / LAMP-like should not show that pattern.
LAMP-like GPU run writes `out/p9_lamp.json` when the GPU is free.
