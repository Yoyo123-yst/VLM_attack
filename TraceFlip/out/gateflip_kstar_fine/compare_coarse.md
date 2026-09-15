# Fine k* vs coarse 3-rung (k>1 cells)

Success is `core_rhc` only. Sealed queries h83–h130 are untouched.
GateFlip artefacts were merged without rewriting `pilot.json`.

Sources:
- `TraceFlip/out/gateflip_kstar_fine/probe_public.json`
- `ref:TraceFlip/out/gateflip_kstar/probe_public.json`

## Per-cell

| cell | TraceFlip | targeted-prefix | GateFlip | k* | ref k* | GF backward | ref backward | TP backward |
|---|---|---|---|---|---|---|---|---|
| h53:c07 | no | RHC | RHC | 2 (k2) | 11 (k_full) | 48 | 120 | 120 |
| h53:c08 | no | RHC | RHC | 3 (k3) | 3 (k_short) | 56 | 64 | 120 |
| h64:c08 | no | RHC | RHC | 4 (k4) | 3 (k_short) | 64 | 64 | 120 |
| h66:c07 | no | RHC | RHC | 1 (k1_token) | 3 (k_short) | 24 | 64 | 120 |
| h72:c08 | no | RHC | RHC | 7 (k7) | 3 (k_short) | 88 | 64 | 120 |

## Summary

- GateFlip cells: **5/5** RHC
- targeted-prefix: **5/5** RHC
- TraceFlip: **0/5** RHC
- GateFlip mean k* on wins: **3.40**  (histogram: {'2': 1, '3': 1, '4': 1, '1': 1, '7': 1})
- mean backward: GateFlip **56.0** vs targeted-prefix **120.0**
- McNemar GateFlip vs targeted-prefix: n10=0 n01=0 p=1.000 (pairs=5; n10 = GF win / TP miss)

## What changed vs the 3-rung ladder

| cell | coarse k* | fine k* | interpretation |
|---|---|---|---|
| h53:c07 | 11 / 120 bw | **2 / 48 bw** | skipped k=2 was the failure; not a long-prefix cell |
| h53:c08 | 3 / 64 | 3 / 56 | unchanged; k=3 won with only 8 extra steps |
| h66:c07 | 3 / 64 | **1 / 24** | k=1 landed this run (4-bit variance vs coarse miss) |
| h64:c08 | 3 / 64 | 4 / 64 | k=3 tried with **8** steps and missed; not a true increase |
| h72:c08 | 3 / 64 | 7 / 88 | same starvation; k=3..6 each got 8 steps |

Keep remains unused. Next knife: give every candidate k a full `INNER_STEPS` until RHC, then only rerun h64:c08 and h72:c08.
