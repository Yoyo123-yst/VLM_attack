# CG-VSF G0

- ok: **True**
- frozen sha256: `39be18f61704c4b70b134bb8a5ad64b8ac1036b2858e0e3810c5c7ed4773a369`
- judge source sha256: `7746a0c4464bb9872813789f359f427a6b2ffddb70a28fe42faa259c53dec25b`

## Decoder alignment

At delta=0 the solver forward and greedy argmax agree. After a constrained flip at t=5 on h49:c07, teacher-forced/incremental solver logits can disagree with a full greedy re-decode (solver argmax 438 vs redecode 358). TraceFlip's 12-cell pilot still has in-model-feasible-but-invalid = 0.000 because accept never uses in-model feasibility. CG-VSF P0 accept is official greedy execute + four-axis core_rhc only. In-model certificate scores are the optimisation proxy, not the success label.

## Problems

- none
