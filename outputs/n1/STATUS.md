# Pipeline status after Fast-Crossed Pilot

N0 behavior labels and the 51 answering pairs remain valid for **output-behavior** analysis.
They are **not** valid for learning a pre-generation safety state.

Fast-Crossed (route B, 96 greedy 16/255 restarts on h01/h07/h11/h21/h31/h41 × c05/c06) **stopped at the pair gate**. Report: `outputs/n1r_fast/FAST_CROSSED_PILOT.md`.

| Stage | Status |
|---|---|
| N0 label schema | pass |
| N0 behavior data | usable for output-behavior analysis |
| N0 pre-generation representations | fail (identical last-prompt inputs on 40/51 pairs) |
| Fast-Crossed same-budget pairs | fail (gray): 4 pairs, query **h11 only**, need ≥12 / ≥4 queries |
| N1 residual scan | blocked |
| N2 causal patching | blocked |
| N3 confirm | blocked; do not backfill confirm |
| Old Attack Gate | closed |
| Next (not started) | ModeSwitch: attack the refusal **mode** bit, not \(U_{\mathrm{safety}}\) |

Gray result: same PGD budget and random restarts mostly switch **ANSWER vs REFUSE**. Both RHC and RELATED_SAFE under one cell appeared only for fraud query h11. Do not scan layers. Do not invent SAFE with a different PGD objective. Do not reopen L24 rank / sealed h91–h130.

Route A (post-generation content state) stays diagnostic only.
