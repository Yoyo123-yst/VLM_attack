# T0_DIAGNOSIS

- pass: **True**
- kind: engineering
- s_mode -112.8947 → -100.9669 (plus convention, answer-like)
- grad_ok: True  linf: 0.062745 <= 0.062745
- hidden_stable: True  weights_frozen: True
- reasons: none

## Engineering notes (not a science change)

First T0 attempt used plain sign-PGD at frozen α=1/255 and **decreased** s_mode (−112.9 → −130.3). Autograd of +⟨h,u⟩ is real and locally correct: a 1e-4 sign step raises s_mode (~+1.1), but the frozen 1/255 step overshoots the 4-bit L24 inner-product landscape (1e-3 already drops s_mode). First-token refusal-margin PGD at the same α remains well-behaved.

Fix kept frozen U, L24, ε, α, seed, and data: signed ascent on +s_mode with Armijo backtracking (halve the proposed 1/255 step until the score does not fall). Official backprop count is still one per PGD step. Diagnostic: `scripts/_otw_t0_grad_diag.py`.

Do not change layer, U, ε, or data if T0 fails again.
