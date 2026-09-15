"""CPU-only math tests for CRAA controllability (no model, no GPU).

Validates the two load-bearing claims behind the CRAA re-design:
  1. First-order projection equivalence: J^T P_col(J) u == J^T u, so
     "project-then-PGD" is redundant (why MRA is dropped).
  2. R_inf = ||J^T u||_1 / ||u||_2 is the exact first-order max displacement
     of h along u under an L-inf pixel budget: max_{||d||_inf<=eps} u^T J d = eps ||J^T u||_1.

Run: python tests/test_craa_cpu.py
"""

from __future__ import annotations

import numpy as np


def _proj_col(J: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto the column space of J (residual space)."""
    U, _, _ = np.linalg.svd(J, full_matrices=False)
    return U @ U.T


def test_projection_equivalence(rng: np.random.Generator, d: int, p: int, n_trial: int = 20) -> None:
    """J^T P_col(J) u must equal J^T u up to float error, for any u."""
    max_err = 0.0
    for _ in range(n_trial):
        J = rng.normal(size=(d, p))
        u = rng.normal(size=(d,))
        P = _proj_col(J)
        g_proj = J.T @ (P @ u)
        g_raw = J.T @ u
        err = float(np.max(np.abs(g_proj - g_raw)))
        max_err = max(max_err, err)
    assert max_err < 1e-10, f"projection equivalence violated, max_err={max_err:.3e}"
    print(f"[PASS] projection equivalence: J^T P_col(J) u == J^T u (max_err={max_err:.3e})")


def test_topk_not_equivalent(rng: np.random.Generator, d: int, p: int) -> None:
    """J^T P_k u must DIFFER from J^T u once low-singular directions are dropped."""
    J = rng.normal(size=(d, p))
    u = rng.normal(size=(d,))
    U, _, _ = np.linalg.svd(J, full_matrices=False)
    k = max(1, U.shape[1] // 2)
    Pk = U[:, :k] @ U[:, :k].T
    g_topk = J.T @ (Pk @ u)
    g_raw = J.T @ u
    diff = float(np.max(np.abs(g_topk - g_raw)))
    assert diff > 1e-6, "top-k projection unexpectedly identical to raw gradient"
    print(f"[PASS] top-k spectral truncation differs from raw gradient (diff={diff:.3e})")


def test_r_inf_formula(rng: np.random.Generator, d: int, p: int, n_trial: int = 20) -> None:
    """max_{||d||_inf<=eps} u^T J d == eps * ||J^T u||_1 (duality of L1 and L-inf)."""
    eps = 16.0 / 255.0
    worst_rel_err = 0.0
    for _ in range(n_trial):
        J = rng.normal(size=(d, p))
        u = rng.normal(size=(d,))
        u = u / np.linalg.norm(u)
        g = J.T @ u  # JVP, no explicit J needed in real code
        analytic = eps * np.linalg.norm(g, ord=1)
        # closed-form optimal d_i = eps * sign(g_i) (saturating L-inf)
        d_opt = eps * np.sign(g)
        numerical = float(u @ J @ d_opt)
        rel_err = abs(numerical - analytic) / (abs(analytic) + 1e-12)
        worst_rel_err = max(worst_rel_err, rel_err)
    assert worst_rel_err < 1e-8, f"R_inf duality failed, worst_rel_err={worst_rel_err:.3e}"
    print(f"[PASS] R_inf duality: max u^T J d = eps ||J^T u||_1 (worst_rel_err={worst_rel_err:.3e})")


def test_r_inf_sign_symmetry(rng: np.random.Generator, d: int, p: int, n_trial: int = 20) -> None:
    """Controllability is sign-blind (R(u)==R(-u)); causality is what carries the sign."""
    worst_rel_err = 0.0
    for _ in range(n_trial):
        J = rng.normal(size=(d, p))
        u = rng.normal(size=(d,))
        r_plus = np.linalg.norm(J.T @ u, ord=1)
        r_minus = np.linalg.norm(J.T @ (-u), ord=1)
        rel_err = abs(r_plus - r_minus) / (abs(r_plus) + 1e-12)
        worst_rel_err = max(worst_rel_err, rel_err)
    assert worst_rel_err < 1e-10, f"R_inf sign symmetry failed"
    print(f"[PASS] R_inf sign symmetry: R(u)==R(-u) (worst_rel_err={worst_rel_err:.3e})")


def test_low_gain_low_controllability(rng: np.random.Generator, d: int, p: int) -> None:
    """A direction living in a low-singular-value mode is hard to push: its R_inf is small
    even when its projection magnitude onto col(J) is 1 (i.e., it is 'reachable' but low-gain)."""
    # Build J with a controlled singular spectrum: one strong mode, rest tiny.
    U = np.linalg.qr(rng.normal(size=(d, d)))[0]
    V = np.linalg.qr(rng.normal(size=(p, p)))[0]
    s = np.concatenate([np.array([10.0]), np.full(min(d, p) - 1, 0.01)])
    S = np.zeros((d, p))
    S[: min(d, p), : min(d, p)] = np.diag(s)
    J = U @ S @ V.T
    u_strong = U[:, 0]          # aligned with high-gain mode
    u_weak = U[:, 1]            # aligned with low-gain mode, fully inside col(J)
    # Both are fully inside col(J) (projection magnitude 1), but controllability differs.
    proj_strong = np.linalg.norm(_proj_col(J) @ u_strong)
    proj_weak = np.linalg.norm(_proj_col(J) @ u_weak)
    assert abs(proj_strong - 1.0) < 1e-8 and abs(proj_weak - 1.0) < 1e-8
    r_strong = np.linalg.norm(J.T @ u_strong, ord=1) / np.linalg.norm(u_strong)
    r_weak = np.linalg.norm(J.T @ u_weak, ord=1) / np.linalg.norm(u_weak)
    assert r_strong > 10 * r_weak, f"gain should matter: r_strong={r_strong:.4f} r_weak={r_weak:.4f}"
    print(f"[PASS] gain matters: fully-reachable low-gain dir is hard to push "
          f"(R_strong={r_strong:.4f}, R_weak={r_weak:.4f})")


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(2026))
    d, p = 64, 256
    test_projection_equivalence(rng, d, p)
    test_topk_not_equivalent(rng, d, p)
    test_r_inf_formula(rng, d, p)
    test_r_inf_sign_symmetry(rng, d, p)
    test_low_gain_low_controllability(rng, d, p)
    print("\nAll CRAA CPU math tests passed.")


if __name__ == "__main__":
    main()
