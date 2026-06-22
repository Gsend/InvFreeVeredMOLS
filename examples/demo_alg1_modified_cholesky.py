"""
Demo 1 — Algorithm 1: Modified Cholesky (closed-form OLS without inversion)
==========================================================================

Formal description
------------------
Given a design matrix X ∈ ℝ^(n × p) and response y ∈ ℝⁿ, the Algorithm 1
recipe for the OLS coefficient vector β = (XᵀX)⁻¹ Xᵀy is:

    1.  Augment X with y as an additional column:    A = [X | y] ∈ ℝ^(n × (p+1))
    2.  Form the augmented Gram matrix:               M = AᵀA ∈ ℝ^((p+1) × (p+1))
    3.  LU-decompose M; let U be its upper-triangular factor
    4.  Row-normalise U so its diagonal becomes 1:    C[i, j] = U[i, j] / U[i, i]
    5.  Back-substitute by solving  C [β; −1]ᵀ = 0   starting from β[p] = −1

Memory: O((p+1)²).  No matrix inversion is computed at any point.

This script
-----------
- Generates a simulated linear system  y = X β_true + ε  with Gaussian noise.
- Solves it using Algorithm 1 (via the Rust binding if installed; NumPy fallback otherwise).
- Solves the *same* system using the textbook reference: Householder QR
  (the standard backward-stable OLS routine, Higham 2002 §19 / Trefethen-Bau Lecture 19).
- Reports coefficient-recovery error vs. β_true and the algorithm-vs-reference
  divergence ‖β_alg1 − β_qr‖₂.

System size: n = 300 samples × p = 100 predictors.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Algorithm dispatch — prefer Rust binding, fall back to NumPy reference      #
# --------------------------------------------------------------------------- #

def _try_rust():
    try:
        import olssm  # noqa: F401
        return True
    except ImportError:
        return False


def alg1_modified_cholesky_rust(X, y):
    import olssm
    return olssm.solve_ols(
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
    )


def alg1_modified_cholesky_numpy(X, y):
    """Pure-NumPy reference implementing the same recipe."""
    A = np.column_stack([X, y])
    M = A.T @ A
    # Cholesky on M (SPD when X has full column rank) — equivalent U via Lᵀ
    try:
        L = np.linalg.cholesky(M)
        U = L.T
    except np.linalg.LinAlgError:
        # Exact-fit case: M is rank-deficient.  Fall back to symbolic LU.
        from scipy.linalg import lu
        _, _, U = lu(M)
    # Row-normalise so diag(C) = 1
    C = U.copy()
    for i in range(C.shape[0]):
        d = C[i, i]
        if abs(d) < 1e-12:
            C[i, i] = 1.0
            continue
        C[i, i:] /= d
    # Back-substitute with betas[p] = -1
    dim = C.shape[0]
    betas = np.zeros(dim)
    betas[-1] = -1.0
    for i in range(dim - 2, -1, -1):
        betas[i] = -(C[i, :] * betas).sum()
    return betas[:dim - 1]


# --------------------------------------------------------------------------- #
# Textbook reference: Householder QR-based OLS                               #
# --------------------------------------------------------------------------- #

def householder_qr_ols(X, y):
    """Standard backward-stable OLS via Householder QR.
       β  =  R⁻¹ Qᵀ y     where X = QR (Householder reflections)."""
    Q, R = np.linalg.qr(X, mode="reduced")
    return np.linalg.solve(R, Q.T @ y)


# --------------------------------------------------------------------------- #
# Main                                                                       #
# --------------------------------------------------------------------------- #

def main():
    rng = np.random.default_rng(seed=42)
    n, p = 300, 100
    print()
    print("=" * 70)
    print(" Demo 1 — Algorithm 1 (Modified Cholesky)")
    print(f" backend: {'Rust olssm extension' if _try_rust() else 'NumPy fallback'}")
    print(f" system : n = {n} samples × p = {p} predictors")
    print("=" * 70)

    # Build a controlled overdetermined system
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    sigma_noise = 0.1
    y_clean = X @ beta_true
    y = y_clean + sigma_noise * rng.standard_normal(n)

    # Condition number (well-conditioned random Gaussian for fair comparison)
    s = np.linalg.svd(X, compute_uv=False)
    kappa_X = s[0] / s[-1]
    print(f"\n  κ(X) = σ_max / σ_min = {kappa_X:.2f}")
    print(f"  Noise σ = {sigma_noise}\n")

    # Run Algorithm 1
    if _try_rust():
        beta_alg1 = alg1_modified_cholesky_rust(X, y)
    else:
        beta_alg1 = alg1_modified_cholesky_numpy(X, y)

    # Run Householder QR reference
    beta_qr = householder_qr_ols(X, y)

    # Reporting
    err_alg1 = float(np.linalg.norm(beta_alg1 - beta_true))
    err_qr   = float(np.linalg.norm(beta_qr   - beta_true))
    div_pair = float(np.linalg.norm(beta_alg1 - beta_qr))
    res_alg1 = float(np.linalg.norm(X @ beta_alg1 - y))
    res_qr   = float(np.linalg.norm(X @ beta_qr   - y))

    print(f"  ‖β_alg1 − β_true‖₂      = {err_alg1:.4e}")
    print(f"  ‖β_QR   − β_true‖₂      = {err_qr:.4e}    ← textbook reference")
    print(f"  ‖β_alg1 − β_QR‖₂        = {div_pair:.4e}    ← algorithm divergence from reference")
    print()
    print(f"  Residual ‖X β_alg1 − y‖₂ = {res_alg1:.4e}")
    print(f"  Residual ‖X β_QR   − y‖₂ = {res_qr:.4e}")

    # Verdict
    print()
    if div_pair < 1e-8 * np.linalg.norm(beta_true):
        print("  ✓ Algorithm 1 matches Householder QR to machine precision.")
    elif div_pair < 1e-4 * np.linalg.norm(beta_true):
        print("  ✓ Algorithm 1 matches Householder QR to expected normal-equations precision")
        print(f"    (~ κ(X)² · ε_fp64 ≈ {kappa_X**2 * 2.2e-16:.2e}).")
    else:
        print("  ⚠ Algorithm 1 diverges from Householder QR by more than expected — investigate.")
    print()


if __name__ == "__main__":
    main()
