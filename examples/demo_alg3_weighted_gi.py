"""
Demo 3 — Algorithm 3: Weighted Generalised Inverse  G = (XᵀWX)⁻¹ Xᵀ W
======================================================================

Formal description
------------------
The weighted generalised inverse of a design matrix X ∈ ℝ^(n × p) under a
symmetric positive-definite weight matrix W ∈ ℝ^(n × n) is:

    G  =  (XᵀWX)⁻¹ Xᵀ W   ∈ ℝ^(p × n)

For a noisy response  y = X β_true + ε  with  Cov(ε) = σ² · W⁻¹  (i.e., known
heteroscedastic noise covariance), the Best Linear Unbiased Estimator (BLUE)
of β by Gauss-Markov is:

    β_GLS  =  G · y  =  (XᵀWX)⁻¹ Xᵀ W y

When W = I, this reduces to ordinary OLS:  β_OLS = (XᵀX)⁻¹ Xᵀ y.

The "no inversion" property: Algorithm 3 computes G via the *LU solve*
    solve(XᵀWX, XᵀW)
so the inverse (XᵀWX)⁻¹ is never formed explicitly — only its action on
the right-hand side is computed.

This script
-----------
- Generates a system with heteroscedastic noise: ε_i ~ N(0, σ_i²) with
  varying σ_i across samples.
- Uses W = diag(1/σ_i²) — the optimal Gauss-Markov weighting.
- Solves via Algorithm 3.
- Compares against the textbook reference: whitening + Householder QR
  (the standard generalised-least-squares pipeline).
- Verifies the BLUE property: weighted estimator has lower variance than
  unweighted OLS in the heteroscedastic regime.

System size: n = 300 samples × p = 100 predictors.
"""
from __future__ import annotations
import numpy as np


def _try_rust():
    try:
        import olssm  # noqa: F401
        return True
    except ImportError:
        return False


def wgi_rust(X, W):
    import olssm
    return olssm.weighted_generalized_inverse(
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(W, dtype=np.float64),
    )


def wgi_numpy(X, W):
    xtw = X.T @ W
    xtwx = xtw @ X
    return np.linalg.solve(xtwx, xtw)


# --------------------------------------------------------------------------- #
# Textbook reference: whitened Householder QR (generalised-least-squares)    #
# --------------------------------------------------------------------------- #

def whitened_qr_gls(X, W, y):
    """Textbook GLS:  apply W^(1/2) to both sides, then standard QR-OLS.
       Mathematically identical to (XᵀWX)⁻¹ XᵀW y."""
    # W is diagonal in this demo, so W^(1/2) = sqrt(W)
    sqrt_W_diag = np.sqrt(np.diag(W))
    Xw = sqrt_W_diag[:, None] * X
    yw = sqrt_W_diag * y
    Q, R = np.linalg.qr(Xw, mode="reduced")
    return np.linalg.solve(R, Q.T @ yw)


# --------------------------------------------------------------------------- #
# Main                                                                       #
# --------------------------------------------------------------------------- #

def main():
    rng = np.random.default_rng(seed=42)
    n, p = 300, 100
    print()
    print("=" * 70)
    print(" Demo 3 — Algorithm 3 (Weighted Generalised Inverse)")
    print(f" backend: {'Rust olssm extension' if _try_rust() else 'NumPy fallback'}")
    print(f" system : n = {n} samples × p = {p} predictors")
    print("=" * 70)

    # Build a design matrix and a known coefficient vector
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)

    # Heteroscedastic noise: each sample i has its own noise std σ_i.
    # We choose σ_i to span a factor of 10 — so the optimal weighting really
    # matters (uniform weighting would be visibly suboptimal).
    sigma_per_sample = np.linspace(0.05, 0.5, n)
    rng_noise = rng.standard_normal(n)
    epsilon = sigma_per_sample * rng_noise
    y = X @ beta_true + epsilon

    # Optimal Gauss-Markov weights: W = diag(1/σ_i²)
    W = np.diag(1.0 / sigma_per_sample**2)
    # And the unit weight matrix, for comparison
    W_identity = np.eye(n)

    print()
    print(f"  Noise std range: σ_i ∈ [{sigma_per_sample.min():.3f}, {sigma_per_sample.max():.3f}]")
    print(f"  Heteroscedasticity ratio: σ_max / σ_min = {sigma_per_sample.max() / sigma_per_sample.min():.1f}")

    # ---- Algorithm 3 with optimal weights ----
    if _try_rust():
        G_weighted = wgi_rust(X, W)
        G_identity = wgi_rust(X, W_identity)
    else:
        G_weighted = wgi_numpy(X, W)
        G_identity = wgi_numpy(X, W_identity)

    beta_alg3_weighted = G_weighted @ y      # optimal weighted
    beta_alg3_uniform  = G_identity @ y       # unweighted OLS (for comparison)

    # ---- Textbook reference: whitened QR ----
    beta_qr_weighted = whitened_qr_gls(X, W, y)

    # Reporting
    print()
    print("  Coefficient-recovery errors  ‖β − β_true‖₂  (lower is better):")
    err_alg3_w = float(np.linalg.norm(beta_alg3_weighted - beta_true))
    err_qr_w   = float(np.linalg.norm(beta_qr_weighted   - beta_true))
    err_alg3_u = float(np.linalg.norm(beta_alg3_uniform  - beta_true))
    print(f"    Algorithm 3 (W=optimal):       {err_alg3_w:.4e}")
    print(f"    Householder QR (whitened):     {err_qr_w:.4e}    ← textbook reference")
    print(f"    Algorithm 3 (W=I, plain OLS):  {err_alg3_u:.4e}    ← naive baseline, no weighting")

    # Reference vs. algorithm divergence
    div = float(np.linalg.norm(beta_alg3_weighted - beta_qr_weighted))
    print()
    print(f"  ‖β_alg3 − β_QR‖₂  =  {div:.4e}    ← algorithm-vs-reference divergence")

    print()
    if div < 1e-6 * np.linalg.norm(beta_true):
        print("  ✓ Algorithm 3 matches whitened Householder QR to high precision.")
    else:
        print("  ⚠ Algorithm 3 diverges from whitened QR more than expected — investigate.")
    if err_alg3_w < err_alg3_u:
        print(f"  ✓ Optimal weighting beats uniform by {err_alg3_u / err_alg3_w:.2f}×")
        print(f"    (expected: BLUE property of GLS under known heteroscedasticity).")
    else:
        print("  ⚠ Optimal weighting did NOT improve over uniform — investigate.")
    print()


if __name__ == "__main__":
    main()
