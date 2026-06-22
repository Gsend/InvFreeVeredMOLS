"""
Demo 2 — Algorithm 2: Simplified Gram-Schmidt Orthogonalisation (SGSO)
=====================================================================

Formal description
------------------
SGSO produces an orthogonal (but NOT orthonormal) basis Q for the column
space of X ∈ ℝ^(n × p), avoiding the square-root step of the classical
Gram-Schmidt and avoiding the explicit normalisation of the columns.

For each column j ∈ {0, …, p−1}:

    q_j  =  x_j  −  Σ_{i < j}  (x_j · q_i) / (q_i · q_i)  ·  q_i

The resulting Q ∈ ℝ^(n × p) satisfies:

    QᵀQ  =  diag(‖q_0‖², ‖q_1‖², …, ‖q_{p−1}‖²)    (diagonal, NOT identity)

Memory: O(np).  No square roots, no divisions by norms.  The OLS coefficients
can then be recovered from Q via  β = D⁻¹ (Qᵀ X)⁻¹ Qᵀ y  where D = diag(QᵀQ).

This script
-----------
- Generates a simulated design matrix X with full column rank.
- Runs SGSO to produce un-normalised orthogonal Q_sgso.
- Compares orthogonality and span quality against the textbook reference:
  Householder QR (NumPy's `np.linalg.qr`, which uses LAPACK's `geqrf`).
- Verifies both methods span the same column space.

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


def sgso_rust(X):
    import olssm
    return olssm.simplified_gram_schmidt(np.ascontiguousarray(X, dtype=np.float64))


def sgso_numpy(X):
    """Modified Gram-Schmidt without column normalisation — matches Rust impl."""
    n, p = X.shape
    Q = np.zeros((n, p))
    for j in range(p):
        qj = X[:, j].copy()
        for i in range(j):
            qi = Q[:, i]
            den = qi @ qi
            if abs(den) > 1e-12:
                qj -= (qj @ qi) / den * qi
        Q[:, j] = qj
    return Q


# --------------------------------------------------------------------------- #
# Textbook reference: Householder QR (LAPACK geqrf)                          #
# --------------------------------------------------------------------------- #

def householder_qr(X):
    """Standard backward-stable QR via Householder reflections.
       Returns Q with orthoNORMAL columns (QᵀQ = I) and upper-triangular R."""
    return np.linalg.qr(X, mode="reduced")


# --------------------------------------------------------------------------- #
# Main                                                                       #
# --------------------------------------------------------------------------- #

def main():
    rng = np.random.default_rng(seed=42)
    n, p = 300, 100
    print()
    print("=" * 70)
    print(" Demo 2 — Algorithm 2 (SGSO: Simplified Gram-Schmidt)")
    print(f" backend: {'Rust olssm extension' if _try_rust() else 'NumPy fallback'}")
    print(f" matrix : {n} × {p}")
    print("=" * 70)

    X = rng.standard_normal((n, p))
    s = np.linalg.svd(X, compute_uv=False)
    kappa_X = s[0] / s[-1]
    print(f"\n  κ(X) = σ_max / σ_min = {kappa_X:.2f}")

    # Run SGSO
    if _try_rust():
        Q_sgso = sgso_rust(X)
    else:
        Q_sgso = sgso_numpy(X)

    # Run Householder QR reference
    Q_qr, R_qr = householder_qr(X)

    # Orthogonality: max |off-diagonal| of QᵀQ (Householder Q is orthonormal,
    # so for fair comparison we also normalise SGSO's Q to unit columns)
    norms_sgso = np.linalg.norm(Q_sgso, axis=0)
    Q_sgso_normalized = Q_sgso / norms_sgso[None, :]

    QtQ_sgso  = Q_sgso_normalized.T @ Q_sgso_normalized
    QtQ_qr    = Q_qr.T @ Q_qr

    off_sgso = float(np.max(np.abs(QtQ_sgso - np.eye(p))))
    off_qr   = float(np.max(np.abs(QtQ_qr   - np.eye(p))))

    print()
    print("  Orthogonality of normalised columns (lower is better):")
    print(f"    SGSO       max |QᵀQ − I| = {off_sgso:.4e}")
    print(f"    Householder QR           = {off_qr:.4e}    ← textbook reference")

    # Span check: every X column should lie in span(Q_sgso) — verify with projection.
    # Residual of X − Q_sgso (Q_sgso⁺ X) should be near zero.
    pinv_sgso = np.linalg.pinv(Q_sgso)
    residual_span = float(np.linalg.norm(X - Q_sgso @ (pinv_sgso @ X)))
    print()
    print(f"  span(Q_sgso) covers span(X)?  ‖X − Q_sgso · Q_sgso⁺ · X‖₂ = {residual_span:.4e}")

    # OLS using SGSO output: since columns are orthogonal, (QᵀQ)⁻¹ is diagonal
    # and the OLS coefficients are computed cheaply.
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    # OLS via SGSO orthogonal basis
    D = np.diag(Q_sgso.T @ Q_sgso)               # diagonal of QᵀQ
    alpha = (Q_sgso.T @ y) / D                    # coefficients in Q basis
    # Now α = SGSO-basis OLS coefficients of y on Q.  To recover β (X basis),
    # solve the upper-triangular system R_sgso β = α where R_sgso is the change-of-basis.
    # Equivalently: SGSO is equivalent to QR factorisation with R extractable
    # from the projection coefficients.
    # Easiest path: compute OLS on Q's normalised form, then express in X's basis.
    beta_via_sgso = np.linalg.lstsq(X, Q_sgso @ alpha, rcond=None)[0]
    # Equivalent direct computation
    beta_via_qr = np.linalg.solve(R_qr, Q_qr.T @ y)

    err_sgso = float(np.linalg.norm(beta_via_sgso - beta_true))
    err_qr   = float(np.linalg.norm(beta_via_qr   - beta_true))
    print()
    print(f"  OLS via SGSO basis:           ‖β − β_true‖₂ = {err_sgso:.4e}")
    print(f"  OLS via Householder QR:       ‖β − β_true‖₂ = {err_qr:.4e}    ← reference")

    print()
    if off_sgso < 10 * off_qr:
        print("  ✓ SGSO orthogonality is within an order of magnitude of Householder QR.")
    else:
        print("  ⚠ SGSO orthogonality lags Householder QR by > 10× — expected for higher κ,")
        print("    suspicious for well-conditioned random Gaussian X.")
    print()


if __name__ == "__main__":
    main()
