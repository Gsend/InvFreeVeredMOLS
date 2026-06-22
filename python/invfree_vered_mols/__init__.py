"""
invfree_vered_mols — Inversion-Free OLS Solvers

Python re-export of the three Senderovich-Sandra algorithms implemented in
the Rust `olssm` crate.

Algorithms
----------
- `modified_cholesky(X, y)`             → C matrix (augmented Gram + LU + row-normalise)
- `back_substitute(C)`                  → β  (back-sub on C with β[p] = −1)
- `solve_ols(X, y)`                     → β  (Modified Cholesky + back-substitute combined)
- `simplified_gram_schmidt(X)`          → Q  (non-normalised orthogonal basis)
- `weighted_generalized_inverse(X, W)`  → G  ((XᵀWX)⁻¹ XᵀW via LU solve)

All arrays must be C-contiguous float64.

Usage
-----
    import numpy as np
    from invfree_vered_mols import solve_ols

    X = np.random.randn(300, 100)
    y = np.random.randn(300)
    beta = solve_ols(X, y)
"""
from __future__ import annotations

try:
    import olssm as _olssm
    _HAVE_RUST = True
except ImportError:
    _HAVE_RUST = False

__version__ = "0.1.0"


def _missing_rust(*_a, **_kw):
    raise ImportError(
        "The Rust `olssm` extension is not installed.  Build it with:\n"
        "    cd rust && maturin develop --features python --release\n"
        "or run the convenience script:\n"
        "    .\\scripts\\build_rust.ps1 -Release      (Windows)\n"
        "    ./scripts/build_rust.sh --release        (Linux/macOS)\n"
    )


if _HAVE_RUST:
    modified_cholesky          = _olssm.modified_cholesky
    back_substitute            = _olssm.back_substitute
    solve_ols                  = _olssm.solve_ols
    simplified_gram_schmidt    = _olssm.simplified_gram_schmidt
    weighted_generalized_inverse = _olssm.weighted_generalized_inverse
else:
    modified_cholesky            = _missing_rust
    back_substitute              = _missing_rust
    solve_ols                    = _missing_rust
    simplified_gram_schmidt      = _missing_rust
    weighted_generalized_inverse = _missing_rust


__all__ = [
    "modified_cholesky",
    "back_substitute",
    "solve_ols",
    "simplified_gram_schmidt",
    "weighted_generalized_inverse",
]
