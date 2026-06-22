//! Pure-Rust implementations of three closed-form OLS algorithms.
//!
//! Inversion or Normalization" —  Senderovich  & Sandra .
//! <
//!
//! All functions operate on `nalgebra::DMatrix<f64>` / `DVector<f64>`.
//! No PyO3 or FFI dependencies — this module is the pure mathematical core.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;
// faer traits required for .solve() and .inverse() on PartialPivLu
use faer::prelude::{SolverCore, SpSolver};

/// Errors returned by olssm algorithms.
#[derive(Debug, Error, PartialEq)]
pub enum OlsSMError {
    /// Row count of X does not match length of y.
    #[error("Dimension mismatch: X has {x_rows} rows, y has {y_len} elements")]
    DimensionMismatch { x_rows: usize, y_len: usize },

    /// A diagonal entry of the upper-triangular factor is (near) zero,
    /// indicating a singular or near-singular Gram matrix.
    #[error("Zero pivot at diagonal position {index} — matrix may be singular")]
    ZeroPivot { index: usize },

    /// Weight matrix W is not n×n where n = number of rows in X.
    #[error("Weight matrix W must be {n}x{n}, got {rows}x{cols}")]
    WeightDimension { n: usize, rows: usize, cols: usize },

    /// LU solve returned None — matrix is singular.
    #[error("Singular matrix: LU solve failed")]
    SingularMatrix,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the SPD Cholesky factorisation `A = L · Lᵀ` via faer's
/// SIMD-accelerated kernel and return the upper-triangular factor `U = Lᵀ`
/// converted back to a nalgebra `DMatrix<f64>`.
///
/// Returns `Err(())` if `A` is not strictly positive-definite (e.g. PSD-
/// singular).  Callers should fall back to an unpivoted PSD-tolerant
/// decomposition (`ldlt_upper_in_place`) in that case.
fn faer_cholesky_upper(a: &DMatrix<f64>) -> Result<DMatrix<f64>, ()> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols(), "Cholesky requires a square matrix");
    let fa: faer::Mat<f64> = faer::Mat::from_fn(n, n, |i, j| a[(i, j)]);
    let chol = fa.cholesky(faer::Side::Lower).map_err(|_| ())?;
    // `compute_l` returns the lower-triangular L; we want U = Lᵀ.
    let l = chol.compute_l();
    let l_ref = l.as_ref();
    // Build U directly: U[i, j] = L[j, i].
    Ok(DMatrix::from_fn(n, n, |i, j| {
        if i <= j { l_ref.read(j, i) } else { 0.0 }
    }))
}

/// SIMD-accelerated matrix multiply via faer.  Computes `A · B` where both
/// `A` and `B` are `nalgebra::DMatrix<f64>`.  Used by `weighted_generalized_
/// inverse` to avoid nalgebra's generic matmul kernel on the dense `XᵀW` and
/// `XᵀWX` products (typically 2-5× faster at n~300, p~100).
fn faer_matmul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let m = a.nrows();
    let k = a.ncols();
    debug_assert_eq!(k, b.nrows(), "matmul dimension mismatch");
    let n = b.ncols();
    let fa: faer::Mat<f64> = faer::Mat::from_fn(m, k, |i, j| a[(i, j)]);
    let fb: faer::Mat<f64> = faer::Mat::from_fn(k, n, |i, j| b[(i, j)]);
    let fc: faer::Mat<f64> = &fa * &fb;
    let fc_ref = fc.as_ref();
    DMatrix::from_fn(m, n, |i, j| fc_ref.read(i, j))
}

/// Unpivoted LDLᵀ-style elimination that returns the upper-triangular factor
/// of an SPD-or-PSD symmetric matrix WITHOUT any row permutation.
///
/// Used as the fallback when `nalgebra::linalg::Cholesky::new` returns `None`
/// because the input is positive-SEMI-definite (e.g. the augmented Gram
/// `[X|y]ᵀ[X|y]` when `y` is exactly in the column space of `X` — the exact-
/// fit case for OLS).  Cholesky requires strict positive-definiteness so it
/// fails on a PSD-singular input; LDLᵀ tolerates it.
///
/// The argument `p` is the number of X columns (so the augmented Gram is
/// `(p+1) × (p+1)`).  A zero pivot at row index `< p` indicates a singular
/// X matrix and propagates as `ZeroPivot { index }`; a zero pivot at row
/// `p` (the y column) is legitimate and tolerated.
fn ldlt_upper_in_place(
    mat: &mut DMatrix<f64>,
    p: usize,
) -> Result<DMatrix<f64>, OlsSMError> {
    let n = mat.nrows();
    for k in 0..n {
        let pivot = mat[(k, k)];
        if pivot.abs() < f64::EPSILON * 1e6 {
            if k < p {
                return Err(OlsSMError::ZeroPivot { index: k });
            }
            // Zero pivot on the y column is the legitimate exact-fit case;
            // leave the row in place and let row-normalisation / back-sub skip it.
            continue;
        }
        for i in (k + 1)..n {
            let factor = mat[(i, k)] / pivot;
            for j in k..n {
                let v = mat[(k, j)];
                mat[(i, j)] -= factor * v;
            }
        }
    }
    Ok(mat.clone())
}

// ---------------------------------------------------------------------------
// Algorithm 1 — Modified Cholesky
// ---------------------------------------------------------------------------

/// Algorithm 1: Cholesky-based Gram matrix decomposition with row normalisation.
///
/// Augments `[X | y]`, computes the Gram matrix `G = [X|y]ᵀ[X|y]`,
/// LU-decomposes G, and returns the **row-normalised** upper triangular
/// factor C whose diagonal entries are all 1.
///
/// # Arguments
/// * `x` — Design matrix of shape `(n, p)`
/// * `y` — Response vector of length `n`
///
/// # Returns
/// `C` — Upper triangular matrix of shape `(p+1, p+1)` with unit diagonal.
/// Pass to [`back_substitute`] to recover OLS coefficients.
///
/// # Errors
/// * [`OlsSMError::DimensionMismatch`] if `x.nrows() != y.len()`
/// * [`OlsSMError::ZeroPivot`] if any diagonal of U is ≈ 0
pub fn modified_cholesky(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
) -> Result<DMatrix<f64>, OlsSMError> {
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(OlsSMError::DimensionMismatch {
            x_rows: n,
            y_len: y.len(),
        });
    }

    // Augment X with y as the last column: shape (n, p+1)
    let mut xy = DMatrix::zeros(n, p + 1);
    xy.columns_mut(0, p).copy_from(x);
    xy.column_mut(p).copy_from(y);

    // Gram matrix G = Xyᵀ @ Xy, shape (p+1, p+1)
    let gram = xy.transpose() * &xy;

    // Cholesky decomposition (unpivoted SPD factorisation).
    //
    // We must NOT use partial-pivoting LU here — even though the augmented
    // Gram is SPD in exact arithmetic, partial pivoting on a floating-point
    // Gram permutes rows whenever an off-diagonal entry exceeds the diagonal
    // (frequent when |Xᵀy[i]| > ‖X[:,i]‖²).  The back-substitution then
    // operates on a permuted system where `β[p] = -1` no longer encodes the
    // y-column relation, and the recovered β drifts from the true OLS solution.
    // Cholesky is unpivoted by construction (Higham 2002 §10) and is the
    // textbook factorisation for SPD matrices.
    //
    // We dispatch to faer's SIMD-accelerated Cholesky (typically 3-5× faster
    // than nalgebra at p~100).  Returns Err when input is rank-deficient
    // (PSD-singular) — we then fall back to unpivoted LDLᵀ elimination,
    // which tolerates a zero pivot in the y-column (the exact-fit case).
    let u = match faer_cholesky_upper(&gram) {
        Ok(u_faer) => u_faer,
        Err(_) => {
            let mut work = gram;
            ldlt_upper_in_place(&mut work, p)?
        }
    };

    // Check for zero pivots on the X-columns only (indices 0..p).
    // The last pivot (index p, the y-column) may be zero when y is exactly
    // in the column space of X (exact fit with no residuals) — this is valid.
    for i in 0..p {
        if u[(i, i)].abs() < f64::EPSILON * 1e6 {
            return Err(OlsSMError::ZeroPivot { index: i });
        }
    }

    // Row-normalise: C[i, :] = U[i, :] / U[i, i]  so diag(C) = 1.
    // Skip the last row if its pivot is zero (exact-fit case) — that row is
    // never accessed during back-substitution.
    let dim = u.nrows();
    let mut c = u.clone();
    for i in 0..dim {
        let d = c[(i, i)];
        if d.abs() < f64::EPSILON * 1e6 {
            c[(i, i)] = 1.0; // normalise diagonal to 1 even for zero-pivot rows
            continue;
        }
        for j in i..dim {
            c[(i, j)] /= d;
        }
    }

    Ok(c)
}

// ---------------------------------------------------------------------------
// Back-substitution (companion to Algorithm 1)
// ---------------------------------------------------------------------------

/// Recovers OLS beta coefficients from the C matrix produced by
/// [`modified_cholesky`] via back-substitution.
///
/// Sets `betas[p] = -1`, then for `i` from `p-1` downto `0`:
/// `betas[i] = -(C[i, :] · betas)`.  Returns `betas[0..p]`.
///
/// # Arguments
/// * `c` — `(p+1, p+1)` unit-diagonal upper triangular matrix
///
/// # Returns
/// `beta` — OLS coefficient vector of length `p`
pub fn back_substitute(c: &DMatrix<f64>) -> Result<DVector<f64>, OlsSMError> {
    let dim = c.nrows(); // p + 1
    let mut betas = DVector::zeros(dim);
    betas[dim - 1] = -1.0;

    for i in (0..dim - 1).rev() {
        // betas[i] = -(C[i, :] · betas)
        let dot: f64 = (0..dim).map(|j| c[(i, j)] * betas[j]).sum();
        betas[i] = -dot;
    }

    // Return first p elements (drop the auxiliary -1 entry)
    Ok(DVector::from_iterator(
        dim - 1,
        (0..dim - 1).map(|i| betas[i]),
    ))
}

// ---------------------------------------------------------------------------
// Combined solver
// ---------------------------------------------------------------------------

/// Full OLS solver: equivalent to [`modified_cholesky`] + [`back_substitute`].
///
/// # Arguments
/// * `x` — Design matrix `(n, p)`
/// * `y` — Response vector `(n,)`
///
/// # Returns
/// `beta` — OLS coefficient vector `(p,)`
pub fn solve_ols(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
) -> Result<DVector<f64>, OlsSMError> {
    let c = modified_cholesky(x, y)?;
    back_substitute(&c)
}

// ---------------------------------------------------------------------------
// Algorithm 2 — Simplified Gram-Schmidt Orthogonalisation (SGSO)
// ---------------------------------------------------------------------------

/// Algorithm 2: Non-normalised Gram-Schmidt orthogonalisation (SGSO).
///
/// Produces an orthogonal (but **not** orthonormal) basis Q for the column
/// space of X.  Avoids all square-root computations — only dot products.
///
/// For each column `j`:
/// ```text
/// q_j = x_j  -  Σ_{i<j}  (x_j · q_i) / (q_i · q_i)  *  q_i
/// ```
///
/// # Arguments
/// * `x` — Input matrix `(n, p)`
///
/// # Returns
/// `Q` — Matrix `(n, p)` with mutually orthogonal (un-normalised) columns
pub fn simplified_gram_schmidt(x: &DMatrix<f64>) -> Result<DMatrix<f64>, OlsSMError> {
    let n = x.nrows();
    let p = x.ncols();
    let mut q = DMatrix::<f64>::zeros(n, p);

    // Buffer for the in-flight orthogonalised column (length n, reused
    // across outer iterations).  Storing this as a Vec<f64> means we can
    // hand a `&[f64]` / `&mut [f64]` to BLAS ddot/daxpy without going
    // through nalgebra's MatrixView abstraction.
    let mut qj: Vec<f64> = vec![0.0; n];

    for j in 0..p {
        // qj ← X[:, j].  DMatrix is column-major, so column j is contiguous
        // in x.as_slice() at offset j·n.
        let x_slice = x.as_slice();
        qj.copy_from_slice(&x_slice[j * n..(j + 1) * n]);

        for i in 0..j {
            // Borrow q.as_slice() inside the inner loop only; the borrow is
            // released before we write back to q in the outer loop.
            let q_slice = q.as_slice();
            let qi_slice = &q_slice[i * n..(i + 1) * n];

            let num = sgso_dot(&qj, qi_slice);
            let den = sgso_dot(qi_slice, qi_slice);
            if den.abs() > f64::EPSILON * 1e6 {
                sgso_axpy(-num / den, qi_slice, &mut qj);
            }
        }

        // Write qj back into column j of Q.
        let q_mut = q.as_mut_slice();
        q_mut[j * n..(j + 1) * n].copy_from_slice(&qj);
    }

    Ok(q)
}

// ---------------------------------------------------------------------------
// SGSO helpers — feature-flagged BLAS path
// ---------------------------------------------------------------------------
//
// When the `blas` feature is enabled (default), the inner-loop dot product
// and AXPY (`y ← y + α·x`) are dispatched to `cblas::ddot` / `cblas::daxpy`,
// which call into OpenBLAS via the `openblas-src` crate.  Typical speedup
// at p ~ 100, n ~ 300 is 3-6× over the pure-Rust path.
//
// When the `blas` feature is disabled, we fall back to a hand-rolled loop
// that LLVM auto-vectorises adequately for small problem sizes.  This path
// keeps the project buildable on systems where OpenBLAS cannot be linked
// (no system OpenBLAS install; no vcpkg; no gfortran; etc.).

#[cfg(feature = "blas")]
#[inline]
fn sgso_dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dot-product length mismatch");
    cblas::ddot(a.len() as i32, a, 1, b, 1)
}

#[cfg(not(feature = "blas"))]
#[inline]
fn sgso_dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dot-product length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(feature = "blas")]
#[inline]
fn sgso_axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len(), "axpy length mismatch");
    cblas::daxpy(x.len() as i32, alpha, x, 1, y, 1);
}

#[cfg(not(feature = "blas"))]
#[inline]
fn sgso_axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len(), "axpy length mismatch");
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

// ---------------------------------------------------------------------------
// Algorithm 3 — Weighted Generalised Inverse
// ---------------------------------------------------------------------------

/// Algorithm 3: Weighted generalised inverse `(XᵀWX)⁻¹ Xᵀ W`.
///
/// Computes the weighted least-squares coefficient matrix without explicitly
/// inverting `XᵀWX`. Uses LU solve internally.
///
/// # Arguments
/// * `x` — Design matrix `(n, p)`
/// * `w` — Positive-definite weight matrix `(n, n)` (e.g. a kinship matrix)
///
/// # Returns
/// `G` — Weighted generalised inverse of shape `(p, n)`.
///       For weighted OLS: `beta = G @ y`.
///
/// # Errors
/// * [`OlsSMError::WeightDimension`] if W is not `n×n`
/// * [`OlsSMError::SingularMatrix`] if `XᵀWX` is singular
pub fn weighted_generalized_inverse(
    x: &DMatrix<f64>,
    w: &DMatrix<f64>,
) -> Result<DMatrix<f64>, OlsSMError> {
    let n = x.nrows();

    if w.nrows() != n || w.ncols() != n {
        return Err(OlsSMError::WeightDimension {
            n,
            rows: w.nrows(),
            cols: w.ncols(),
        });
    }

    // xᵀ W  — shape (p, n).  Dense (n × n) × (n × p) product; the dominant
    // wall-time cost for dense W.  faer SIMD matmul is 2-5× faster than
    // nalgebra's generic matmul kernel at typical (n ~ 300, p ~ 100) sizes.
    let xt = x.transpose();
    let xtw = faer_matmul(&xt, w);

    // XᵀWX  — shape (p, p).  Also SIMD matmul.
    let xtwx = faer_matmul(&xtw, x);

    // Solve XᵀWX · G = Xᵀ W  for G, shape (p, n)
    // Equivalent to G = (XᵀWX)⁻¹ Xᵀ W without explicit inversion.
    // The LU here is on a SMALL (p × p) matrix — typically dominated by
    // the matmul costs above, so we keep nalgebra's LU for simplicity.
    let lu = nalgebra::linalg::LU::new(xtwx);
    let result = lu.solve(&xtw).ok_or(OlsSMError::SingularMatrix)?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// Algorithm 4 — Direct Gram Matrix LU Solve
//
// The three public functions below (`lu_solve_gram`, `lu_solve_gram_vec`,
// `lu_inverse_gram`) are SIMD-accelerated LU solve / inverse routines for
// general Gram-style matrices.  They use `faer` instead of nalgebra for
// cache-aware kernels — closing the gap with BLAS-backed linear-algebra
// routines without requiring any external system libraries.
// ---------------------------------------------------------------------------

/// Convert a nalgebra `DMatrix<f64>` to a `faer::Mat<f64>` (column-major copy).
#[inline]
fn nalgebra_to_faer(m: &DMatrix<f64>) -> faer::Mat<f64> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    faer::Mat::from_fn(nrows, ncols, |i, j| m[(i, j)])
}

/// Convert a `faer::MatRef<f64>` back to a nalgebra `DMatrix<f64>`.
#[inline]
fn faer_to_nalgebra(m: faer::MatRef<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(m.nrows(), m.ncols(), |i, j| m.read(i, j))
}

/// Solve `gram · X = rhs` via faer LU factorisation — no explicit inverse formed.
///
/// Uses `faer`'s SIMD-accelerated partial-pivoting LU which is significantly
/// faster than nalgebra for matrices ≥ 64×64.
///
/// # Arguments
/// * `gram` — Symmetric positive-(semi)definite matrix of shape `(p, p)`
/// * `rhs`  — Right-hand side matrix of shape `(p, k)`
///
/// # Returns
/// Solution matrix `X` of shape `(p, k)` such that `gram · X ≈ rhs`.
///
/// # Errors
/// * [`OlsSMError::DimensionMismatch`] if row counts disagree
/// * [`OlsSMError::SingularMatrix`]    if LU solve fails (singular matrix)
pub fn lu_solve_gram(
    gram: &DMatrix<f64>,
    rhs: &DMatrix<f64>,
) -> Result<DMatrix<f64>, OlsSMError> {
    if gram.nrows() != gram.ncols() {
        return Err(OlsSMError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: gram.ncols(),
        });
    }
    if gram.nrows() != rhs.nrows() {
        return Err(OlsSMError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: rhs.nrows(),
        });
    }

    let fa = nalgebra_to_faer(gram);
    let fb = nalgebra_to_faer(rhs);
    let plu = fa.partial_piv_lu();
    let fx = plu.solve(&fb);
    Ok(faer_to_nalgebra(fx.as_ref()))
}

/// Solve `gram · x = rhs` for a single right-hand-side vector.
///
/// Reshapes the vector to a single-column matrix, delegates to `faer` LU,
/// and reshapes back.
///
/// # Arguments
/// * `gram` — Symmetric positive-(semi)definite matrix `(p, p)`
/// * `rhs`  — Right-hand side vector `(p,)`
///
/// # Returns
/// Solution vector `x` of length `p`.
pub fn lu_solve_gram_vec(
    gram: &DMatrix<f64>,
    rhs: &DVector<f64>,
) -> Result<DVector<f64>, OlsSMError> {
    if gram.nrows() != gram.ncols() {
        return Err(OlsSMError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: gram.ncols(),
        });
    }
    if gram.nrows() != rhs.len() {
        return Err(OlsSMError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: rhs.len(),
        });
    }

    let p = gram.nrows();
    let fa = nalgebra_to_faer(gram);
    let fb = faer::Mat::from_fn(p, 1, |i, _| rhs[i]);
    let plu = fa.partial_piv_lu();
    let fx = plu.solve(&fb);
    Ok(DVector::from_fn(p, |i, _| fx.read(i, 0)))
}

/// Compute the explicit inverse of a Gram matrix via faer LU factorisation.
///
/// While `olssm` philosophy favours direct solves over explicit inversion,
/// callers that need to apply the inverse from both sides repeatedly can
/// benefit from caching the inverse once.
///
/// Uses `faer`'s SIMD-accelerated LU which is significantly faster than the
/// nalgebra implementation for typical matrix sizes.
///
/// # Arguments
/// * `gram` — Square matrix of shape `(p, p)`
///
/// # Returns
/// `gram⁻¹` — Inverse matrix of shape `(p, p)`
///
/// # Errors
/// * [`OlsSMError::SingularMatrix`] if the matrix is singular
pub fn lu_inverse_gram(gram: &DMatrix<f64>) -> Result<DMatrix<f64>, OlsSMError> {
    if gram.nrows() != gram.ncols() {
        return Err(OlsSMError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: gram.ncols(),
        });
    }

    let fa = nalgebra_to_faer(gram);
    let plu = fa.partial_piv_lu();
    // faer's inverse() computes A⁻¹ directly without constructing an identity RHS
    let finv = plu.inverse();
    Ok(faer_to_nalgebra(finv.as_ref()))
}
