//! `olssm` — Closed-form OLS without inversion or normalisation.
//!
//! Exposes three algorithms from the paper
//! "Closed-form OLS without inversion or normalisation" — Senderovich & Sandra:
//!
//!   1. Modified Cholesky          — augmented Gram + LU + row-normalise + back-substitute
//!   2. Simplified Gram-Schmidt    — non-normalised orthogonalisation (SGSO)
//!   3. Weighted generalised inverse — `(XᵀWX)⁻¹ Xᵀ W` via LU solve
//!
//! Plus general SIMD-accelerated LU solve / inverse utilities for Gram-style
//! matrices (`lu_solve_gram`, `lu_solve_gram_vec`, `lu_inverse_gram`).
//!
//! # Usage
//! - **Rust**: import `olssm::algorithms::*` directly.
//! - **Python**: `import olssm` after `maturin develop --features python`.
//! - **C/C++**: link against `libolssm` and include `olssm.h`.

pub mod algorithms;
pub mod ffi;

// ---------------------------------------------------------------------------
// Python bindings (feature-gated — activated by maturin via extension-module)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python_bindings {
    use nalgebra::{DMatrix, DVector};
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use crate::algorithms;

    // -----------------------------------------------------------------------
    // Layout conversion helpers
    // -----------------------------------------------------------------------

    /// Convert a numpy (row-major / C-order) 2-D array to a nalgebra DMatrix.
    /// Always copies because nalgebra uses column-major storage internally.
    fn py_to_dmatrix(arr: &PyReadonlyArray2<f64>) -> DMatrix<f64> {
        let shape = arr.shape();
        let slice = arr
            .as_slice()
            .expect("numpy array must be C-contiguous (call .copy() if needed)");
        DMatrix::from_row_slice(shape[0], shape[1], slice)
    }

    /// Convert a numpy 1-D array to a nalgebra DVector.
    fn py_to_dvector(arr: &PyReadonlyArray1<f64>) -> DVector<f64> {
        DVector::from_column_slice(
            arr.as_slice()
                .expect("numpy array must be contiguous"),
        )
    }

    // -----------------------------------------------------------------------
    // Algorithm 1 — Modified Cholesky
    // -----------------------------------------------------------------------

    /// Algorithm 1: LU-based Gram matrix decomposition with row normalisation.
    ///
    /// Args:
    ///     x: numpy float64 array of shape (n, p)
    ///     y: numpy float64 array of shape (n,)
    ///
    /// Returns:
    ///     C matrix of shape (p+1, p+1), dtype float64, diagonal = 1.
    ///     Pass to ``back_substitute`` to recover OLS coefficients.
    #[pyfunction]
    pub fn modified_cholesky<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let xm = py_to_dmatrix(&x);
        let yv = py_to_dvector(&y);
        let c = algorithms::modified_cholesky(&xm, &yv)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = c.nrows();
        let cols = c.ncols();
        let data: Vec<f64> = c.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Back-substitute C to recover OLS coefficients.
    ///
    /// Args:
    ///     c: numpy float64 array of shape (p+1, p+1) — output of ``modified_cholesky``
    ///
    /// Returns:
    ///     beta: numpy float64 array of shape (p,)
    #[pyfunction]
    pub fn back_substitute<'py>(
        py: Python<'py>,
        c: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let cm = py_to_dmatrix(&c);
        let beta = algorithms::back_substitute(&cm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(beta.as_slice().to_vec().into_pyarray(py))
    }

    /// Full OLS solver — equivalent to ``modified_cholesky`` + ``back_substitute``.
    ///
    /// Args:
    ///     x: numpy float64 array of shape (n, p)
    ///     y: numpy float64 array of shape (n,)
    ///
    /// Returns:
    ///     beta: numpy float64 array of shape (p,), the OLS coefficients.
    #[pyfunction]
    pub fn solve_ols<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let xm = py_to_dmatrix(&x);
        let yv = py_to_dvector(&y);
        let beta = algorithms::solve_ols(&xm, &yv)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(beta.as_slice().to_vec().into_pyarray(py))
    }

    // -----------------------------------------------------------------------
    // Algorithm 2 — Simplified Gram-Schmidt (SGSO)
    // -----------------------------------------------------------------------

    /// Algorithm 2: Non-normalised Gram-Schmidt orthogonalisation (SGSO).
    ///
    /// Args:
    ///     x: numpy float64 array of shape (n, p)
    ///
    /// Returns:
    ///     Q: numpy float64 array of shape (n, p) — orthogonal columns, not normalised.
    #[pyfunction]
    pub fn simplified_gram_schmidt<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let xm = py_to_dmatrix(&x);
        let q = algorithms::simplified_gram_schmidt(&xm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = q.nrows();
        let cols = q.ncols();
        let data: Vec<f64> = q.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Algorithm 3 — Weighted generalised inverse
    // -----------------------------------------------------------------------

    /// Algorithm 3: Weighted generalised inverse ``(XᵀWX)⁻¹ Xᵀ W``.
    ///
    /// Args:
    ///     x: numpy float64 array of shape (n, p)
    ///     w: numpy float64 array of shape (n, n), positive-definite weight matrix
    ///
    /// Returns:
    ///     G: numpy float64 array of shape (p, n).
    ///        Weighted OLS solution: ``beta = G @ y``.
    #[pyfunction]
    pub fn weighted_generalized_inverse<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        w: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let xm = py_to_dmatrix(&x);
        let wm = py_to_dmatrix(&w);
        let g = algorithms::weighted_generalized_inverse(&xm, &wm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = g.nrows();
        let cols = g.ncols();
        let data: Vec<f64> = g.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // General Gram-LU utilities — useful primitives for OLS workflows.
    // -----------------------------------------------------------------------

    /// Solve ``gram · X = rhs`` via LU factorisation — no explicit inverse.
    #[pyfunction]
    pub fn lu_solve_gram<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f64>,
        rhs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let gm = py_to_dmatrix(&gram);
        let rm = py_to_dmatrix(&rhs);
        let result = algorithms::lu_solve_gram(&gm, &rm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = result.nrows();
        let cols = result.ncols();
        let data: Vec<f64> = result.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Solve ``gram · x = rhs`` for a single vector RHS.
    #[pyfunction]
    pub fn lu_solve_gram_vec<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f64>,
        rhs: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let gm = py_to_dmatrix(&gram);
        let rv = py_to_dvector(&rhs);
        let result = algorithms::lu_solve_gram_vec(&gm, &rv)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.as_slice().to_vec().into_pyarray(py))
    }

    /// Compute ``gram⁻¹`` via LU factorisation.
    #[pyfunction]
    pub fn lu_inverse_gram<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let gm = py_to_dmatrix(&gram);
        let result = algorithms::lu_inverse_gram(&gm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = result.nrows();
        let cols = result.ncols();
        let data: Vec<f64> = result.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Module registration
    // -----------------------------------------------------------------------

    #[pymodule]
    pub fn olssm(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(modified_cholesky, m)?)?;
        m.add_function(wrap_pyfunction!(back_substitute, m)?)?;
        m.add_function(wrap_pyfunction!(solve_ols, m)?)?;
        m.add_function(wrap_pyfunction!(simplified_gram_schmidt, m)?)?;
        m.add_function(wrap_pyfunction!(weighted_generalized_inverse, m)?)?;
        m.add_function(wrap_pyfunction!(lu_solve_gram, m)?)?;
        m.add_function(wrap_pyfunction!(lu_solve_gram_vec, m)?)?;
        m.add_function(wrap_pyfunction!(lu_inverse_gram, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::olssm;
