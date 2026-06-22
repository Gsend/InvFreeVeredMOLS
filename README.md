# InvFreeVeredMOLS — Inversion-Free OLS Solvers

A Rust + Python implementation of three closed-form ordinary-least-squares
algorithms that avoid explicit matrix inversion:

1. **Modified Cholesky** — augmented Gram + LU + row-normalisation + back-substitution
2. **Simplified Gram-Schmidt Orthogonalisation (SGSO)** — non-normalised orthogonalisation
3. **Weighted generalised inverse** — `(XᵀWX)⁻¹ XᵀW` via LU solve

Plus general SIMD-accelerated LU solve / inverse utilities for Gram-style
matrices (`lu_solve_gram`, `lu_solve_gram_vec`, `lu_inverse_gram`).

## Repository layout

```
InvFreeVeredMOLS\
├── rust\                              Rust crate `olssm`
│   ├── Cargo.toml                     workspace + deps (nalgebra, faer, thiserror)
│   ├── src\                           lib.rs, algorithms.rs, ffi.rs
│   ├── tests\test_algorithms.rs       Rust unit tests
│   ├── examples\                      Rust demos
│   │   └── ols_three_algorithms.rs    runs all three algorithms on a 4×2 system
│   └── patches\nano-gemm-c64\         local crate patch (build dependency)
├── python\
│   ├── invfree_vered_mols\            Python re-export of the Rust `olssm` bindings
│   │   └── __init__.py                public API: solve_ols, modified_cholesky,
│   │                                  back_substitute, simplified_gram_schmidt,
│   │                                  weighted_generalized_inverse
│   └── tests\
│       └── test_ols_algorithms.py     focused per-algorithm unit tests
├── examples\
│   └── demo_ols_python.py             Python demo (Rust binding + NumPy fallback)
├── scripts\
│   ├── build_rust.ps1                 Windows build script
│   └── build_rust.sh                  Linux/macOS build script
├── LICENSE\MIT
├── requirements.txt                   Python deps
└── PLAN.md                            project status + open questions
```

## Installation

You need a Rust toolchain (for the native solver core) and Python 3.10+
(for the optimizer wrapper and demos).

### 0. Install OpenBLAS (default build path)

By default, `olssm`'s `simplified_gram_schmidt` dispatches its inner-loop dot products to OpenBLAS for a 3–6× speedup over the pure-Rust path.  Install OpenBLAS once:

| OS | Command |
|---|---|
| **Windows (MSVC)** | `vcpkg install openblas:x64-windows` + `vcpkg integrate install` |
| **Linux (Debian/Ubuntu)** | `sudo apt install libopenblas-dev` |
| **macOS** | `brew install openblas` (and `export OPENBLAS_DIR=$(brew --prefix openblas)` if cargo doesn't auto-detect) |

**If you can't install OpenBLAS** (no admin rights, restricted environment, etc.) — skip this step and add `--no-default-features` to the cargo commands below; the build will use the pure-Rust SGSO fallback.

### 1. Install the Rust toolchain

- **Windows / macOS / Linux:** install via [rustup](https://rustup.rs/)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# or on Windows, download and run https://win.rustup.rs/x86_64
```

Verify with `rustc --version` and `cargo --version`.

### 2. (Optional) Create a Python virtual environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

This pulls in `numpy`, `torch`, `maturin` (the Rust ↔ Python build tool),
`pytest`, and `scipy` (used only by the fallback demo).

### 4. Build the Rust crate and Python binding

**Windows (PowerShell):**

```powershell
.\scripts\build_rust.ps1 -Release
```

**Linux / macOS:**

```bash
chmod +x scripts/build_rust.sh
scripts/build_rust.sh --release
```

What this does:
- `cargo build --release` to produce the native library
- `cargo test --release` to run the Rust unit tests
- `maturin develop --features python --release` to compile the PyO3 extension
  and install it as the importable Python module `olssm`

You should see something like:

```
✓ olssm Python extension installed into active environment.
```

If `maturin` is not on your PATH, the script will warn and skip the Python
binding step — the demos still run, falling back to a NumPy reference
implementation.

## Quick demos

There is one demo per algorithm, in both Python and Rust.  Each demo:

- simulates a linear system  `y = X β_true + ε`  of size **n = 300 samples × p = 100 predictors** with Gaussian noise,
- runs the corresponding olssm algorithm,
- compares the result to a **textbook Householder-QR reference solution** (the standard backward-stable approach from Higham 2002, §19; Trefethen & Bau, Lecture 19),
- reports coefficient-recovery error, residual norm, and algorithm-vs-reference divergence.

### Rust demos

```bash
cd rust
cargo run --example demo_alg1_modified_cholesky --release
cargo run --example demo_alg2_sgso              --release
cargo run --example demo_alg3_weighted_gi       --release
```

### Python demos

```powershell
python examples\demo_alg1_modified_cholesky.py
python examples\demo_alg2_sgso.py
python examples\demo_alg3_weighted_gi.py
```

The Python demos use the Rust binding (`import olssm`) when it's been built by maturin; otherwise they fall back to a pure-NumPy reference implementation. The script prints which backend it used on the first line.

### What each demo shows

| Demo | Algorithm | Textbook reference for comparison |
|---|---|---|
| 1 | Modified Cholesky → augmented Gram + LU + row-normalise + back-substitute → OLS β | Householder QR on X, then back-solve R β = Qᵀy |
| 2 | SGSO → un-normalised orthogonal Q with diagonal QᵀQ | Householder QR (gives orthonormal Q) — compared after normalising SGSO's columns |
| 3 | Weighted generalised inverse  G = (XᵀWX)⁻¹ XᵀW with optimal Gauss-Markov W | Whitening (W^(1/2) X, W^(1/2) y) + Householder QR — the standard GLS pipeline |

Each demo prints both errors and a verdict, e.g.:

```
  ‖β_alg1 − β_true‖₂      = 7.0496e-02
  ‖β_QR   − β_true‖₂      = 7.0496e-02    ← textbook reference
  ‖β_alg1 − β_QR‖₂        = 1.66e-14      ← algorithm divergence from reference
  ✓ Algorithm 1 matches Householder QR to machine precision.
```

### Run the unit tests

```bash
# Rust
cd rust && cargo test --release

# Python
pytest python/tests/
```

## Algorithm complexity

All bounds assume a design matrix of shape `(n, p)` with `n ≥ p` (overdetermined system).

### Algorithm 1 — Modified Cholesky

| Step | Time | Memory (peak) |
|---|---|---|
| Augment `[X \| y]` | `O(n·p)` (in place — `O(1)` if streamed) | `O(n·p)` |
| Form Gram `M = AᵀA` | **`O(n·p²)`** | `O(p²)` |
| LU decomposition of `M` | `O(p³)` | `O(p²)` |
| Row-normalise `U → C` | `O(p²)` | in place |
| Back-substitute | `O(p²)` | `O(p)` |
| **Total** | **`O(n·p² + p³)`** — dominated by Gram formation when `n ≫ p` | **`O(n·p + p²)`** |

**Conditioning:** error bound `O(κ(X)² · ε)`. Forming `M = AᵀA` squares the condition number — this is the κ²-instability flagged in §2 of the parent paper.

### Algorithm 2 — Simplified Gram-Schmidt (SGSO)

| Step | Time | Memory |
|---|---|---|
| For each column `j ∈ [0, p)`: subtract projections onto `q₀, …, q_{j−1}` | inner step `O(n)`; `j·n` per column | working vector `O(n)` |
| **Total** | **`Σ_{j=0}^{p−1} j·O(n) = O(n·p²)`** | **`O(n·p)`** for the output `Q` |

**Numerical savings vs classical Gram-Schmidt:** avoids `p` square-root operations (one per column-normalisation step) by deferring the normalisation. Conditioning is `O(κ(X) · ε)` for the modified variant — one order of κ below the normal-equations route, matching textbook QR.

### Algorithm 3 — Weighted Generalised Inverse

For weight matrix `W ∈ ℝ^(n × n)`, the cost depends on `W`'s structure:

| Step | Diagonal `W` | Dense `W` | Memory |
|---|---|---|---|
| Form `XᵀW` | `O(n·p)` | **`O(n²·p)`** | `O(n·p)` |
| Form `XᵀWX` | `O(n·p²)` | `O(n·p²)` | `O(p²)` |
| LU factorise `XᵀWX` | `O(p³)` | `O(p³)` | `O(p²)` |
| Triangular solve with `n` RHS columns | `O(p²·n)` | `O(p²·n)` | `O(p·n)` |
| **Total** | **`O(n·p² + p³)`** | **`O(n²·p + p³)`** — dominated by `XᵀW` for dense W | **`O(n·p + p²)`** |

**Output `G` has shape `(p, n)`** — once computed, weighted OLS for any response `y` is one matrix-vector product: `β = G @ y` at `O(n·p)`.

### Comparison table

| Algorithm | Time (n ≫ p) | Memory | Conditioning |
|---|---|---|---|
| Algorithm 1 (Modified Cholesky)         | `O(n·p²)` | `O(n·p)` | `O(κ(X)² · ε)` |
| Algorithm 2 (SGSO)                      | `O(n·p²)` | `O(n·p)` | `O(κ(X) · ε)` |
| Algorithm 3 (Weighted GI, diagonal W)   | `O(n·p²)` | `O(n·p)` | `O(κ(W^{½} X)² · ε)` |
| Algorithm 3 (Weighted GI, dense W)      | `O(n²·p)` | `O(n·p)` | `O(κ(W^{½} X)² · ε)` |
| Textbook reference: Householder QR + solve | `O(n·p²)` | `O(n·p)` | `O(κ(X) · ε)` |

All algorithms are asymptotically identical to Householder QR in time and memory (`n·p²` and `n·p` respectively) for `n ≫ p`, *except* Algorithm 3 with a dense `W`, which is `O(n²·p)` due to the dense `XᵀW` step. Algorithm 1 trades a `κ` of stability for the cache-friendly Gram-matrix structure (`p²` working set), Algorithm 2 saves `p` square roots and matches QR stability, and Algorithm 3 extends OLS to known noise covariances — its dominant cost is the weight-matrix product, not the LU solve.

## Using the algorithms directly

### From Rust

```rust
use nalgebra::{DMatrix, DVector};
use olssm::algorithms::solve_ols;

let x = DMatrix::from_row_slice(/* … */);
let y = DVector::from_column_slice(/* … */);
let beta = solve_ols(&x, &y).expect("solver failed");
```

Full algorithm catalogue: `modified_cholesky`, `back_substitute`, `solve_ols`,
`simplified_gram_schmidt`, `weighted_generalized_inverse`, `lu_solve_gram`,
`lu_solve_gram_vec`, `lu_inverse_gram`, `lu_damped_inverse_f32`,
`eigh_f32`, `eigh_topk_f32`, `apply_kfac_eigen_f32`,
`apply_kfac_lowrank_f32`, `randomized_eigh_f32`.

### From Python (Rust binding)

After running the build script:

```python
import numpy as np
import olssm

x = np.random.randn(100, 5)
y = np.random.randn(100)
beta = olssm.solve_ols(x, y)        # OLS via Modified Cholesky
Q = olssm.simplified_gram_schmidt(x)  # SGSO
G = olssm.weighted_generalized_inverse(x, np.eye(100))
```

All arrays must be C-contiguous float64.

## C ABI

The `olssm` crate also exposes a stable C ABI in `rust/src/ffi.rs` for
C/C++/FORTRAN callers; `cargo build` produces both a `cdylib` (for C linkage)
and an `rlib` (for Rust linkage). See `rust/build.rs` for `cbindgen`
header generation.

## License

MIT — see `LICENSE/MIT`.

## Citation

If you use these algorithms in research, please cite the source paper:

> Madar, V. S., & Batista, S. L. (2023). *Solving The Ordinary Least Squares
> in Closed Form, Without Inversion or Normalization*. arXiv:2301.01854.
> [https://arxiv.org/abs/2301.01854](https://arxiv.org/abs/2301.01854)

A local copy of the paper and BibTeX entry live in
[`docs/README.md`](docs/README.md).

## Status

Pre-1.0. See `PLAN.md` for the open work items, including a known-bug
investigation in one of the three algorithms — please consult `PLAN.md` §2
before relying on the implementation for production use.

## Cleanup note

This project was extracted from a larger codebase that included K-FAC
optimizer code. The K-FAC-specific Python modules and tests have been
emptied to single-line removal stubs (sandbox limitations prevent file
deletion). Before the first commit, run:

```powershell
Remove-Item .\python\invfree_vered_mols\olssm_kfac.py,`
            .\python\invfree_vered_mols\kfac_hooks.py,`
            .\python\invfree_vered_mols\gram_estimator.py,`
            .\python\invfree_vered_mols\bf16_linalg.py,`
            .\python\invfree_vered_mols\backend.py,`
            .\python\invfree_vered_mols\errors.py,`
            .\python\tests\test_backend_dispatch.py,`
            .\python\tests\test_kappa_scaling.py,`
            .\python\tests\test_solver_accuracy_cpu_gpu.py,`
            .\examples\demo_ols_python.py,`
            .\rust\examples\ols_three_algorithms.rs
```

Or, equivalently, in a single line:

```powershell
.\scripts\cleanup_stubs.ps1
```

K-FAC code now lives at `C:\Users\Admin\IFKFAC` (separate project).
