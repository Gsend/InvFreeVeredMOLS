# InvFreeVeredMOLS — Build & Bug-Fix Plan

This document describes what I have *already done* in this directory, what I still need to confirm with you, and the proposed scope for the remaining work. **No code will be changed until you sign off on the bug identification (§2) and the work breakdown (§4).**

---

## 0. Performance migration to faer SIMD (in progress — 2026-06-22)

Three optimisations requested.  Items 1 and 3 implemented; Item 2 deferred pending design decision.

### ✓ Item 1 — `modified_cholesky` uses faer Cholesky (DONE)

Replaced `nalgebra::linalg::Cholesky::new(gram)` with a `faer_cholesky_upper` helper that converts the augmented Gram to a `faer::Mat<f64>`, runs faer's SIMD-accelerated Cholesky, and converts the upper-triangular factor back to `nalgebra::DMatrix<f64>`.  LDLᵀ fallback for the PSD-singular (exact-fit) case is preserved.  Expected speedup: 3–5× at p~100.

### ✓ Item 3 — `weighted_generalized_inverse` matmuls use faer (DONE)

Replaced the two large dense matmuls (`Xᵀ · W` and `(XᵀW) · X`) with a `faer_matmul` helper.  The small (p×p) LU solve is kept on nalgebra because it is dominated by the matmul costs.  Expected speedup: 2–5× at n~300, p~100 (dense W), driven by the `Xᵀ · W` step which is `O(n²·p)`.

### ✓ Item 2 — `simplified_gram_schmidt` BLAS dot products (DONE — Path A)

**Path A chosen.**  Added `cblas` + `openblas-src` as dependencies behind a default-on `blas` feature flag.  `simplified_gram_schmidt` was rewritten to extract a `&[f64]` slice for each column of Q and feed it to `cblas::ddot` (dot product) and `cblas::daxpy` (`y ← y + α·x`).  Expected speedup: 3–6× at p ~ 100, n ~ 300.

Two helper functions `sgso_dot` and `sgso_axpy` dispatch at compile time via `#[cfg(feature = "blas")]`:

- `feature = "blas"` (default): calls `cblas::ddot` / `cblas::daxpy` → OpenBLAS SIMD-tuned kernel.
- `feature = "blas"` disabled: hand-rolled Rust loops that LLVM auto-vectorises adequately for small problem sizes.

**Build prerequisite (default path):** OpenBLAS must be installed and on the link search path.

| OS | Install command | Notes |
|---|---|---|
| Linux | `sudo apt install libopenblas-dev` | Or `dnf`, `pacman`, etc. |
| macOS | `brew install openblas`           | Set `OPENBLAS_DIR=$(brew --prefix openblas)` if cargo doesn't find it |
| Windows (MSVC) | `vcpkg install openblas` + `vcpkg integrate install` | Or download a pre-built binary from the OpenBLAS releases page and set `LIB`/`PATH` accordingly |

**Fallback for machines without OpenBLAS:**

```bash
cargo build --release --no-default-features --features python
```

This drops the BLAS path and uses the pure-Rust auto-vectorised fallback.  Same correctness, ~3–6× slower at the SGSO inner loop, but no system dependency.

---

---

## 1. What's already in place

I copied source files from `C:\Users\Admin\OlsVered` and applied a minimal Python rename (`from optimizer.X` → `from invfree_vered_mols.X`). The current tree:

```
C:\Users\Admin\InvFreeVeredMOLS\
├── rust\
│   ├── Cargo.toml                    # `olssm` crate, edition 2021
│   ├── Cargo.lock                    # pinned crate versions
│   ├── build.rs                      # cbindgen header generation
│   ├── src\
│   │   ├── lib.rs                    # 528 lines — crate root, Python FFI exports
│   │   ├── algorithms.rs             # 784 lines — eigh / Lanczos / randomized eigh
│   │   └── ffi.rs                    # 249 lines — C ABI wrappers
│   ├── tests\
│   │   └── test_algorithms.rs        # 340 lines — Rust unit tests
│   └── patches\nano-gemm-c64\        # local crate patch
└── python\
    ├── invfree_vered_mols\
    │   ├── __init__.py               # public API surface
    │   ├── olssm_kfac.py             # main optimizer (OlsSmKFAC)
    │   ├── backend.py                # Python ↔ Rust dispatcher
    │   ├── gram_estimator.py         # Gram-matrix utility
    │   ├── kfac_hooks.py             # forward/backward hooks
    │   ├── bf16_linalg.py            # bf16 linear-algebra primitives
    │   └── errors.py                 # ConfigurationError
    └── tests\                        # ported from OlsVered/tests
        ├── test_backend_dispatch.py
        ├── test_kappa_scaling.py
        └── test_solver_accuracy_cpu_gpu.py
```

Status: imports rewritten, Python files syntax-check clean. **The Rust crate has not been built**, and **the bug you mentioned has not been fixed** — that's what this document is for.

---

## 2. Bug confirmed and fixed (2026-06-22)

### The bug

`rust/src/algorithms.rs::modified_cholesky` used `nalgebra::linalg::LU::new(gram)` and read `lu.u()` directly. **Partial-pivoting LU computes `P·A = L·U`**: the returned `U` is the upper-triangular factor of `P·A`, *not* of `A`. The augmented Gram `M = [X|y]ᵀ[X|y]` is SPD in exact arithmetic, but in floating point partial pivoting still swaps rows whenever an off-diagonal entry exceeds the diagonal — which happens routinely for the augmented Gram, because `|Xᵀy[i]|` is frequently larger than `‖X[:,i]‖²`. When the swap puts the y-column row into a non-final position, the back-substitution invariant ("set `β[p] = −1`, the last C row encodes the y-relation") breaks, and the recovered β drifts from the true OLS solution.

### How it slipped past the existing tests

- `test_solve_ols_exact_system` uses a system where y is exactly in span(X). In that case the augmented Gram is singular and the last U row becomes all zeros regardless of pivoting → the back-substitution gets a correct answer by accident.
- `test_solve_ols_5x3_system` has a tolerance of `< 3.0` on residuals — almost certainly widened in response to the (then-unexplained) drift caused by this very bug.

### How it was exposed

The user ran `examples/demo_alg1_modified_cholesky.py` at n=300, p=100, σ_noise=0.1 (the first test case that is both full-rank with noise and large enough for pivoting to swap):

| Path | ‖β − β_true‖ | ‖β − β_QR‖ |
|---|---|---|
| Householder QR (textbook reference) | 7.05e-02 | 0 |
| NumPy fallback (uses Cholesky)      | 7.05e-02 | **1.66e-14** |
| Rust olssm (uses LU)                | 7.12e-02 | **5.64e-03** ← drift |

The NumPy fallback was already using Cholesky (because the demo's pure-NumPy reference implementation tries Cholesky first and falls back to scipy LU only on failure). The Rust path was the only one using LU. That's exactly the contrast the demo measured.

### The fix

Replace `LU::new(gram)` with `Cholesky::new(gram)`, with an explicit no-pivot LDLᵀ fallback for the singular (exact-fit) case:

```rust
let u = match nalgebra::linalg::Cholesky::new(gram.clone()) {
    Some(chol) => chol.l().transpose(),     // U = L^T from gram = LL^T
    None => {
        let mut work = gram;
        ldlt_upper_in_place(&mut work, p)?  // PSD-singular fallback, also unpivoted
    }
};
```

Cholesky is unpivoted by construction (Higham 2002 §10), exploits the SPD structure (≈ 2× faster than LU), and is the textbook factorisation for SPD matrices — so it both fixes the correctness bug and is the algorithmically natural choice for an algorithm named "Modified *Cholesky*".

### Regression test added

`rust/tests/test_algorithms.rs::test_solve_ols_matches_householder_qr_at_scale` builds n=300, p=100 well-conditioned random Gaussian X via a deterministic LCG, solves OLS via `solve_ols`, and compares against nalgebra's Householder-QR reference. Fails the build if the LU-pivoting bug regresses.

### Verify the fix

```powershell
cd C:\Users\Admin\InvFreeVeredMOLS\rust
cargo test --release                 # the new regression test should pass
maturin develop --features python --release
cd ..
python examples\demo_alg1_modified_cholesky.py
# Expected: '‖β_alg1 − β_QR‖₂ ~ 1e-14' and the '✓ matches Householder QR to
# machine precision' verdict
```

---

## 2-old. Original candidate list (kept for context)

You said "we had a bug in this project that you pointed out to me 2 days ago." I checked git log on `C:\Users\Admin\OlsVered` for commits and notes from 2026-06-19 through 2026-06-21 touching the OLS-solver-relevant files (`olssm_kfac.py`, `gram_estimator.py`, `backend.py`, `hooks.py`, `src/`). I did not find an obvious unfixed issue with that label. My honest candidates, ordered by my best guess at what you mean:

### Candidate A — Augmented-R baked into the stored factor (vered_kfac, not olssm)

**Where:** `optimizer/vered_kfac.py` / `ifkfac.py` in the OlsVered and IFKFAC projects. The Vered hook stores `R` such that `R^T R = A + λ_train I` (training-damping baked into the factor), while Classic K-FAC stores raw `A`. I flagged this on **2026-06-21 / 22** during the §5.8 Laplace work — see task #90 (`§5.8 control: λ_train→0 to test damping-bias hypothesis`).

**Important:** This bug is **not in `olssm_kfac.py`** — the OlsSmKFAC class uses eigendecomposition on the captured Gram, with damping added to eigenvalues after the fact (`inv_lam = 1.0 / (eigenvalues + self.damping)`). It does not have the augmented-R issue.

So if this is the bug you meant: it lives in the **IFKFAC project**, not `InvFreeVeredMOLS`. Fix would land in `C:\Users\Admin\IFKFAC\ifkfac\ifkfac.py` (and the source `vered_kfac.py` back in OlsVered).

### Candidate B — `capture_fp64_reference` Conv2d normalization mismatch

**Where:** `benchmark/laplace_fp64_fidelity.py` in OlsVered. The fp64 reference path divides accumulated Gram matrices by `n_samples` (total training examples), but K-FAC for Conv2d layers expects normalization by `n_samples × n_locations`. I noticed this on **2026-06-21** while debugging the fp64-fidelity diagnostic. We abandoned the diagnostic (task #89) rather than fix it.

**Important:** This bug is **not in the OLS solver code either** — it's in the diagnostic script for §5.8. It does not live in `InvFreeVeredMOLS`.

### Candidate C — A bug specifically inside the OLS solver code that I'm not remembering

If neither A nor B is the one, I may have missed something in our June 20 session. Possibilities I can investigate:

- The `eigh_topk_f32` / `randomized_eigh_f32` dispatch in `backend.py`
- A normalization or scaling issue in `gram_estimator.py`
- A Rust algorithm correctness issue in `src/algorithms.rs`

**Please tell me which it is.** I will not guess and apply a wrong fix.

---

## 3. Outstanding scope (in addition to the bug fix)

You asked for: Python + Rust demos + unit tests. Current state:

| Item | Status | Action needed |
|---|---|---|
| Python source modules | ✓ copied, renamed | none |
| Python unit tests | ✓ ported 3 tests | add 1–2 more demos-as-tests for clarity |
| Python demo | ✗ missing | write `examples/demo_train_mlp.py` showing OlsSmKFAC on a small MLP |
| Rust source modules | ✓ copied | none |
| Rust unit tests | ✓ copied `tests/test_algorithms.rs` | none |
| Rust example/demo | ✗ missing | write `rust/examples/eigh_kappa_sweep.rs` showing the eigh primitive on synthetic κ-controlled matrices |
| `requirements.txt` | ✗ missing | torch ≥ 2.0, numpy, pytest |
| `pyproject.toml` (Python package) | ✗ missing | one with package + test deps |
| `pyproject.toml` for maturin (Python ↔ Rust binding build) | ✗ missing | needed if user wants `pip install .` to compile the Rust |
| Build scripts | ✗ missing | `scripts/build_rust.ps1` (Windows), `scripts/build_rust.sh` (Linux/macOS) |
| README.md | ✗ missing | overview, install (Rust + Python), quick start, demo links |
| LICENSE/MIT | ✗ missing | MIT license file |
| `.gitignore` | ✗ missing | target/, __pycache__/, *.pyc |

---

## 4. Proposed sequence

I will execute in this order, each step preceded by a single confirmation question if there's any judgment call. **Nothing happens until you confirm §2.**

1. **You tell me which bug** (A, B, C, or none-of-the-above with details).
2. **I explain the bug in detail** in this PLAN.md (root cause, where in code, what the fix is, what tests would catch it). No code change yet.
3. **You sign off on the fix.**
4. **I apply the fix** to the source-of-truth file in OlsVered or IFKFAC (as appropriate) AND to the copy in this project.
5. **I write a regression test** that would have caught it.
6. **I write the missing infrastructure**: README, LICENSE, requirements.txt, pyproject (both flavors), build scripts.
7. **I write the demos**:
   - `examples/demo_train_mlp.py` (Python) — small MLP trained with `OlsSmKFAC` showing convergence and the eigh primitive being exercised
   - `rust/examples/eigh_kappa_sweep.rs` (Rust) — call `olssm::algorithms::eigh_f32` on synthetic matrices of controlled κ; print accuracy
8. **I write a build verification step**: a single `scripts/verify.ps1` / `scripts/verify.sh` that runs `cargo test --release`, `cargo build --release`, `cargo run --example eigh_kappa_sweep`, `pytest python/tests/`, and `python examples/demo_train_mlp.py`. This is the "did the build succeed" gate for users.

---

## 5. Open questions for you

1. **Which bug?** (Candidate A, B, C from §2.)
2. **Naming:** the Python package is currently `invfree_vered_mols` (snake_case). Is that what you want, or `invfree_mols`, `ifvm`, something else? The Rust crate stays `olssm`.
3. **Rust binding:** Do you want the Rust core to be importable from Python automatically via `pip install` (requires `maturin` and a `pyo3` extension)? Or is "build with cargo, set PYTHONPATH manually, fall back to the pure-Python path otherwise" sufficient? The current code already has a PyTorch fallback if the Rust extension isn't found, so both options work — but option 1 makes the experience much smoother for users on systems without a Rust toolchain.
4. **License:** Confirm MIT, same as the IFKFAC project, with `Gilad Senderovich` as copyright holder?

---

Once you answer §5 (and especially §5 question 1 — which bug), I will fill in §2's "explain the bug in detail" section and then await your sign-off before any code change.
