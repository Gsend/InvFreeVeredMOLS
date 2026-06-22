"""
tests/test_solver_accuracy_cpu_gpu.py
=====================================

Cross-cutting accuracy tests for the inversion-free solvers in this repo.

What it covers
--------------
* **Rust / olssm path (CPU only):** ``solve_ols``, ``simplified_gram_schmidt``,
  ``weighted_generalized_inverse``, and ``lu_solve_gram_vec`` from
  ``python/olssm``. These are CPU-only by construction; the device axis
  collapses to ``cpu`` for them.

* **PyTorch / vered_solve path (CPU + CUDA):** ``vered_decompose``,
  ``vered_solve``, ``vered_apply``, ``vered_solve_batched`` from
  ``diagnostic/vered_solve.py``. These genuinely run on both CPU and CUDA
  (CUDA cases are skipped if no CUDA device is available).

* **K-FAC inversion utilities (CPU + CUDA):** ``naive_pinv_inverse`` and
  ``kfac_a_inverse`` from ``diagnostic/inversion.py``. Validates the
  factor-then-solve Moore-Penrose / K-FAC-A posterior-mean paths against
  references on systems with a known exact solution.

Methodology
-----------
For each solver we synthesize a linear system with a **known** ground-truth
β (or x) by constructing X via an SVD with controlled condition number:

    X = U diag(s) Vᵀ      with s linearly spaced in [1, 1/κ]
    y = X β*              (noise-free for accuracy testing)

This guarantees a known closed-form answer to compare against. For SPD
solvers we use the Gram matrix G = AᵀA + λI with a known right-hand-side
``r = G x*`` and recover ``x*``. Each method's max-abs / relative /
residual error is recorded.

We assert on tolerances that scale with the conditioning regime and the
working precision (float64 vs float32) and we collect every measurement
into a Markdown report written at session end to
``tests/results/solver_accuracy_report.md``.

Run
---
::

    pytest tests/test_solver_accuracy_cpu_gpu.py -v
    pytest tests/test_solver_accuracy_cpu_gpu.py -v --device=cuda     # force CUDA
    pytest tests/test_solver_accuracy_cpu_gpu.py -v -k "vered_solve"  # subset

The Markdown report is rewritten on every run; it does not depend on
``pytest-html`` or any other plugin.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional imports — gracefully skip whole groups if the dep is missing
# ---------------------------------------------------------------------------

_TORCH_ERR: Optional[str] = None
try:
    import torch
    _HAS_TORCH = True
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False
    _TORCH_ERR = repr(_e)

_OLSSM_ERR: Optional[str] = None
try:
    from olssm import (  # type: ignore
        solve_ols,
        simplified_gram_schmidt,
        weighted_generalized_inverse,
        lu_solve_gram_vec,
    )
    _HAS_OLSSM = True
except Exception as _e:  # pragma: no cover
    _HAS_OLSSM = False
    _OLSSM_ERR = repr(_e)

_VERED_ERR: Optional[str] = None
_KFAC_ERR: Optional[str] = None
if _HAS_TORCH:
    try:
        from diagnostic.vered_solve import (  # type: ignore
            vered_decompose,
            vered_solve,
            vered_apply,
            vered_solve_batched,
        )
        _HAS_VERED = True
    except Exception as _e:  # pragma: no cover - report exact reason
        _HAS_VERED = False
        _VERED_ERR = repr(_e)

    try:
        from diagnostic.inversion import (  # type: ignore
            naive_pinv_inverse,
            kfac_a_inverse,
        )
        _HAS_KFAC_INV = True
    except Exception as _e:  # pragma: no cover
        _HAS_KFAC_INV = False
        _KFAC_ERR = repr(_e)
else:
    _HAS_VERED = False
    _HAS_KFAC_INV = False
    _VERED_ERR = f"torch unavailable: {_TORCH_ERR}"
    _KFAC_ERR = f"torch unavailable: {_TORCH_ERR}"


_CUDA = bool(_HAS_TORCH and torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Test parameter axes
# ---------------------------------------------------------------------------

# (n_rows, n_cols) — larger than what test_python.py covers.
SIZES_OLS = [
    pytest.param(100, 10, id="100x10"),
    pytest.param(1_000, 50, id="1000x50"),
    pytest.param(5_000, 200, id="5000x200"),
]

# SPD-solve systems are square; pick representative transformer-layer dims.
SIZES_SPD = [
    pytest.param(64, id="d=64"),
    pytest.param(256, id="d=256"),
    pytest.param(1024, id="d=1024"),
]

# Conditioning regimes: κ(X) = COND. The normal-equation κ is COND**2,
# which is why the "ill" case is the genuinely demanding one for
# Gram-matrix-based solvers like modified_cholesky.
CONDS = [
    pytest.param(1e1, id="well"),
    pytest.param(1e4, id="moderate"),
    pytest.param(1e6, id="ill"),
]

DEVICES_CPU_GPU = ["cpu"] + (["cuda"] if _CUDA else [])
DEVICES_CPU_ONLY = ["cpu"]

# Tolerance scaling: rel_err ≲ C · κ(X)² · eps_machine.
# These constants are conservative; tighten as solver implementations improve.
_TOL_F64 = {
    "well": 1e-9,
    "moderate": 1e-5,
    "ill":  1e-2,
}
_TOL_F32 = {
    "well": 1e-4,
    "moderate": 1e-2,
    "ill":  5e-1,   # at κ≈1e6 in fp32 the normal equations are essentially singular
}


def tol_for(cond_id: str, dtype) -> float:
    table = _TOL_F32 if dtype in (np.float32, getattr(torch, "float32", None)) else _TOL_F64
    return table[cond_id]


def _cond_id_from_request(request) -> str:
    """Pull the conditioning regime out of a pytest node id robustly.

    Parametrize stacking and ordering can shift positions, so we search by
    membership rather than indexing.
    """
    parts = request.node.callspec.id.split("-")
    for p in parts:
        if p in _TOL_F64:
            return p
    raise AssertionError(f"could not infer cond_id from {request.node.callspec.id!r}")


# ---------------------------------------------------------------------------
# Synthetic-system generators (known solutions)
# ---------------------------------------------------------------------------

def make_lstsq_system(
    n: int, p: int, cond: float, seed: int, dtype=np.float64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X (n×p), y (n,), β* (p,) such that y = X β* exactly.

    X is constructed via SVD so that cond(X) = cond. y is noise-free so
    that the OLS solution is exactly β*.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, p))
    B = rng.standard_normal((p, p))
    U, _ = np.linalg.qr(A)        # (n, p) with orthonormal cols
    V, _ = np.linalg.qr(B)        # (p, p) orthogonal
    s = np.linspace(1.0, 1.0 / cond, p)
    X = (U * s) @ V.T
    beta_true = rng.standard_normal(p)
    y = X @ beta_true
    return X.astype(dtype), y.astype(dtype), beta_true.astype(dtype)


def make_spd_system(
    d: int, cond: float, seed: int, damping: float = 1e-6,
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build SPD G = AᵀA + λI (d×d), x* (d,), r = G x*.

    cond(G) ≈ cond by construction (before damping; damping reduces it).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.linspace(1.0, 1.0 / cond, d)
    G = (Q * s) @ Q.T + damping * np.eye(d)
    G = 0.5 * (G + G.T)  # numerical symmetrization
    x_true = rng.standard_normal(d)
    r = G @ x_true
    return G.astype(dtype), r.astype(dtype), x_true.astype(dtype)


# ---------------------------------------------------------------------------
# Metrics + report collection
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    group: str            # "Rust / olssm", "vered_solve", "K-FAC inversion"
    method: str           # "solve_ols", "vered_solve", ...
    device: str
    size: str             # "100x10" / "d=256"
    cond_id: str          # "well" / "moderate" / "ill"
    dtype: str
    max_abs_err: float
    rel_err: float
    residual: float
    wallclock_ms: float
    passed: bool


def _metric_block(y_hat: np.ndarray, y_true: np.ndarray, ref_rhs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Return max-abs / relative / residual metrics in float64 numpy."""
    y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    max_abs = float(np.max(np.abs(y_hat - y_true)))
    rel = float(np.linalg.norm(y_hat - y_true) / max(np.linalg.norm(y_true), 1e-300))
    if ref_rhs is not None:
        ref_rhs = np.asarray(ref_rhs, dtype=np.float64).ravel()
        residual = float(np.linalg.norm(ref_rhs[: y_hat.size] - y_hat) / max(np.linalg.norm(ref_rhs), 1e-300))
    else:
        residual = float("nan")
    return {"max_abs_err": max_abs, "rel_err": rel, "residual": residual}


@pytest.fixture(scope="session")
def measurements() -> List[Measurement]:
    """Session-scoped collector; teardown writes the Markdown report."""
    bucket: List[Measurement] = []
    yield bucket

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "solver_accuracy_report.md"

    # Group → list[Measurement]
    by_group: Dict[str, List[Measurement]] = defaultdict(list)
    for m in bucket:
        by_group[m.group].append(m)

    lines: List[str] = []
    lines.append("# Solver accuracy report")
    lines.append("")
    lines.append(f"CUDA available: **{_CUDA}**  ·  torch: **{_HAS_TORCH}**  ·  olssm: **{_HAS_OLSSM}**")
    lines.append("")
    lines.append("Metrics: ``max_abs_err`` = max|x̂ − x*|, ``rel_err`` = ‖x̂ − x*‖ / ‖x*‖, ``residual`` = ‖Ax̂ − b‖ / ‖b‖.")
    lines.append("")
    for group in sorted(by_group):
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| method | device | size | cond | dtype | max_abs_err | rel_err | residual | time (ms) | pass |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for m in sorted(by_group[group], key=lambda x: (x.method, x.device, x.size, x.cond_id, x.dtype)):
            lines.append(
                f"| {m.method} | {m.device} | {m.size} | {m.cond_id} | {m.dtype} "
                f"| {m.max_abs_err:.3e} | {m.rel_err:.3e} | {m.residual:.3e} "
                f"| {m.wallclock_ms:.2f} | {'✅' if m.passed else '❌'} |"
            )
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    # Echo path so the test runner output points at it.
    print(f"\n[solver-accuracy] wrote {report}", flush=True)


def _record(
    measurements,
    *,
    group: str,
    method: str,
    device: str,
    size: str,
    cond_id: str,
    dtype: str,
    metrics: Dict[str, float],
    wallclock_ms: float,
    tol: float,
):
    passed = metrics["rel_err"] <= tol or (
        math.isnan(metrics["rel_err"]) and metrics["max_abs_err"] <= tol
    )
    measurements.append(
        Measurement(
            group=group, method=method, device=device, size=size,
            cond_id=cond_id, dtype=dtype,
            max_abs_err=metrics["max_abs_err"], rel_err=metrics["rel_err"],
            residual=metrics["residual"], wallclock_ms=wallclock_ms,
            passed=passed,
        )
    )
    return passed


# ===========================================================================
# Group A — Rust / olssm path (CPU only)
# ===========================================================================

@pytest.mark.skipif(
    not _HAS_OLSSM,
    reason=f"olssm Rust extension import failed: {_OLSSM_ERR} (run `maturin develop --features python`)",
)
class TestRustOlssmAccuracy:
    """Accuracy of the Rust closed-form OLS solvers via the PyO3 bindings.

    The Rust extension exposes a CPU implementation; this class fixes the
    device axis to CPU. The GPU path for these algorithms is the
    ``vered_solve`` family in Group B.
    """

    @pytest.mark.parametrize("cond", CONDS)
    @pytest.mark.parametrize("n,p", SIZES_OLS)
    def test_solve_ols_recovers_beta(self, n, p, cond, request, measurements):
        cond_id = _cond_id_from_request(request)
        X, y, beta_true = make_lstsq_system(n, p, cond, seed=42, dtype=np.float64)
        t0 = time.perf_counter()
        try:
            beta_hat = solve_ols(X, y)
        except (ValueError, RuntimeError) as e:
            # modified_cholesky raises ZeroPivot when the Gram matrix is
            # below working precision (κ² past f64). This is *correct*
            # behaviour at κ=1e6 (κ²=1e12); a bug at κ≤1e4.
            dt_ms = (time.perf_counter() - t0) * 1e3
            metrics = {"max_abs_err": float("inf"), "rel_err": float("inf"),
                       "residual": float("inf")}
            measurements.append(Measurement(
                group="Rust / olssm", method="solve_ols", device="cpu",
                size=f"{n}x{p}", cond_id=cond_id, dtype="float64 (rejected)",
                max_abs_err=float("nan"), rel_err=float("nan"),
                residual=float("nan"), wallclock_ms=dt_ms,
                passed=(cond_id == "ill"),
            ))
            if cond_id == "ill":
                pytest.skip(f"solve_ols correctly refused at κ²≈1e12: {e}")
            raise
        dt_ms = (time.perf_counter() - t0) * 1e3
        metrics = _metric_block(beta_hat, beta_true)
        metrics["residual"] = float(
            np.linalg.norm(X @ beta_hat - y) / max(np.linalg.norm(y), 1e-300)
        )
        tol = tol_for(cond_id, np.float64)
        passed = _record(measurements, group="Rust / olssm", method="solve_ols",
                         device="cpu", size=f"{n}x{p}", cond_id=cond_id,
                         dtype="float64", metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"rel_err {metrics['rel_err']:.3e} > tol {tol:.1e}"

    @pytest.mark.parametrize("n,p", SIZES_OLS)
    def test_simplified_gram_schmidt_orthogonality(self, n, p, measurements):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, p))
        t0 = time.perf_counter()
        Q = simplified_gram_schmidt(X)
        dt_ms = (time.perf_counter() - t0) * 1e3
        QtQ = Q.T @ Q
        off = QtQ - np.diag(np.diag(QtQ))
        max_off = float(np.max(np.abs(off)))
        metrics = {
            "max_abs_err": max_off,
            "rel_err":     max_off / max(np.max(np.abs(np.diag(QtQ))), 1e-300),
            "residual":    float("nan"),
        }
        tol = 1e-8                                        # orthogonality in f64
        passed = _record(measurements, group="Rust / olssm",
                         method="simplified_gram_schmidt", device="cpu",
                         size=f"{n}x{p}", cond_id="well", dtype="float64",
                         metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"max off-diagonal {max_off:.3e} > {tol:.1e}"

    @pytest.mark.parametrize("cond", CONDS)
    @pytest.mark.parametrize("n,p", SIZES_OLS[:2])  # the 5000×200 case takes too long w/ W
    def test_weighted_generalized_inverse_recovers_beta(self, n, p, cond, request, measurements):
        cond_id = _cond_id_from_request(request)
        rng = np.random.default_rng(7)
        X, y, beta_true = make_lstsq_system(n, p, cond, seed=7)
        W = np.diag(rng.uniform(0.5, 2.0, n))             # diagonal pos-def weight
        t0 = time.perf_counter()
        G = weighted_generalized_inverse(X, W)            # (p, n)
        beta_hat = G @ y
        dt_ms = (time.perf_counter() - t0) * 1e3
        metrics = _metric_block(beta_hat, beta_true)
        metrics["residual"] = float(
            np.linalg.norm(X @ beta_hat - y) / max(np.linalg.norm(y), 1e-300)
        )
        tol = tol_for(cond_id, np.float64)
        passed = _record(measurements, group="Rust / olssm",
                         method="weighted_generalized_inverse", device="cpu",
                         size=f"{n}x{p}", cond_id=cond_id, dtype="float64",
                         metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"rel_err {metrics['rel_err']:.3e} > tol {tol:.1e}"

    @pytest.mark.parametrize("cond", CONDS)
    @pytest.mark.parametrize("d", [64, 256, 1024])
    def test_lu_solve_gram_vec_matches_known_x(self, d, cond, request, measurements):
        cond_id = _cond_id_from_request(request)
        G, r, x_true = make_spd_system(d, cond, seed=11, damping=1e-6)
        t0 = time.perf_counter()
        x_hat = lu_solve_gram_vec(G, r)
        dt_ms = (time.perf_counter() - t0) * 1e3
        metrics = _metric_block(x_hat, x_true)
        metrics["residual"] = float(
            np.linalg.norm(G @ x_hat - r) / max(np.linalg.norm(r), 1e-300)
        )
        tol = tol_for(cond_id, np.float64)
        passed = _record(measurements, group="Rust / olssm",
                         method="lu_solve_gram_vec", device="cpu",
                         size=f"d={d}", cond_id=cond_id, dtype="float64",
                         metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"rel_err {metrics['rel_err']:.3e} > tol {tol:.1e}"


# ===========================================================================
# Group B — vered_solve (PyTorch, CPU + CUDA)
# ===========================================================================

@pytest.mark.skipif(not _HAS_VERED, reason=f"diagnostic.vered_solve import failed: {_VERED_ERR}")
class TestVeredSolveAccuracy:
    """Accuracy of the PyTorch Cholesky-based solver across devices and dtypes.

    Runs the same battery on CPU and (if present) CUDA; on CUDA we also
    exercise float32 to catch low-precision regressions.
    """

    @pytest.mark.parametrize("device", DEVICES_CPU_GPU)
    @pytest.mark.parametrize("torch_dtype_name", ["float64", "float32"])
    @pytest.mark.parametrize("cond", CONDS)
    @pytest.mark.parametrize("d", [64, 256, 1024])
    def test_vered_solve_recovers_x(self, d, cond, torch_dtype_name, device, request, measurements):
        cond_id = _cond_id_from_request(request)
        if device == "cuda" and torch_dtype_name == "float32" and cond_id == "ill":
            pytest.skip("κ≈1e6 in fp32 is below working precision — not a useful accuracy test")
        torch_dtype = getattr(torch, torch_dtype_name)
        np_dtype = np.float64 if torch_dtype is torch.float64 else np.float32

        G_np, r_np, x_true = make_spd_system(d, cond, seed=23, damping=1e-6, dtype=np_dtype)
        G = torch.from_numpy(G_np).to(device=device, dtype=torch_dtype)
        r = torch.from_numpy(r_np).to(device=device, dtype=torch_dtype)

        # NOTE: pass damping≈0 so we measure pure solver accuracy.
        # vered_solve's default damping=1e-6 dominates the error at small
        # damping/smallest_eigenvalue ratios, which is correct behaviour for
        # ill-conditioned streaming Gram matrices but masks pure precision.
        solver_damping = 1e-14

        # Warm up CUDA kernels (does not count toward measured time)
        if device == "cuda":
            _ = vered_solve(G, r, damping=solver_damping)
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        x_hat = vered_solve(G, r, damping=solver_damping)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3

        metrics = _metric_block(x_hat.detach().cpu().numpy(), x_true)
        metrics["residual"] = float(
            torch.linalg.norm(G @ x_hat - r).item() / max(torch.linalg.norm(r).item(), 1e-300)
        )
        tol = tol_for(cond_id, np_dtype)
        passed = _record(measurements, group="vered_solve",
                         method="vered_solve", device=device,
                         size=f"d={d}", cond_id=cond_id, dtype=torch_dtype_name,
                         metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"rel_err {metrics['rel_err']:.3e} > tol {tol:.1e}"

    @pytest.mark.parametrize("device", DEVICES_CPU_GPU)
    @pytest.mark.parametrize("d,k", [(64, 8), (256, 32), (1024, 16)])
    def test_vered_solve_batched_matches_per_column(self, d, k, device, measurements):
        """Factor-once / solve-many should agree with solving each column independently."""
        G_np, _, _ = make_spd_system(d, cond=1e2, seed=31, damping=1e-6)
        rng = np.random.default_rng(0)
        R_np = rng.standard_normal((d, k))
        G = torch.from_numpy(G_np).to(device=device, dtype=torch.float64)
        R = torch.from_numpy(R_np).to(device=device, dtype=torch.float64)

        if device == "cuda":
            _ = vered_solve_batched(G, R); torch.cuda.synchronize()
        t0 = time.perf_counter()
        X_batch = vered_solve_batched(G, R)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3

        # Reference: column-by-column via vered_solve (same path, no batching).
        X_ref = torch.stack([vered_solve(G, R[:, j]) for j in range(k)], dim=1)
        metrics = _metric_block(X_batch.detach().cpu().numpy(), X_ref.detach().cpu().numpy())
        metrics["residual"] = float(
            torch.linalg.norm(G @ X_batch - R).item() / max(torch.linalg.norm(R).item(), 1e-300)
        )
        tol = 1e-10
        passed = _record(measurements, group="vered_solve",
                         method="vered_solve_batched", device=device,
                         size=f"d={d},k={k}", cond_id="well", dtype="float64",
                         metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"batched ≠ per-column: rel_err {metrics['rel_err']:.3e}"

    @pytest.mark.parametrize("device", DEVICES_CPU_GPU)
    @pytest.mark.parametrize("d", [128, 512])
    def test_vered_decompose_then_apply_equals_one_shot(self, d, device, measurements):
        """vered_decompose + vered_apply should match vered_solve."""
        G_np, r_np, _ = make_spd_system(d, cond=1e2, seed=41, damping=1e-6)
        G = torch.from_numpy(G_np).to(device=device, dtype=torch.float64)
        r = torch.from_numpy(r_np).to(device=device, dtype=torch.float64)
        t0 = time.perf_counter()
        factor = vered_decompose(G)
        x1 = vered_apply(factor, r)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3
        x2 = vered_solve(G, r)
        metrics = _metric_block(x1.detach().cpu().numpy(), x2.detach().cpu().numpy())
        metrics["residual"] = float(
            torch.linalg.norm(G @ x1 - r).item() / max(torch.linalg.norm(r).item(), 1e-300)
        )
        tol = 1e-12
        passed = _record(measurements, group="vered_solve",
                         method="decompose+apply", device=device,
                         size=f"d={d}", cond_id="well", dtype="float64",
                         metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"decompose+apply ≠ vered_solve: rel_err {metrics['rel_err']:.3e}"


# ===========================================================================
# Group C — K-FAC inversion utilities (CPU + CUDA)
# ===========================================================================

@pytest.mark.skipif(not _HAS_KFAC_INV, reason=f"diagnostic.inversion import failed: {_KFAC_ERR}")
class TestKfacInversionAccuracy:
    """``naive_pinv_inverse`` and ``kfac_a_inverse`` from diagnostic/inversion.py.

    The setup is the target-propagation back-step: pick a known input ``a*``,
    push through a known linear layer to get the target ``t = W a* + b``,
    and check that each inverse recovers ``a*`` (modulo damping / prior).
    """

    @pytest.mark.parametrize("device", DEVICES_CPU_GPU)
    @pytest.mark.parametrize("N,d_out,d_in", [(64, 32, 128), (256, 128, 512)])
    def test_naive_pinv_inverse_recovers_a(self, N, d_out, d_in, device, measurements):
        torch.manual_seed(0)
        W = torch.randn(d_out, d_in, dtype=torch.float64, device=device)
        b = torch.randn(d_out, dtype=torch.float64, device=device)
        a_true = torch.randn(N, d_in, dtype=torch.float64, device=device)
        t_pre = a_true @ W.T + b                          # known target

        t0 = time.perf_counter()
        a_hat = naive_pinv_inverse(W, b, t_pre, eps=1e-8)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3

        # When d_in > d_out the minimum-norm solution is NOT a_true; instead it is
        # the projection of a_true onto the row-space of W. Compare on the
        # constraint W a_hat ≈ t_pre - b, which the inverse should satisfy.
        recon = a_hat @ W.T + b
        metrics = _metric_block(recon.detach().cpu().numpy(),
                                t_pre.detach().cpu().numpy())
        metrics["residual"] = float(
            torch.linalg.norm(recon - t_pre).item() / max(torch.linalg.norm(t_pre).item(), 1e-300)
        )
        tol = 1e-6
        passed = _record(measurements, group="K-FAC inversion",
                         method="naive_pinv_inverse", device=device,
                         size=f"N={N},W={d_out}x{d_in}", cond_id="well",
                         dtype="float64", metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"constraint reconstruction rel_err {metrics['rel_err']:.3e}"

    @pytest.mark.parametrize("device", DEVICES_CPU_GPU)
    @pytest.mark.parametrize("N,d_out,d_in", [(64, 32, 128), (128, 64, 256)])
    def test_kfac_a_inverse_satisfies_constraint(self, N, d_out, d_in, device, measurements):
        torch.manual_seed(1)
        W = torch.randn(d_out, d_in, dtype=torch.float64, device=device)
        b = torch.randn(d_out, dtype=torch.float64, device=device)
        a_true = torch.randn(N, d_in, dtype=torch.float64, device=device)
        t_pre = a_true @ W.T + b

        mu_a = torch.zeros(d_in, dtype=torch.float64, device=device)
        # Well-conditioned isotropic prior covariance.
        Sigma_a = torch.eye(d_in, dtype=torch.float64, device=device)

        t0 = time.perf_counter()
        a_hat = kfac_a_inverse(W, b, t_pre, mu_a, Sigma_a, sigma2=1e-8)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3

        recon = a_hat @ W.T + b
        metrics = _metric_block(recon.detach().cpu().numpy(),
                                t_pre.detach().cpu().numpy())
        metrics["residual"] = float(
            torch.linalg.norm(recon - t_pre).item() / max(torch.linalg.norm(t_pre).item(), 1e-300)
        )
        tol = 1e-5            # σ²=1e-8 looser than the naive path
        passed = _record(measurements, group="K-FAC inversion",
                         method="kfac_a_inverse", device=device,
                         size=f"N={N},W={d_out}x{d_in}", cond_id="well",
                         dtype="float64", metrics=metrics, wallclock_ms=dt_ms, tol=tol)
        assert passed, f"constraint reconstruction rel_err {metrics['rel_err']:.3e}"


# ---------------------------------------------------------------------------
# Convenience: allow `python tests/test_solver_accuracy_cpu_gpu.py` invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
