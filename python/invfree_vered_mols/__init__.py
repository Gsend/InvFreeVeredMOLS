"""
InvFreeVeredMOLS — Inversion-Free Vered Method for OLS-Stable K-FAC

Public API
----------
- OlsSmKFAC                 : the SM K-FAC optimizer (PyTorch torch.optim.Optimizer)
- GramMatrixEstimator       : utility for incremental Gram estimation under bf16
- KFACHooks                 : forward/backward hook wrapper used by the optimizer
- ConfigurationError        : raised on invalid optimizer config
- backend module            : low-level eigh / apply primitives (auto-selects
                              Rust olssm backend if compiled, else PyTorch fallback)

Usage
-----
    from invfree_vered_mols import OlsSmKFAC
    optimizer = OlsSmKFAC(model, lr=1e-3, damping=1e-2)
    # standard PyTorch train loop:
    loss.backward()
    optimizer.step()

For the Rust accelerated backend, see scripts/build_rust.ps1 (Windows) or
scripts/build_rust.sh (Linux/macOS) to compile the `olssm` cdylib and place
it on PYTHONPATH.
"""
from .olssm_kfac import OlsSmKFAC
from .gram_estimator import GramMatrixEstimator
from .kfac_hooks import KFACHooks
from .errors import ConfigurationError
from . import backend
from . import bf16_linalg

__version__ = "0.1.0"
__all__ = [
    "OlsSmKFAC",
    "GramMatrixEstimator",
    "KFACHooks",
    "ConfigurationError",
    "backend",
    "bf16_linalg",
]
