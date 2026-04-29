"""Validate active-set IRLS against the full-Gram baseline.

At convergence, both solvers must produce the same (β, intercept) up
to numerical tolerance. The active-set solver should also be faster
once the active set is much smaller than p.

Run:
    pytest gpu_hal/tests/test_active_set_vs_full.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Provide a no-op @pytest.mark.parametrize so the test file still runs
    # under plain `python -m`. The __main__ block below loops over the same
    # parameter list explicitly.
    class _PytestStub:
        class mark:
            @staticmethod
            def parametrize(*_args, **_kwargs):
                def decorator(fn):
                    return fn
                return decorator
    pytest = _PytestStub()  # type: ignore
from scipy.sparse import csr_matrix


def _make_synthetic(n=2000, p=200, density=0.20, n_signal=10, seed=42):
    rng = np.random.default_rng(seed)
    # Sparse 0/1 indicator-like features
    mask = rng.random((n, p)) < density
    X = csr_matrix(mask.astype(np.float64))
    # True coefs: only n_signal columns active
    beta_true = np.zeros(p)
    signal_idx = rng.choice(p, n_signal, replace=False)
    beta_true[signal_idx] = rng.standard_normal(n_signal) * 1.5
    eta = X @ beta_true + 0.5
    p_true = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < p_true).astype(np.float64)
    return X, y, beta_true


@pytest.mark.parametrize("lam", [0.05, 0.01, 0.005])
def test_active_set_matches_full_gram(lam):
    """Active-set IRLS should produce the same β as full-Gram IRLS at λ."""
    from gpu_hal.cd_logistic import logistic_lasso
    from gpu_hal.cd_logistic_active_set import logistic_lasso_active_set

    X, y, _ = _make_synthetic(n=1500, p=120, density=0.20, n_signal=8)

    full = logistic_lasso(
        X, y, lam, max_irls=30, irls_tol=1e-8,
        cd_max_sweeps=500, cd_tol=1e-9, fit_intercept=True, verbose=False,
    )
    actset = logistic_lasso_active_set(
        X, y, lam, max_irls=30, irls_tol=1e-8,
        cd_max_sweeps=500, cd_tol=1e-9, kkt_tol=1e-5,
        initial_full_irls=2, fit_intercept=True, verbose=False,
    )

    # Coefficient agreement (rel L2)
    rel_l2 = np.linalg.norm(full.beta - actset.beta) / max(np.linalg.norm(full.beta), 1e-12)
    int_diff = abs(full.intercept - actset.intercept)
    print(f"  λ={lam}: full active={int(np.sum(np.abs(full.beta) > 1e-10))}, "
          f"as active={int(np.sum(np.abs(actset.beta) > 1e-10))}, "
          f"rel-L2={rel_l2:.3e}, int-diff={int_diff:.3e}")

    # Active sets should match
    full_active  = set(np.where(np.abs(full.beta)   > 1e-9)[0].tolist())
    as_active    = set(np.where(np.abs(actset.beta) > 1e-9)[0].tolist())
    sym_diff = (full_active ^ as_active)
    assert len(sym_diff) == 0, (
        f"active-set mismatch at λ={lam}: full={sorted(full_active)} "
        f"vs as={sorted(as_active)}, symmetric diff = {sorted(sym_diff)}"
    )

    # Coefficient values should agree to a few digits
    assert rel_l2 < 1e-3, f"rel-L2 = {rel_l2:.3e} > 1e-3 at λ={lam}"
    assert int_diff < 1e-3, f"intercept diff = {int_diff:.3e} > 1e-3 at λ={lam}"


if __name__ == "__main__":
    for lam in [0.05, 0.01, 0.005]:
        test_active_set_matches_full_gram(lam)
        print(f"✓ λ={lam} passed")
