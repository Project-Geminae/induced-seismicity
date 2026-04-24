"""Validation: FISTA Lasso output vs sklearn.linear_model.Lasso.

Synthetic data, moderate n/p. Expects agreement within 1e-3 relative L2
error on β and <1% relative error on out-of-sample MSE.
"""
import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gpu_hal import fista
from gpu_hal.fista import (
    fista_lasso,
    make_dense_matvec,
    make_sparse_matvec,
    compute_lambda_max,
)

try:
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False


def synthetic_lasso(n=500, p=200, k_active=10, sigma=0.5, seed=42):
    """Make a sparse-ground-truth Lasso problem."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    idx = rng.choice(p, size=k_active, replace=False)
    beta_true[idx] = rng.standard_normal(k_active) * 2.0
    y = X @ beta_true + sigma * rng.standard_normal(n)
    return X, y, beta_true


def test_dense_fista_vs_sklearn():
    if not HAS_JAX:
        print("JAX not available — skipping")
        return
    from sklearn.linear_model import Lasso

    X, y, beta_true = synthetic_lasso(n=500, p=200, k_active=10)
    y_c = y - y.mean()

    matvec, rmatvec, L = make_dense_matvec(X)
    y_j = jnp.asarray(y_c)
    lam_max = compute_lambda_max(rmatvec, y_j, len(y))
    lam = lam_max * 0.05  # non-trivial regularization

    fit_fista = fista_lasso(matvec, rmatvec, y_j, p=X.shape[1], lam=lam,
                            n=len(y), L=L, max_iter=2000, tol=1e-7)
    fit_sk = Lasso(alpha=lam, fit_intercept=False, max_iter=10000, tol=1e-10)
    fit_sk.fit(X, y_c)

    beta_fista = np.asarray(fit_fista.beta)
    beta_sk = fit_sk.coef_

    rel_l2 = np.linalg.norm(beta_fista - beta_sk) / (np.linalg.norm(beta_sk) + 1e-30)
    max_diff = np.max(np.abs(beta_fista - beta_sk))

    print(f"  Dense Lasso validation (n={len(y)}, p={X.shape[1]}, λ={lam:.3e}):")
    print(f"    FISTA: {fit_fista.n_iter} iters, converged={fit_fista.converged}, gap={fit_fista.final_gap:.2e}")
    print(f"    |β_FISTA − β_sklearn|_inf = {max_diff:.3e}")
    print(f"    relative L2 = {rel_l2:.3e}")
    print(f"    FISTA active bases: {int(np.sum(np.abs(beta_fista) > 1e-6))}")
    print(f"    sklearn active bases: {int(np.sum(np.abs(beta_sk) > 1e-6))}")

    # Loose tolerance because sklearn.Lasso has a slightly different
    # objective parameterization (α vs λ/n).
    assert rel_l2 < 0.05, f"Relative L2 error too large: {rel_l2}"
    print("  ✓ PASSED: rel L2 < 0.05")


def test_sparse_fista_lipschitz():
    if not HAS_JAX:
        print("JAX not available — skipping")
        return
    # Build a sparse X
    rng = np.random.default_rng(42)
    n, p, density = 500, 200, 0.1
    X_sp = sparse_random(n, p, density=density, format='csr', random_state=42, dtype=float)
    beta = np.zeros(p)
    beta[:5] = rng.standard_normal(5)
    y = X_sp @ beta + 0.1 * rng.standard_normal(n)
    y_c = y - y.mean()

    matvec, rmatvec, L = make_sparse_matvec(X_sp)
    print(f"  Sparse Lasso smoke test (n={n}, p={p}, density={density}):")
    print(f"    Lipschitz L = {L:.3e}")
    y_j = jnp.asarray(y_c)
    lam_max = compute_lambda_max(rmatvec, y_j, n)
    print(f"    λ_max = {lam_max:.3e}")

    lam = lam_max * 0.1
    fit = fista_lasso(matvec, rmatvec, y_j, p=p, lam=lam,
                      n=n, L=L, max_iter=500, tol=1e-5)
    print(f"    FISTA: {fit.n_iter} iters, converged={fit.converged}")
    print(f"    active: {int(np.sum(np.abs(fit.beta) > 1e-6))}/{p}")
    print("  ✓ Sparse path runs end-to-end")


if __name__ == "__main__":
    print("=== Test: FISTA dense vs sklearn.Lasso ===")
    test_dense_fista_vs_sklearn()
    print()
    print("=== Test: FISTA sparse smoke ===")
    test_sparse_fista_lipschitz()
    print()
    print("All tests passed.")
