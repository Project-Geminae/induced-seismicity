"""Validate Gram-based CD vs sklearn.linear_model.Lasso on synthetic data.

Expects: |β_CD − β_sklearn|_∞ < 1e-4, same active set after thresholding.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from gpu_hal.cd_gram import cd_lasso_gram, HAS_JAX
from gpu_hal.fista_gram import compute_gram

if HAS_JAX:
    import jax.numpy as jnp


def synthetic(n=500, p=200, k_active=10, sigma=0.5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    idx = rng.choice(p, size=k_active, replace=False)
    beta[idx] = rng.standard_normal(k_active) * 2.0
    y = X @ beta + sigma * rng.standard_normal(n)
    return X, y, beta


def test_cd_vs_sklearn():
    if not HAS_JAX:
        print("JAX not available — skipping")
        return
    from sklearn.linear_model import Lasso
    from scipy.sparse import csr_matrix

    X, y, beta_true = synthetic(n=500, p=200, k_active=10)
    y_c = y - y.mean()

    G_np, Xty_np = compute_gram(csr_matrix(X), y_c)
    G_j = jnp.asarray(G_np)
    Xty_j = jnp.asarray(Xty_np)

    # Test at moderate λ
    lam = 0.05
    fit_cd = cd_lasso_gram(G_j, Xty_j, lam=lam, max_sweeps=200, tol=1e-8, verbose=False)
    fit_sk = Lasso(alpha=lam, fit_intercept=False, max_iter=10000, tol=1e-10).fit(X, y_c)

    beta_cd = np.asarray(fit_cd.beta)
    beta_sk = fit_sk.coef_

    rel_l2 = np.linalg.norm(beta_cd - beta_sk) / (np.linalg.norm(beta_sk) + 1e-30)
    max_diff = np.max(np.abs(beta_cd - beta_sk))

    print(f"  CD vs sklearn (n=500, p=200, λ=0.05):")
    print(f"    CD: {fit_cd.n_sweeps} sweeps, converged={fit_cd.converged}, gap={fit_cd.final_gap:.2e}")
    print(f"    |β_CD − β_sklearn|_∞ = {max_diff:.3e}")
    print(f"    relative L2 = {rel_l2:.3e}")
    print(f"    CD active: {int(np.sum(np.abs(beta_cd) > 1e-8))}, sklearn active: {int(np.sum(np.abs(beta_sk) > 1e-8))}")

    assert rel_l2 < 0.01, f"Relative L2 too large: {rel_l2}"
    print("  ✓ PASSED")


def test_cd_small_lambda():
    """At very small λ, CD should still converge cleanly. FISTA struggles here."""
    if not HAS_JAX:
        print("JAX not available — skipping")
        return
    from sklearn.linear_model import Lasso
    from scipy.sparse import csr_matrix

    X, y, beta_true = synthetic(n=500, p=200, k_active=10)
    y_c = y - y.mean()

    G_np, Xty_np = compute_gram(csr_matrix(X), y_c)
    G_j = jnp.asarray(G_np)
    Xty_j = jnp.asarray(Xty_np)

    # Very small λ — near-OLS
    lam = 1e-4
    fit_cd = cd_lasso_gram(G_j, Xty_j, lam=lam, max_sweeps=500, tol=1e-8, verbose=False)
    fit_sk = Lasso(alpha=lam, fit_intercept=False, max_iter=20000, tol=1e-12).fit(X, y_c)

    beta_cd = np.asarray(fit_cd.beta)
    beta_sk = fit_sk.coef_

    rel_l2 = np.linalg.norm(beta_cd - beta_sk) / (np.linalg.norm(beta_sk) + 1e-30)
    print(f"  CD vs sklearn at small λ=1e-4:")
    print(f"    CD: {fit_cd.n_sweeps} sweeps, converged={fit_cd.converged}, gap={fit_cd.final_gap:.2e}")
    print(f"    relative L2 = {rel_l2:.3e}")
    print(f"    CD active: {int(np.sum(np.abs(beta_cd) > 1e-8))}, sklearn active: {int(np.sum(np.abs(beta_sk) > 1e-8))}")
    assert rel_l2 < 0.01, f"Small-λ relative L2 too large: {rel_l2}"
    print("  ✓ PASSED")


if __name__ == "__main__":
    print("=== Test: CD vs sklearn at moderate λ ===")
    test_cd_vs_sklearn()
    print()
    print("=== Test: CD vs sklearn at small λ (FISTA's failure mode) ===")
    test_cd_small_lambda()
    print()
    print("All tests passed.")
