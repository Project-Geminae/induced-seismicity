"""
Logistic Lasso via IRLS + Gram-based coordinate descent (GPU).

Problem:
    minimize_β  (1/n) Σ -[y log(p) + (1-y) log(1-p)]  +  λ ||β||_1
    where p = σ(X β + β_0)

Algorithm (IRLS with weighted-Lasso inner solve, glmnet-style):

    Repeat until convergence:
        η  = X β + β_0
        p  = σ(η)
        w  = p (1-p)                 (working weights)
        z  = η + (y - p) / w         (working response)
        Solve weighted Lasso:
            β ← argmin (1/n) Σ w_i (z_i - X_i β)^2 / 2 + λ ||β||_1
        β_0 ← (Σ w (z - X β)) / Σ w  (weighted intercept)

The weighted Lasso reduces to gaussian Lasso on transformed data:
    X' = sqrt(w) ⊙ X  (rescale rows by √w_i)
    z' = sqrt(w) z
    β solves    (1/n) ||z' - X' β||² / 2 + λ ||β||_1

For the Gram form:
    G_w = X^T diag(w) X / n
    b_w = X^T diag(w) z / n  −  β_0 X^T diag(w) 1 / n  (intercept correction)

Each IRLS iter requires recomputing G_w (sparse-sparse mult with row weights).
We minimize this cost by warm-starting β across iterations, so most IRLS
iters need only a few CD sweeps to re-converge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import cd_gram

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False


@dataclass
class LogisticResult:
    beta:         np.ndarray
    intercept:    float
    n_irls:       int
    n_total_cd:   int
    converged:    bool
    final_loglik: float


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _build_weighted_gram_cpu(X_csr, w: np.ndarray):
    """Compute G_w = X^T diag(w) X / n on CPU. Returns dense p × p numpy."""
    from scipy.sparse import diags
    n, p = X_csr.shape
    # X_w_rows = sqrt(w) ⊙ X (row-scaled X)
    sw = np.sqrt(np.maximum(w, 1e-12))
    X_w = diags(sw) @ X_csr
    G = (X_w.T @ X_w).toarray().astype(np.float32) / float(n)
    return G


def _weighted_xty_cpu(X_csr, w: np.ndarray, z: np.ndarray):
    """Compute b_w = X^T diag(w) z / n. Returns 1D numpy."""
    n = X_csr.shape[0]
    return (X_csr.T @ (w * z)).astype(np.float32) / float(n)


def logistic_lasso(
    X_csr,
    y: np.ndarray,
    lam: float,
    max_irls: int = 25,
    irls_tol: float = 1e-6,
    cd_max_sweeps: int = 200,
    cd_tol: float = 1e-7,
    beta0: Optional[np.ndarray] = None,
    intercept0: float = 0.0,
    fit_intercept: bool = True,
    verbose: bool = False,
) -> LogisticResult:
    """Solve logistic Lasso via IRLS + Gram CD."""
    if not HAS_JAX:
        raise RuntimeError("JAX required.")

    n, p = X_csr.shape
    y = np.asarray(y, dtype=np.float64)

    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=np.float64).copy()
    intercept = float(intercept0)
    total_cd_sweeps = 0
    converged = False

    prev_loglik = -np.inf

    for irls_iter in range(1, max_irls + 1):
        # Linear predictor
        eta = X_csr @ beta + intercept
        p_hat = _sigmoid(eta)
        # Working weights and response
        w = np.maximum(p_hat * (1.0 - p_hat), 1e-6)  # floor for numerical stability
        z = eta + (y - p_hat) / w

        # Weighted Gram + RHS (move y_w to centered form by removing intercept)
        # If we fit intercept, center z by weighted mean
        if fit_intercept:
            wmean = float(np.sum(w * z) / np.sum(w))
            z_c = z - wmean
        else:
            wmean = 0.0
            z_c = z
        G_w = _build_weighted_gram_cpu(X_csr, w)
        b_w = _weighted_xty_cpu(X_csr, w, z_c)

        # CD on (G_w, b_w) — warm-start from current β
        G_j = jnp.asarray(G_w)
        b_j = jnp.asarray(b_w)
        beta_j = jnp.asarray(beta.astype(np.float32))
        cd_res = cd_gram.cd_lasso_gram(
            G=G_j, Xty=b_j, lam=float(lam),
            max_sweeps=cd_max_sweeps, tol=cd_tol,
            beta0=beta_j, verbose=False,
        )
        beta_new = np.asarray(cd_res.beta).astype(np.float64)

        # Recompute intercept as weighted residual mean
        if fit_intercept:
            res = z - X_csr @ beta_new
            new_intercept = float(np.sum(w * res) / np.sum(w))
        else:
            new_intercept = 0.0

        # Check IRLS convergence via change in negative log-likelihood
        eta_new = X_csr @ beta_new + new_intercept
        p_new = _sigmoid(eta_new)
        loglik = float(np.sum(y * np.log(np.clip(p_new, 1e-12, 1.0))
                              + (1 - y) * np.log(np.clip(1.0 - p_new, 1e-12, 1.0))))
        delta_beta = float(np.max(np.abs(beta_new - beta)))
        delta_int = abs(new_intercept - intercept)

        beta = beta_new
        intercept = new_intercept
        total_cd_sweeps += cd_res.n_sweeps

        if verbose:
            n_active = int(np.sum(np.abs(beta) > 1e-10))
            print(f"  IRLS iter {irls_iter}: loglik = {loglik:+.4e}, "
                  f"|Δβ|_∞ = {delta_beta:.2e}, intercept = {intercept:.3e}, "
                  f"active = {n_active}, CD sweeps this iter = {cd_res.n_sweeps}",
                  flush=True)

        if irls_iter >= 2 and delta_beta < irls_tol and delta_int < irls_tol:
            converged = True
            break
        prev_loglik = loglik

    return LogisticResult(
        beta=beta,
        intercept=intercept,
        n_irls=irls_iter,
        n_total_cd=total_cd_sweeps,
        converged=converged,
        final_loglik=loglik,
    )


def logistic_lasso_path(
    X_csr,
    y: np.ndarray,
    lambdas: np.ndarray,
    max_irls: int = 25,
    irls_tol: float = 1e-6,
    cd_max_sweeps: int = 200,
    cd_tol: float = 1e-7,
    fit_intercept: bool = True,
    verbose: bool = False,
):
    """λ-path with warm starts (decreasing λ)."""
    results = []
    beta_warm = None
    int_warm = 0.0
    for i, lam in enumerate(lambdas):
        res = logistic_lasso(
            X_csr, y, lam=float(lam),
            max_irls=max_irls, irls_tol=irls_tol,
            cd_max_sweeps=cd_max_sweeps, cd_tol=cd_tol,
            beta0=beta_warm, intercept0=int_warm,
            fit_intercept=fit_intercept, verbose=False,
        )
        results.append(res)
        beta_warm = res.beta
        int_warm = res.intercept
        if verbose:
            n_active = int(np.sum(np.abs(res.beta) > 1e-10))
            print(f"  λ[{i}]={lam:.3e}: {res.n_irls} IRLS, {res.n_total_cd} CD sweeps, "
                  f"loglik={res.final_loglik:+.3e}, active={n_active}", flush=True)
    return results
