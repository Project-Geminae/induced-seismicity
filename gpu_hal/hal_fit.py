"""
End-to-end GPU HAL: build basis via hal9001 (R), solve Lasso path on GPU
via FISTA, select λ via CV, support undersmoothing.

Public API:
    fit = fit_hal_gpu(X, y, ...)
    pred = fit.predict(X_new)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import backend as bk
from . import cv as cv_mod
from . import fista


@dataclass
class GPUHALFit:
    """Result of fit_hal_gpu."""
    # Configuration
    max_degree:         int
    num_knots:          tuple
    smoothness_orders:  int
    n_lambdas:          int
    n_folds:            int

    # Data dimensions
    n:                  int
    p:                  int          # number of basis functions

    # λ selection
    lambdas:            np.ndarray   # (n_lambdas,) decreasing
    cv_mean_mse:        np.ndarray
    cv_sem_mse:         np.ndarray
    lambda_cv:          float        # argmin λ
    lambda_1se:         float        # 1-SE rule λ
    lambda_used:        float        # the λ we used for the final fit

    # Final fit
    beta:               np.ndarray   # (p,) coefficients at lambda_used
    intercept:          float        # y_mean used during centering
    active_idx:         np.ndarray   # indices of nonzero coefficients

    # Book-keeping
    basis_list:         object       # R object for prediction at new data
    fit_time_sec:       float
    undersmoothing:     float        # factor applied to λ_cv (1.0 if no undersmoothing)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        phi = bk.apply_basis(X_new, self.basis_list)
        return phi @ self.beta + self.intercept

    @property
    def coef_active(self) -> np.ndarray:
        return self.beta[self.active_idx]


def fit_hal_gpu(
    X: np.ndarray,
    y: np.ndarray,
    max_degree: int = 2,
    num_knots: tuple = (25, 10),
    smoothness_orders: int = 1,
    lambda_grid: Optional[np.ndarray] = None,
    n_lambdas: int = 50,
    lambda_ratio: float = 1e-3,
    n_folds: int = 10,
    undersmoothing: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
) -> GPUHALFit:
    """Fit HAL with Lasso solved on GPU via FISTA.

    Parameters
    ----------
    X : array (n, d)
    y : array (n,)
    max_degree, num_knots, smoothness_orders : HAL configuration
    lambda_grid : optional pre-specified λ grid (decreasing)
    n_lambdas : length of auto-generated λ grid
    lambda_ratio : λ_min / λ_max for auto grid
    n_folds : CV folds
    undersmoothing : factor < 1 to shrink λ_cv toward smaller regularization.
                     None → use λ_cv as-is. Common choice: 1.0/sqrt(log(n)).
    """
    t0 = time.time()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    # ── Step 1: build basis (R/hal9001) ─────────────────────────────
    if verbose:
        print(f"[GPU-HAL] Building basis on n={n}, d={d}, max_degree={max_degree}, num_knots={num_knots}...")
    phi_csr, basis_list = bk.build_hal_basis(
        X, max_degree=max_degree, num_knots=num_knots,
        smoothness_orders=smoothness_orders,
    )
    p = phi_csr.shape[1]

    # Scale each basis column to unit L2 norm. This is a no-op for pure
    # indicator HAL (smoothness_orders=0) but essential when smoothness=1
    # produces splines that can take values up to the raw covariate scale
    # (e.g., cumulative volume up to 1e8 BBL). Without this, λ selection
    # is dominated by the largest-scale bases and all small-scale bases
    # are zeroed out.
    from scipy.sparse import diags
    col_norms = np.sqrt(np.asarray(phi_csr.multiply(phi_csr).sum(axis=0)).ravel())
    col_norms = np.maximum(col_norms, 1e-12)
    D_inv = diags(1.0 / col_norms)
    phi_csr = phi_csr @ D_inv
    phi_csr = phi_csr.tocsr()
    # We'll unscale β at the end.
    if verbose:
        print(f"[GPU-HAL] Scaled bases to unit L2 norm (max col-norm before scaling: {col_norms.max():.3e})")
    if verbose:
        print(f"[GPU-HAL] Basis matrix: ({n} × {p}), nnz = {phi_csr.nnz}, density = {phi_csr.nnz/(n*p):.2%}")

    # ── Step 2: CV λ selection via Gram-based coordinate descent ────
    # CD converges in O(log 1/ε) iterations vs FISTA's O(1/sqrt(ε)) — for
    # sparse problems with very small λ, CD is dramatically faster and
    # produces glmnet-equivalent solutions.
    from . import cv_cd, fista_gram, cd_gram
    if verbose:
        print(f"[GPU-HAL] k={n_folds}-fold CV λ selection (Gram-CD, n_lambdas={n_lambdas})...")
    y_mean = float(y.mean())
    y_centered = y - y_mean
    cv_res = cv_cd.cv_cd_gram_sparse(
        phi_csr, y, lambdas=lambda_grid,
        n_lambdas=n_lambdas, ratio=lambda_ratio,
        n_folds=n_folds, max_sweeps=max_iter, tol=tol, seed=seed,
        verbose=verbose,
    )
    if verbose:
        print(f"[GPU-HAL] λ_cv = {cv_res.lambda_cv:.4e},  λ_1se = {cv_res.lambda_1se:.4e}")

    # ── Step 3: final fit at λ_used (with optional undersmoothing) ─
    factor = 1.0 if undersmoothing is None else float(undersmoothing)
    lambda_used = cv_res.lambda_cv * factor
    if verbose:
        print(f"[GPU-HAL] Final fit at λ = {lambda_used:.4e} (undersmoothing factor = {factor:.3f})")
    G_full, Xty_full = fista_gram.compute_gram(phi_csr, y_centered)
    import jax.numpy as jnp
    G_j = jnp.asarray(G_full)
    Xty_j = jnp.asarray(Xty_full)
    final_fit = cd_gram.cd_lasso_gram(
        G=G_j, Xty=Xty_j, lam=lambda_used,
        max_sweeps=max_iter, tol=tol, verbose=False,
    )

    # Unscale β to the original (unnormalized) basis
    beta_final = np.asarray(final_fit.beta) / col_norms

    elapsed = time.time() - t0
    if verbose:
        active = int(np.sum(np.abs(beta_final) > 0))
        n_iters = getattr(final_fit, "n_sweeps", getattr(final_fit, "n_iter", 0))
        print(f"[GPU-HAL] Final fit: {n_iters} CD sweeps, {active} active bases, "
              f"gap={final_fit.final_gap:.2e} ({elapsed:.0f}s total)")

    active_idx = np.where(np.abs(beta_final) > 0)[0]

    return GPUHALFit(
        max_degree=max_degree,
        num_knots=num_knots,
        smoothness_orders=smoothness_orders,
        n_lambdas=len(cv_res.lambdas),
        n_folds=n_folds,
        n=n,
        p=p,
        lambdas=cv_res.lambdas,
        cv_mean_mse=cv_res.mean_mse,
        cv_sem_mse=cv_res.sem_mse,
        lambda_cv=cv_res.lambda_cv,
        lambda_1se=cv_res.lambda_1se,
        lambda_used=lambda_used,
        beta=beta_final,
        intercept=y_mean,
        active_idx=active_idx,
        basis_list=basis_list,
        fit_time_sec=elapsed,
        undersmoothing=factor,
    )
