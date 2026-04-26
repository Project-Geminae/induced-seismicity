"""Hurdle-HAL on GPU: combines logistic + gaussian HAL into a hurdle estimator.

Mirrors `undersmoothed_hal.HurdleHAL` (which uses CPU hal9001), but uses
our GPU CD pipeline for both stages.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import backend, cd_gram, cd_logistic, fista_gram


@dataclass
class HurdleHALFit:
    """Fitted hurdle HAL with both stages."""
    # Stage 1 (logistic):
    beta_pos:        np.ndarray
    intercept_pos:   float
    lambda_pos:      float
    n_active_pos:    int
    # Stage 2 (gaussian on positives):
    beta_mag:        np.ndarray
    intercept_mag:   float
    lambda_mag:      float
    n_active_mag:    int
    # Common:
    basis_list:      object
    col_norms:       np.ndarray
    n:               int
    p:               int
    fit_time_sec:    float = 0.0
    notes:           dict = field(default_factory=dict)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Compose hurdle prediction: Q(x) = P(Y>0|x) * expm1(E[log(1+Y)|Y>0,x])."""
        from scipy.sparse import diags
        phi = backend.apply_basis(X_new, self.basis_list).tocsr()
        # Apply same column scaling as during fit
        D_inv = diags(1.0 / self.col_norms)
        phi_scaled = (phi @ D_inv).tocsr()
        # Stage 1: logistic prediction
        eta_pos = phi_scaled @ self.beta_pos + self.intercept_pos
        p_pos = 1.0 / (1.0 + np.exp(-np.clip(eta_pos, -50.0, 50.0)))
        # Stage 2: log-magnitude prediction (only meaningful when scaled)
        log_mag = phi_scaled @ self.beta_mag + self.intercept_mag
        return p_pos * np.expm1(log_mag)


def fit_hurdle_hal_gpu(
    X: np.ndarray,
    y: np.ndarray,
    max_degree: int = 2,
    num_knots: tuple = (25, 10),
    smoothness_orders: int = 1,
    n_lambdas: int = 30,
    lambda_ratio: float = 1e-3,
    n_folds: int = 5,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
    verbose: bool = True,
) -> HurdleHALFit:
    """Fit hurdle HAL on GPU."""
    t0 = time.time()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    # ── Build basis once (shared across stages) ────────────────────
    if verbose:
        print(f"[Hurdle-GPU-HAL] Building basis on n={n} ...", flush=True)
    phi_csr, basis_list = backend.build_hal_basis(
        X, max_degree=max_degree, num_knots=num_knots,
        smoothness_orders=smoothness_orders,
    )
    p = phi_csr.shape[1]
    if verbose:
        print(f"  basis ({n} × {p}), nnz={phi_csr.nnz}, density={phi_csr.nnz/(n*p):.2%}", flush=True)

    # Column-normalize (unit L2)
    from scipy.sparse import diags
    col_norms = np.sqrt(np.asarray(phi_csr.multiply(phi_csr).sum(axis=0)).ravel())
    col_norms = np.maximum(col_norms, 1e-12)
    phi_scaled = (phi_csr @ diags(1.0 / col_norms)).tocsr()

    # ── Stage 1: logistic on Y > 0 ─────────────────────────────────
    is_pos = (y > 0).astype(np.float64)
    if verbose:
        print(f"[Hurdle-GPU-HAL] Stage 1: logistic on P(Y>0), positives = {int(is_pos.sum())}/{n}", flush=True)

    # Lambda grid for logistic — similar to gaussian via |X^T (y-ȳ)|/n
    Xty1 = (phi_scaled.T @ (is_pos - is_pos.mean())) / float(n)
    lam_max1 = float(np.max(np.abs(Xty1)))
    lambdas1 = lam_max1 * np.logspace(0, np.log10(lambda_ratio), n_lambdas)

    # CV for logistic
    from .cv import kfold_indices
    folds = kfold_indices(n, n_folds, seed=seed)
    all_dev = np.zeros((n_folds, len(lambdas1)))
    for f, ho in enumerate(folds):
        train_mask = np.ones(n, dtype=bool); train_mask[ho] = False
        X_tr = phi_scaled[train_mask]
        y_tr = is_pos[train_mask]
        X_ho = phi_scaled[~train_mask]
        y_ho = is_pos[~train_mask]
        results = cd_logistic.logistic_lasso_path(
            X_tr, y_tr, lambdas1,
            max_irls=12, irls_tol=1e-5,
            cd_max_sweeps=max_iter, cd_tol=tol,
            fit_intercept=True, verbose=False,
        )
        for i, res in enumerate(results):
            eta_ho = X_ho @ res.beta + res.intercept
            p_ho = 1.0 / (1.0 + np.exp(-np.clip(eta_ho, -50.0, 50.0)))
            all_dev[f, i] = -2.0 * float(np.mean(
                y_ho * np.log(np.clip(p_ho, 1e-12, 1.0))
                + (1 - y_ho) * np.log(np.clip(1 - p_ho, 1e-12, 1.0))
            ))
        if verbose:
            print(f"    Fold {f+1}/{n_folds} (logistic): best dev = {all_dev[f].min():.4e}", flush=True)
    mean_dev = all_dev.mean(axis=0)
    min_idx1 = int(np.argmin(mean_dev))
    lambda_pos = float(lambdas1[min_idx1])
    if verbose:
        print(f"  λ_cv (logistic) = {lambda_pos:.4e}", flush=True)

    # Final logistic fit at λ_cv on full data
    final_pos = cd_logistic.logistic_lasso(
        phi_scaled, is_pos, lam=lambda_pos,
        max_irls=25, irls_tol=1e-6,
        cd_max_sweeps=max_iter, cd_tol=tol,
        fit_intercept=True, verbose=False,
    )
    n_active_pos = int(np.sum(np.abs(final_pos.beta) > 1e-10))
    if verbose:
        print(f"  final logistic: {final_pos.n_irls} IRLS, active={n_active_pos}", flush=True)

    # ── Stage 2: gaussian on log(1+Y) | Y>0 ────────────────────────
    pos_mask = y > 0
    if verbose:
        print(f"[Hurdle-GPU-HAL] Stage 2: gaussian on log(1+Y) | Y>0, n={int(pos_mask.sum())}", flush=True)

    if pos_mask.sum() < 50:
        # Too few positives for stage 2; fall back to constant
        beta_mag = np.zeros(p)
        intercept_mag = float(np.log1p(y[pos_mask]).mean()) if pos_mask.any() else 0.0
        lambda_mag = float("nan")
        n_active_mag = 0
    else:
        phi_pos = phi_scaled[pos_mask]
        y_pos = np.log1p(y[pos_mask])

        # Lambda grid for gaussian
        Xty2 = (phi_pos.T @ (y_pos - y_pos.mean())) / float(pos_mask.sum())
        lam_max2 = float(np.max(np.abs(Xty2)))
        lambdas2 = lam_max2 * np.logspace(0, np.log10(lambda_ratio), n_lambdas)

        # CV for gaussian (5-fold on positives)
        from .cv_cd import cv_cd_gram_sparse
        cv2 = cv_cd_gram_sparse(
            phi_pos, y_pos, lambdas=lambdas2,
            n_folds=n_folds, max_sweeps=max_iter, tol=tol, seed=seed,
            verbose=verbose,
        )
        lambda_mag = cv2.lambda_cv

        # Final gaussian fit
        import jax.numpy as jnp
        G_full, Xty_full = fista_gram.compute_gram(phi_pos, y_pos - y_pos.mean())
        G_j = jnp.asarray(G_full); Xty_j = jnp.asarray(Xty_full)
        final_mag = cd_gram.cd_lasso_gram(
            G=G_j, Xty=Xty_j, lam=lambda_mag,
            max_sweeps=max_iter, tol=tol, verbose=False,
        )
        beta_mag = np.asarray(final_mag.beta)
        intercept_mag = float(y_pos.mean())
        n_active_mag = int(np.sum(np.abs(beta_mag) > 1e-10))
        if verbose:
            print(f"  λ_cv (gaussian) = {lambda_mag:.4e}, active={n_active_mag}", flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"[Hurdle-GPU-HAL] Total time: {elapsed:.0f}s", flush=True)

    return HurdleHALFit(
        beta_pos=final_pos.beta,
        intercept_pos=final_pos.intercept,
        lambda_pos=lambda_pos,
        n_active_pos=n_active_pos,
        beta_mag=beta_mag,
        intercept_mag=intercept_mag,
        lambda_mag=lambda_mag,
        n_active_mag=n_active_mag,
        basis_list=basis_list,
        col_norms=col_norms,
        n=n,
        p=p,
        fit_time_sec=elapsed,
    )
