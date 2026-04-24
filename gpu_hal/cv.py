"""
K-fold cross-validation for λ selection on the FISTA Lasso path.

For each fold (holdout set H):
  - Fit FISTA path on training fold
  - Evaluate MSE on holdout set for each λ
Return λ_cv = argmin over λ of mean-across-folds holdout MSE, and the
"1-SE" λ (largest λ whose MSE is within 1 SE of the minimum).

This module accepts a generic matvec interface so the same code handles
dense numpy, scipy.sparse, or JAX BCOO backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .fista import (
    FISTAResult,
    fista_lasso,
    fista_lasso_path,
    make_dense_matvec,
    make_sparse_matvec,
    compute_lambda_max,
    make_lambda_grid,
    HAS_JAX,
)


if HAS_JAX:
    import jax.numpy as jnp
else:
    jnp = None


@dataclass
class CVResult:
    lambdas:    np.ndarray      # (n_lambdas,) candidate λ values (decreasing)
    mean_mse:   np.ndarray      # (n_lambdas,) mean holdout MSE
    sem_mse:    np.ndarray      # (n_lambdas,) SE of holdout MSE across folds
    lambda_cv:  float           # argmin
    lambda_1se: float           # 1-SE rule: largest λ with MSE within 1·SE of min
    lambda_min_idx:  int
    lambda_1se_idx:  int
    fold_indices: list          # list of fold index arrays used


def kfold_indices(n: int, k: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds


def cv_fista_lasso_dense(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    n_lambdas: int = 50,
    ratio: float = 1e-3,
    n_folds: int = 10,
    max_iter: int = 500,
    tol: float = 1e-5,
    seed: int = 42,
    verbose: bool = False,
) -> CVResult:
    """k-fold CV λ selection for a dense-matrix Lasso.

    Uses make_dense_matvec on each training fold to build matvecs and
    compute its own Lipschitz constant.
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required.")
    n, p = X.shape
    y = np.asarray(y, dtype=np.float32)

    # Compute full-data λ_max for the grid (proper bound for any subset)
    matvec_all, rmatvec_all, L_all = make_dense_matvec(X)
    if lambdas is None:
        lam_max = compute_lambda_max(rmatvec_all, jnp.asarray(y - y.mean()), n)
        lambdas = make_lambda_grid(lam_max, n_lambdas, ratio)

    folds = kfold_indices(n, n_folds, seed=seed)
    all_mses = np.zeros((n_folds, len(lambdas)))
    for f, ho in enumerate(folds):
        train = np.setdiff1d(np.arange(n), ho)
        X_tr, y_tr = X[train], y[train]
        X_ho, y_ho = X[ho], y[ho]
        # Center y on training fold
        y_tr_c = y_tr - y_tr.mean()
        matvec, rmatvec, L = make_dense_matvec(X_tr)
        y_tr_j = jnp.asarray(y_tr_c)
        fits = fista_lasso_path(
            matvec, rmatvec, y_tr_j, p, lambdas, len(train),
            L=L, max_iter=max_iter, tol=tol, verbose=False,
        )
        # Evaluate on holdout
        for i, fit in enumerate(fits):
            pred = X_ho @ fit.beta + y_tr.mean()
            all_mses[f, i] = float(np.mean((y_ho - pred) ** 2))
        if verbose:
            print(f"  Fold {f+1}/{n_folds} complete  best MSE = {all_mses[f].min():.4e}")

    mean_mse = all_mses.mean(axis=0)
    sem_mse = all_mses.std(axis=0, ddof=1) / np.sqrt(n_folds)
    min_idx = int(np.argmin(mean_mse))
    threshold = mean_mse[min_idx] + sem_mse[min_idx]
    # 1-SE rule: largest λ whose MSE is within threshold
    allowed = mean_mse <= threshold
    se1_idx = int(np.max(np.where(allowed)[0]))
    return CVResult(
        lambdas=np.asarray(lambdas),
        mean_mse=mean_mse,
        sem_mse=sem_mse,
        lambda_cv=float(lambdas[min_idx]),
        lambda_1se=float(lambdas[se1_idx]),
        lambda_min_idx=min_idx,
        lambda_1se_idx=se1_idx,
        fold_indices=folds,
    )


def cv_fista_lasso_sparse(
    X_sparse,
    y: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    n_lambdas: int = 50,
    ratio: float = 1e-3,
    n_folds: int = 10,
    max_iter: int = 500,
    tol: float = 1e-5,
    seed: int = 42,
    verbose: bool = False,
) -> CVResult:
    """k-fold CV λ selection for a sparse-matrix Lasso (CSR or CSC)."""
    if not HAS_JAX:
        raise RuntimeError("JAX required.")
    from scipy.sparse import csr_matrix
    if not hasattr(X_sparse, "tocsr"):
        raise TypeError("X_sparse must be a scipy.sparse matrix.")
    X_sparse = X_sparse.tocsr()
    n, p = X_sparse.shape
    y = np.asarray(y, dtype=np.float64)

    matvec_all, rmatvec_all, L_all = make_sparse_matvec(X_sparse)
    if lambdas is None:
        lam_max = compute_lambda_max(rmatvec_all, jnp.asarray(y - y.mean()), n)
        lambdas = make_lambda_grid(lam_max, n_lambdas, ratio)

    folds = kfold_indices(n, n_folds, seed=seed)
    all_mses = np.zeros((n_folds, len(lambdas)))
    for f, ho in enumerate(folds):
        train_mask = np.ones(n, dtype=bool)
        train_mask[ho] = False
        X_tr = X_sparse[train_mask]
        y_tr = y[train_mask]
        X_ho = X_sparse[~train_mask]
        y_ho = y[~train_mask]
        y_tr_mean = float(y_tr.mean())
        y_tr_c = y_tr - y_tr_mean
        matvec, rmatvec, L = make_sparse_matvec(X_tr)
        y_tr_j = jnp.asarray(y_tr_c, dtype=jnp.float32)
        fits = fista_lasso_path(
            matvec, rmatvec, y_tr_j, p, lambdas, len(y_tr),
            L=L, max_iter=max_iter, tol=tol, verbose=False,
        )
        for i, fit in enumerate(fits):
            pred = X_ho @ fit.beta + y_tr_mean
            all_mses[f, i] = float(np.mean((y_ho - pred) ** 2))
        if verbose:
            print(f"  Fold {f+1}/{n_folds} complete  best MSE = {all_mses[f].min():.4e}")

    mean_mse = all_mses.mean(axis=0)
    sem_mse = all_mses.std(axis=0, ddof=1) / np.sqrt(n_folds)
    min_idx = int(np.argmin(mean_mse))
    threshold = mean_mse[min_idx] + sem_mse[min_idx]
    allowed = mean_mse <= threshold
    se1_idx = int(np.max(np.where(allowed)[0]))
    return CVResult(
        lambdas=np.asarray(lambdas),
        mean_mse=mean_mse,
        sem_mse=sem_mse,
        lambda_cv=float(lambdas[min_idx]),
        lambda_1se=float(lambdas[se1_idx]),
        lambda_min_idx=min_idx,
        lambda_1se_idx=se1_idx,
        fold_indices=folds,
    )
