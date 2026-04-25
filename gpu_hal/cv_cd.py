"""K-fold CV using Gram-based coordinate descent (cv_gram.py companion)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .fista_gram import compute_gram
from .cd_gram import cd_lasso_gram_path
from .cv import CVResult, kfold_indices

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False


def cv_cd_gram_sparse(
    X_sparse,
    y: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    n_lambdas: int = 50,
    ratio: float = 1e-3,
    n_folds: int = 10,
    max_sweeps: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
    verbose: bool = False,
) -> CVResult:
    """k-fold CV using Gram-based CD. Per-fold Gram on CPU, CD on GPU."""
    if not HAS_JAX:
        raise RuntimeError("JAX required.")
    if not hasattr(X_sparse, "tocsr"):
        raise TypeError("X_sparse must be a scipy.sparse matrix.")
    X = X_sparse.tocsr()
    n, p = X.shape
    y = np.asarray(y, dtype=np.float64)

    if lambdas is None:
        Xty_full = (X.T @ (y - y.mean())).astype(np.float32) / float(n)
        lam_max = float(np.max(np.abs(Xty_full)))
        lambdas = lam_max * np.logspace(0, np.log10(ratio), n_lambdas)

    folds = kfold_indices(n, n_folds, seed=seed)
    all_mses = np.zeros((n_folds, len(lambdas)))

    for f, ho in enumerate(folds):
        train_mask = np.ones(n, dtype=bool)
        train_mask[ho] = False
        X_tr = X[train_mask]
        y_tr = y[train_mask]
        X_ho = X[~train_mask]
        y_ho = y[~train_mask]
        y_tr_mean = float(y_tr.mean())
        y_tr_c = y_tr - y_tr_mean

        G_np, Xty_np = compute_gram(X_tr, y_tr_c)
        G_j = jnp.asarray(G_np)
        Xty_j = jnp.asarray(Xty_np)

        fits = cd_lasso_gram_path(
            G=G_j, Xty=Xty_j, lambdas=lambdas,
            max_sweeps=max_sweeps, tol=tol, verbose=False,
        )
        for i, fit in enumerate(fits):
            pred = X_ho @ fit.beta + y_tr_mean
            all_mses[f, i] = float(np.mean((y_ho - pred) ** 2))
        if verbose:
            print(f"  Fold {f+1}/{n_folds} done  best MSE = {all_mses[f].min():.4e}", flush=True)

    mean_mse = all_mses.mean(axis=0)
    sem_mse = all_mses.std(axis=0, ddof=1) / np.sqrt(n_folds)
    min_idx = int(np.argmin(mean_mse))
    threshold = mean_mse[min_idx] + sem_mse[min_idx]
    se1_idx = int(np.max(np.where(mean_mse <= threshold)[0]))

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
