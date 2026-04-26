"""
Gram-based FISTA Lasso. For sparse Lasso problems where p is moderate
(< 10k) and the dense Gram fits on GPU, this is dramatically faster
than per-iter SpMV: precompute G = X^T X / n and Xty = X^T y / n once
on CPU (one sparse-sparse mult and one SpMV), upload as dense to GPU,
then iterate.

The objective gradient at β is
    ∇f(β) = -(1/n) X^T (y - Xβ) = G β - Xty
which is computed as a single dense matvec on GPU per iteration.

For HAL on n=451k with p~3000-5000:
  - Sparse matvec via JAX BCOO: 50-200ms per iter (with copies)
  - Gram-based dense matvec on GPU: ~1ms per iter
  - 100-200x speedup of the hot loop
  - Memory: dense Gram ~200 MB on GPU at p=5000

Trade-off: one-time Gram cost is 5-30 sec on CPU at full n. CV with
5 folds means 5 Grams, each on a sub-panel — still under a minute total.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False


@dataclass
class FISTAGramResult:
    beta:         np.ndarray
    n_iter:       int
    converged:    bool
    final_gap:    float
    lipschitz:    float


def compute_gram(X_sparse, y: np.ndarray):
    """Compute G = (X^T X) / n and Xty = (X^T y) / n for a CSR sparse X.

    Uses scipy's optimized CSR @ CSC multiplication. Returns numpy
    arrays ready for GPU upload.
    """
    from scipy.sparse import csr_matrix
    if not hasattr(X_sparse, "tocsr"):
        raise TypeError("X_sparse must be a scipy.sparse matrix.")
    X = X_sparse.tocsr()
    n, p = X.shape
    y = np.asarray(y, dtype=np.float64)

    # Sparse Gram: X^T @ X. Result is p × p, can be sparse or dense
    # depending on basis structure. For HAL bases this is usually fairly
    # dense (sub-bases overlap heavily) so materialize as dense float32.
    XtX = (X.T @ X).toarray().astype(np.float32) / float(n)
    Xty = (X.T @ y).astype(np.float32) / float(n)
    return XtX, Xty


def fista_lasso_gram(
    G: "jnp.ndarray",
    Xty: "jnp.ndarray",
    lam: float,
    n: int,
    L: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    beta0: Optional["jnp.ndarray"] = None,
    verbose: bool = False,
) -> FISTAGramResult:
    """Run FISTA for one λ value using a precomputed Gram matrix.

    The objective is (1/2) β^T G β - β^T Xty + λ ||β||_1
    (equivalent to the standard Lasso objective up to a constant).

    Gradient:  ∇f(β) = G β - Xty
    Lipschitz: L = largest eigenvalue of G
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required.")
    p = G.shape[0]
    if L is None:
        # Power iteration on G
        v = jnp.ones(p, dtype=G.dtype) / jnp.sqrt(float(p))
        for _ in range(40):
            v = G @ v
            v = v / (jnp.linalg.norm(v) + 1e-30)
        L = float(v @ G @ v / (jnp.dot(v, v) + 1e-30))

    beta = jnp.zeros(p, dtype=G.dtype) if beta0 is None else jnp.asarray(beta0, dtype=G.dtype)
    y_prev = beta
    t_prev = 1.0
    n_iter = 0
    gap = float("inf")

    for k in range(1, max_iter + 1):
        n_iter = k
        grad = G @ y_prev - Xty
        # Proximal step
        u = y_prev - grad / L
        thresh = lam / L
        beta_new = jnp.sign(u) * jnp.maximum(jnp.abs(u) - thresh, 0.0)
        # Momentum
        t_new = (1.0 + jnp.sqrt(1.0 + 4.0 * t_prev ** 2)) / 2.0
        y_prev = beta_new + ((t_prev - 1.0) / t_new) * (beta_new - beta)
        gap = float(jnp.max(jnp.abs(beta_new - beta)))
        beta = beta_new
        t_prev = t_new

        if verbose and (k % 100 == 0 or k == max_iter):
            print(f"  FISTA iter {k}: gap = {gap:.2e}", flush=True)

        if gap < tol:
            break

    return FISTAGramResult(
        beta=np.asarray(beta),
        n_iter=n_iter,
        converged=(gap < tol),
        final_gap=gap,
        lipschitz=float(L),
    )


def fista_lasso_gram_path(
    G: "jnp.ndarray",
    Xty: "jnp.ndarray",
    lambdas: np.ndarray,
    n: int,
    L: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
):
    """Fit Lasso along a λ path with warm starts (decreasing λ)."""
    if not HAS_JAX:
        raise RuntimeError("JAX required.")
    p = G.shape[0]
    if L is None:
        v = jnp.ones(p, dtype=G.dtype) / jnp.sqrt(float(p))
        for _ in range(40):
            v = G @ v
            v = v / (jnp.linalg.norm(v) + 1e-30)
        L = float(v @ G @ v / (jnp.dot(v, v) + 1e-30))

    results = []
    beta_warm = None
    for i, lam in enumerate(lambdas):
        res = fista_lasso_gram(
            G=G, Xty=Xty, lam=float(lam), n=n, L=L,
            max_iter=max_iter, tol=tol, beta0=beta_warm, verbose=False,
        )
        results.append(res)
        beta_warm = res.beta
        if verbose:
            n_active = int(np.sum(np.abs(res.beta) > 1e-10))
            print(f"  λ[{i}]={lam:.3e} {res.n_iter} iters, active={n_active}", flush=True)
    return results
