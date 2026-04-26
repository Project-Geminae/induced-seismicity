"""
Gram-based coordinate descent Lasso on GPU (JAX).

Solves the same problem as fista_gram.py but with cyclic coordinate
descent instead of FISTA. CD converges in O(log 1/ε) iterations vs
FISTA's O(1/sqrt(ε)) — for sparse problems with small λ, that means
order-of-magnitude fewer iterations.

Algorithm (cyclic Gauss-Seidel CD on the Gram-form Lasso):
    initialize: β = 0, r = -Xty  (residual G β - Xty for β=0)
    repeat:
        for j = 1..p:
            z_j = -r_j + G_jj β_j
            β_j_new = soft_threshold(z_j, λ) / G_jj
            δ = β_j_new - β_j
            r ← r + G[:, j] · δ            (incremental residual update)
            β_j ← β_j_new
        until max|Δβ| < tol

This is what glmnet does (modulo active-set + sparse SpMV optimizations).
We do it on the GPU using JAX's fori_loop. The inner per-coordinate loop
is inherently sequential, but with all p×p Gram and p-vector residual
on-device the per-iter cost is O(p) memory accesses — fast on a GPU.

Convergence: empirically converges in 50-300 outer sweeps for HAL-like
problems, vs 5000+ FISTA iterations at the same tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    HAS_JAX = True
except Exception:
    HAS_JAX = False


@dataclass
class CDGramResult:
    beta:       np.ndarray
    n_sweeps:   int
    converged:  bool
    final_gap:  float


def _soft_threshold(u, tau):
    return jnp.sign(u) * jnp.maximum(jnp.abs(u) - tau, 0.0)


if HAS_JAX:

    @jax.jit
    def _cd_one_sweep(beta: jnp.ndarray, r: jnp.ndarray,
                      G: jnp.ndarray, G_diag: jnp.ndarray,
                      Xty: jnp.ndarray, lam: float):
        """One full pass over all p coordinates. Updates β and r in place
        (via JAX functional update). Returns (β_new, r_new)."""
        p = beta.shape[0]

        def body(j, state):
            beta_v, r_v = state
            beta_j_old = beta_v[j]
            G_jj = G_diag[j]
            # z_j = -r_j + G_jj β_j_old   (r = G β − Xty so −r_j = Xty_j − Σ_k G_jk β_k;
            # adding G_jj β_j gives Xty_j − Σ_{k≠j} G_jk β_k, i.e. the
            # "raw" partial residual coordinate)
            z_j = -r_v[j] + G_jj * beta_j_old
            beta_j_new = _soft_threshold(z_j, lam) / jnp.where(G_jj > 0, G_jj, 1.0)
            delta = beta_j_new - beta_j_old
            beta_v = beta_v.at[j].set(beta_j_new)
            r_v = r_v + G[:, j] * delta
            return (beta_v, r_v)

        beta_new, r_new = lax.fori_loop(0, p, body, (beta, r))
        return beta_new, r_new


def cd_lasso_gram(
    G: "jnp.ndarray",
    Xty: "jnp.ndarray",
    lam: float,
    max_sweeps: int = 200,
    tol: float = 1e-6,
    beta0: Optional["jnp.ndarray"] = None,
    verbose: bool = False,
) -> CDGramResult:
    """Coordinate descent Lasso on a precomputed Gram matrix.

    G:    (p, p) Gram = X^T X / n
    Xty:  (p,)   X^T y / n
    lam:  λ
    max_sweeps: maximum outer iterations (each sweep updates all p coords)
    tol:  convergence threshold on max|Δβ| per sweep
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required.")
    p = G.shape[0]
    G_diag = jnp.diag(G)
    beta = jnp.zeros(p, dtype=G.dtype) if beta0 is None else jnp.asarray(beta0, dtype=G.dtype)
    # r = G β − Xty (residual gradient minus -Xty)
    r = G @ beta - Xty

    n_sweeps = 0
    gap = float("inf")
    for k in range(1, max_sweeps + 1):
        n_sweeps = k
        beta_old = beta
        beta, r = _cd_one_sweep(beta, r, G, G_diag, Xty, float(lam))
        gap = float(jnp.max(jnp.abs(beta - beta_old)))
        if verbose and (k % 10 == 0 or k == max_sweeps):
            n_active = int(jnp.sum(jnp.abs(beta) > 1e-12))
            print(f"  CD sweep {k}: gap = {gap:.2e}  active = {n_active}", flush=True)
        if gap < tol:
            break

    return CDGramResult(
        beta=np.asarray(beta),
        n_sweeps=n_sweeps,
        converged=(gap < tol),
        final_gap=gap,
    )


def cd_lasso_gram_path(
    G: "jnp.ndarray",
    Xty: "jnp.ndarray",
    lambdas: np.ndarray,
    max_sweeps: int = 200,
    tol: float = 1e-6,
    verbose: bool = False,
):
    """λ-path with warm starts (decreasing λ). Each fit reuses the previous β
    as initialization, which dramatically speeds up convergence at small λ.
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required.")
    results = []
    beta_warm = None
    for i, lam in enumerate(lambdas):
        res = cd_lasso_gram(
            G=G, Xty=Xty, lam=float(lam),
            max_sweeps=max_sweeps, tol=tol, beta0=beta_warm, verbose=False,
        )
        results.append(res)
        beta_warm = res.beta
        if verbose:
            n_active = int(np.sum(np.abs(res.beta) > 1e-12))
            print(f"  λ[{i}]={lam:.3e}: {res.n_sweeps} sweeps, gap={res.final_gap:.2e}, active={n_active}",
                  flush=True)
    return results
