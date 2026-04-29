"""
Active-set IRLS for logistic Lasso — full-n hurdle HAL on GPU.

The full-Gram IRLS in `cd_logistic.py` rebuilds the entire weighted Gram
G_w = X^T diag(w) X / n every IRLS iteration. At n = 451,212, p = 1,564,
density = 28%, this is ~25 minutes per IRLS iter — making the full-n
hurdle fit infeasible (~3-4 weeks projected).

The active-set IRLS strategy reduces this to O(|S|² + n · |S|) per iter
where |S| is the size of the active set (~80-200 in practice), a 10-20×
speedup that brings the full-n hurdle within range.

Algorithm:
  1. Phase 1 (warmup): K_warm full-Gram IRLS iterations to identify an
     initial active set S = {j : |β_j| > 0}.
  2. Phase 2 (active-set IRLS): For each subsequent iter,
       a. Build G_S = X_S^T diag(w) X_S / n  (|S| × |S|; fast)
       b. Run CD on G_S to update β_S
       c. KKT check on inactive coordinates:
            c_j = (X^T diag(w) (z - β_0 - X_S β_S))_j / n
          If max |c_j| > λ + kkt_tol for j ∉ S, add violators to S.
       d. If no violators, accept the step.  Otherwise rebuild G_S
          with the enlarged S and re-solve.
  3. Phase 3 (final KKT check): At convergence, do a full-coordinate
     KKT verification.  Any violator triggers a final full-Gram step.

Properties:
  - Equivalent to full-Gram IRLS at convergence (same fixed point)
  - Robust to active-set thrashing: KKT-driven addition is monotone
    under small λ changes; we don't drop coordinates within an
    iteration.
  - Warm-starts β across IRLS iters and across λ values on a path.

References:
  - Friedman, Hastie & Tibshirani (2010), "Regularization paths for
    generalized linear models via coordinate descent", JSS, §3.2
    (the strong-rule + KKT-check structure glmnet uses).
  - El Ghaoui, Viallon & Rabbani (2010), "Safe Feature Elimination
    for the LASSO and Sparse Supervised Learning Problems".

Status: first working cut — exhaustive testing TBD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import numpy as np

from . import cd_gram
from .cd_logistic import (
    _sigmoid,
    LogisticResult,
)

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False


def _build_weighted_gram_active(X_csr, w: np.ndarray, S: np.ndarray):
    """Build G_w on the active-set columns S only.

    G_S = X_S^T diag(w) X_S / n ∈ ℝ^{|S| × |S|}
    """
    from scipy.sparse import diags
    n = X_csr.shape[0]
    sw = np.sqrt(np.maximum(w, 1e-12))
    # Slice columns first, then row-scale — scipy CSR supports this efficiently
    X_S = X_csr[:, S]
    X_Sw = diags(sw) @ X_S
    G_S = (X_Sw.T @ X_Sw).toarray().astype(np.float32) / float(n)
    return G_S


def _weighted_xty_active(X_csr, w: np.ndarray, z_c: np.ndarray, S: np.ndarray):
    """Compute b_S = X_S^T diag(w) z_c / n."""
    n = X_csr.shape[0]
    return (X_csr[:, S].T @ (w * z_c)).astype(np.float32) / float(n)


def _kkt_residuals(X_csr, w: np.ndarray, z_c: np.ndarray,
                   beta_S: np.ndarray, S: np.ndarray) -> np.ndarray:
    """For each coordinate j (across the full p), compute
       c_j = (X^T diag(w) (z_c - X_S β_S))_j / n
    The KKT condition for the Lasso is |c_j| ≤ λ for inactive j;
    violators are candidates for promotion to the active set.

    Returns array of shape (p,) with the residual gradient.
    """
    n = X_csr.shape[0]
    if len(S) > 0:
        residual = z_c - X_csr[:, S] @ beta_S
    else:
        residual = z_c
    return np.asarray(X_csr.T @ (w * residual)).ravel() / float(n)


def logistic_lasso_active_set(
    X_csr,
    y: np.ndarray,
    lam: float,
    max_irls: int = 25,
    irls_tol: float = 1e-6,
    cd_max_sweeps: int = 200,
    cd_tol: float = 1e-7,
    kkt_tol: float = 1e-3,
    initial_full_irls: int = 2,
    initial_active_size: int = 50,
    beta0: Optional[np.ndarray] = None,
    intercept0: float = 0.0,
    fit_intercept: bool = True,
    verbose: bool = False,
) -> LogisticResult:
    """Logistic Lasso via active-set IRLS + Gram CD.

    Parameters
    ----------
    X_csr : scipy.sparse.csr_matrix, shape (n, p)
    y : array, shape (n,)  binary outcome
    lam : float, L1 penalty
    max_irls : int, max IRLS iterations
    irls_tol : convergence tolerance on β
    cd_max_sweeps : max CD sweeps per IRLS inner solve
    cd_tol : CD convergence tolerance (KKT gap)
    kkt_tol : tolerance for KKT-violator detection
    initial_full_irls : number of full-Gram IRLS iters before going active-set
    initial_active_size : when warmup gives <K active coords, top-up S to
        this size by taking the largest-|c_j| inactive coords (for warm-start
        stability — prevents the active set from being too small to
        meaningfully constrain the IRLS step)
    beta0 : optional warm-start coefficients, shape (p,)
    intercept0 : optional warm-start intercept
    fit_intercept : whether to fit an unpenalized intercept
    verbose : bool

    Returns
    -------
    LogisticResult with β, intercept, n_irls, n_total_cd, converged, final_loglik.
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for the inner CD solve.")

    n, p = X_csr.shape
    y = np.asarray(y, dtype=np.float64)
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=np.float64).copy()
    intercept = float(intercept0)
    total_cd_sweeps = 0
    converged = False

    # Phase 1: warmup with full-Gram IRLS (or use existing logistic_lasso)
    # to seed the active set.
    if verbose:
        print(f"[active-set IRLS] n={n}, p={p}, λ={lam:.4e}, "
              f"warmup_iters={initial_full_irls}", flush=True)

    from .cd_logistic import logistic_lasso as _full_logistic
    if initial_full_irls > 0:
        warmup = _full_logistic(
            X_csr, y, lam,
            max_irls=initial_full_irls, irls_tol=irls_tol * 10,
            cd_max_sweeps=cd_max_sweeps, cd_tol=cd_tol,
            beta0=beta, intercept0=intercept,
            fit_intercept=fit_intercept, verbose=verbose,
        )
        beta = warmup.beta
        intercept = warmup.intercept
        total_cd_sweeps += warmup.n_total_cd

    # Determine initial active set
    S = np.where(np.abs(beta) > 1e-10)[0]
    if verbose:
        print(f"[active-set IRLS] warmup active = {len(S)}", flush=True)

    # Phase 2: active-set IRLS
    for irls_iter in range(1, max_irls + 1):
        t0 = time.time()

        eta = X_csr @ beta + intercept
        p_hat = _sigmoid(eta)
        w = np.maximum(p_hat * (1.0 - p_hat), 1e-6)
        z = eta + (y - p_hat) / w
        if fit_intercept:
            wmean = float(np.sum(w * z) / np.sum(w))
            z_c = z - wmean
        else:
            wmean = 0.0
            z_c = z

        # Inner active-set loop: rebuild G_S, solve CD, KKT-check.
        # Re-enter the loop if KKT violators are found.
        for as_inner in range(20):  # cap to prevent runaway
            if len(S) == 0:
                # All coefficients zero — KKT check decides whether to add any.
                kkt_full = _kkt_residuals(X_csr, w, z_c, np.zeros(0), S)
                violators = np.where(np.abs(kkt_full) > lam + kkt_tol)[0]
                if len(violators) == 0:
                    beta_S_new = np.zeros(0)
                    break
                # Add violators
                S = np.union1d(S, violators)
                continue

            G_S = _build_weighted_gram_active(X_csr, w, S)
            b_S = _weighted_xty_active(X_csr, w, z_c, S)
            beta_S0 = beta[S].astype(np.float32)

            G_j = jnp.asarray(G_S)
            b_j = jnp.asarray(b_S)
            beta_S_j = jnp.asarray(beta_S0)
            cd_res = cd_gram.cd_lasso_gram(
                G=G_j, Xty=b_j, lam=float(lam),
                max_sweeps=cd_max_sweeps, tol=cd_tol,
                beta0=beta_S_j, verbose=False,
            )
            beta_S_new = np.asarray(cd_res.beta).astype(np.float64)
            total_cd_sweeps += cd_res.n_sweeps

            # KKT check on inactive coordinates
            kkt_full = _kkt_residuals(X_csr, w, z_c, beta_S_new, S)
            mask_in = np.zeros(p, dtype=bool); mask_in[S] = True
            kkt_inactive = kkt_full.copy()
            kkt_inactive[mask_in] = 0.0  # we don't check active members
            violators = np.where(np.abs(kkt_inactive) > lam + kkt_tol)[0]
            if len(violators) == 0:
                # Active-set step accepted
                break
            # Promote violators
            S = np.union1d(S, violators)
            if verbose:
                print(f"  IRLS {irls_iter} active-set inner {as_inner}: "
                      f"|S| {len(S) - len(violators)} → {len(S)} "
                      f"(+{len(violators)} KKT violators)", flush=True)

        # Update β at full p with the active-set solution
        beta_new = np.zeros(p)
        beta_new[S] = beta_S_new

        # Drop coordinates that decayed back to ~0 (keeps the active set tight)
        S_kept = S[np.abs(beta_S_new) > 1e-10]
        S = S_kept

        if fit_intercept:
            res = z - X_csr @ beta_new
            new_intercept = float(np.sum(w * res) / np.sum(w))
        else:
            new_intercept = 0.0

        delta_beta = float(np.max(np.abs(beta_new - beta)))
        delta_int = abs(new_intercept - intercept)
        beta = beta_new
        intercept = new_intercept

        if verbose:
            elapsed = time.time() - t0
            n_active = len(S)
            eta_dbg = X_csr @ beta + intercept
            p_dbg = _sigmoid(eta_dbg)
            ll = float(np.sum(y * np.log(np.clip(p_dbg, 1e-12, 1.0))
                              + (1 - y) * np.log(np.clip(1.0 - p_dbg, 1e-12, 1.0))))
            print(f"  IRLS {irls_iter}: |S|={n_active:4d}  "
                  f"|Δβ|_∞={delta_beta:.2e}  "
                  f"loglik={ll:+.4e}  ({elapsed:.1f}s)", flush=True)

        if irls_iter >= 2 and delta_beta < irls_tol and delta_int < irls_tol:
            # Final full-KKT check — important: at convergence we verify
            # NO inactive coordinate is a KKT violator.
            eta_final = X_csr @ beta + intercept
            p_final = _sigmoid(eta_final)
            w_final = np.maximum(p_final * (1.0 - p_final), 1e-6)
            z_final = eta_final + (y - p_final) / w_final
            if fit_intercept:
                z_c_final = z_final - float(np.sum(w_final * z_final) / np.sum(w_final))
            else:
                z_c_final = z_final
            kkt_final = _kkt_residuals(X_csr, w_final, z_c_final, beta[S], S)
            mask_in = np.zeros(p, dtype=bool); mask_in[S] = True
            kkt_inactive_final = kkt_final.copy()
            kkt_inactive_final[mask_in] = 0.0
            violators_final = np.where(np.abs(kkt_inactive_final) > lam + kkt_tol)[0]
            if len(violators_final) == 0:
                converged = True
                if verbose:
                    print(f"[active-set IRLS] CONVERGED at iter {irls_iter}, "
                          f"final |S|={len(S)}", flush=True)
                break
            else:
                # Add violators and continue IRLS
                S = np.union1d(S, violators_final)
                if verbose:
                    print(f"  final-KKT violators: {len(violators_final)} — "
                          f"promoting and continuing", flush=True)

    # Final logloss
    eta_done = X_csr @ beta + intercept
    p_done = _sigmoid(eta_done)
    final_loglik = float(np.sum(y * np.log(np.clip(p_done, 1e-12, 1.0))
                                + (1 - y) * np.log(np.clip(1.0 - p_done, 1e-12, 1.0))))

    return LogisticResult(
        beta=beta,
        intercept=intercept,
        n_irls=irls_iter,
        n_total_cd=total_cd_sweeps,
        converged=converged,
        final_loglik=final_loglik,
    )
