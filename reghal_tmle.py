"""
reghal_tmle.py
──────────────
Delta-method regularized HAL-TMLE for the shift-intervention estimand.

Reference: Li, Qiu, Wang & van der Laan (2025),
  "Regularized Targeted Maximum Likelihood Estimation in Highly
  Adaptive Lasso Implied Working Models" — arXiv:2506.17214

This module implements the Delta-method variant (Section 3.2 of the paper).
The core idea: after a HAL fit, we have a working model
    Q_β(x) = β_0 + Σ_{j=1..p} β_j · φ_j(x)
where {φ_j} are the basis functions selected by HAL's L1 penalty with non-zero
coefficients. Standard TMLE targets Q by fluctuating along a clever-covariate
direction; regHAL-TMLE instead targets Q within this finite-dimensional
HAL-implied working model via a ridge-regularized Newton step along the
direction of the parametric efficient influence curve.

The parametric EIC for target parameter Ψ(Q) under the working model is:
    D*_β(O_i) = (∂Ψ/∂β)^T · (I_n(β) + η·I)^(-1) · S^β_i(O)
where
    I_n(β) = (1/n) Σ_i S^β_i(O) · S^β_i(O)^T      is the empirical Fisher information
    S^β_i(O)                                       is the score at observation i
    η > 0                                          is a small ridge (stabilizes inversion)

For the Gaussian outcome model Y_i = Q_β(X_i) + ε_i with ε ~ N(0, σ²):
    S^β_i(O) = (Y_i - Q_β(X_i)) · [1, φ_1(X_i), ..., φ_p(X_i)]^T / σ²
    Q_β(X_i) is linear in β so ∂Q/∂β_j = (1 if j=0 else φ_j(X_i))

For the shift target parameter Ψ(Q) = (1/n) Σ_i [Q(1.1·A_i, L_i) - Q(A_i, L_i)]:
    ∂Ψ/∂β_j = (1/n) Σ_i [φ_j(1.1·A_i, L_i) - φ_j(A_i, L_i)]  for j ≥ 1
    ∂Ψ/∂β_0 = 0                                               (intercept drops)

The targeting loop iteratively updates β along the EIC-implied direction
until |P_n D*_β| < SE(D*_β) / (√n · log n), which is the paper's stopping
criterion for asymptotic linearity.

This module provides:
    - extract_hal_basis(fit, X): compute the design matrix [1, φ_1(X), ..., φ_p(X)]
    - reghal_tmle_shift(df, ...): full targeting loop + plug-in updates
    - Influence-function-based CI using the targeted EIC

STATUS: scaffolded implementation. Core logic in place; the hal9001 basis
extraction via rpy2 is the riskiest piece and may need adjustment after
the first test run. Test first on a single radius at smaller n before
scaling to the 20-radius sweep.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegHALResult:
    """Delta-method regHAL-TMLE estimate for a shift intervention."""
    psi_plugin:   float    # HAL plug-in before targeting
    psi_targeted: float    # after targeting within working model
    se_if:        float    # SE from targeted parametric EIC
    ci_low:       float
    ci_high:      float
    pval:         float
    n:            int
    n_clusters:   int
    n_basis:      int      # number of HAL bases with non-zero coefficients
    n_iter:       int      # targeting iterations
    converged:    bool
    elapsed_sec:  float
    notes:        dict


def extract_hal_basis(hal_fit, X, ro, hal9001):
    """Extract the HAL design matrix at X using the fitted model's bases.

    Returns an (n, p+1) numpy array where the first column is a constant 1
    (intercept) and columns 1..p correspond to the p selected basis
    functions (those with non-zero coefficient at the CV-selected lambda).

    hal9001's `make_design_matrix` re-creates the basis matrix at new_data
    using the stored knots and basis list. We then subset to the active
    bases (non-zero coefficients) for efficiency.
    """
    X = np.asarray(X, dtype=float)
    # Convert X to R matrix
    flat = ro.FloatVector(X.T.reshape(-1).tolist())
    r_X = ro.r["matrix"](flat, nrow=X.shape[0], ncol=X.shape[1])
    # Ask hal9001 for the design matrix at new data.
    # make_design_matrix signature: (X, blist, p_reserve=0.5). Pass the
    # basis list via R's generic call; result is a dgCMatrix which we
    # convert to dense via base::as.matrix.
    r_blist = ro.r("function(f) f$basis_list")(hal_fit)
    r_make = ro.r("function(X, blist) as.matrix(hal9001::make_design_matrix(X, blist))")
    r_design = r_make(r_X, r_blist)
    design = np.asarray(r_design)
    # Get the coefficient vector at lambda_star to identify active bases
    r_coefs = ro.r("function(f) as.numeric(stats::coef(f$lasso_fit, s=f$lambda_star))")(hal_fit)
    coefs = np.asarray(r_coefs)
    # coefs[0] is intercept; coefs[1:] correspond to basis columns in design
    active = np.abs(coefs[1:]) > 0
    n = design.shape[0]
    # Prepend intercept column
    basis_mat = np.column_stack([np.ones(n), design[:, active]])
    active_coefs = np.concatenate([[coefs[0]], coefs[1:][active]])
    return basis_mat, active_coefs


def reghal_tmle_shift(
    df: pd.DataFrame,
    A_col: str,
    L_cols: list[str],
    Y_col: str,
    cluster_col: str,
    shift_pct: float = 0.10,
    ridge_eta: float = 1e-4,
    step_size: float = 1e-4,
    max_iter: int = 100,
    hal_kwargs: dict | None = None,
    verbose: bool = True,
) -> RegHALResult:
    """Delta-method regHAL-TMLE for the multiplicative shift d(a) = a(1+δ).

    Hurdle simplification: to keep this tractable as a first cut, we fit
    HAL directly on Y (gaussian family). Two-part hurdle can be added
    later by running the targeting loop on P(Y>0) and E[Y|Y>0] separately
    and composing the EICs via the product rule.
    """
    import undersmoothed_hal as uhal

    hal_kwargs = hal_kwargs or {}
    t0 = time.time()

    A = df[A_col].to_numpy(dtype=float)
    L = df[L_cols].to_numpy(dtype=float)
    Y = df[Y_col].to_numpy(dtype=float)
    clusters = df[cluster_col].to_numpy()
    n = len(df)
    n_clusters = int(pd.Series(clusters).nunique())

    # ── 1. Fit HAL on (A, L) → Y (gaussian plug-in) ──────────────────
    if verbose:
        print(f"  Fitting HAL on n={n} ...", flush=True)
    hal = uhal.UndersmoothedHAL(family="gaussian", **hal_kwargs)
    AL = np.column_stack([A, L])
    hal.fit(AL, Y)
    hal._init_r()  # ensures _ro, _hal9001 are populated for basis extraction
    ro = hal._ro
    hal9001 = hal._hal9001

    # ── 2. Extract design matrix (active bases) ──────────────────────
    Phi_obs, beta = extract_hal_basis(hal._hal_fit, AL, ro, hal9001)
    n_basis = len(beta) - 1
    if verbose:
        print(f"  HAL active bases: {n_basis}", flush=True)

    # Basis at shifted treatment
    A_post = A * (1.0 + shift_pct)
    AL_post = np.column_stack([A_post, L])
    Phi_post, _ = extract_hal_basis(hal._hal_fit, AL_post, ro, hal9001)

    # ── 3. Plug-in estimate ──────────────────────────────────────────
    Q_obs = Phi_obs @ beta
    Q_post = Phi_post @ beta
    psi_plugin = float(np.mean(Q_post - Q_obs))
    if verbose:
        print(f"  Plug-in psi = {psi_plugin:+.4e}", flush=True)

    # ── 4. Parametric EIC under Gaussian working model ───────────────
    # ∂Ψ/∂β = mean(Phi_post - Phi_obs) along axis=0  (vector of length p+1)
    d_psi = (Phi_post - Phi_obs).mean(axis=0)

    # Score S_i = (Y_i - Q_β(X_i)) * Phi_i / sigma^2
    def compute_eic_components(beta_cur):
        Q_pred = Phi_obs @ beta_cur
        resid = Y - Q_pred
        sigma2 = max(np.var(resid), 1e-12)
        # Score matrix: n × (p+1)
        S = resid[:, None] * Phi_obs / sigma2
        # Fisher info with ridge: (p+1) × (p+1)
        I_n = (S.T @ S) / n + ridge_eta * np.eye(len(beta_cur))
        # Direction: I_n^-1 · d_psi
        try:
            direction = np.linalg.solve(I_n, d_psi)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(I_n, d_psi, rcond=None)[0]
        # EIC per observation
        D = S @ direction  # n-vector
        return D, direction, sigma2

    # ── 5. Targeting loop ────────────────────────────────────────────
    # Solve P_n(D*_β) = 0 via a line-search Newton step. The paper's
    # recommendation of fixed step_size = 1e-4 with sign-scaled gradient
    # descent is too conservative for our covariate structure — at n=50k
    # with 79 active bases we hit max_iter without meaningful progress.
    # Line search along the Newton direction converges in ~5 iterations.
    beta_cur = beta.copy()
    converged = False
    n_iter = 0
    D_history = []
    for n_iter in range(1, max_iter + 1):
        D, direction, sigma2 = compute_eic_components(beta_cur)
        P_n_D = float(np.mean(D))
        se_D = float(np.std(D, ddof=1) / np.sqrt(n))
        threshold = se_D / np.sqrt(n) / np.log(max(n, 2))
        D_history.append((P_n_D, se_D))
        if verbose and n_iter % 5 == 0:
            print(f"    iter {n_iter}: P_n(D*) = {P_n_D:+.3e}  thresh = {threshold:.3e}", flush=True)
        if abs(P_n_D) < threshold:
            converged = True
            break

        # Backtracking line search: start with a Newton-like step (α = P_n_D)
        # and halve until |P_n(D*)| strictly decreases.
        alpha = P_n_D
        best_beta = beta_cur
        best_abs = abs(P_n_D)
        for _ in range(20):
            trial_beta = beta_cur - alpha * direction
            D_trial, _, _ = compute_eic_components(trial_beta)
            abs_trial = abs(float(np.mean(D_trial)))
            if abs_trial < best_abs:
                best_beta = trial_beta
                best_abs = abs_trial
                break
            alpha *= 0.5
        beta_cur = best_beta

        # Safety: bail out if no α improves at all → stuck at optimum
        if best_abs >= abs(P_n_D):
            if verbose:
                print(f"    iter {n_iter}: line search stuck at |P_n(D*)| = {best_abs:.3e}", flush=True)
            break

    if verbose:
        print(f"  Targeting {'converged' if converged else 'hit max_iter'} at iter {n_iter}", flush=True)

    # ── 6. Recompute targeted ψ ──────────────────────────────────────
    Q_obs_t = Phi_obs @ beta_cur
    Q_post_t = Phi_post @ beta_cur
    psi_targeted = float(np.mean(Q_post_t - Q_obs_t))

    # ── 7. Final EIC and cluster-robust SE ───────────────────────────
    D_final, _, _ = compute_eic_components(beta_cur)
    # Center the EIC for SE calculation (targeting enforces P_n(D*) ≈ 0 up to threshold)
    D_centered = D_final - D_final.mean()
    se_iid = float(np.std(D_final, ddof=1) / np.sqrt(n))

    # Cluster-IF SE (Bessel-corrected)
    s = pd.Series(D_centered).groupby(pd.Series(clusters).values).sum().to_numpy()
    n_c = len(s)
    if n_c >= 2:
        se_cluster = float(np.sqrt((n_c / (n_c - 1)) * np.sum(s ** 2) / (n ** 2)))
    else:
        se_cluster = float("nan")

    z = 1.959963984540054
    ci_low = psi_targeted - z * se_cluster
    ci_high = psi_targeted + z * se_cluster
    from math import erf, sqrt
    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    z_stat = psi_targeted / max(se_cluster, 1e-15)
    pval = 2 * (1 - _norm_cdf(abs(z_stat)))

    elapsed = time.time() - t0
    if verbose:
        print(f"  psi_targeted = {psi_targeted:+.4e}  CI=[{ci_low:+.3e}, {ci_high:+.3e}]  p={pval:.4f}  ({elapsed:.0f}s)", flush=True)

    return RegHALResult(
        psi_plugin=psi_plugin,
        psi_targeted=psi_targeted,
        se_if=se_cluster,
        ci_low=ci_low,
        ci_high=ci_high,
        pval=pval,
        n=n,
        n_clusters=n_clusters,
        n_basis=n_basis,
        n_iter=n_iter,
        converged=converged,
        elapsed_sec=elapsed,
        notes={
            "shift_pct":       shift_pct,
            "se_iid":          se_iid,
            "sigma2":          float(np.var(Y - Phi_obs @ beta_cur)),
            "beta_shift_l2":   float(np.linalg.norm(beta_cur - beta)),
            "P_n_D_final":     float(np.mean(D_final)),
            "SE_D_final":      float(np.std(D_final, ddof=1)),
        },
    )
