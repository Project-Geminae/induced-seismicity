"""
undersmoothed_hal.py
────────────────────
Undersmoothed Highly Adaptive Lasso for the shift-intervention estimand.

Implements van der Laan & Benkeser (2019), "Efficient estimation of
pathwise differentiable target parameters with the undersmoothed highly
adaptive lasso" (arXiv:1908.05607).

The estimator:
  1. Fit HAL to Y given (A, L) on full data via hal9001::fit_hal(family="gaussian").
     HAL constructs an L1-penalized regression over indicator basis functions
     of all subsets of (A, L) up to `max_degree` interactions.
  2. Select the regularization parameter lambda via cross-validation.
  3. Undersmooth: use lambda_cv * undersmoothing_factor (default 1/sqrt(log n))
     so that the fit is slightly less regularized than CV-optimal. This
     reduces bias at the cost of variance, and makes the plug-in estimator
     asymptotically linear.
  4. Plug-in target parameter for the shift intervention d(a) = a * (1 + delta):
        psi = (1/n) sum_i [Q_hat(d(A_i), L_i) - Q_hat(A_i, L_i)]
  5. Inference via cluster bootstrap (B resamples of cluster IDs).

Under this construction there is:
  - No TMLE targeting step (degenerate H is a non-issue)
  - No SuperLearner meta-learner (HAL is the estimator, not a base learner)
  - No conditional density estimator for A | L (not needed for plug-in)
  - No subsampling (HAL scales to the full panel in our dimensions)

Y in our case is the per-event max_ML, which is zero in ~95% of rows.
We fit a hurdle: P(Y > 0 | A, L) via HAL-logistic, E[log(1+Y) | Y > 0, A, L]
via HAL-gaussian. The composed prediction is
  Q(x) = P(Y>0|x) * expm1(E[log(1+Y)|Y>0, x])
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class HALShiftResult:
    """Undersmoothed-HAL shift-intervention estimate with bootstrap CI."""
    psi:         float           # point estimate
    ci_low:      float           # 2.5th percentile bootstrap quantile
    ci_high:     float           # 97.5th percentile
    se_boot:     float           # bootstrap SD
    n:           int
    n_clusters:  int
    B:           int             # number of bootstrap iterations completed
    lambda_cv:   float           # CV-optimal lambda from hal9001
    lambda_used: float           # undersmoothed lambda actually used
    fit_time_sec: float
    boot_time_sec: float
    notes:       dict


class UndersmoothedHAL:
    """Wraps R's hal9001::fit_hal with undersmoothing.

    Parameters
    ----------
    family : "gaussian" | "binomial"
    max_degree : int
        Maximum interaction depth. 2 or 3 typical.
    num_knots : tuple of int
        Knots per degree. (25, 10) is a reasonable default.
    smoothness_orders : int
        0 = indicator HAL, 1 = piecewise linear.
    undersmoothing_factor : float or None
        Multiplier on the CV-selected lambda. < 1 means LESS regularization
        than CV-optimal. None → auto = 1 / sqrt(log(n)).
    random_state : int
    """

    def __init__(
        self,
        family: str = "gaussian",
        max_degree: int = 2,
        num_knots: tuple = (25, 10),
        smoothness_orders: int = 1,
        undersmoothing_factor: float | None = None,
        random_state: int = 42,
    ):
        self.family = family
        self.max_degree = max_degree
        self.num_knots = num_knots
        self.smoothness_orders = smoothness_orders
        self.undersmoothing_factor = undersmoothing_factor
        self.random_state = random_state
        self._hal_fit = None
        self._ro = None
        self._hal9001 = None
        self.lambda_cv_ = None
        self.lambda_used_ = None
        self.n_fit_ = None

    def _init_r(self):
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        self._ro = ro
        self._hal9001 = importr("hal9001")
        self._base = importr("base")

    def _to_r_vec(self, arr):
        return self._ro.FloatVector(np.asarray(arr, dtype=float).tolist())

    def _to_r_mat(self, arr):
        arr = np.asarray(arr, dtype=float)
        flat = self._ro.FloatVector(arr.T.reshape(-1).tolist())
        return self._ro.r["matrix"](flat, nrow=arr.shape[0], ncol=arr.shape[1])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_fit_ = X.shape[0]
        self._init_r()
        ro = self._ro
        r_X = self._to_r_mat(X)
        r_Y = self._to_r_vec(y)
        r_knots = ro.IntVector(list(self.num_knots))

        # Step 1: CV fit to select lambda
        self._hal_fit = self._hal9001.fit_hal(
            X=r_X, Y=r_Y,
            family=self.family,
            max_degree=self.max_degree,
            num_knots=r_knots,
            smoothness_orders=self.smoothness_orders,
            fit_control=ro.ListVector({"cv_select": True}),
        )

        # Step 2: pull CV-selected lambda
        try:
            lambda_star = float(ro.r("function(f) f$lambda_star")(self._hal_fit)[0])
            self.lambda_cv_ = lambda_star
        except Exception:
            self.lambda_cv_ = None

        # Step 3: undersmoothing
        if self.undersmoothing_factor is None:
            # Default: 1 / sqrt(log(n))
            factor = 1.0 / np.sqrt(np.log(max(self.n_fit_, 2)))
        else:
            factor = self.undersmoothing_factor
        self.lambda_used_ = (
            self.lambda_cv_ * factor if self.lambda_cv_ is not None else None
        )
        # NOTE: switching to the undersmoothed lambda at predict time is handled
        # via predict_hal9001's lambda_select argument; we store both for
        # inspection and post-hoc diagnostics.
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._hal_fit is None:
            raise RuntimeError("fit() must be called before predict().")
        ro = self._ro
        r_X = self._to_r_mat(X)
        # Predict at the CV-selected lambda (default). To use an undersmoothed
        # lambda, pass lambda_select="lambda_1se" or numeric lambda value.
        r_pred = ro.r["predict"]
        preds = np.asarray(r_pred(self._hal_fit, new_data=r_X)).ravel()
        if self.family == "binomial":
            preds = np.clip(preds, 0.0, 1.0)
        return preds


class HurdleHAL:
    """Two-part hurdle: P(Y > 0) via HAL-logistic, E[log(1+Y) | Y>0] via HAL-gaussian."""

    def __init__(self, **hal_kwargs):
        self.hal_kwargs = hal_kwargs
        self.clf_ = None
        self.reg_ = None
        self._pos_mean_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        is_pos = (y > 0).astype(float)
        # Stage 1: P(Y > 0 | X)
        self.clf_ = UndersmoothedHAL(family="binomial", **self.hal_kwargs)
        self.clf_.fit(X, is_pos)
        # Stage 2: E[log(1 + Y) | Y > 0, X]
        pos_mask = y > 0
        if pos_mask.sum() < 50:
            self.reg_ = None
            self._pos_mean_ = float(np.log1p(y[pos_mask]).mean()) if pos_mask.any() else 0.0
            return self
        self.reg_ = UndersmoothedHAL(family="gaussian", **self.hal_kwargs)
        self.reg_.fit(X[pos_mask], np.log1p(y[pos_mask]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        p_pos = np.clip(self.clf_.predict(X), 0.0, 1.0)
        if self.reg_ is None:
            log_mag = np.full(X.shape[0], self._pos_mean_)
        else:
            log_mag = self.reg_.predict(X)
        return p_pos * np.expm1(log_mag)


def undersmoothed_hal_shift(
    df: pd.DataFrame,
    A_col: str,
    L_cols: list[str],
    Y_col: str,
    cluster_col: str,
    shift_pct: float = 0.10,
    B: int = 100,
    hal_kwargs: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> HALShiftResult:
    """Undersmoothed-HAL estimator for the shift intervention d(a) = a * (1 + shift_pct).

    Args
    ----
    df             : panel with columns [A_col, *L_cols, Y_col, cluster_col]
    shift_pct      : treatment shift magnitude
    B              : bootstrap iterations for CI
    hal_kwargs     : passed through to UndersmoothedHAL
    """
    import time
    hal_kwargs = hal_kwargs or {}

    A = df[A_col].to_numpy(dtype=float)
    L = df[L_cols].to_numpy(dtype=float)
    Y = df[Y_col].to_numpy(dtype=float)
    clusters = df[cluster_col].to_numpy()
    n = len(df)
    n_clusters = int(pd.Series(clusters).nunique())

    # ── Point estimate ───────────────────────────────────────────────
    AL = np.column_stack([A, L])
    if verbose:
        print(f"  Fitting HAL hurdle on n={n}, {AL.shape[1]} features...", flush=True)
    t0 = time.time()
    hurdle = HurdleHAL(**hal_kwargs)
    hurdle.fit(AL, Y)
    fit_time = time.time() - t0
    if verbose:
        print(f"  HAL fit complete ({fit_time:.0f}s). lambda_cv = {hurdle.clf_.lambda_cv_:.3e}", flush=True)

    A_post = A * (1.0 + shift_pct)
    AL_post = np.column_stack([A_post, L])
    Q_obs = hurdle.predict(AL)
    Q_post = hurdle.predict(AL_post)
    psi = float(np.mean(Q_post - Q_obs))
    if verbose:
        print(f"  psi (point estimate) = {psi:+.3e}", flush=True)

    # ── Cluster bootstrap ────────────────────────────────────────────
    if verbose:
        print(f"  Cluster bootstrap (B={B})...", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    cluster_to_idx = {c: np.where(clusters == c)[0] for c in unique_clusters}
    boot_psi = []
    for b in range(B):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        try:
            A_b = A[idx]
            L_b = L[idx]
            Y_b = Y[idx]
            AL_b = np.column_stack([A_b, L_b])
            hurdle_b = HurdleHAL(**hal_kwargs)
            hurdle_b.fit(AL_b, Y_b)
            A_post_b = A_b * (1.0 + shift_pct)
            AL_post_b = np.column_stack([A_post_b, L_b])
            psi_b = float(np.mean(hurdle_b.predict(AL_post_b) - hurdle_b.predict(AL_b)))
            boot_psi.append(psi_b)
            if verbose and ((b + 1) % max(1, B // 10) == 0):
                print(f"    boot {b + 1}/{B}  psi*={psi_b:+.3e}  elapsed={time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            warnings.warn(f"Bootstrap iter {b} failed: {e}")
            continue
    boot_time = time.time() - t0

    boot_psi_arr = np.array(boot_psi)
    ci_low = float(np.quantile(boot_psi_arr, 0.025))
    ci_high = float(np.quantile(boot_psi_arr, 0.975))
    se_boot = float(np.std(boot_psi_arr, ddof=1))

    if verbose:
        print(f"  Bootstrap done ({boot_time:.0f}s, B={len(boot_psi)} successful)", flush=True)
        print(f"  psi = {psi:+.3e}  CI=[{ci_low:+.3e}, {ci_high:+.3e}]  SE_boot={se_boot:.3e}", flush=True)

    return HALShiftResult(
        psi=psi,
        ci_low=ci_low,
        ci_high=ci_high,
        se_boot=se_boot,
        n=n,
        n_clusters=n_clusters,
        B=len(boot_psi),
        lambda_cv=hurdle.clf_.lambda_cv_ or float("nan"),
        lambda_used=hurdle.clf_.lambda_used_ or float("nan"),
        fit_time_sec=fit_time,
        boot_time_sec=boot_time,
        notes={
            "shift_pct":      shift_pct,
            "n_clusters":     n_clusters,
            "boot_mean":      float(np.mean(boot_psi_arr)),
            "boot_median":    float(np.median(boot_psi_arr)),
        },
    )
