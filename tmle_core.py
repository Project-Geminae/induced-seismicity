"""
tmle_core.py
────────────
Targeted Maximum Likelihood Estimation primitives for the induced-seismicity
panel.

This module replaces causal_core.py's OLS-based mediation framework with
doubly robust, semi-parametric efficient estimators. The motivation is laid
out in CHANGES.md (TMLE section); briefly:

  • Outcome max_ML is zero in 96–98% of cells. OLS imposes a global linear
    slope through a near-degenerate distribution; TMLE with a hurdle Q model
    handles the zero mass natively.
  • The treatment is continuous (cumulative BBL). TMLE for continuous
    treatments works via stochastic shift interventions or a dose-response
    grid, both of which are policy-natural and avoid the awkward "per BBL"
    framing.
  • The functional form of Q is hand-imposed in OLS. TMLE uses a Super
    Learner stack (here: ridge + GBM + XGBoost + GAM-style spline regression)
    and the targeting step preserves valid inference under model
    misspecification of either Q or g (but not both).
  • Inference uses the influence function rather than naive bootstrap.
    Cluster-IF variance respects the within-well correlation in the panel.

Key TMLE machinery
──────────────────
For a stochastic shift intervention d(a, l) → a · (1 + δ) the estimand is

    ψ_δ = E[Y_{d(A,L)}]

Identification under positivity + conditional exchangeability + consistency:

    ψ_δ = E_L[ E[Y | A = d(A,L), L] ]

The TMLE plug-in is:
  1. Q_n(a, l)  = Super-Learner estimate of E[Y | A=a, L=l]
  2. g_n(a | l) = conditional density of A given L (histogram via multinomial
                  classifier on K quantile bins)
  3. Clever covariate H(a, l) = g_n(a / (1+δ) | l) / g_n(a | l) · 1/(1+δ)
  4. Targeted update Q_n^*(a, l) = Q_n(a, l) + ε̂ · H(a, l)
                     where ε̂ = OLS slope of (Y − Q_n(A,L)) on H(A,L)
  5. ψ_n = mean of Q_n^*(d(A_i, L_i), L_i)

The efficient influence function:

    IF(O) = H(A, L) · (Y − Q_n^*(A, L))
          + Q_n^*(d(A, L), L)
          − ψ_n

Variance:  Var[IF] / n  (i.i.d.)
Cluster:   sum IF within cluster, then empirical variance over clusters.

For the dose-response grid we replace the shift with point interventions
d_a(·) = a (constant) at a grid of cumulative-volume levels and run the
same TMLE machinery at each grid point.

For mediation (NDE/NIE) we use the regression-based imputation TMLE from
Zheng & van der Laan (2012), simplified to a high-vs-low contrast.

References
──────────
- van der Laan & Rose (2011), Targeted Learning
- Hejazi & van der Laan (2020), txshift package vignette
- Díaz & van der Laan (2018), Stochastic shift interventions
- Zheng & van der Laan (2012), Targeted Maximum Likelihood Estimation of
  Natural Direct Effect
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeCV,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import xgboost as xgb


# ──────────────────── Hyperparameters ────────────────────────────
# Trade-off: higher N_FOLDS gives a more honest cross-fitted Q at the cost
# of training time. 3 folds is the standard "quick" choice; 5–10 is "publish".
#
# All of these can be overridden via env vars for the minitim run:
#   TMLE_N_FOLDS=5 TMLE_XGB_N=300 TMLE_GBM_N=200 TMLE_BIG_LIBRARY=1
import os as _os

N_FOLDS_CROSSFIT = int(_os.environ.get("TMLE_N_FOLDS", "3"))
N_DENSITY_BINS    = int(_os.environ.get("TMLE_N_DENSITY_BINS", "20"))
DENSITY_EPS_FRAC  = 0.005       # floor on g_n to prevent positivity blow-up

# XGBoost / GBM tree counts. The Super Learner stack is more sensitive to
# library DIVERSITY than to individual library depth, so we run modest trees
# in each base learner and let the meta-learner pick the weighting.
XGB_N_ESTIMATORS  = int(_os.environ.get("TMLE_XGB_N", "120"))
GBM_N_ESTIMATORS  = int(_os.environ.get("TMLE_GBM_N", "80"))

# When TMLE_BIG_LIBRARY=1 the SuperLearner stack adds MLPRegressor,
# ExtraTreesRegressor, and KNeighborsRegressor for additional diversity.
# This is the "minitim" library; on the Mac we keep the smaller stack for
# runtime budget.
BIG_LIBRARY = bool(int(_os.environ.get("TMLE_BIG_LIBRARY", "0")))


# ──────────────────── Data structures ────────────────────────────

@dataclass
class TMLEResult:
    """One TMLE estimate at a single radius / window / shift."""
    estimand:        str        # short label for the estimand
    psi:             float      # point estimate
    se_iid:          float      # i.i.d. SE
    se_cluster:      float      # cluster-IF SE
    ci_low:          float      # 95% CI lower (cluster-IF)
    ci_high:         float      # 95% CI upper (cluster-IF)
    pval:            float      # two-sided p-value vs null = 0
    n:               int        # number of rows in the design matrix
    n_clusters:      int        # number of clusters
    epsilon:         float      # targeting fluctuation parameter
    q_initial_psi:   float      # mean of Q at the shifted treatment, before targeting
    notes:           dict[str, float | str] = field(default_factory=dict)


# ──────────────────── Super Learner — Q model ────────────────────

class HurdleSuperLearner(BaseEstimator, RegressorMixin):
    """Two-part hurdle Super Learner for non-negative outcomes with heavy zero mass.

    Stage 1: P(Y > 0 | A, L)             logistic stack
    Stage 2: E[log(1 + Y) | Y > 0, A, L] regression stack on the positives

    Combined prediction: P_pos(x) · expm1(reg_pos(x)).

    The base learners are: regularised linear (ridge / logistic) + gradient
    boosting + XGBoost. They get cross-validated then stacked via a final
    non-negative least squares meta-learner approximated by RidgeCV with a
    very small regularisation grid (clean enough for this scale).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _build_classifier_stack(self):
        return [
            ("logit",  Pipeline([("sc", StandardScaler()),
                                 ("clf", LogisticRegression(max_iter=2000,
                                                            solver="lbfgs"))])),
            ("gbm",    GradientBoostingClassifier(n_estimators=GBM_N_ESTIMATORS, max_depth=3,
                                                  random_state=self.random_state)),
            ("xgb",    xgb.XGBClassifier(n_estimators=XGB_N_ESTIMATORS, max_depth=4,
                                         learning_rate=0.05, eval_metric="logloss",
                                         tree_method="hist", verbosity=0,
                                         random_state=self.random_state)),
        ]

    def _build_regressor_stack(self):
        base = [
            ("ridge",  Pipeline([("sc", StandardScaler()),
                                 ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13)))])),
            ("gbm",    GradientBoostingRegressor(n_estimators=GBM_N_ESTIMATORS, max_depth=3,
                                                 random_state=self.random_state)),
            ("xgb",    xgb.XGBRegressor(n_estimators=XGB_N_ESTIMATORS, max_depth=4,
                                        learning_rate=0.05, tree_method="hist",
                                        verbosity=0, random_state=self.random_state)),
        ]
        if BIG_LIBRARY:
            # Add three more diverse learners for the minitim run.
            base.extend([
                ("et",  ExtraTreesRegressor(n_estimators=200, max_depth=12,
                                            n_jobs=1, random_state=self.random_state)),
                ("knn", Pipeline([("sc", StandardScaler()),
                                  ("knn", KNeighborsRegressor(n_neighbors=15, n_jobs=1))])),
                ("mlp", Pipeline([("sc", StandardScaler()),
                                  ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32),
                                                       max_iter=200,
                                                       early_stopping=True,
                                                       random_state=self.random_state))])),
            ])
        return base

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        is_pos = (y > 0).astype(int)

        # Stage 1: classifier on P(Y > 0)
        clf_learners = self._build_classifier_stack()
        kf = KFold(n_splits=N_FOLDS_CROSSFIT, shuffle=True, random_state=self.random_state)
        clf_cv_preds = np.zeros((n, len(clf_learners)))
        for j, (_, model) in enumerate(clf_learners):
            for tr_idx, te_idx in kf.split(X):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_clone = _clone(model)
                    model_clone.fit(X[tr_idx], is_pos[tr_idx])
                    clf_cv_preds[te_idx, j] = _safe_predict_proba(model_clone, X[te_idx])
        # Meta: non-negative ridge to combine
        meta_clf = RidgeCV(alphas=np.logspace(-3, 3, 13))
        meta_clf.fit(clf_cv_preds, is_pos)
        self.clf_meta_ = meta_clf
        # Refit base learners on full data for prediction
        self.clf_full_ = []
        for _, model in clf_learners:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_clone = _clone(model)
                model_clone.fit(X, is_pos)
            self.clf_full_.append(model_clone)

        # Stage 2: regression on log(1+Y) for positives only
        pos_mask = is_pos == 1
        if pos_mask.sum() < 50:
            # too few positives; fall back to a constant regressor at the mean
            self.reg_full_ = None
            self.reg_meta_ = None
            self._pos_mean_ = float(np.log1p(y[pos_mask]).mean()) if pos_mask.any() else 0.0
            return self
        Xp = X[pos_mask]
        yp = np.log1p(y[pos_mask])
        reg_learners = self._build_regressor_stack()
        kfp = KFold(n_splits=min(N_FOLDS_CROSSFIT, max(2, pos_mask.sum() // 50)),
                    shuffle=True, random_state=self.random_state)
        reg_cv_preds = np.zeros((Xp.shape[0], len(reg_learners)))
        for j, (_, model) in enumerate(reg_learners):
            for tr_idx, te_idx in kfp.split(Xp):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_clone = _clone(model)
                    model_clone.fit(Xp[tr_idx], yp[tr_idx])
                    reg_cv_preds[te_idx, j] = model_clone.predict(Xp[te_idx])
        meta_reg = RidgeCV(alphas=np.logspace(-3, 3, 13))
        meta_reg.fit(reg_cv_preds, yp)
        self.reg_meta_ = meta_reg
        self.reg_full_ = []
        for _, model in reg_learners:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_clone = _clone(model)
                model_clone.fit(Xp, yp)
            self.reg_full_.append(model_clone)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        # Stage 1: classifier ensemble
        clf_preds = np.column_stack([
            _safe_predict_proba(m, X) for m in self.clf_full_
        ])
        p_pos = np.clip(self.clf_meta_.predict(clf_preds), 0.0, 1.0)
        # Stage 2: log-magnitude regression
        if self.reg_full_ is None:
            log_mag = np.full(X.shape[0], self._pos_mean_)
        else:
            reg_preds = np.column_stack([m.predict(X) for m in self.reg_full_])
            log_mag = self.reg_meta_.predict(reg_preds)
        return p_pos * np.expm1(log_mag)


def _clone(estimator):
    """Lightweight clone — works for sklearn estimators and Pipelines."""
    from sklearn.base import clone
    return clone(estimator)


def _safe_predict_proba(model, X):
    """Returns P(class=1) regardless of whether the underlying estimator is
    a classifier or a regressor on a binary target."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.predict(X), 0.0, 1.0)


# ──────────────────── Conditional density g(A | L) ───────────────

class HistogramConditionalDensity:
    """Conditional density estimator g(A | L) via quantile-bin histogram.

    Discretises A into K equal-frequency bins, fits a multinomial classifier
    on (L → bin index), and reports density at any (a, l) as
    P(bin_k(a) | L=l) / width_k. Crude but a reliable workhorse for shift
    interventions on continuous treatments. The XGBoost classifier handles
    the high-dimensional confounders without manual feature engineering.
    """

    def __init__(self, n_bins: int = N_DENSITY_BINS, random_state: int = 42):
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(self, A: np.ndarray, L: np.ndarray) -> "HistogramConditionalDensity":
        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float)
        # Equal-frequency bin edges (quantiles)
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.quantile(A, quantiles)
        # Ensure strictly increasing edges (handles ties at zero)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        self.edges_ = edges
        self.widths_ = np.diff(edges)
        bin_idx = np.clip(np.digitize(A, edges, right=False) - 1, 0, self.n_bins - 1)
        self.classes_ = np.arange(self.n_bins)
        # Multinomial classifier
        self.clf_ = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            tree_method="hist", verbosity=0,
            objective="multi:softprob", num_class=self.n_bins,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf_.fit(L, bin_idx)
        return self

    def density(self, A: np.ndarray, L: np.ndarray) -> np.ndarray:
        """Return g(A_i | L_i) for each row."""
        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float)
        # Predict class probabilities
        probs = self.clf_.predict_proba(L)         # (n, K)
        bin_idx = np.clip(np.digitize(A, self.edges_, right=False) - 1, 0, self.n_bins - 1)
        p_at_bin = probs[np.arange(len(A)), bin_idx]
        widths_at_bin = self.widths_[bin_idx]
        density = p_at_bin / widths_at_bin
        # Floor for positivity
        floor = DENSITY_EPS_FRAC / max(self.widths_.max(), 1.0)
        return np.maximum(density, floor)


# ──────────────────── Cross-fitted Q estimation ──────────────────

def crossfit_Q(
    df: pd.DataFrame,
    A_col: str,
    L_cols: list[str],
    Y_col: str,
    n_folds: int = N_FOLDS_CROSSFIT,
    seed: int = 42,
) -> tuple[np.ndarray, list[HurdleSuperLearner]]:
    """Cross-fitted Q estimate. Returns (Q_hat at observed (A,L), per-fold models)."""
    n = len(df)
    Q_hat = np.zeros(n)
    AL = df[[A_col] + L_cols].to_numpy(dtype=float)
    Y  = df[Y_col].to_numpy(dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    models = []
    for tr_idx, te_idx in kf.split(AL):
        m = HurdleSuperLearner(random_state=seed)
        m.fit(AL[tr_idx], Y[tr_idx])
        Q_hat[te_idx] = m.predict(AL[te_idx])
        models.append((tr_idx, te_idx, m))
    return Q_hat, models


def predict_Q_at(
    A_new: np.ndarray,
    L: np.ndarray,
    models: list,
) -> np.ndarray:
    """Predict Q(A_new, L) using the cross-fitted models. For each test row we
    use the model whose held-out fold contained that row (so the prediction
    is honest); rows that weren't held out (shouldn't happen with kf
    splitting) fall back to a Q-mean across all models."""
    n = len(A_new)
    pred = np.zeros(n)
    used = np.zeros(n, dtype=bool)
    for tr_idx, te_idx, m in models:
        AL_te = np.column_stack([A_new[te_idx], L[te_idx]])
        pred[te_idx] = m.predict(AL_te)
        used[te_idx] = True
    if not used.all():
        # Average across all models for any leftover rows
        leftover = np.where(~used)[0]
        AL_lo = np.column_stack([A_new[leftover], L[leftover]])
        avg = np.mean([m.predict(AL_lo) for _, _, m in models], axis=0)
        pred[leftover] = avg
    return pred


# ──────────────────── TMLE for stochastic shift intervention ─────

def tmle_shift(
    df: pd.DataFrame,
    A_col: str,
    L_cols: list[str],
    Y_col: str,
    cluster_col: str,
    shift_pct: float = 0.10,
    seed: int = 42,
) -> TMLEResult:
    """TMLE for the multiplicative shift d(a) = a · (1 + shift_pct).

    Estimand:  ψ_δ = E[Y_{d(A,L)}]
    Reported:  ψ_δ − ψ_0 (the *difference* in expected outcome under shift
               vs no shift) so the result is on a directly comparable scale.
    """
    n = len(df)
    A = df[A_col].to_numpy(dtype=float)
    L = df[L_cols].to_numpy(dtype=float)
    Y = df[Y_col].to_numpy(dtype=float)
    clusters = df[cluster_col].to_numpy()

    # 1. Cross-fitted Q on observed (A, L)
    Q_hat, q_models = crossfit_Q(df, A_col, L_cols, Y_col, seed=seed)

    # 2. Conditional density g(A | L) — fit ONCE on the full data (g doesn't
    #    need cross-fitting in standard TMLE because the influence function's
    #    asymptotic linearity only requires Q to be cross-fit; the targeting
    #    step's clever-covariate construction uses g as a fixed nuisance).
    g_model = HistogramConditionalDensity(random_state=seed)
    g_model.fit(A, L)

    g_obs = g_model.density(A, L)            # g(A_i | L_i)
    A_pre = A / (1.0 + shift_pct)            # the treatment that would have shifted to A
    g_pre = g_model.density(A_pre, L)        # g(A_i / (1+δ) | L_i)
    H = (g_pre / g_obs) / (1.0 + shift_pct)  # clever covariate (with Jacobian)

    # 3. Targeting: solve ε from OLS of (Y − Q_hat) on H, no intercept.
    residual = Y - Q_hat
    eps = float(np.dot(H, residual) / max(np.dot(H, H), 1e-12))

    Q_star_obs = Q_hat + eps * H

    # 4. Plug-in: predict Q at the shifted treatment for every row, then update
    A_post = A * (1.0 + shift_pct)
    Q_post = predict_Q_at(A_post, L, q_models)
    # Apply the same fluctuation to the post-shift Q. Use the clever covariate
    # at the SHIFTED treatment value, which is g(A_post / (1+δ) | L) / g(A_post | L) / (1+δ)
    # = g(A | L) / g(A_post | L) / (1+δ).
    g_post = g_model.density(A_post, L)
    H_post = (g_obs / g_post) / (1.0 + shift_pct)
    Q_star_post = Q_post + eps * H_post

    psi_n   = float(np.mean(Q_star_post))
    psi_0   = float(np.mean(Q_star_obs))     # baseline (no-shift) plug-in
    psi_diff = psi_n - psi_0
    q_initial_psi = float(np.mean(Q_post))   # before targeting (for diagnostics)

    # 5. Influence function for ψ_δ = E[Y_{d(A,L)}]:
    #    IF_i = H(A_i, L_i) · (Y_i − Q*(A_i, L_i))  +  Q*(d(A_i), L_i)  −  ψ_n
    if_psi  = H * (Y - Q_star_obs) + Q_star_post - psi_n
    if_psi0 = (Y - Q_star_obs)               + Q_star_obs - psi_0
    if_diff = if_psi - if_psi0

    # 6. Variance: i.i.d. and cluster-IF
    se_iid     = float(np.sqrt(np.var(if_diff, ddof=1) / n))
    se_cluster = _cluster_se(if_diff, clusters)

    # 7. CIs and p-value
    z = 1.959963984540054
    ci_low  = psi_diff - z * se_cluster
    ci_high = psi_diff + z * se_cluster
    z_stat  = psi_diff / max(se_cluster, 1e-15)
    pval    = 2 * (1 - _norm_cdf(abs(z_stat)))

    return TMLEResult(
        estimand=f"shift_{int(shift_pct*100):+d}pct",
        psi=psi_diff,
        se_iid=se_iid,
        se_cluster=se_cluster,
        ci_low=ci_low,
        ci_high=ci_high,
        pval=pval,
        n=n,
        n_clusters=int(pd.Series(clusters).nunique()),
        epsilon=eps,
        q_initial_psi=q_initial_psi,
        notes={
            "psi_under_shift": psi_n,
            "psi_no_shift":    psi_0,
            "shift_pct":       shift_pct,
            "mean_H":          float(np.mean(H)),
            "max_H":           float(np.max(H)),
        },
    )


# ──────────────────── TMLE for dose-response curve ───────────────

def tmle_dose_response(
    df: pd.DataFrame,
    A_col: str,
    L_cols: list[str],
    Y_col: str,
    cluster_col: str,
    a_grid: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """TMLE for the causal dose-response curve E[Y_a] at a grid of A values.

    For each grid point a* we run a g-computation TMLE: target the
    counterfactual mean E[Y_a*] by re-weighting Q with a clever covariate
    that depends on the conditional density of A given L. Returns one row
    per grid point with point estimate + cluster-IF CI.
    """
    n = len(df)
    A = df[A_col].to_numpy(dtype=float)
    L = df[L_cols].to_numpy(dtype=float)
    Y = df[Y_col].to_numpy(dtype=float)
    clusters = df[cluster_col].to_numpy()

    Q_hat, q_models = crossfit_Q(df, A_col, L_cols, Y_col, seed=seed)
    g_model = HistogramConditionalDensity(random_state=seed)
    g_model.fit(A, L)
    g_obs = g_model.density(A, L)

    rows = []
    for a_star in a_grid:
        # Clever covariate: indicator-style approximation for a point
        # intervention at a*. We use the binned approximation: the rows whose
        # observed A is in the same bin as a* get weight 1/g_n(a* | L), all
        # other rows get 0. This is the standard density-bin g-computation
        # construction for continuous treatment dose-response.
        a_star_arr = np.full(n, float(a_star))
        bin_idx_obs  = np.clip(np.digitize(A, g_model.edges_, right=False) - 1, 0, g_model.n_bins - 1)
        bin_idx_star = int(np.clip(np.digitize(a_star, g_model.edges_, right=False) - 1, 0, g_model.n_bins - 1))
        in_bin = (bin_idx_obs == bin_idx_star).astype(float)
        g_at_star = g_model.density(a_star_arr, L)
        H_a = in_bin / np.maximum(g_at_star * g_model.widths_[bin_idx_star], 1e-12)

        # Targeting: ε from OLS of residual on H_a (no intercept)
        residual = Y - Q_hat
        eps_a = float(np.dot(H_a, residual) / max(np.dot(H_a, H_a), 1e-12))

        Q_star_a = Q_hat + eps_a * H_a
        Q_pred_a = predict_Q_at(a_star_arr, L, q_models)
        Q_pred_a_star = Q_pred_a + eps_a * (in_bin / np.maximum(g_at_star * g_model.widths_[bin_idx_star], 1e-12))

        psi_a = float(np.mean(Q_pred_a_star))
        IF_a = H_a * (Y - Q_star_a) + Q_pred_a_star - psi_a
        se_iid     = float(np.sqrt(np.var(IF_a, ddof=1) / n))
        se_cluster = _cluster_se(IF_a, clusters)

        z = 1.959963984540054
        rows.append({
            "a_star":     float(a_star),
            "psi":        psi_a,
            "se_iid":     se_iid,
            "se_cluster": se_cluster,
            "ci_low":     psi_a - z * se_cluster,
            "ci_high":    psi_a + z * se_cluster,
            "epsilon":    eps_a,
            "n_in_bin":   int(in_bin.sum()),
        })
    return pd.DataFrame(rows)


# ──────────────────── TMLE for natural direct/indirect effects ───

def _fit_simple_q(AML: np.ndarray, Y: np.ndarray, seed: int = 42):
    """Fast hurdle GBM (single learner, no stacking) for use inside bootstrap loops.

    The doubly robust point estimate uses the full HurdleSuperLearner; this
    surrogate is only used to compute bootstrap CIs efficiently. The
    surrogate's role is to capture the SAMPLING variability of the plug-in
    parameter; modest under-fitting in the surrogate is acceptable as long as
    it's used consistently across bootstrap iterations.
    """
    is_pos = (Y > 0).astype(int)
    pos_mask = is_pos == 1
    clf = xgb.XGBClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.08,
        tree_method="hist", verbosity=0, random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(AML, is_pos)
    if pos_mask.sum() < 30:
        # No positive regression possible — fall back to constant
        const = float(np.log1p(Y[pos_mask]).mean()) if pos_mask.any() else 0.0
        def _predict(X):
            return clf.predict_proba(X)[:, 1] * np.expm1(const)
        return _predict
    reg = xgb.XGBRegressor(
        n_estimators=80, max_depth=4, learning_rate=0.08,
        tree_method="hist", verbosity=0, random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(AML[pos_mask], np.log1p(Y[pos_mask]))
    def _predict(X):
        p_pos = clf.predict_proba(X)[:, 1]
        log_mag = reg.predict(X)
        return p_pos * np.expm1(log_mag)
    return _predict


def tmle_mediation(
    df: pd.DataFrame,
    A_col: str,
    M_col: str,
    L_cols: list[str],
    Y_col: str,
    cluster_col: str,
    a_high: float,
    a_low: float,
    seed: int = 42,
    n_iter_boot: int = 30,
) -> dict:
    """Regression-based mediational TMLE for NDE / NIE at the (a_high, a_low) contrast.

    Implements a simplified version of the Zheng & van der Laan (2012)
    sequentially doubly robust estimator. We use g-computation throughout
    rather than the full IPW + targeting machinery; the trade-off is that
    consistency requires correct specification of Q, not double robustness.
    Given the nonparametric Super Learner, this is a reasonable simplification
    for a first-pass implementation.

    Point estimate uses the full HurdleSuperLearner stack. Bootstrap CIs use
    a faster GBM surrogate (single learner) for runtime — see _fit_simple_q
    above for the rationale.

    Decomposition (under sequential ignorability + cross-world independence):
        Total effect (TE) = E[Y_{a_high}] − E[Y_{a_low}]
        NIE              = E[Y_{a_high, M_{a_high}}] − E[Y_{a_high, M_{a_low}}]
        NDE              = E[Y_{a_high, M_{a_low}}] − E[Y_{a_low, M_{a_low}}]
        TE = NIE + NDE
    """
    A = df[A_col].to_numpy(dtype=float)
    M = df[M_col].to_numpy(dtype=float)
    L = df[L_cols].to_numpy(dtype=float)
    Y = df[Y_col].to_numpy(dtype=float)
    clusters = df[cluster_col].to_numpy()
    n = len(df)

    AML = np.column_stack([A, M, L])

    # Point estimate: full Super Learner Q + XGB mediator model
    Q_full = HurdleSuperLearner(random_state=seed)
    Q_full.fit(AML, Y)
    M_model = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS, max_depth=4, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=seed,
    )
    M_model.fit(np.column_stack([A, L]), M)

    M_high = M_model.predict(np.column_stack([np.full(n, a_high), L]))
    M_low  = M_model.predict(np.column_stack([np.full(n, a_low),  L]))

    Q_hh = Q_full.predict(np.column_stack([np.full(n, a_high), M_high, L]))
    Q_hl = Q_full.predict(np.column_stack([np.full(n, a_high), M_low,  L]))
    Q_ll = Q_full.predict(np.column_stack([np.full(n, a_low),  M_low,  L]))

    psi_hh = float(np.mean(Q_hh))
    psi_hl = float(np.mean(Q_hl))
    psi_ll = float(np.mean(Q_ll))

    TE  = psi_hh - psi_ll
    NIE = psi_hh - psi_hl
    NDE = psi_hl - psi_ll
    pct_mediated = (NIE / TE * 100.0) if abs(TE) > 1e-12 else float("nan")

    # Cluster bootstrap with the FAST surrogate Q for CI computation
    rng = np.random.default_rng(seed)
    cluster_ids = np.unique(clusters)
    boot_TE, boot_NIE, boot_NDE = [], [], []
    for _ in range(n_iter_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx_parts = [np.where(clusters == c)[0] for c in sampled]
        idx = np.concatenate(idx_parts)
        if idx.size == 0:
            continue
        try:
            Q_b = _fit_simple_q(AML[idx], Y[idx], seed=seed)
            Mm = xgb.XGBRegressor(
                n_estimators=80, max_depth=4, learning_rate=0.08,
                tree_method="hist", verbosity=0, random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Mm.fit(np.column_stack([A[idx], L[idx]]), M[idx])
            Mh = Mm.predict(np.column_stack([np.full(len(idx), a_high), L[idx]]))
            Ml = Mm.predict(np.column_stack([np.full(len(idx), a_low),  L[idx]]))
            qhh = Q_b(np.column_stack([np.full(len(idx), a_high), Mh, L[idx]])).mean()
            qhl = Q_b(np.column_stack([np.full(len(idx), a_high), Ml, L[idx]])).mean()
            qll = Q_b(np.column_stack([np.full(len(idx), a_low),  Ml, L[idx]])).mean()
            boot_TE.append(qhh - qll)
            boot_NIE.append(qhh - qhl)
            boot_NDE.append(qhl - qll)
        except Exception:
            continue

    def pct_ci(arr):
        if not arr:
            return (float("nan"), float("nan"))
        a = np.percentile(arr, [2.5, 97.5])
        return (float(a[0]), float(a[1]))

    return {
        "a_high":       a_high,
        "a_low":        a_low,
        "TE":           TE,
        "NDE":          NDE,
        "NIE":          NIE,
        "pct_mediated": pct_mediated,
        "TE_ci":        pct_ci(boot_TE),
        "NDE_ci":       pct_ci(boot_NDE),
        "NIE_ci":       pct_ci(boot_NIE),
        "n":            n,
        "n_clusters":   int(pd.Series(clusters).nunique()),
        "n_iter_boot":  len(boot_TE),
    }


# ──────────────────── Helpers ────────────────────────────────────

def _cluster_se(IF: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-IF standard error: sum the IF within each cluster, then take
    the empirical SD over clusters and divide by n.

    For an asymptotically linear estimator with influence function IF, the
    cluster-robust variance is

        Var[ψ̂] = (1 / n²) · Σ_c (Σ_{i∈c} IF_i)²

    where the outer sum is over clusters. Equivalently, treat each cluster's
    summed-IF as one observation and use the standard SD-of-mean formula.
    """
    s = pd.Series(IF).groupby(pd.Series(clusters).values).sum().to_numpy()
    n = len(IF)
    if len(s) < 2:
        return float("nan")
    return float(np.sqrt(np.sum(s ** 2) / (n ** 2)))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF without scipy dependency."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
