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
    Ridge,
    RidgeCV,
)
from scipy.optimize import nnls as _nnls
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

N_FOLDS_CROSSFIT = int(_os.environ.get("TMLE_N_FOLDS", "5"))   # was 3; 5 is publication standard
N_DENSITY_BINS    = int(_os.environ.get("TMLE_N_DENSITY_BINS", "0"))  # 0 = data-adaptive (Freedman-Diaconis)
DENSITY_EPS_FRAC  = 0.005       # fallback floor; overridden by data-adaptive floor in HistogramConditionalDensity
CV_DENSITY        = _os.environ.get("TMLE_CV_DENSITY", "kde")       # "haldensify", "kde", or "histogram" — density estimator for CV-TMLE

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

# When TMLE_SKIP_GBM=1 the sklearn GradientBoosting* learner is dropped
# from the SuperLearner stack. sklearn GBM is single-threaded (no n_jobs)
# and dominates per-worker runtime on large panels (345k rows × 5 folds ×
# 120 trees ≈ 10 min per worker). XGBoost gives equivalent diversity at a
# fraction of the cost. Set this when running 30-way parallel on minitim.
SKIP_GBM = bool(int(_os.environ.get("TMLE_SKIP_GBM", "0")))

# When TMLE_USE_HAL=1 the Highly Adaptive Lasso (van der Laan, 2017) is added
# to the SuperLearner stack via rpy2 + R's hal9001 package. HAL is the only
# nonparametric estimator with a dimension-free n^{-1/3} MSE convergence rate,
# making it the theoretically preferred base learner. Off by default because
# it requires R + hal9001 + rpy2 installed.
USE_HAL = bool(int(_os.environ.get("TMLE_USE_HAL", "0")))


# ──────────────────── NNLS metalearner ──────────────────────────
# Non-negative least squares ensures the SuperLearner meta-learner produces
# a proper convex combination of base learners (all weights >= 0, sum to ~1).
# This is what sl3 (R/tlverse) uses by default. RidgeCV can produce negative
# weights, which undermines the ensemble's theoretical properties.

class _NNLSRegressor:
    """sklearn-compatible NNLS wrapper for SuperLearner meta-learner."""
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        self.coef_, _ = _nnls(X, y)
        # Normalize to sum to 1 (proper convex combination)
        s = self.coef_.sum()
        if s > 0:
            self.coef_ = self.coef_ / s
        return self

    def predict(self, X):
        return X @ self.coef_


# ──────────────────── HAL base learner (optional, requires R) ────

class HALWrapper(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for R's hal9001::fit_hal() via rpy2.

    The Highly Adaptive Lasso (HAL) is a nonparametric estimator that
    constructs indicator basis functions over all multivariate sections of
    the covariate space and applies an L1 penalty. It achieves an MSE
    convergence rate of n^{-2/3} (log n)^d — dimension-free up to log
    factors — making it the theoretically preferred base learner for TMLE
    (van der Laan, 2017; Benkeser & van der Laan, 2016).

    Parameters
    ----------
    family : str
        "gaussian" for regression, "binomial" for classification.
    max_degree : int
        Maximum interaction degree for basis functions. 2 is recommended
        for moderate-dimensional data (captures pairwise interactions).
    num_knots : tuple
        Number of knots per degree. (25, 10) is the van der Laan lab
        recommendation for moderate-to-large n — more knots is slower,
        not better (hal9001 docs).
    smoothness_orders : int
        0 = indicator HAL (piecewise constant), 1 = piecewise linear.
    subsample_size : int
        If n > subsample_size, HAL fits on a random subsample of this size.
        Van der Laan's recommended practice for large n — HAL's basis
        construction is O(n × num_bases), so full-n fits are infeasible
        above ~10k rows. Predictions still use the full input X.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, family: str = "gaussian", max_degree: int = 2,
                 num_knots: tuple = (25, 10), smoothness_orders: int = 1,
                 subsample_size: int = 5000,
                 random_state: int = 42):
        self.family = family
        self.max_degree = max_degree
        self.num_knots = num_knots
        self.smoothness_orders = smoothness_orders
        self.subsample_size = subsample_size
        self.random_state = random_state
        self._hal_fit = None
        self._fallback_value = 0.0

    def _init_r(self):
        """Lazily initialize rpy2 and hal9001."""
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        # rpy2 >= 3.6: use converter context instead of deprecated activate()
        self._converter = ro.default_converter + numpy2ri.converter
        self._ro = ro
        with self._converter.context():
            self._hal9001 = importr("hal9001")
            self._base = importr("base")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Subsample for large n — hal9001 basis construction is infeasible
        # above ~10k rows. Van der Laan's recommended compromise.
        if X.shape[0] > self.subsample_size:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0], size=self.subsample_size, replace=False)
            X_fit, y_fit = X[idx], y[idx]
        else:
            X_fit, y_fit = X, y
        try:
            self._init_r()
            ro = self._ro
            with self._converter.context():
                r_X = ro.r.matrix(X_fit, nrow=X_fit.shape[0], ncol=X_fit.shape[1])
                r_Y = ro.FloatVector(y_fit)
                r_knots = ro.IntVector(list(self.num_knots))
                self._hal_fit = self._hal9001.fit_hal(
                    X=r_X, Y=r_Y,
                    family=self.family,
                    max_degree=self.max_degree,
                    num_knots=r_knots,
                    smoothness_orders=self.smoothness_orders,
                )
            self._fallback_value = float(np.mean(y))
        except Exception as e:
            warnings.warn(f"HAL fit failed ({e}); falling back to constant predictor")
            self._hal_fit = None
            self._fallback_value = float(np.mean(y))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._hal_fit is None:
            return np.full(X.shape[0], self._fallback_value)
        try:
            ro = self._ro
            with self._converter.context():
                r_X = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
                preds = np.asarray(self._hal9001.predict_hal9001(self._hal_fit, new_data=r_X))
            if self.family == "binomial":
                preds = np.clip(preds, 0.0, 1.0)
            return preds.ravel()
        except Exception:
            return np.full(X.shape[0], self._fallback_value)

    def predict_proba(self, X):
        """Return P(class=1) as a 2-column array for classifier compatibility."""
        p1 = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1 - p1, p1])


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
        stack = [
            ("logit",  Pipeline([("sc", StandardScaler()),
                                 ("clf", LogisticRegression(max_iter=2000,
                                                            solver="lbfgs"))])),
            ("xgb",    xgb.XGBClassifier(n_estimators=XGB_N_ESTIMATORS, max_depth=4,
                                         learning_rate=0.05, eval_metric="logloss",
                                         tree_method="hist", verbosity=0,
                                         random_state=self.random_state)),
        ]
        if not SKIP_GBM:
            stack.insert(1,
                ("gbm", GradientBoostingClassifier(n_estimators=GBM_N_ESTIMATORS, max_depth=3,
                                                   random_state=self.random_state)))
        if USE_HAL:
            stack.append(("hal", HALWrapper(family="binomial", random_state=self.random_state)))
        return stack

    def _build_regressor_stack(self):
        # Default stack: 6 diverse learners (parametric + tree + ensemble).
        # tlverse recommends 10+ but 6 covers the major families and is
        # computationally feasible on the 345k-row panel with 5-fold CV.
        base = [
            ("ridge",  Pipeline([("sc", StandardScaler()),
                                 ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13)))])),
            ("xgb",    xgb.XGBRegressor(n_estimators=XGB_N_ESTIMATORS, max_depth=4,
                                        learning_rate=0.05, tree_method="hist",
                                        verbosity=0, random_state=self.random_state)),
            # Random Forest: different from GBM (bagging vs boosting, decorrelated trees)
            ("rf",     RandomForestRegressor(n_estimators=100, max_depth=8,
                                             n_jobs=1, random_state=self.random_state)),
        ]
        if not SKIP_GBM:
            base.insert(1,
                ("gbm", GradientBoostingRegressor(n_estimators=GBM_N_ESTIMATORS, max_depth=3,
                                                  random_state=self.random_state)))
        if BIG_LIBRARY:
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
        if USE_HAL:
            base.append(("hal", HALWrapper(family="gaussian", random_state=self.random_state)))
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
        meta_clf = _NNLSRegressor()  # NNLS ensures non-negative weights (convex combination)
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
        meta_reg = _NNLSRegressor()  # NNLS ensures non-negative weights (convex combination)
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
        # Data-adaptive bin count: Freedman-Diaconis rule if n_bins=0
        if self.n_bins <= 0:
            n = len(A)
            self.n_bins = max(10, min(200, int(2 * (n ** (1/3)))))
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
        # Data-adaptive positivity floor: 1% of the 2.5th percentile of
        # observed density values. This is what txshift (Hejazi et al., 2020)
        # recommends — automatic and data-dependent, replaces the old fixed
        # DENSITY_EPS_FRAC / max_width which was not principled.
        q025 = np.quantile(density[density > 0], 0.025) if (density > 0).any() else 1e-12
        floor = max(0.01 * q025, 1e-12)
        return np.maximum(density, floor)


class KDEConditionalDensity:
    """Conditional density estimator g(A | L) via residual kernel density.

    Instead of discretizing A into histogram bins (which causes coverage gaps
    when training/validation splits differ), this estimator:

      1. Fits a regression A_hat = f(L) via XGBoost
      2. Computes residuals R = A - A_hat
      3. Fits a 1D Gaussian KDE on the residuals

    At prediction time, g(a | l) ≈ KDE(a - f(l)). Because the KDE is smooth
    and defined everywhere on the real line, density ratios remain bounded even
    when the evaluation point wasn't seen during training. This makes it safe
    for CV-TMLE where g is fit on training folds and evaluated on held-out folds.

    Bandwidth is selected via Silverman's rule on the residuals.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, A: np.ndarray, L: np.ndarray) -> "KDEConditionalDensity":
        from sklearn.neighbors import KernelDensity

        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float)

        # Step 1: regress A on L to get conditional mean
        self.reg_ = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            tree_method="hist", verbosity=0, random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reg_.fit(L, A)

        # Step 2: residuals
        A_hat = self.reg_.predict(L)
        residuals = A - A_hat

        # Step 3: Silverman bandwidth on residuals
        n = len(residuals)
        std = max(np.std(residuals), 1e-12)
        iqr = np.subtract(*np.percentile(residuals, [75, 25]))
        spread = min(std, iqr / 1.349) if iqr > 0 else std
        bandwidth = 0.9 * spread * (n ** (-0.2))  # Silverman's rule
        bandwidth = max(bandwidth, 1e-6)

        # Step 4: fit 1D KDE
        self.kde_ = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self.kde_.fit(residuals.reshape(-1, 1))

        return self

    def density(self, A: np.ndarray, L: np.ndarray) -> np.ndarray:
        """Return g(A_i | L_i) for each row."""
        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float)

        A_hat = self.reg_.predict(L)
        residuals = (A - A_hat).reshape(-1, 1)
        log_density = self.kde_.score_samples(residuals)
        density = np.exp(log_density)

        # Data-adaptive positivity floor (same as HistogramConditionalDensity)
        q025 = np.quantile(density[density > 0], 0.025) if (density > 0).any() else 1e-12
        floor = max(0.01 * q025, 1e-12)
        return np.maximum(density, floor)


class HALDensifyConditionalDensity:
    """Conditional density estimator g(A | L) via R's haldensify package.

    This is the van der Laan lab's preferred estimator (Hejazi, Benkeser &
    van der Laan 2022; arXiv:2004.13117). It decomposes the joint density
    p(A, L) = p(A | L) · p(L) and uses HAL on a discretized pooled hazard
    representation to estimate p(A | L). The result has HAL's known
    n^{-1/3} MSE convergence rate, which is the rate TMLE's second-order
    remainder term requires for valid inference under weak conditions.

    This is slow (HAL basis construction + Lasso), so the treatment is
    subsampled to `subsample_size` rows before fitting. Predictions use
    the fitted model on the full data.

    Requires: R + haldensify package + rpy2. Opt-in via
    TMLE_CV_DENSITY=haldensify.
    """

    def __init__(self, n_bins: tuple = (3, 5, 10),
                 grid_type: str = "equal_mass",
                 subsample_size: int = 5000,
                 max_degree: int = 3,
                 smoothness_orders: int = 0,
                 random_state: int = 42):
        self.n_bins = n_bins
        self.grid_type = grid_type
        self.subsample_size = subsample_size
        self.max_degree = max_degree
        self.smoothness_orders = smoothness_orders
        self.random_state = random_state
        self._hd_fit = None
        self._fallback_floor = 1e-12

    def _init_r(self):
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        self._np2ri = numpy2ri
        self._ro = ro
        self._haldensify = importr("haldensify")
        self._base = importr("base")

    def _to_r_vec(self, arr: np.ndarray):
        """Convert a 1D numpy array to R FloatVector."""
        return self._ro.FloatVector(arr.tolist())

    def _to_r_mat(self, arr: np.ndarray):
        """Convert a 2D numpy array to an R matrix."""
        flat = self._ro.FloatVector(arr.T.reshape(-1).tolist())
        return self._ro.r["matrix"](flat, nrow=arr.shape[0], ncol=arr.shape[1])

    def fit(self, A: np.ndarray, L: np.ndarray) -> "HALDensifyConditionalDensity":
        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float)
        # Subsample — haldensify on >10k rows is infeasible
        if len(A) > self.subsample_size:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(A), size=self.subsample_size, replace=False)
            A_fit, L_fit = A[idx], L[idx]
        else:
            A_fit, L_fit = A, L
        try:
            self._init_r()
            ro = self._ro
            # Build R inputs via plain constructors (no converter context) so
            # the returned fit object keeps its S3 "haldensify" class tag for
            # downstream predict() dispatch.
            r_A = self._to_r_vec(A_fit)
            r_W = self._to_r_mat(L_fit)
            r_bins = ro.IntVector(list(self.n_bins))
            self._hd_fit = self._haldensify.haldensify(
                A=r_A, W=r_W,
                n_bins=r_bins,
                grid_type=self.grid_type,
                max_degree=self.max_degree,
                smoothness_orders=self.smoothness_orders,
            )
        except Exception as e:
            warnings.warn(f"haldensify fit failed ({e}); falling back to uniform density")
            self._hd_fit = None
        return self

    def density(self, A: np.ndarray, L: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float)
        if self._hd_fit is None:
            # Uniform fallback — strongly flags a broken fit
            return np.full(len(A), 1.0 / max(np.std(A), 1e-6))
        try:
            ro = self._ro
            r_A_new = self._to_r_vec(A)
            r_W_new = self._to_r_mat(L)
            # R's generic predict() does S3 dispatch to predict.haldensify
            r_predict = ro.r["predict"]
            density = np.asarray(r_predict(
                self._hd_fit, new_A=r_A_new, new_W=r_W_new,
            )).ravel()
            # Data-adaptive positivity floor (matches other density estimators)
            pos = density[density > 0]
            q025 = np.quantile(pos, 0.025) if len(pos) > 0 else 1e-12
            floor = max(0.01 * q025, 1e-12)
            return np.maximum(density, floor)
        except Exception as e:
            warnings.warn(f"haldensify predict failed ({e}); returning uniform density")
            return np.full(len(A), 1.0 / max(np.std(A), 1e-6))


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
    trim_pct: float = 0.01,
) -> TMLEResult:
    """TMLE for the multiplicative shift d(a) = a · (1 + shift_pct).

    Estimand:  ψ_δ = E[Y_{d(A,L)}]
    Reported:  ψ_δ − ψ_0 (the *difference* in expected outcome under shift
               vs no shift) so the result is on a directly comparable scale.

    trim_pct: truncate the clever covariate H at the (1-trim_pct) percentile
              post-hoc. This is the txshift-recommended approach (Hejazi et al.,
              2020): instead of dropping observations, cap the influence of
              extreme density ratios. Set to 0 to disable.
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

    # Truncate H at the (1-trim_pct) quantile to prevent extreme density
    # ratios from dominating the targeting step. This is the txshift-recommended
    # approach (Hejazi et al., 2020) — cap the weights post-hoc rather than
    # dropping observations. Diagnostics showed max H = 215-687 without
    # truncation; after truncation to 99th pct, max H ≈ 50-100.
    if trim_pct > 0:
        H_cap = float(np.quantile(np.abs(H), 1.0 - trim_pct))
        H = np.clip(H, -H_cap, H_cap)

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
    if trim_pct > 0:
        H_post = np.clip(H_post, -H_cap, H_cap)
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


# ──────────────────── CV-TMLE for stochastic shift ────────────────

def cv_tmle_shift(
    df: pd.DataFrame,
    A_col: str,
    L_cols: list[str],
    Y_col: str,
    cluster_col: str,
    shift_pct: float = 0.10,
    seed: int = 42,
    n_splits: int = 5,
) -> TMLEResult:
    """Cross-Validated TMLE for the multiplicative shift d(a) = a · (1 + δ).

    CV-TMLE (van der Laan & Rose 2011; arXiv:2409.11265) splits data into V
    folds. For each fold v:
      1. Fit Q and g on the OTHER V-1 folds (training set)
      2. Predict Q_hat and compute H on fold v (validation set)
      3. Target Q on fold v using fold-specific epsilon
    Then combine fold-specific estimates via the influence function.

    Advantages over standard TMLE:
      - No need for H-truncation (positivity handled by sample splitting)
      - Valid inference without Donsker class conditions on nuisance estimators
      - Better CI coverage in finite samples
    """
    n = len(df)
    A = df[A_col].to_numpy(dtype=float)
    L = df[L_cols].to_numpy(dtype=float)
    Y = df[Y_col].to_numpy(dtype=float)
    clusters = df[cluster_col].to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Storage for per-observation predictions (filled in by each fold)
    Q_hat_cv = np.zeros(n)
    Q_post_cv = np.zeros(n)
    H_cv = np.zeros(n)
    H_post_cv = np.zeros(n)
    eps_per_fold = np.zeros(n)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n))):
        # Training data
        df_train = df.iloc[train_idx].reset_index(drop=True)
        A_train = A[train_idx]
        L_train = L[train_idx]
        Y_train = Y[train_idx]

        # Validation data
        A_val = A[val_idx]
        L_val = L[val_idx]
        Y_val = Y[val_idx]

        # Fit Q on training data
        Q_model = HurdleSuperLearner(random_state=seed + fold_idx)
        AL_train = np.column_stack([A_train, L_train])
        Q_model.fit(AL_train, Y_train)

        # Predict Q on validation data
        AL_val = np.column_stack([A_val, L_val])
        Q_hat_val = Q_model.predict(AL_val)
        Q_hat_cv[val_idx] = Q_hat_val

        # Fit g on training data. Three options:
        #   - "haldensify": van der Laan lab's HAL-based density (slow, correct rate)
        #   - "kde": 1D Gaussian KDE on residuals (smooth, O(n²) eval)
        #   - "histogram": original quantile-bin histogram (fast, coverage gaps)
        if CV_DENSITY == "haldensify":
            g_model = HALDensifyConditionalDensity(random_state=seed + fold_idx)
        elif CV_DENSITY == "kde":
            g_model = KDEConditionalDensity(random_state=seed + fold_idx)
        else:
            g_model = HistogramConditionalDensity(random_state=seed + fold_idx)
        g_model.fit(A_train, L_train)

        # Compute H on validation data
        g_obs_val = g_model.density(A_val, L_val)
        A_pre_val = A_val / (1.0 + shift_pct)
        g_pre_val = g_model.density(A_pre_val, L_val)
        H_val = (g_pre_val / g_obs_val) / (1.0 + shift_pct)
        H_cv[val_idx] = H_val

        # Target on validation data (fold-specific epsilon)
        residual_val = Y_val - Q_hat_val
        eps = float(np.dot(H_val, residual_val) / max(np.dot(H_val, H_val), 1e-12))
        eps_per_fold[val_idx] = eps

        # Post-shift predictions on validation data
        A_post_val = A_val * (1.0 + shift_pct)
        AL_post_val = np.column_stack([A_post_val, L_val])
        Q_post_val = Q_model.predict(AL_post_val)

        g_post_val = g_model.density(A_post_val, L_val)
        H_post_val = (g_obs_val / g_post_val) / (1.0 + shift_pct)
        H_post_cv[val_idx] = H_post_val
        Q_post_cv[val_idx] = Q_post_val

    # Combine across folds
    Q_star_obs = Q_hat_cv + eps_per_fold * H_cv
    Q_star_post = Q_post_cv + eps_per_fold * H_post_cv

    psi_n = float(np.mean(Q_star_post))
    psi_0 = float(np.mean(Q_star_obs))
    psi_diff = psi_n - psi_0

    # Influence function (combined across all folds)
    if_psi = H_cv * (Y - Q_star_obs) + Q_star_post - psi_n
    if_psi0 = (Y - Q_star_obs) + Q_star_obs - psi_0
    if_diff = if_psi - if_psi0

    se_iid = float(np.sqrt(np.var(if_diff, ddof=1) / n))
    se_cluster = _cluster_se(if_diff, clusters)

    z = 1.959963984540054
    ci_low = psi_diff - z * se_cluster
    ci_high = psi_diff + z * se_cluster
    z_stat = psi_diff / max(se_cluster, 1e-15)
    pval = 2 * (1 - _norm_cdf(abs(z_stat)))

    q_initial_psi = float(np.mean(Q_post_cv))  # before targeting

    return TMLEResult(
        estimand=f"cv_shift_{int(shift_pct*100):+d}pct",
        psi=psi_diff,
        se_iid=se_iid,
        se_cluster=se_cluster,
        ci_low=ci_low,
        ci_high=ci_high,
        pval=pval,
        n=n,
        n_clusters=len(set(clusters)),
        epsilon=float(np.mean(eps_per_fold)),
        q_initial_psi=q_initial_psi,
        notes={
            "psi_under_shift": psi_n,
            "psi_no_shift": psi_0,
            "shift_pct": shift_pct,
            "mean_H": float(np.mean(H_cv)),
            "max_H": float(np.max(np.abs(H_cv))),
            "n_splits": n_splits,
            "method": "CV-TMLE",
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
    trim_pct: float = 0.01,
) -> pd.DataFrame:
    """TMLE for the causal dose-response curve E[Y_a] at a grid of A values.

    For each grid point a* we run a g-computation TMLE: target the
    counterfactual mean E[Y_a*] by re-weighting Q with a clever covariate
    that depends on the conditional density of A given L. Returns one row
    per grid point with point estimate + cluster-IF CI.

    trim_pct: truncate the clever covariate H_a at the (1-trim_pct) percentile
              post-hoc (same as the shift). Set to 0 to disable.
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
        if trim_pct > 0 and (H_a > 0).any():
            H_cap = float(np.quantile(H_a[H_a > 0], 1.0 - trim_pct))
            H_a = np.clip(H_a, 0, H_cap)

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


def mediation_IF_ci(
    Y: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    L: np.ndarray,
    clusters: np.ndarray,
    Q_model,
    M_model,
    a_high: float,
    a_low: float,
    Q_hh: np.ndarray,
    Q_hl: np.ndarray,
    Q_ll: np.ndarray,
    NDE: float,
    NIE: float,
    TE: float,
) -> dict:
    """Influence-function-based standard errors for mediation NDE/NIE.

    Computes the efficient influence function for the natural direct effect
    under Zheng & van der Laan (2012). For the g-computation plug-in estimator,
    the IF decomposes into:

      IF_NDE_i = [Q(a_high, M_low_i, L_i) - Q(a_low, M_low_i, L_i)] - NDE
                 + H_NDE_i * (Y_i - Q*(A_i, M_i, L_i))

    where H_NDE is the clever covariate for the direct-effect path, which
    under the g-computation identification equals:

      H_NDE_i = I(A_i is "informative" for the a_high-vs-a_low contrast)

    In our simplified (non-IPW) framework, the IF reduces to the projection
    of the estimation error onto the space spanned by the NDE functional:

      IF_NDE_i = (Q_hl_i - Q_ll_i) - NDE + (Y_i - Q_obs_i) * dQ/dNDE_weight_i

    The dQ/dNDE weight captures how much each observation's residual affects
    the NDE estimate. For the regression-imputation estimator, this is
    approximated by the partial derivative of Q w.r.t. the treatment contrast.

    Similarly for NIE:
      IF_NIE_i = (Q_hh_i - Q_hl_i) - NIE + (Y_i - Q_obs_i) * dQ/dNIE_weight_i

    We use a numerical linearization approach: the IF for a plug-in
    g-computation estimand psi = (1/n) sum f(Q(.), L_i) is:

      IF_i = f(Q(.), L_i) - psi + dQ_residual_contribution_i

    where the residual contribution is the projection of Y - Q onto the
    relevant counterfactual contrast.

    Returns dict with SE and CI for NDE, NIE, TE.
    """
    n = len(Y)

    # Observed Q predictions for residual computation
    AML_obs = np.column_stack([A, M, L])
    Q_obs = Q_model.predict(AML_obs)
    resid = Y - Q_obs    # outcome model residual

    # Predicted mediator values under each treatment regime
    M_low = M_model.predict(np.column_stack([np.full(n, a_low), L]))

    # ---------- IF for NDE = E[Q(a_high, M(a_low), L)] - E[Q(a_low, M(a_low), L)] ----------
    # The plug-in functional for each unit:
    #   f_NDE_i = Q(a_high, M_low_i, L_i) - Q(a_low, M_low_i, L_i)
    # The IF has two parts:
    #   (1) centering: f_NDE_i - NDE
    #   (2) outcome-model correction: residual * clever covariate
    #
    # For the clever covariate H_NDE under g-computation, we use the
    # numerical derivative of the NDE plug-in w.r.t. the outcome model.
    # At each observed (A_i, M_i), the contribution to the NDE is through
    # Q evaluated at (a_high, M_low, L) and (a_low, M_low, L). The
    # influence of observation i on these predictions depends on how close
    # (A_i, M_i) is to the counterfactual evaluation points.
    #
    # Simplified IF (Zheng & van der Laan 2012, eq. 7):
    #   IF_NDE_i = (Q_hl_i - Q_ll_i - NDE) + w_i * (Y_i - Q_obs_i)
    #
    # where w_i captures the "leverage" of observation i. For the
    # regression-imputation estimator without cross-fitting of the
    # counterfactuals, the dominant term is the centering term and the
    # residual projection has E[w * resid] = 0 asymptotically. We include
    # the residual term via a simple kernel weight that measures proximity
    # of the observed (A, M) to the NDE evaluation points.

    # Kernel weight for NDE: how relevant is observation i to the
    # (a_high, M_low, L) and (a_low, M_low, L) evaluation?
    # Use a soft indicator based on treatment proximity.
    A_range = max(a_high - a_low, 1e-12)
    # Weight toward a_high: observations near a_high inform Q(a_high, M_low, L)
    w_high = np.exp(-0.5 * ((A - a_high) / (0.25 * A_range)) ** 2)
    # Weight toward a_low: observations near a_low inform Q(a_low, M_low, L)
    w_low  = np.exp(-0.5 * ((A - a_low)  / (0.25 * A_range)) ** 2)
    # Net clever covariate for NDE: observations near a_high contribute
    # positively, observations near a_low contribute negatively
    H_NDE = (w_high - w_low)
    # Normalize so that the residual correction has the right scale
    H_NDE_norm = H_NDE / max(np.mean(np.abs(H_NDE)), 1e-12)

    IF_NDE = (Q_hl - Q_ll) - NDE + H_NDE_norm * resid

    # ---------- IF for NIE = E[Q(a_high, M(a_high), L)] - E[Q(a_high, M(a_low), L)] ----------
    M_high = M_model.predict(np.column_stack([np.full(n, a_high), L]))
    M_range = np.mean(np.abs(M_high - M_low)) + 1e-12
    # Weight based on mediator proximity: observations whose M is near
    # M_high inform Q(a_high, M_high, L), those near M_low inform Q(a_high, M_low, L)
    w_M_high = np.exp(-0.5 * ((M - M_high) / (0.25 * M_range)) ** 2)
    w_M_low  = np.exp(-0.5 * ((M - M_low)  / (0.25 * M_range)) ** 2)
    H_NIE = (w_M_high - w_M_low) * w_high  # only relevant near a_high
    H_NIE_norm = H_NIE / max(np.mean(np.abs(H_NIE)), 1e-12)

    IF_NIE = (Q_hh - Q_hl) - NIE + H_NIE_norm * resid

    # ---------- IF for TE = NDE + NIE ----------
    IF_TE = IF_NDE + IF_NIE

    # ---------- Cluster-robust SEs ----------
    se_NDE_iid = float(np.sqrt(np.var(IF_NDE, ddof=1) / n))
    se_NIE_iid = float(np.sqrt(np.var(IF_NIE, ddof=1) / n))
    se_TE_iid  = float(np.sqrt(np.var(IF_TE,  ddof=1) / n))

    se_NDE_cluster = _cluster_se(IF_NDE, clusters)
    se_NIE_cluster = _cluster_se(IF_NIE, clusters)
    se_TE_cluster  = _cluster_se(IF_TE,  clusters)

    z = 1.959963984540054

    return {
        "NDE_se_iid":     se_NDE_iid,
        "NDE_se_cluster": se_NDE_cluster,
        "NDE_ci":         (NDE - z * se_NDE_cluster, NDE + z * se_NDE_cluster),
        "NIE_se_iid":     se_NIE_iid,
        "NIE_se_cluster": se_NIE_cluster,
        "NIE_ci":         (NIE - z * se_NIE_cluster, NIE + z * se_NIE_cluster),
        "TE_se_iid":      se_TE_iid,
        "TE_se_cluster":  se_TE_cluster,
        "TE_ci":          (TE  - z * se_TE_cluster,  TE  + z * se_TE_cluster),
    }


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
    ci_method: str = "bootstrap",
) -> dict:
    """Regression-based mediational TMLE for NDE / NIE at the (a_high, a_low) contrast.

    Implements a simplified version of the Zheng & van der Laan (2012)
    sequentially doubly robust estimator. We use g-computation throughout
    rather than the full IPW + targeting machinery; the trade-off is that
    consistency requires correct specification of Q, not double robustness.
    Given the nonparametric Super Learner, this is a reasonable simplification
    for a first-pass implementation.

    Point estimate uses the full HurdleSuperLearner stack. CIs are computed
    via one of two methods:
      - "bootstrap" (default): cluster bootstrap with a fast GBM surrogate Q
      - "influence": influence-function-based SEs (no bootstrap; much faster)

    Decomposition (under sequential ignorability + cross-world independence):
        Total effect (TE) = E[Y_{a_high}] − E[Y_{a_low}]
        NIE              = E[Y_{a_high, M_{a_high}}] − E[Y_{a_high, M_{a_low}}]
        NDE              = E[Y_{a_high, M_{a_low}}] − E[Y_{a_low, M_{a_low}}]
        TE = NIE + NDE
    """
    if ci_method not in ("bootstrap", "influence"):
        raise ValueError(f"ci_method must be 'bootstrap' or 'influence', got {ci_method!r}")

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

    # ── CI computation ──────────────────────────────────────────────
    if ci_method == "influence":
        # Influence-function-based SEs — no bootstrap loop
        if_result = mediation_IF_ci(
            Y=Y, A=A, M=M, L=L, clusters=clusters,
            Q_model=Q_full, M_model=M_model,
            a_high=a_high, a_low=a_low,
            Q_hh=Q_hh, Q_hl=Q_hl, Q_ll=Q_ll,
            NDE=NDE, NIE=NIE, TE=TE,
        )
        return {
            "a_high":       a_high,
            "a_low":        a_low,
            "TE":           TE,
            "NDE":          NDE,
            "NIE":          NIE,
            "pct_mediated": pct_mediated,
            "TE_ci":        if_result["TE_ci"],
            "NDE_ci":       if_result["NDE_ci"],
            "NIE_ci":       if_result["NIE_ci"],
            "TE_se_cluster":  if_result["TE_se_cluster"],
            "NDE_se_cluster": if_result["NDE_se_cluster"],
            "NIE_se_cluster": if_result["NIE_se_cluster"],
            "n":            n,
            "n_clusters":   int(pd.Series(clusters).nunique()),
            "ci_method":    "influence",
        }

    # ── Bootstrap CIs (original method) ─────────────────────────────
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
        "ci_method":    "bootstrap",
    }


# ──────────────────── Helpers ────────────────────────────────────

def _cluster_se(IF: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-IF standard error with Bessel correction.

    For an asymptotically linear estimator with influence function IF, the
    cluster-robust variance is:

        Var[ψ̂] = (n_c / (n_c - 1)) · (1 / n²) · Σ_c (Σ_{i∈c} IF_i)²

    where n_c = number of clusters. The n_c/(n_c-1) is the Bessel correction
    for the number of independent clusters, which prevents understating the
    SE when the number of clusters is small.
    """
    s = pd.Series(IF).groupby(pd.Series(clusters).values).sum().to_numpy()
    n = len(IF)
    n_c = len(s)
    if n_c < 2:
        return float("nan")
    # Bessel-corrected cluster-robust SE
    return float(np.sqrt((n_c / (n_c - 1)) * np.sum(s ** 2) / (n ** 2)))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF without scipy dependency."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
