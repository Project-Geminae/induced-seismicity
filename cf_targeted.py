#!/usr/bin/env python3
"""
cf_targeted.py
──────────────
Three improvements for the dashboard's Causal Forest:

  1. cluster_robust_total(...)   — well-cluster-aware SE for Σ CATE
  2. tmle_target_total(...)      — TMLE-targeted population shift mean
                                    using CF nuisances, with cluster IF SE
  3. hurdle_decompose_total(...) — frequency/magnitude/cross channels
                                    when given a logistic CF + Gaussian CF

Usage
-----
    from cf_targeted import (
        cluster_robust_total,
        tmle_target_total,
        hurdle_decompose_total,
    )
    import pickle, pandas as pd

    with open("cf_cate_7km.pkl", "rb") as f:
        cf = pickle.load(f)

    panel = pd.read_csv("panel_with_faults_7km.csv", low_memory=False)

    # 1. Cluster-robust population total
    out = cluster_robust_total(cf, panel, well_id_col="API Number")
    # → {"psi_plugin", "se_cluster", "ci_low", "ci_high", "n", "n_clusters"}

    # 2. TMLE-targeted (shift A → 0)
    out = tmle_target_total(cf, panel, well_id_col="API Number")
    # → {"psi_plugin", "psi_targeted", "epsilon", "se_cluster", "ci_low", "ci_high"}

    # 3. Hurdle channels (requires two CFs from build_hurdle_cf.py)
    out = hurdle_decompose_total(cf_log, cf_mag, panel, well_id_col="API Number")
    # → {"psi_freq", "psi_mag", "psi_cross", "psi_total", "se_*", "ci_*"}

Cluster bootstrap (B=500) is provided as `cluster_bootstrap_total()` — a
refit-free resampling that uses the fitted CF's pre-computed CATE rather
than re-training (so 500 reps cost ~5 sec, not ~11 hours).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from column_maps import COL_API, COL_OUTCOME_MAX_ML, cum_volume_col


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

_Z95 = 1.959963984540054


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _to_X(cf_bundle, panel: pd.DataFrame) -> np.ndarray:
    """Build the confounder design matrix using the bundle's _to_X."""
    return cf_bundle._to_X(panel)


def _cluster_sum_var(per_row: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust variance of Σ_i per_row[i].

    For a sum statistic S = Σ_i x_i with cluster structure, the cluster-
    robust variance is

        Var(S) = (n_c / (n_c - 1)) · Σ_c (Σ_{i ∈ c} x_i)²   (centered)

    This is the natural cluster bootstrap variance for Σ-style estimators
    (Liang & Zeger 1986; Wooldridge 2003).
    """
    df = pd.DataFrame({"x": per_row, "c": clusters})
    by_c = df.groupby("c", sort=False)["x"].sum().to_numpy()
    n_c = len(by_c)
    if n_c <= 1:
        return float("nan")
    centered = by_c - by_c.mean()
    return float((n_c / (n_c - 1)) * np.sum(centered ** 2))


def _cluster_se_mean(IF: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust SE for a mean-style estimator with influence values IF.

    Var(ψ̂) = (n_c / (n_c - 1)) · (1/n²) · Σ_c (Σ_{i ∈ c} IF_i)²
    """
    n = len(IF)
    df = pd.DataFrame({"IF": IF, "c": clusters})
    by_c = df.groupby("c", sort=False)["IF"].sum().to_numpy()
    n_c = len(by_c)
    if n_c <= 1:
        return float("nan")
    return float(np.sqrt((n_c / (n_c - 1)) * np.sum(by_c ** 2) / (n ** 2)))


# ──────────────────────────────────────────────────────────────────
# 1. Cluster-robust population total
# ──────────────────────────────────────────────────────────────────

def cluster_robust_total(
    cf_bundle,
    panel: pd.DataFrame,
    well_id_col: str = COL_API,
    treatment_col: str | None = None,
    outcome_col: str = COL_OUTCOME_MAX_ML,
    drop_na: bool = True,
) -> dict:
    """Population total Σ CATE with cluster-robust (well-level) SE.

    The dashboard currently shows Σ CATE = Σ_i τ̂(L_i) · A_i with no SE,
    or with i.i.d. honest-tree CIs that ignore the well-level clustering.
    This computes the same point estimate but with cluster-robust SE.

    Returns
    -------
    dict with:
      psi_plugin    Σ_i τ̂(L_i) · A_i    (current dashboard headline)
      psi_per_row   mean of τ̂(L_i) · A_i (per well-day average effect)
      se_cluster    cluster-robust SE for psi_plugin
      se_iid        i.i.d. SE for psi_plugin (for comparison)
      ci_low, ci_high   95% cluster-robust CI for psi_plugin
      design_effect    se_cluster / se_iid (Kish ratio)
      n             number of rows
      n_clusters    number of unique well IDs
    """
    treat_col = treatment_col or cf_bundle.treatment_col

    df = panel
    if drop_na:
        df = df.dropna(subset=[treat_col, outcome_col]).reset_index(drop=True)

    X = _to_X(cf_bundle, df)
    A = df[treat_col].astype(float).to_numpy()
    clusters = df[well_id_col].astype(int).to_numpy()
    n = len(df)
    n_c = int(pd.Series(clusters).nunique())

    tau = cf_bundle.cf.const_marginal_effect(X).flatten()
    contrib = tau * A
    psi_plugin = float(contrib.sum())

    var_iid = float(np.var(contrib, ddof=1) * n)  # Var(Σ_i x_i) = n · Var(x)
    var_cluster = _cluster_sum_var(contrib, clusters)
    se_iid = math.sqrt(var_iid)
    se_cluster = math.sqrt(var_cluster) if not np.isnan(var_cluster) else float("nan")

    ci_low = psi_plugin - _Z95 * se_cluster
    ci_high = psi_plugin + _Z95 * se_cluster
    design_effect = (se_cluster / se_iid) if (se_iid > 0 and not np.isnan(se_cluster)) else float("nan")

    return {
        "psi_plugin":    psi_plugin,
        "psi_per_row":   float(contrib.mean()),
        "se_iid":        se_iid,
        "se_cluster":    se_cluster,
        "ci_low":        ci_low,
        "ci_high":       ci_high,
        "design_effect": design_effect,
        "n":             n,
        "n_clusters":    n_c,
        "treatment_col": treat_col,
    }


def cluster_bootstrap_total(
    cf_bundle,
    panel: pd.DataFrame,
    well_id_col: str = COL_API,
    treatment_col: str | None = None,
    outcome_col: str = COL_OUTCOME_MAX_ML,
    B: int = 500,
    seed: int = 42,
) -> dict:
    """Refit-free cluster bootstrap CI for Σ CATE.

    Uses the already-fitted CF (no re-training) and resamples wells
    (clusters) with replacement. For each replicate, sums the per-row
    contributions for the resampled set of wells.

    Returns dict with bootstrap percentile CI and SE.
    """
    treat_col = treatment_col or cf_bundle.treatment_col
    df = panel.dropna(subset=[treat_col, outcome_col]).reset_index(drop=True)

    X = _to_X(cf_bundle, df)
    A = df[treat_col].astype(float).to_numpy()
    clusters = df[well_id_col].astype(int).to_numpy()
    tau = cf_bundle.cf.const_marginal_effect(X).flatten()
    contrib = tau * A

    # Per-cluster aggregate
    df_c = pd.DataFrame({"contrib": contrib, "c": clusters})
    by_c = df_c.groupby("c", sort=False)["contrib"].sum()
    cluster_ids = by_c.index.to_numpy()
    cluster_totals = by_c.to_numpy()
    n_c = len(cluster_ids)

    rng = np.random.default_rng(seed)
    psis = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n_c, size=n_c)
        psis[b] = cluster_totals[idx].sum()

    psi_point = float(cluster_totals.sum())
    return {
        "psi_plugin":  psi_point,
        "boot_mean":   float(psis.mean()),
        "boot_se":     float(psis.std(ddof=1)),
        "ci_low":      float(np.quantile(psis, 0.025)),
        "ci_high":     float(np.quantile(psis, 0.975)),
        "B":           B,
        "n_clusters":  n_c,
    }


# ──────────────────────────────────────────────────────────────────
# 2. TMLE-targeted population shift mean using CF nuisances
# ──────────────────────────────────────────────────────────────────

def _fit_q_from_residuals(cf_bundle, X: np.ndarray, A: np.ndarray, Y: np.ndarray):
    """Reconstruct Q̂(A,L) = m̂(L) + τ̂(L)·(A - ê(L)) from the CF.

    econml's CausalForestDML stores the cross-fitted nuisance models as
    `models_y` (lists of fold-wise E[Y|L] regressors) and `models_t`
    (E[A|L]). We average across folds to get a "deployment" m̂ and ê
    that we can evaluate at any (A_query, L) pair.

    For finite n this is approximate (the cross-fitting structure is
    lost), but it's the standard way to reconstruct Q̂ for a downstream
    targeting step when the CF was the fit object.
    """
    cf = cf_bundle.cf

    def _ensemble_predict(models, X_query):
        if not isinstance(models, (list, tuple)):
            return models.predict(X_query)
        preds = [m.predict(X_query) for m in models]
        return np.mean(np.stack(preds, axis=0), axis=0)

    # econml stores models as List[List[model]]: outer over folds, inner over fits
    def _flatten(maybe_list):
        out = []
        if isinstance(maybe_list, (list, tuple)):
            for x in maybe_list:
                out.extend(_flatten(x))
        else:
            out.append(maybe_list)
        return out

    y_models = _flatten(getattr(cf, "models_y", None) or [])
    t_models = _flatten(getattr(cf, "models_t", None) or [])
    if not y_models or not t_models:
        # Fall back: refit quick xgb on (X, Y) and (X, A)
        import xgboost as xgb
        m_y = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                tree_method="hist", verbosity=0, n_jobs=2)
        m_y.fit(X, Y)
        m_t = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                tree_method="hist", verbosity=0, n_jobs=2)
        m_t.fit(X, A)
        y_models = [m_y]
        t_models = [m_t]

    m_hat = _ensemble_predict(y_models, X)
    e_hat = _ensemble_predict(t_models, X)
    tau = cf.const_marginal_effect(X).flatten()

    def Q(A_query: np.ndarray) -> np.ndarray:
        return m_hat + tau * (A_query - e_hat)

    return Q, m_hat, e_hat, tau


def tmle_target_total(
    cf_bundle,
    panel: pd.DataFrame,
    well_id_col: str = COL_API,
    treatment_col: str | None = None,
    outcome_col: str = COL_OUTCOME_MAX_ML,
    shift_factor: float = 0.90,
    trim_pct: float = 0.01,
    subsample_n: int | None = 100_000,
    seed: int = 42,
) -> dict:
    """TMLE-targeted shift-intervention mean using CF-derived Q̂.

    Parameters
    ----------
    shift_factor : float
        Multiplicative shift: A_post = shift_factor · A_obs. Defaults to 0.9
        (a 10% reduction in injection volume — matches the rest of the TMLE
        pipeline). For a 10% increase use 1.10. shift_factor=0 ("shut off
        everything") is degenerate for continuous A under density-ratio
        targeting; use g-computation for that.
    subsample_n : int | None
        Cluster-aware subsample size for the density and targeting step.
        The CF was fit on 200k rows; using the full 903k row panel for the
        density fit is overkill and triggers degenerate quantile bins on
        zero-inflated treatments.

    Returns dict with:
        psi_plugin    Σ_i [Q̂(A_post) - Q̂(A_obs)] (population total over n_used)
        psi_targeted  same after one TMLE fluctuation step
        epsilon       fluctuation parameter
        IF, se_cluster, se_iid, ci_low, ci_high
        n, n_clusters
    """
    from tmle_core import KDEConditionalDensity

    treat_col = treatment_col or cf_bundle.treatment_col
    df = panel.dropna(subset=[treat_col, outcome_col]).reset_index(drop=True)

    # Cluster-aware subsample so density estimation is tractable
    if subsample_n is not None and len(df) > subsample_n:
        rng = np.random.default_rng(seed)
        clusters_all = df[well_id_col].to_numpy()
        unique = pd.unique(clusters_all)
        rng.shuffle(unique)
        kept = []
        kept_n = 0
        for c in unique:
            n_c = int((clusters_all == c).sum())
            if kept_n + n_c > subsample_n and kept_n > 0:
                break
            kept.append(c)
            kept_n += n_c
        df = df.loc[df[well_id_col].isin(kept)].reset_index(drop=True)

    X = _to_X(cf_bundle, df)
    A = df[treat_col].astype(float).to_numpy()
    Y = df[outcome_col].astype(float).to_numpy()
    clusters = df[well_id_col].astype(int).to_numpy()
    n = len(df)

    Q, m_hat, e_hat, tau = _fit_q_from_residuals(cf_bundle, X, A, Y)

    # KDE-based conditional density (smoother than histogram for zero-inflated A)
    g_model = KDEConditionalDensity(random_state=seed).fit(A, X)
    g_obs = g_model.density(A, X)
    A_post = A * shift_factor

    # For shift_factor = 0, we want E[Q(0, L)] (the "shut everything off"
    # counterfactual mean). The IF is the shift-mean canonical IF specialized
    # to a deterministic shift d(A,L) = c (a constant — here c = 0).
    # Influence function: IF = Q*(A_post,L) − ψ + clever * (Y − Q*(A_obs,L))
    # where clever = g(A_post|L) / g(A_obs|L).
    g_post = g_model.density(A_post, X)
    H = g_post / np.maximum(g_obs, 1e-12)

    if trim_pct > 0:
        H_cap = float(np.quantile(np.abs(H), 1.0 - trim_pct))
        H = np.clip(H, -H_cap, H_cap)

    Q_obs = Q(A)
    Q_post = Q(A_post)

    # Fluctuation: ε from OLS of (Y − Q_obs) on H
    residual = Y - Q_obs
    eps = float(np.dot(H, residual) / max(np.dot(H, H), 1e-12))
    Q_obs_star = Q_obs + eps * H
    # Apply same fluctuation at A_post — at A_post, the clever covariate is
    # g(A_post|L)/g(A_post|L) = 1 (the targeting is at the shifted level).
    # For a deterministic shift, the targeting move at A_post is just +eps·1
    # if we took the simple working submodel logit(Q*) = logit(Q) + ε·H, but
    # here we use the linear fluctuation Q* = Q + ε·H so:
    Q_post_star = Q_post + eps * 1.0

    psi_plugin   = float((Q_obs - Q_post).sum())          # plug-in total: Σ [Q̂(A_obs) - Q̂(A_post)]
    psi_targeted = float((Q_obs_star - Q_post_star).sum())

    # Influence function for the shift-mean difference E[Q(A_obs) - Q(A_post)],
    # written for a SUM (multiply each row's IF by 1, sum them).
    # Per-row mean-version IF:  H·(Y−Q_obs_star) + Q_obs_star − Q_post_star − μ_diff
    mu_diff = (Q_obs_star - Q_post_star).mean()
    IF_mean = H * (Y - Q_obs_star) + (Q_obs_star - Q_post_star) - mu_diff
    se_cluster_mean = _cluster_se_mean(IF_mean, clusters)
    se_iid_mean = float(np.sqrt(np.var(IF_mean, ddof=1) / n))

    se_cluster = se_cluster_mean * n  # SE for SUM = n · SE for MEAN
    se_iid = se_iid_mean * n

    ci_low = psi_targeted - _Z95 * se_cluster
    ci_high = psi_targeted + _Z95 * se_cluster
    z = psi_targeted / max(se_cluster, 1e-15)
    pval = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return {
        "psi_plugin":    psi_plugin,
        "psi_targeted":  psi_targeted,
        "epsilon":       eps,
        "se_iid":        se_iid,
        "se_cluster":    se_cluster,
        "ci_low":        ci_low,
        "ci_high":       ci_high,
        "z":             z,
        "pval":          pval,
        "shift_factor":  shift_factor,
        "n":             n,
        "n_clusters":    int(pd.Series(clusters).nunique()),
        "max_H":         float(np.max(np.abs(H))),
        "mean_H":        float(np.mean(H)),
    }


# ──────────────────────────────────────────────────────────────────
# 3. Hurdle decomposition
# ──────────────────────────────────────────────────────────────────

@dataclass
class HurdleChannelResult:
    psi_freq:   float    # E[(P_post − P_obs) · M_obs]    (frequency channel)
    psi_mag:    float    # E[P_obs · (M_post − M_obs)]    (magnitude channel)
    psi_cross:  float    # E[(P_post − P_obs) · (M_post − M_obs)]
    psi_total:  float    # = psi_freq + psi_mag + psi_cross
    se_freq:    float
    se_mag:     float
    se_cross:   float
    se_total:   float
    ci_low:     float
    ci_high:    float
    n:          int
    n_clusters: int


def hurdle_decompose_total(
    cf_logistic,
    cf_magnitude,
    panel: pd.DataFrame,
    well_id_col: str = COL_API,
    treatment_col: str | None = None,
    outcome_col: str = COL_OUTCOME_MAX_ML,
    shift_factor: float = 0.0,
) -> HurdleChannelResult:
    """Decompose population shift effect into frequency × magnitude channels.

    For a hurdle outcome Y = 1{Y > 0} · Y, the conditional mean is
        Q(W) = P(Y > 0 | W) · E[Y | Y > 0, W]
             = P(W) · M(W)
    where P̂(W) = cf_logistic.predict_proba(...)[:, 1] and
          M̂(W) = expm1(cf_magnitude.predict(W))    (we modeled log(1+Y) on Y>0).

    Total effect under shift A → A·shift_factor:
        ψ_total = Σ_i [P̂(A_obs)·M̂(A_obs) − P̂(A_post)·M̂(A_post)]
                = Σ_i ψ_freq_i + ψ_mag_i + ψ_cross_i
    where for each row i (using ΔP_i = P̂(A_post) − P̂(A_obs), etc.)
        ψ_freq_i  = -ΔP_i · M̂(A_obs)
        ψ_mag_i   = -P̂(A_obs) · ΔM_i
        ψ_cross_i = -ΔP_i · ΔM_i

    Cluster-robust SE on each channel via the per-cluster sum variance.
    """
    treat_col = treatment_col or cf_logistic.treatment_col
    df = panel.dropna(subset=[treat_col, outcome_col]).reset_index(drop=True)
    X_log = cf_logistic._to_X(df)
    X_mag = cf_magnitude._to_X(df)
    A = df[treat_col].astype(float).to_numpy()
    clusters = df[well_id_col].astype(int).to_numpy()
    n = len(df)
    A_post = A * shift_factor

    # Frequency channel: logistic CF predicts P(Y>0 | A, L). For a CausalForestDML
    # with a binary outcome via XGBClassifier model_y, .effect(X, T0, T1) gives
    # the marginal effect on the predicted probability scale. Reconstruct the
    # conditional probability at A_obs and A_post.
    #
    # We approximate P(A, L) using the fitted CF as
    #   P̂(A, L) ≈ baseline_prob(L) + τ_log(L) · (A - e_log(L))
    # using the same Q-from-residuals trick.
    Q_log, _, _, _ = _fit_q_from_residuals(cf_logistic, X_log, A,
                                            (df[outcome_col].astype(float) > 0).astype(float).to_numpy())
    Q_mag, _, _, _ = _fit_q_from_residuals(cf_magnitude, X_mag, A,
                                            np.log1p(df[outcome_col].astype(float).clip(lower=0)).to_numpy())

    P_obs  = np.clip(Q_log(A),       0.0, 1.0)
    P_post = np.clip(Q_log(A_post),  0.0, 1.0)
    M_obs  = np.expm1(Q_mag(A))         # E[Y | Y>0, W] inverse of log(1+·)
    M_post = np.expm1(Q_mag(A_post))

    dP = P_post - P_obs
    dM = M_post - M_obs

    freq_per_row  = -dP * M_obs
    mag_per_row   = -P_obs * dM
    cross_per_row = -dP * dM
    total_per_row = freq_per_row + mag_per_row + cross_per_row

    psi_freq  = float(freq_per_row.sum())
    psi_mag   = float(mag_per_row.sum())
    psi_cross = float(cross_per_row.sum())
    psi_total = float(total_per_row.sum())

    se_freq  = math.sqrt(_cluster_sum_var(freq_per_row,  clusters))
    se_mag   = math.sqrt(_cluster_sum_var(mag_per_row,   clusters))
    se_cross = math.sqrt(_cluster_sum_var(cross_per_row, clusters))
    se_total = math.sqrt(_cluster_sum_var(total_per_row, clusters))

    return HurdleChannelResult(
        psi_freq=psi_freq, psi_mag=psi_mag, psi_cross=psi_cross,
        psi_total=psi_total,
        se_freq=se_freq, se_mag=se_mag, se_cross=se_cross, se_total=se_total,
        ci_low=psi_total - _Z95 * se_total,
        ci_high=psi_total + _Z95 * se_total,
        n=n, n_clusters=int(pd.Series(clusters).nunique()),
    )


# ──────────────────────────────────────────────────────────────────
# CLI: validate + write a JSON cache for the dashboard
# ──────────────────────────────────────────────────────────────────

def _summary_dict(result: HurdleChannelResult | dict) -> dict:
    if isinstance(result, HurdleChannelResult):
        return asdict(result)
    return dict(result)


def main():
    import argparse, json, pickle
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--cf",  default=None,  help="CF pickle path (default: cf_cate_<R>km.pkl)")
    p.add_argument("--cf-logistic",  default=None, help="hurdle logistic CF pickle (optional)")
    p.add_argument("--cf-magnitude", default=None, help="hurdle Gaussian CF pickle (optional)")
    p.add_argument("--panel", default=None, help="panel CSV (default: panel_with_faults_<R>km.csv)")
    p.add_argument("--out",   default=None, help="output JSON (default: cf_targeted_<R>km.json)")
    p.add_argument("--bootstrap-B", type=int, default=500)
    p.add_argument("--shift-factor", type=float, default=0.0,
                   help="A_post = shift_factor · A_obs; 0.0 = shut-off counterfactual")
    args = p.parse_args()

    R = args.radius
    cf_path     = Path(args.cf or f"cf_cate_{R}km.pkl")
    panel_path  = Path(args.panel or f"panel_with_faults_{R}km.csv")
    out_path    = Path(args.out or f"cf_targeted_{R}km.json")

    print(f"[{R}km] loading {cf_path} …")
    import sys
    from build_causal_forest import CausalForestBundle
    sys.modules["__main__"].CausalForestBundle = CausalForestBundle
    with cf_path.open("rb") as f:
        cf = pickle.load(f)

    print(f"[{R}km] loading {panel_path} …")
    panel = pd.read_csv(panel_path, low_memory=False)
    print(f"[{R}km] panel: n={len(panel)} cols={panel.shape[1]}")

    print(f"[{R}km] computing cluster-robust total …")
    cluster_out = cluster_robust_total(cf, panel)
    print(f"           psi_plugin    = {cluster_out['psi_plugin']:+.4e}")
    print(f"           se_iid        = {cluster_out['se_iid']:.4e}")
    print(f"           se_cluster    = {cluster_out['se_cluster']:.4e}")
    print(f"           design effect = {cluster_out['design_effect']:.2f}")
    print(f"           CI95          = [{cluster_out['ci_low']:+.3e}, {cluster_out['ci_high']:+.3e}]")
    print(f"           n             = {cluster_out['n']:,}  clusters = {cluster_out['n_clusters']}")

    print(f"\n[{R}km] cluster bootstrap B={args.bootstrap_B} …")
    boot_out = cluster_bootstrap_total(cf, panel, B=args.bootstrap_B)
    print(f"           boot CI95     = [{boot_out['ci_low']:+.3e}, {boot_out['ci_high']:+.3e}]")
    print(f"           boot SE       = {boot_out['boot_se']:.4e}")

    print(f"\n[{R}km] TMLE-targeted (shift_factor={args.shift_factor}) …")
    tmle_out = tmle_target_total(cf, panel, shift_factor=args.shift_factor)
    print(f"           psi_plugin    = {tmle_out['psi_plugin']:+.4e}")
    print(f"           psi_targeted  = {tmle_out['psi_targeted']:+.4e}")
    print(f"           epsilon       = {tmle_out['epsilon']:+.4e}")
    print(f"           se_cluster    = {tmle_out['se_cluster']:.4e}")
    print(f"           CI95          = [{tmle_out['ci_low']:+.3e}, {tmle_out['ci_high']:+.3e}]")
    print(f"           p             = {tmle_out['pval']:.3e}")

    output = {
        "radius_km":         R,
        "cluster_robust":    cluster_out,
        "cluster_bootstrap": boot_out,
        "tmle_targeted":     tmle_out,
    }

    if args.cf_logistic and args.cf_magnitude:
        print(f"\n[{R}km] hurdle decomposition …")
        with open(args.cf_logistic, "rb") as f:
            cf_log = pickle.load(f)
        with open(args.cf_magnitude, "rb") as f:
            cf_mag = pickle.load(f)
        hurdle_out = hurdle_decompose_total(cf_log, cf_mag, panel,
                                             shift_factor=args.shift_factor)
        print(f"           psi_freq      = {hurdle_out.psi_freq:+.4e}  (SE {hurdle_out.se_freq:.3e})")
        print(f"           psi_mag       = {hurdle_out.psi_mag:+.4e}  (SE {hurdle_out.se_mag:.3e})")
        print(f"           psi_cross     = {hurdle_out.psi_cross:+.4e}  (SE {hurdle_out.se_cross:.3e})")
        print(f"           psi_total     = {hurdle_out.psi_total:+.4e}  (SE {hurdle_out.se_total:.3e})")
        output["hurdle"] = _summary_dict(hurdle_out)

    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[{R}km] wrote {out_path}")


if __name__ == "__main__":
    main()
