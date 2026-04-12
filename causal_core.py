"""
causal_core.py
──────────────
Shared causal-inference primitives for the analysis driver scripts.

The four driver scripts (dowhy_simple_all.py, dowhy_simple_all_aggregate.py,
dowhy_ci.py, dowhy_ci_aggregated.py) all share the same core operations:
loading a panel, building a design matrix, fitting OLS for total / direct /
indirect effects, computing cluster bootstrap CIs, running refutations, and
producing summary rows. This module is the single home for those primitives.

Key methodological choices (vs the old pipeline)
-------------------------------------------------
1. **Treatment**       = cumulative injection volume over a configurable
                         lookback window (default 30d), NOT same-day BBL.
2. **Mediator**        = depth-corrected BHP estimated from the
                         volume-weighted WHP over the same window.
3. **Confounders**     = nearest fault distance, fault segment count within R,
                         injection interval midpoint depth, days_active, plus
                         one-hot formation indicators (top-K levels).
4. **Bootstrap**       = CLUSTER bootstrap by API Number (well-day rows are
                         not independent within a well), 50 iterations, single
                         OLS fit per iteration. The old pipeline did 3
                         CausalModel constructions per iteration which was
                         pure overhead — `backdoor.linear_regression` IS OLS.
5. **Refutations**     = placebo (shuffle treatment), random_common_cause
                         averaged over 20 iters, plus a sensitivity sweep
                         that adds an unobserved confounder of varying
                         strength. PASS/FLAG/FAIL based on absolute placebo
                         magnitude rather than the brittle 10% threshold.
6. **Misspecification** flag when |indirect| > |total| (rather than
                         silently printing "152% mediated by pressure").
7. **Mediation**       reports two indirect estimates: c − c′ and a × b.
                         Linear OLS makes these algebraically identical
                         except for one floating-point rounding step; if
                         they differ by more than ~1e-6 something is wrong.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from column_maps import (
    COL_API,
    COL_OUTCOME_MAX_ML,
    COL_PERF_DEPTH_FT,
    confounder_columns,
    fault_segment_col,
    mediator_column,
    treatment_column,
)


# ──────────────────── Data loading & cleaning ────────────────────

def load_panel(path: str, radius_km: int) -> pd.DataFrame:
    """Load a panel_with_faults_<R>km.csv and ensure required columns exist.

    Drops rows where treatment OR outcome is missing/NaN. Median-fills the
    confounders.
    """
    df = pd.read_csv(path, low_memory=False)

    required = [
        COL_API, COL_OUTCOME_MAX_ML, COL_PERF_DEPTH_FT,
        treatment_column(30), treatment_column(90), treatment_column(180), treatment_column(365),
        mediator_column(30), mediator_column(90), mediator_column(180), mediator_column(365),
        *confounder_columns(radius_km),
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")
    return df


def build_design_matrix(
    df: pd.DataFrame,
    radius_km: int,
    window_days: int,
    use_bhp: bool = True,
) -> tuple[pd.DataFrame, str, str, str, list[str], pd.Series]:
    """Construct (data, W, P, S, confounder_cols, cluster_id).

    Drops rows with NaN treatment / mediator / outcome. Median-fills the
    numeric confounders. Uses depth-class proxy (shallow/mid/deep bins from
    measured perf_depth_ft) instead of operator-reported formation labels.
    """
    W = treatment_column(window_days)
    P = mediator_column(window_days, use_bhp=use_bhp)
    S = COL_OUTCOME_MAX_ML
    base_confs = confounder_columns(radius_km)

    # COL_PERF_DEPTH_FT is already in base_confs; avoid duplicate columns
    cols = list(dict.fromkeys([COL_API, W, P, S, *base_confs]))
    sub = df[cols].copy()
    sub = sub.dropna(subset=[W, P, S])

    # Median-fill numeric confounders
    for c in base_confs:
        med = sub[c].median()
        sub[c] = sub[c].fillna(med)

    # Depth-class proxy instead of operator-reported formation
    # (formation labels are unreliable — operators self-report, RRC doesn't validate)
    depth = sub[COL_PERF_DEPTH_FT].fillna(sub[COL_PERF_DEPTH_FT].median())
    sub["depth_shallow"] = (depth < 6000).astype(float)
    sub["depth_mid"]     = ((depth >= 6000) & (depth < 10000)).astype(float)
    # drop_first=True equivalent: omit depth_deep (it's the reference category)

    confounders = base_confs + ["depth_shallow", "depth_mid"]
    cluster_id = sub[COL_API]

    return sub, W, P, S, confounders, cluster_id


# ──────────────────── OLS effect estimation ──────────────────────

@dataclass
class EffectFit:
    total_effect: float
    total_pval: float
    direct_effect: float
    direct_pval: float
    indirect_effect_diff: float       # c − c′
    indirect_effect_product: float    # a × b
    path_a: float
    path_b: float
    path_a_pval: float
    path_b_pval: float
    n: int
    causal_r2: float
    misspecified: bool                # |indirect| > |total|
    extras: dict[str, Any] = field(default_factory=dict)


def fit_effects(
    data: pd.DataFrame,
    W: str,
    P: str,
    S: str,
    confounders: list[str],
    cluster_id: pd.Series | None = None,
) -> EffectFit:
    """Run the four OLS fits underlying the mediation decomposition.

    If cluster_id is given, p-values use cluster-robust (HC1) standard errors.
    """
    X_total = sm.add_constant(data[[W, *confounders]].astype(float), has_constant="add")
    X_full  = sm.add_constant(data[[P, W, *confounders]].astype(float), has_constant="add")
    X_a     = sm.add_constant(data[[W, *confounders]].astype(float), has_constant="add")
    y_S = data[S].astype(float)
    y_P = data[P].astype(float)

    fit_kwargs: dict[str, Any] = {}
    if cluster_id is not None:
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": cluster_id.values}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_total = sm.OLS(y_S, X_total).fit(**fit_kwargs)
        m_full  = sm.OLS(y_S, X_full).fit(**fit_kwargs)
        m_a     = sm.OLS(y_P, X_a).fit(**fit_kwargs)

    c       = float(m_total.params[W])
    c_prime = float(m_full.params[W])
    a       = float(m_a.params[W])
    b       = float(m_full.params[P])

    indirect_diff    = c - c_prime
    indirect_product = a * b
    # Mediation framework only meaningful when total effect is non-trivial.
    # When |total| ~ 0 the indirect/direct decomposition is unstable noise.
    DEGENERATE_TOTAL_THRESHOLD = 1e-9
    misspecified = (
        abs(c) > DEGENERATE_TOTAL_THRESHOLD
        and abs(indirect_diff) > abs(c)
    )

    return EffectFit(
        total_effect          = c,
        total_pval            = float(m_total.pvalues[W]),
        direct_effect         = c_prime,
        direct_pval           = float(m_full.pvalues[W]),
        indirect_effect_diff  = indirect_diff,
        indirect_effect_product = indirect_product,
        path_a                = a,
        path_b                = b,
        path_a_pval           = float(m_a.pvalues[W]),
        path_b_pval           = float(m_full.pvalues[P]),
        n                     = int(len(data)),
        causal_r2             = float(m_total.rsquared),
        misspecified          = misspecified,
    )


# ──────────────────── Cluster bootstrap ──────────────────────────

def cluster_bootstrap_ci(
    data: pd.DataFrame,
    W: str,
    P: str,
    S: str,
    confounders: list[str],
    n_iter: int = 50,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Cluster bootstrap by API Number → 95% CIs for total / direct / indirect.

    Resamples clusters with replacement, then takes ALL rows for each
    sampled cluster. This respects the panel's within-cluster correlation.

    Optimized: precomputes a row-index array per cluster, then assembles
    bootstrap samples by integer index concatenation rather than per-cluster
    boolean scans of the full DataFrame (~50× speedup for event-level).
    """
    rng = np.random.default_rng(seed)
    cluster_ids = data[COL_API].unique()
    n_clusters = len(cluster_ids)

    # Pre-compute row indices per cluster (one numpy array per cluster_id)
    row_idx_by_cluster: dict = {}
    pos = data.reset_index(drop=True)
    for cid, sub in pos.groupby(COL_API, sort=False):
        row_idx_by_cluster[cid] = sub.index.to_numpy()

    # Pre-extract numeric arrays for the design matrices to avoid repeated
    # DataFrame column lookups inside the loop
    Y     = pos[S].astype(float).to_numpy()
    Wcol  = pos[W].astype(float).to_numpy()
    Pcol  = pos[P].astype(float).to_numpy()
    Cmat  = pos[confounders].astype(float).to_numpy()
    n_conf = Cmat.shape[1]

    totals, directs, indirects = [], [], []
    for _ in range(n_iter):
        sampled = rng.choice(cluster_ids, size=n_clusters, replace=True)
        idx_parts = [row_idx_by_cluster[c] for c in sampled]
        if not idx_parts:
            continue
        idx = np.concatenate(idx_parts)
        if idx.size == 0:
            continue

        try:
            # Total-effect design: [const, W, *confounders]
            n = idx.size
            X_total = np.empty((n, 2 + n_conf))
            X_total[:, 0] = 1.0
            X_total[:, 1] = Wcol[idx]
            X_total[:, 2:] = Cmat[idx]

            # Full design: [const, P, W, *confounders]
            X_full = np.empty((n, 3 + n_conf))
            X_full[:, 0] = 1.0
            X_full[:, 1] = Pcol[idx]
            X_full[:, 2] = Wcol[idx]
            X_full[:, 3:] = Cmat[idx]

            y = Y[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tot  = sm.OLS(y, X_total).fit()
                full = sm.OLS(y, X_full).fit()
            c  = float(tot.params[1])
            cp = float(full.params[2])
            totals.append(c); directs.append(cp); indirects.append(c - cp)
        except Exception:
            continue

    def pct_ci(arr):
        if not arr:
            return (np.nan, np.nan)
        a = np.percentile(arr, [2.5, 97.5])
        return (float(a[0]), float(a[1]))

    return {
        "total_ci":    pct_ci(totals),
        "direct_ci":   pct_ci(directs),
        "indirect_ci": pct_ci(indirects),
        "n_iter_ok":   len(totals),
    }


# ──────────────────── Refutations ────────────────────────────────

def placebo_treatment_refutation(
    data: pd.DataFrame,
    W: str,
    P: str,
    S: str,
    confounders: list[str],
    seed: int = 42,
) -> float:
    """Shuffle the treatment column and refit. Effect should be ≈ 0."""
    rng = np.random.default_rng(seed)
    boot = data.copy()
    boot[W] = rng.permutation(boot[W].values)
    X = sm.add_constant(boot[[W, *confounders]].astype(float), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sm.OLS(boot[S].astype(float), X).fit()
    return float(m.params[W])


def random_common_cause_refutation(
    data: pd.DataFrame,
    W: str,
    P: str,
    S: str,
    confounders: list[str],
    n_iter: int = 20,
    seed: int = 42,
) -> tuple[float, float]:
    """Add a random Gaussian confounder and refit. Effect should be unchanged.

    Returns (mean_effect, std_effect) over n_iter iterations.
    """
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(n_iter):
        noise = rng.standard_normal(len(data))
        X = sm.add_constant(
            np.column_stack([data[W].astype(float).values, noise,
                             data[confounders].astype(float).values]),
            has_constant="add",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.OLS(data[S].astype(float).values, X).fit()
        effects.append(float(m.params[1]))  # column 1 is W (after const at 0)
    return float(np.mean(effects)), float(np.std(effects))


def unobserved_confounder_sensitivity(
    data: pd.DataFrame,
    W: str,
    P: str,
    S: str,
    confounders: list[str],
    confounder_strengths: tuple[float, ...] = (0.1, 0.3, 0.5),
    seed: int = 42,
) -> dict[str, float]:
    """Inject an unobserved confounder of varying strength on both W and S.

    Strength is the std-dev of the unobserved variable expressed as a
    fraction of the std-dev of W (and a separate effect on S equal to the
    std-dev of S). Returns the implied effect estimate at each strength.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    sd_W = float(np.std(data[W].astype(float)))
    sd_S = float(np.std(data[S].astype(float)))
    for s in confounder_strengths:
        u = rng.standard_normal(len(data))
        # Construct a contaminated W and S that share the unobserved cause
        W_contam = data[W].astype(float).values + s * sd_W * u
        S_contam = data[S].astype(float).values + s * sd_S * u
        X = sm.add_constant(
            np.column_stack([W_contam, data[confounders].astype(float).values]),
            has_constant="add",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.OLS(S_contam, X).fit()
        out[f"sensitivity_strength_{s:.2f}"] = float(m.params[1])
    return out


def refutation_status(
    original: float,
    placebo: float,
    rcc_mean: float,
    placebo_threshold_frac: float = 0.10,
    rcc_threshold_frac: float = 0.05,
    degenerate_threshold: float = 1e-9,
) -> str:
    """Classify refutation outcome.

    NULL  : original effect is below degenerate_threshold; ratio-based tests
            are uninformative when there's nothing to refute
    PASS  : placebo within threshold AND random_common_cause within threshold
    FLAG  : random_common_cause within threshold but placebo too large
    FAIL  : random_common_cause moves the estimate substantially
    """
    if abs(original) < degenerate_threshold:
        return "NULL"
    placebo_frac = abs(placebo / original)
    rcc_frac = abs((rcc_mean - original) / original)
    if placebo_frac < placebo_threshold_frac and rcc_frac < rcc_threshold_frac:
        return "PASS"
    if rcc_frac < rcc_threshold_frac:
        return "FLAG"
    return "FAIL"


# ──────────────────── VIF (multicollinearity) ────────────────────

def compute_vif(data: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """VIF for each column in `columns`. Higher = more collinear."""
    X = sm.add_constant(data[columns].astype(float).values, has_constant="add")
    out = {}
    for i, name in enumerate(columns):
        try:
            out[name] = float(variance_inflation_factor(X, i + 1))  # +1 for const
        except Exception:
            out[name] = float("nan")
    return out


# ──────────────────── Event-level aggregation ────────────────────

def aggregate_panel_to_event_level(
    panel: pd.DataFrame,
    radius_km: int,
    window_days: int,
) -> pd.DataFrame:
    """Aggregate the (well, day) panel to one row per (date, location-cluster).

    For the event-level analysis we collapse all wells active near a given
    earthquake into a single row. Aggregation rules:
      - Volume:    SUM (additive)
      - Pressure:  VOLUME-WEIGHTED MEAN (Σ P_i V_i / Σ V_i)
      - Fault distance: MIN (closest well drives the geology)
      - Fault segment count: MEAN (NOT sum — sum double-counts shared faults)
      - Depth:     VOLUME-WEIGHTED MEAN (operationally weighted)
      - Days active: MEAN
      - Outcome:   FIRST (event magnitude is identical for all wells in the cluster)
      - Formation: MODE (most common across the cluster)
      - well_count: COUNT (used as a confounder)

    Returns one row per (event-day, well-cluster). Unit-of-analysis = "the
    event", controls = days when no event occurred at any well location.
    """
    W30 = treatment_column(30)
    W90 = treatment_column(90)
    W180 = treatment_column(180)
    W365 = treatment_column(365)

    P30 = mediator_column(30)
    P90 = mediator_column(90)
    P180 = mediator_column(180)
    P365 = mediator_column(365)

    G_dist = "Nearest Fault Dist (km)"
    G_seg  = fault_segment_col(radius_km)
    G_dep  = "perf_depth_ft"
    G_age  = "days_active"

    panel = panel.copy()

    # Group by (date, geographic cluster). For simplicity we cluster by
    # snapping each row's coordinates to a coarse grid (~5 km cell). Wells
    # within the same cell on the same day are treated as one event location.
    # NOTE: this is approximate. A more rigorous version would use the actual
    # event hypocenter from texnet_events_filtered.csv as the cluster centroid.
    LAT_BIN = 0.05  # degrees ≈ 5.5 km
    LON_BIN = 0.05
    panel["_lat_bin"] = (panel["Surface Latitude"]  / LAT_BIN).round() * LAT_BIN
    panel["_lon_bin"] = (panel["Surface Longitude"] / LON_BIN).round() * LON_BIN

    # Pre-compute volume-weighted numerators (Σ P_i V_i) so we can use a
    # plain groupby+sum and divide by the group volume sum at the end. This
    # avoids the slow per-group Python loop in the previous implementation.
    for win, P_col, V_col in [(30, P30, W30), (90, P90, W90), (180, P180, W180), (365, P365, W365)]:
        panel[f"_pv_{win}"] = panel[P_col].fillna(0) * panel[V_col]

    # For depth weighting: weight by the 30-day cumulative volume
    depth_med = panel[G_dep].median()
    panel[f"_dep_v30"] = panel[G_dep].fillna(depth_med) * panel[W30]

    grp_cols = ["Date of Injection", "_lat_bin", "_lon_bin"]
    agg = panel.groupby(grp_cols, sort=False).agg(
        well_count=(COL_API, "size"),
        **{
            W30:  (W30, "sum"),
            W90:  (W90, "sum"),
            W180: (W180, "sum"),
            W365: (W365, "sum"),
            "_pv_30":  ("_pv_30",  "sum"),
            "_pv_90":  ("_pv_90",  "sum"),
            "_pv_180": ("_pv_180", "sum"),
            "_pv_365": ("_pv_365", "sum"),
            "_dep_v30": ("_dep_v30", "sum"),
            G_dist: (G_dist, "min"),
            G_seg:  (G_seg,  "mean"),
            G_age:  (G_age,  "mean"),
            COL_OUTCOME_MAX_ML: (COL_OUTCOME_MAX_ML, "max"),
        },
    ).reset_index()

    # Volume-weighted pressure means: divide accumulated PV by accumulated V
    def safe_div(num, den):
        return np.where(den > 0, num / den, np.nan)
    agg[P30]  = safe_div(agg["_pv_30"],  agg[W30])
    agg[P90]  = safe_div(agg["_pv_90"],  agg[W90])
    agg[P180] = safe_div(agg["_pv_180"], agg[W180])
    agg[P365] = safe_div(agg["_pv_365"], agg[W365])
    agg[G_dep] = safe_div(agg["_dep_v30"], agg[W30])
    agg[G_dep] = agg[G_dep].fillna(depth_med)

    # Formation label dropped — operator-reported and unreliable. The
    # design matrix uses depth-class proxy from measured perf_depth_ft.
    # If the panel has a "formation" column, carry it through for backward
    # compatibility but it's not used by build_design_matrix().
    if "formation" in panel.columns:
        formation_mode = (
            panel.groupby(grp_cols, sort=False)["formation"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
            .reset_index(name="formation")
        )
        agg = agg.merge(formation_mode, on=grp_cols, how="left")

    # Cluster pseudo-id (used as the "API Number" for downstream cluster bootstrap)
    agg[COL_API] = (
        "cluster_" + agg["_lat_bin"].round(2).astype(str) +
        "_" + agg["_lon_bin"].round(2).astype(str)
    )

    # Drop intermediate volume-product columns
    agg = agg.drop(columns=[c for c in agg.columns if c.startswith("_pv_") or c.startswith("_dep_")])
    agg = agg.rename(columns={"_lat_bin": "lat_bin", "_lon_bin": "lon_bin"})
    return agg
