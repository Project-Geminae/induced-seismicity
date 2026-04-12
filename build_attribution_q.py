#!/usr/bin/env python3
"""
build_attribution_q.py
──────────────────────
Fit a per-radius hurdle SuperLearner Q model at the WELL-DAY level for use
by the dashboard's per-well attribution endpoint.

This is conceptually distinct from the TMLE Q model used in dowhy_ci.py /
tmle_*_analysis.py. Those Qs are fit on the (date, ~5km cluster) aggregation
where multiple wells get pooled into one row before the model sees them. They
target population-level estimands like "the dose-response of cumulative
volume on max ML across all cluster-days."

The dashboard wants something different: "given THIS well on THIS day, with
its actual injection history and geology, what does the model predict the
local seismic outcome to be? And what would it predict if this well had been
shut off?"

For that, we need a Q that operates on per-well features directly:

  Q(W=cum_vol_365d_well, L=well_features) → E[outcome_max_ML]

where outcome_max_ML is the per-well outcome already computed by
spatiotemporal_join.py at this radius (max ML within R km on this day, or 0).

We fit one such Q per radius (1..20 km) since the outcome definition changes
with radius. Each Q is pickled to ~/induced-seismicity/q_attribution_<R>km.pkl
and loaded by the dashboard on startup.

For each well-day cell, the per-well contribution to the model's prediction is:

  contribution_i = Q(W=actual, L=this well's features)
                 - Q(W=0,      L=this well's features)

This is an in-model g-computation, not an identified causal effect. It's
defensible as "the model's estimate of this well's marginal contribution to
expected nearby seismicity, holding the well's geology constant." See the
methodology page in the dashboard for the full disclaimer.

Run on minitim:

    .venv/bin/python build_attribution_q.py                       # all 20 radii
    .venv/bin/python build_attribution_q.py --radii 7             # one radius
    .venv/bin/python build_attribution_q.py --workers 10          # parallel

Outputs (one per radius):
    q_attribution_<R>km.pkl       — pickled HurdleSuperLearner + metadata
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import causal_core as cc
import tmle_core as tmle
from column_maps import (
    COL_API,
    COL_DAYS_ACTIVE,
    COL_FORMATION,
    COL_NEAREST_FAULT_KM,
    COL_OUTCOME_MAX_ML,
    COL_PERF_DEPTH_FT,
    cum_volume_col,
    bhp_vw_avg_col,
    fault_segment_col,
)


# ──────────────────── Constants ──────────────────────────────────
PANEL_FMT     = "panel_with_faults_{R}km.csv"
OUT_FMT       = "q_attribution_{R}km.pkl"
RADII_KM      = list(range(1, 21))
WINDOW_DAYS   = 365
TOP_K_FORMATIONS = 6  # match causal_core.TOP_K_FORMATIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────── Pickled wrapper ────────────────────────────
@dataclass
class AttributionQ:
    """A pickled bundle: fitted Q, fitted g, and the training-sample arrays
    needed for **localized TMLE** (van der Laan & Luedtke 2015) at serve time.

    Instead of a single global ε that over-corrects individual CATEs, the
    localized TMLE solves a separate ε(l₀) for each query well by
    weighting the efficient influence function equation with a Gaussian
    kernel centered at the query well's covariate profile:

        H_i(l₀) = K_h(L_i, l₀) / (g(A_i | L_i) · bin_width_i)

        ε(l₀) = Σ_i K_i · H_i · (Y_i − Q̂_i) / Σ_i K_i · H_i²

    where K_i = K_h(L_i, l₀) = exp(−||L̃_i − l̃₀||² / (2h²)) and L̃ is
    the standardized confounder vector.

    This gives a CATE estimate that's locally targeted to the specific
    well's geology/injection profile, rather than globally optimized for
    the population mean (which is what caused the nonsensical results for
    edge-of-basin events).

    The IF-based SE is also localized:
        SE(l₀) = sqrt(Σ_i K²_i · resid²_i / (Σ_i K_i)²)
    """
    radius_km:      int
    window_days:    int
    feature_cols:   list[str]
    formation_cols: list[str]
    top_formations: list[str]
    q:              tmle.HurdleSuperLearner
    g:              tmle.HistogramConditionalDensity
    n_train:        int
    n_pos:          int
    fit_time_sec:   float
    # Pre-computed training-sample arrays for localized targeting at serve time
    L_std:          np.ndarray    # standardized confounders (n_train × n_conf)
    L_mean:         np.ndarray    # confounder means (for standardizing query)
    L_scale:        np.ndarray    # confounder stds (for standardizing query)
    A_train:        np.ndarray    # treatment values (n_train,)
    Y_train:        np.ndarray    # outcome values (n_train,)
    Q_hat_train:    np.ndarray    # Q predictions on training data (n_train,)
    g_obs_train:    np.ndarray    # g(A|L) on training data (n_train,)
    bin_idx_train:  np.ndarray    # histogram bin index per training row
    bandwidth:      float         # kernel bandwidth (in standardized units)

    def _to_X(self, well_rows: pd.DataFrame) -> np.ndarray:
        """Convert a DataFrame of per-well features into the design matrix."""
        df = well_rows.copy()
        df["_form"] = np.where(
            df[COL_FORMATION].isin(self.top_formations),
            df[COL_FORMATION],
            "OTHER",
        )
        for col in self.formation_cols:
            label = col[len("form_"):]
            df[col] = (df["_form"] == label).astype(float)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_cols].astype(float).to_numpy()
        if np.isnan(X).any():
            col_medians = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_medians, inds[1])
        return X

    def predict(self, well_rows: pd.DataFrame) -> np.ndarray:
        """Plain g-computation prediction Q̂(features) — no targeting."""
        return self.q.predict(self._to_X(well_rows))

    def _solve_local_epsilon(self, l_query_std: np.ndarray) -> tuple[float, float]:
        """Solve the localized targeting parameter ε(l₀) and return (ε, SE).

        l_query_std: standardized confounder vector for the query well (1D).
        Returns (epsilon_local, se_local).
        """
        # Gaussian kernel weights: K_i = exp(-||L̃_i - l̃₀||² / (2h²))
        diff = self.L_std - l_query_std[np.newaxis, :]
        sq_dist = np.sum(diff ** 2, axis=1)
        K = np.exp(-sq_dist / (2.0 * self.bandwidth ** 2))

        # Floor tiny weights to avoid numerical noise from distant observations
        K = np.where(K > 1e-10, K, 0.0)
        K_sum = K.sum()
        if K_sum < 1.0:
            # Essentially no nearby training data — return zero epsilon
            return 0.0, float("nan")

        # Clever covariate: H_i = K_i / (g(A_i|L_i) · bin_width_i)
        bin_w = self.g.widths_[self.bin_idx_train]
        H = K / np.maximum(self.g_obs_train * bin_w, 1e-12)

        # Residual from the initial Q fit
        resid = self.Y_train - self.Q_hat_train

        # Weighted OLS of residual on H (no intercept), weights = K
        # ε = Σ K·H·resid / Σ K·H²
        num = np.dot(K * H, resid)
        den = np.dot(K * H, H)
        eps_local = float(num / max(den, 1e-15))

        # Localized SE: sqrt(Σ K²·(resid − ε·H)² / (Σ K)²)
        targeted_resid = resid - eps_local * H
        se_local = float(np.sqrt(
            np.dot(K ** 2, targeted_resid ** 2) / max(K_sum ** 2, 1e-15)
        ))

        return eps_local, se_local

    def predict_targeted(self, well_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Localized TMLE prediction.

        For each query row, solves a local ε(l₀) and returns:
          (q_star, epsilon_per_well, se_per_well)

        q_star: Q̂*(a, l₀) = Q̂(a, l₀) + ε(l₀) · H_query(a, l₀)
        """
        X = self._to_X(well_rows)
        n = X.shape[0]
        A_query = X[:, 0]
        L_query = X[:, 1:]

        q_hat = self.q.predict(X)

        # Standardize query confounders using the training-time means/scales
        L_query_std = (L_query - self.L_mean[np.newaxis, :]) / np.maximum(self.L_scale[np.newaxis, :], 1e-12)

        # g density at the query treatment values
        g_query = self.g.density(A_query, L_query)
        bin_idx_query = np.clip(
            np.digitize(A_query, self.g.edges_, right=False) - 1,
            0, self.g.n_bins - 1,
        )
        bin_w_query = self.g.widths_[bin_idx_query]
        H_query = 1.0 / np.maximum(g_query * bin_w_query, 1e-12)

        q_star = np.zeros(n)
        eps_arr = np.zeros(n)
        se_arr  = np.full(n, float("nan"))

        for i in range(n):
            eps_i, se_i = self._solve_local_epsilon(L_query_std[i])
            q_star[i] = q_hat[i] + eps_i * H_query[i]
            eps_arr[i] = eps_i
            se_arr[i] = se_i

        return q_star, eps_arr, se_arr


# ──────────────────── Per-radius training ────────────────────────
def fit_one_radius(R: int, window_days: int = WINDOW_DAYS) -> AttributionQ | None:
    """Fit and return an AttributionQ for radius R."""
    panel_path = Path(PANEL_FMT.format(R=R))
    if not panel_path.exists():
        log.warning("⚠️   %s missing — skipping", panel_path)
        return None

    log.info("[%2dkm] loading %s …", R, panel_path.name)
    panel = pd.read_csv(panel_path, low_memory=False)
    log.info("[%2dkm] %d rows × %d cols", R, *panel.shape)

    W   = cum_volume_col(window_days)
    P   = bhp_vw_avg_col(window_days)
    G_dep  = COL_PERF_DEPTH_FT
    G_age  = COL_DAYS_ACTIVE
    G_fdist = COL_NEAREST_FAULT_KM
    G_fseg = fault_segment_col(R)
    Y   = COL_OUTCOME_MAX_ML

    required = [W, P, G_dep, G_age, G_fdist, G_fseg, Y, COL_FORMATION, COL_API]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        log.error("[%2dkm] missing columns: %s", R, missing)
        return None

    sub = panel[required].copy()
    n_before = len(sub)
    sub = sub.dropna(subset=[W, P, Y])
    log.info("[%2dkm] dropped %d rows with NaN treatment/mediator/outcome",
             R, n_before - len(sub))

    # Median-fill numeric confounders
    for c in [G_dep, G_age, G_fdist, G_fseg]:
        med = sub[c].median()
        sub[c] = sub[c].fillna(med)

    # One-hot formation: top-K + OTHER
    top_forms = sub[COL_FORMATION].value_counts().head(TOP_K_FORMATIONS).index.tolist()
    sub["_form"] = np.where(sub[COL_FORMATION].isin(top_forms), sub[COL_FORMATION], "OTHER")
    form_dummies = pd.get_dummies(sub["_form"], prefix="form", drop_first=False, dtype=float)
    formation_cols = list(form_dummies.columns)
    sub = pd.concat([sub.drop(columns=["_form"]), form_dummies], axis=1)

    # Final feature column order
    feature_cols = [W, P, G_dep, G_age, G_fdist, G_fseg, *formation_cols]

    X = sub[feature_cols].astype(float).to_numpy()
    y = sub[Y].astype(float).to_numpy()
    n_pos = int((y > 0).sum())

    log.info("[%2dkm] fitting HurdleSuperLearner Q: n=%d, n_pos=%d (%.2f%%)",
             R, len(X), n_pos, 100 * n_pos / max(len(X), 1))

    t0 = time.time()

    # ── Step 1: fit Q (outcome regression) ──
    q = tmle.HurdleSuperLearner(random_state=42)
    q.fit(X, y)
    Q_hat = q.predict(X)
    log.info("[%2dkm] Q fit done (%.1fs)", R, time.time() - t0)

    # ── Step 2: fit g (conditional density of treatment A | L) ──
    A = X[:, 0]   # treatment = first feature column (cum_vol_365d_BBL)
    L = X[:, 1:]  # confounders = the rest
    g = tmle.HistogramConditionalDensity(random_state=42)
    g.fit(A, L)
    g_obs = g.density(A, L)
    bin_idx = np.clip(np.digitize(A, g.edges_, right=False) - 1, 0, g.n_bins - 1)
    log.info("[%2dkm] g fit done", R)

    # ── Step 3: standardize confounders + choose bandwidth ──
    L_mean = L.mean(axis=0)
    L_scale = L.std(axis=0)
    L_scale = np.where(L_scale > 1e-12, L_scale, 1.0)
    L_std = (L - L_mean[np.newaxis, :]) / L_scale[np.newaxis, :]

    # Bandwidth: Silverman's rule of thumb on the standardized confounders
    # h = n^(-1/(d+4)) where d = number of confounders
    n_conf = L.shape[1]
    bandwidth = float(len(X) ** (-1.0 / (n_conf + 4)))
    log.info("[%2dkm] kernel bandwidth h = %.4f (Silverman, d=%d)", R, bandwidth, n_conf)

    # ── Step 4: subsample training arrays for serve-time kernel queries ──
    # 800k rows is too many for per-query kernel evaluation at serve time
    # (~10ms per query × 20 wells = 200ms, tolerable). But if we want to
    # keep memory reasonable in the pickle, subsample to ~50k rows. The
    # kernel will still find enough neighbors.
    MAX_TRAIN_ROWS = 50_000
    if len(X) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), MAX_TRAIN_ROWS, replace=False)
        L_std_keep   = L_std[idx]
        A_keep       = A[idx]
        Y_keep       = y[idx]
        Q_hat_keep   = Q_hat[idx]
        g_obs_keep   = g_obs[idx]
        bin_idx_keep = bin_idx[idx]
        log.info("[%2dkm] subsampled training arrays: %d → %d rows", R, len(X), MAX_TRAIN_ROWS)
    else:
        L_std_keep   = L_std
        A_keep       = A
        Y_keep       = y
        Q_hat_keep   = Q_hat
        g_obs_keep   = g_obs
        bin_idx_keep = bin_idx

    fit_time = time.time() - t0
    log.info("[%2dkm] ✅ localized TMLE fit in %.1fs (h=%.4f, n_sub=%d)",
             R, fit_time, bandwidth, len(A_keep))

    return AttributionQ(
        radius_km      = R,
        window_days    = window_days,
        feature_cols   = feature_cols,
        formation_cols = formation_cols,
        top_formations = top_forms,
        q              = q,
        g              = g,
        n_train        = len(X),
        n_pos          = n_pos,
        fit_time_sec   = fit_time,
        L_std          = L_std_keep.astype(np.float32),
        L_mean         = L_mean.astype(np.float32),
        L_scale        = L_scale.astype(np.float32),
        A_train        = A_keep.astype(np.float32),
        Y_train        = Y_keep.astype(np.float32),
        Q_hat_train    = Q_hat_keep.astype(np.float32),
        g_obs_train    = g_obs_keep.astype(np.float32),
        bin_idx_train  = bin_idx_keep.astype(np.int16),
        bandwidth      = bandwidth,
    )


def _worker(args: dict) -> dict:
    """ProcessPoolExecutor worker entry point."""
    # Force single-threaded xgboost / BLAS per worker to avoid core contention
    # when running under ProcessPoolExecutor with multiple workers.
    import os
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    R = args["R"]
    window_days = args["window_days"]
    aq = fit_one_radius(R, window_days)
    if aq is None:
        return {"R": R, "ok": False}
    out_path = Path(OUT_FMT.format(R=R))
    with out_path.open("wb") as f:
        pickle.dump(aq, f)
    return {
        "R": R,
        "ok": True,
        "out": str(out_path),
        "n": aq.n_train,
        "n_pos": aq.n_pos,
        "fit_sec": aq.fit_time_sec,
    }


# ──────────────────── CLI ────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radii", type=int, nargs="+", default=RADII_KM,
                   help="Radii (km) to fit.")
    p.add_argument("--window", type=int, default=WINDOW_DAYS,
                   help=f"Lookback window in days. Default {WINDOW_DAYS}.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (use 5+ on minitim, 1 on Mac).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("ATTRIBUTION Q TRAINING")
    log.info("  radii: %s", args.radii)
    log.info("  window: %d days  workers: %d", args.window, args.workers)
    log.info("  TMLE_SKIP_GBM=%d  TMLE_BIG_LIBRARY=%d  TMLE_XGB_N=%d",
             int(tmle.SKIP_GBM), int(tmle.BIG_LIBRARY), tmle.XGB_N_ESTIMATORS)

    jobs = [{"R": R, "window_days": args.window} for R in args.radii]
    t_start = time.time()

    if args.workers == 1:
        for job in jobs:
            res = _worker(job)
            if res["ok"]:
                log.info("✓ R=%dkm done (n=%d, n_pos=%d, %.0fs)",
                         res["R"], res["n"], res["n_pos"], res["fit_sec"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, j): j for j in jobs}
            for fut in as_completed(futures):
                res = fut.result()
                if res["ok"]:
                    log.info("✓ R=%dkm done (n=%d, n_pos=%d, %.0fs)",
                             res["R"], res["n"], res["n_pos"], res["fit_sec"])

    log.info("✅ total wall-clock %.1fs (%.1f min)",
             time.time() - t_start, (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
