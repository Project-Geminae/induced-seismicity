#!/usr/bin/env python3
"""
build_causal_forest.py
──────────────────────
Fit a per-radius Causal Forest (DML) for per-well CATE estimation.

Replaces the TMLE-targeting approach (build_attribution_q.py) which was
numerically unstable for per-well CATEs in high-dimensional confounder
space. The Causal Forest (Athey, Tibshirani & Wager 2019) handles this
natively: it splits on treatment-effect heterogeneity, uses cross-fitted
nuisance models for double robustness, and produces honest per-unit CIs
via sample splitting.

For each radius R, fits:

    CausalForestDML(
        model_y = XGBRegressor  (outcome nuisance: E[Y | L])
        model_t = XGBRegressor  (treatment nuisance: E[A | L])
        n_estimators = 200
        honest = True           (sample splitting for valid CIs)
    )

on the well-day panel. Treatment A = cum_vol_365d_BBL, outcome Y =
outcome_max_ML, confounders L = [perf_depth_ft, days_active, fault_dist,
fault_count, formation_onehot].

At serve time, the dashboard calls:
    cate = cf.effect(X_query, T0=0, T1=actual_vol)
    ci   = cf.effect_interval(X_query, T0=0, T1=actual_vol, alpha=0.05)

Outputs (one per radius):
    cf_cate_<R>km.pkl — pickled CausalForestDML + metadata

Run on minitim:
    OMP_NUM_THREADS=2 .venv/bin/python build_causal_forest.py --workers 10

References:
    Athey, S., Tibshirani, J., & Wager, S. (2019). "Generalized Random
    Forests." Annals of Statistics.
    Chernozhukov, V. et al. (2018). "Double/debiased machine learning for
    treatment and structural parameters." Econometrics Journal.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from column_maps import (
    COL_API,
    COL_DAYS_ACTIVE,
    COL_FORMATION,
    COL_NEAREST_FAULT_KM,
    COL_OUTCOME_MAX_ML,
    COL_PERF_DEPTH_FT,
    cum_volume_col,
    fault_segment_col,
)

PANEL_FMT      = "panel_with_faults_{R}km.csv"
OUT_FMT        = "cf_cate_{R}km.pkl"
RADII_KM       = list(range(1, 21))
WINDOW_DAYS    = 365
TOP_K_FORMATIONS = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


@dataclass
class CausalForestBundle:
    """Pickled bundle: fitted CausalForestDML + metadata for serve-time use."""
    radius_km:      int
    window_days:    int
    confounder_cols: list[str]    # column order for X (confounders)
    treatment_col:  str           # column name for T (treatment)
    formation_cols: list[str]     # one-hot dummy names
    top_formations: list[str]     # which formations got their own dummy
    cf:             object        # the fitted CausalForestDML
    n_train:        int
    n_pos:          int
    fit_time_sec:   float

    def _to_X(self, well_rows: pd.DataFrame) -> np.ndarray:
        """Build the confounder design matrix from a DataFrame of well features."""
        df = well_rows.copy()
        # If the model used formation one-hots (legacy), build them
        if self.top_formations:
            df["_form"] = np.where(
                df[COL_FORMATION].isin(self.top_formations),
                df[COL_FORMATION],
                "OTHER",
            )
            for col in self.formation_cols:
                label = col[len("form_"):]
                df[col] = (df["_form"] == label).astype(float)
        else:
            # Depth-class proxy: compute from perf_depth_ft
            depth = pd.to_numeric(df.get(COL_PERF_DEPTH_FT, 7000), errors="coerce").fillna(7000)
            df["depth_shallow"] = (depth < 6000).astype(float)
            df["depth_mid"]     = ((depth >= 6000) & (depth < 10000)).astype(float)
            df["depth_deep"]    = (depth >= 10000).astype(float)
        for col in self.confounder_cols:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.confounder_cols].astype(float).to_numpy()
        if np.isnan(X).any():
            col_medians = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_medians, inds[1])
        return X

    def estimate_cate(self, well_rows: pd.DataFrame,
                      treatment_values: np.ndarray) -> dict:
        """Estimate per-well CATE(l) = E[Y(a) - Y(0) | L=l] with CIs.

        Args:
            well_rows: DataFrame with confounder columns + formation
            treatment_values: array of actual cum_vol_365d per well

        Returns dict with arrays:
            cate:    (n,) point estimates
            ci_low:  (n,) lower 95% CI
            ci_high: (n,) upper 95% CI
        """
        X = self._to_X(well_rows)
        T1 = np.asarray(treatment_values, dtype=float).ravel()
        T0 = np.zeros_like(T1)

        cate = self.cf.effect(X, T0=T0, T1=T1).flatten()
        ci_low, ci_high = self.cf.effect_interval(
            X, T0=T0, T1=T1, alpha=0.05
        )
        return {
            "cate":    cate.flatten(),
            "ci_low":  ci_low.flatten(),
            "ci_high": ci_high.flatten(),
        }


def fit_one_radius(R: int, window_days: int = WINDOW_DAYS) -> CausalForestBundle | None:
    """Fit CausalForestDML for one radius."""
    panel_path = Path(PANEL_FMT.format(R=R))
    if not panel_path.exists():
        log.warning("⚠️   %s missing — skipping", panel_path)
        return None

    log.info("[%2dkm] loading %s …", R, panel_path.name)
    panel = pd.read_csv(panel_path, low_memory=False)
    log.info("[%2dkm] %d rows × %d cols", R, *panel.shape)

    W = cum_volume_col(window_days)
    G_dep  = COL_PERF_DEPTH_FT
    G_age  = COL_DAYS_ACTIVE
    G_fdist = COL_NEAREST_FAULT_KM
    G_fseg = fault_segment_col(R)
    Y = COL_OUTCOME_MAX_ML

    required = [W, G_dep, G_age, G_fdist, G_fseg, Y, COL_FORMATION, COL_API]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        log.error("[%2dkm] missing columns: %s", R, missing)
        return None

    sub = panel[required].copy()
    n_before = len(sub)
    sub = sub.dropna(subset=[W, Y])
    log.info("[%2dkm] dropped %d rows with NaN treatment/outcome", R, n_before - len(sub))

    # Median-fill confounders
    for c in [G_dep, G_age, G_fdist, G_fseg]:
        sub[c] = sub[c].fillna(sub[c].median())

    # ── Depth-class proxy instead of self-reported formation ──
    # The `Current Injection Formations` field is operator-reported and
    # unreliable. Replace with a measured proxy: depth bins that capture
    # the same physical distinction (shallow carbonate vs mid-depth vs
    # basement-coupled) without depending on the formation label.
    depth_med = sub[G_dep].median()
    sub["depth_shallow"] = (sub[G_dep] < 6000).astype(float)     # < 6000 ft (SAN ANDRES / GRAYBURG class)
    sub["depth_mid"]     = ((sub[G_dep] >= 6000) & (sub[G_dep] < 10000)).astype(float)  # 6000-10000 ft
    sub["depth_deep"]    = (sub[G_dep] >= 10000).astype(float)    # > 10000 ft (ELLENBURGER / DEVONIAN class)
    formation_cols = ["depth_shallow", "depth_mid", "depth_deep"]
    top_forms = []  # no formation one-hots — using depth proxy instead
    log.info("[%2dkm] using depth-class proxy: shallow=%d mid=%d deep=%d",
             R, int(sub["depth_shallow"].sum()),
             int(sub["depth_mid"].sum()), int(sub["depth_deep"].sum()))

    confounder_cols = [G_dep, G_age, G_fdist, G_fseg, *formation_cols]

    X = sub[confounder_cols].astype(float).to_numpy()
    T = sub[W].astype(float).to_numpy()
    y = sub[Y].astype(float).to_numpy()
    n_pos = int((y > 0).sum())

    # Subsample for training speed — CausalForest on 800k rows is slow
    MAX_ROWS = 200_000
    if len(X) > MAX_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), MAX_ROWS, replace=False)
        X, T, y = X[idx], T[idx], y[idx]
        log.info("[%2dkm] subsampled: %d → %d rows", R, len(sub), MAX_ROWS)

    log.info("[%2dkm] fitting CausalForestDML: n=%d, n_pos=%d (%.2f%%)",
             R, len(X), n_pos, 100 * n_pos / max(len(sub), 1))

    import xgboost as xgb
    from econml.dml import CausalForestDML

    t0 = time.time()
    cf = CausalForestDML(
        model_y=xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            tree_method="hist", verbosity=0, n_jobs=1,
        ),
        model_t=xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            tree_method="hist", verbosity=0, n_jobs=1,
        ),
        n_estimators=200,
        min_samples_leaf=50,
        max_depth=None,
        honest=True,
        inference=True,
        random_state=42,
        n_jobs=1,  # single-threaded per worker (outer parallelism)
    )
    cf.fit(y, T, X=X)
    fit_time = time.time() - t0
    log.info("[%2dkm] ✅ CausalForestDML fit in %.1fs", R, fit_time)

    return CausalForestBundle(
        radius_km       = R,
        window_days     = window_days,
        confounder_cols = confounder_cols,
        treatment_col   = W,
        formation_cols  = formation_cols,
        top_formations  = top_forms,
        cf              = cf,
        n_train         = len(X),
        n_pos           = n_pos,
        fit_time_sec    = fit_time,
    )


def _worker(args: dict) -> dict:
    """ProcessPoolExecutor worker."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "2"  # 2 threads for xgb nuisance, not more
    R = args["R"]
    bundle = fit_one_radius(R, args.get("window_days", WINDOW_DAYS))
    if bundle is None:
        return {"R": R, "ok": False}
    out_path = Path(OUT_FMT.format(R=R))
    with out_path.open("wb") as f:
        pickle.dump(bundle, f)
    return {
        "R": R, "ok": True, "out": str(out_path),
        "n": bundle.n_train, "fit_sec": bundle.fit_time_sec,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radii", type=int, nargs="+", default=RADII_KM)
    p.add_argument("--window", type=int, default=WINDOW_DAYS)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    log.info("CAUSAL FOREST DML BUILD")
    log.info("  radii: %s  window: %dd  workers: %d", args.radii, args.window, args.workers)

    jobs = [{"R": R, "window_days": args.window} for R in args.radii]
    t0 = time.time()

    if args.workers <= 1:
        for j in jobs:
            res = _worker(j)
            if res["ok"]:
                log.info("✓ R=%dkm (n=%d, %.0fs)", res["R"], res["n"], res["fit_sec"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, j): j for j in jobs}
            for fut in as_completed(futs):
                res = fut.result()
                if res["ok"]:
                    log.info("✓ R=%dkm (n=%d, %.0fs)", res["R"], res["n"], res["fit_sec"])

    log.info("✅ total: %.1fs (%.1f min)", time.time() - t0, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
