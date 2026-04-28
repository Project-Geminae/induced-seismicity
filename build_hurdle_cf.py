#!/usr/bin/env python3
"""
build_hurdle_cf.py
──────────────────
Train hurdle Causal Forests for the dashboard's frequency × magnitude
decomposition. Companion to build_causal_forest.py.

For each radius R, fits TWO CausalForestDML models:

  cf_hurdle_log_<R>km.pkl   — logistic stage: outcome = 1{Y > 0}
                              CATE = ∂P(Y > 0 | A, L) / ∂A
                              ("frequency channel": effect on event probability)

  cf_hurdle_mag_<R>km.pkl   — magnitude stage: outcome = log(1+Y) on rows Y > 0
                              CATE = ∂E[log(1+Y) | Y>0, A, L] / ∂A
                              ("magnitude channel": effect on conditional event size)

Compose with cf_targeted.hurdle_decompose_total() to recover

  ψ_total = ψ_freq + ψ_mag + ψ_cross

This decomposition matches the science paper's regHAL-TMLE channel split
(53/34/13 at 7 km in the paper).

Usage
-----
    python build_hurdle_cf.py --radii 7 --workers 1
    python build_hurdle_cf.py --radii 1 2 3 4 5 6 7 8 9 10 --workers 4

References:
    Mullahy (1986). "Specification and testing of some modified count data
    models." Journal of Econometrics — original hurdle model.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from build_causal_forest import CausalForestBundle  # reuse the dataclass
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
OUT_LOG_FMT    = "cf_hurdle_log_{R}km.pkl"
OUT_MAG_FMT    = "cf_hurdle_mag_{R}km.pkl"
RADII_KM       = list(range(1, 21))
WINDOW_DAYS    = 365
MAX_TRAIN_ROWS = 200_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _build_features(R: int, panel: pd.DataFrame, window_days: int):
    """Mirror build_causal_forest._build_features but expose Y separately."""
    W = cum_volume_col(window_days)
    G_dep   = COL_PERF_DEPTH_FT
    G_age   = COL_DAYS_ACTIVE
    G_fdist = COL_NEAREST_FAULT_KM
    G_fseg  = fault_segment_col(R)
    Y       = COL_OUTCOME_MAX_ML
    G_rate     = "avg_rate_365d"
    G_neighbor = "neighbor_cum_vol_7km"

    required = [W, G_dep, G_age, G_fdist, G_fseg, Y, COL_FORMATION, COL_API]
    has_rate     = G_rate in panel.columns
    has_neighbor = G_neighbor in panel.columns
    if has_rate:     required.append(G_rate)
    if has_neighbor: required.append(G_neighbor)
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    sub = panel[required].copy()
    sub = sub.dropna(subset=[W, Y])
    fill_cols = [G_dep, G_age, G_fdist, G_fseg]
    if has_rate:     fill_cols.append(G_rate)
    if has_neighbor: fill_cols.append(G_neighbor)
    for c in fill_cols:
        sub[c] = sub[c].fillna(sub[c].median())

    # Depth-class proxy (matches build_causal_forest.py)
    sub["depth_shallow"] = (sub[G_dep] < 6000).astype(float)
    sub["depth_mid"]     = ((sub[G_dep] >= 6000) & (sub[G_dep] < 10000)).astype(float)
    sub["depth_deep"]    = (sub[G_dep] >= 10000).astype(float)
    formation_cols = ["depth_shallow", "depth_mid", "depth_deep"]

    confounder_cols = [G_dep, G_age, G_fdist, G_fseg, *formation_cols]
    if has_rate:     confounder_cols.append(G_rate)
    if has_neighbor: confounder_cols.append(G_neighbor)

    return sub, confounder_cols, formation_cols, W, Y


def _fit_cf(X, T, y, *, n_estimators=200, min_samples_leaf=50):
    """Fit a CausalForestDML on continuous outcome y."""
    import xgboost as xgb
    from econml.dml import CausalForestDML
    cf = CausalForestDML(
        model_y=xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            tree_method="hist", verbosity=0, n_jobs=1,
        ),
        model_t=xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            tree_method="hist", verbosity=0, n_jobs=1,
        ),
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=None,
        honest=True,
        inference=True,
        random_state=42,
        n_jobs=1,
    )
    cf.fit(y, T, X=X)
    return cf


def fit_one_radius(R: int, window_days: int = WINDOW_DAYS) -> dict | None:
    panel_path = Path(PANEL_FMT.format(R=R))
    if not panel_path.exists():
        log.warning("⚠️   %s missing — skipping", panel_path)
        return None

    log.info("[%2dkm] loading %s …", R, panel_path.name)
    panel = pd.read_csv(panel_path, low_memory=False)

    sub, confounder_cols, formation_cols, W, Y = _build_features(R, panel, window_days)
    log.info("[%2dkm] design rows=%d cols=%s", R, len(sub), confounder_cols)

    X_full = sub[confounder_cols].astype(float).to_numpy()
    T_full = sub[W].astype(float).to_numpy()
    y_full = sub[Y].astype(float).to_numpy()

    # Subsample to keep training time reasonable, mirroring build_causal_forest.py
    if len(X_full) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_full), MAX_TRAIN_ROWS, replace=False)
        X_full, T_full, y_full = X_full[idx], T_full[idx], y_full[idx]

    # ── Logistic stage: 1{Y > 0} ──
    y_log = (y_full > 0).astype(float)
    n_pos_log = int(y_log.sum())
    log.info("[%2dkm] logistic stage: n=%d, P(Y>0)=%.3f", R, len(y_log), n_pos_log / len(y_log))
    t0 = time.time()
    cf_log = _fit_cf(X_full, T_full, y_log)
    fit_log_sec = time.time() - t0
    log.info("[%2dkm] ✅ logistic fit in %.1fs", R, fit_log_sec)

    # ── Magnitude stage: log(1+Y) on Y > 0 subset ──
    mask = y_full > 0
    X_mag = X_full[mask]
    T_mag = T_full[mask]
    y_mag = np.log1p(y_full[mask])
    log.info("[%2dkm] magnitude stage: n=%d (Y>0 subset)", R, len(y_mag))
    if len(y_mag) < 1000:
        log.warning("[%2dkm] magnitude subset too small (n=%d) — skipping mag stage", R, len(y_mag))
        cf_mag = None
        fit_mag_sec = 0.0
    else:
        t0 = time.time()
        cf_mag = _fit_cf(X_mag, T_mag, y_mag, min_samples_leaf=20)
        fit_mag_sec = time.time() - t0
        log.info("[%2dkm] ✅ magnitude fit in %.1fs", R, fit_mag_sec)

    bundle_log = CausalForestBundle(
        radius_km       = R,
        window_days     = window_days,
        confounder_cols = confounder_cols,
        treatment_col   = W,
        formation_cols  = formation_cols,
        top_formations  = [],
        cf              = cf_log,
        n_train         = len(X_full),
        n_pos           = n_pos_log,
        fit_time_sec    = fit_log_sec,
    )
    bundle_mag = None
    if cf_mag is not None:
        bundle_mag = CausalForestBundle(
            radius_km       = R,
            window_days     = window_days,
            confounder_cols = confounder_cols,
            treatment_col   = W,
            formation_cols  = formation_cols,
            top_formations  = [],
            cf              = cf_mag,
            n_train         = len(X_mag),
            n_pos           = len(X_mag),
            fit_time_sec    = fit_mag_sec,
        )

    return {
        "log": bundle_log,
        "mag": bundle_mag,
        "n_pos": n_pos_log,
    }


def _worker(args: dict) -> dict:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "2"
    R = args["R"]
    res = fit_one_radius(R, args.get("window_days", WINDOW_DAYS))
    if res is None:
        return {"R": R, "ok": False}
    Path(OUT_LOG_FMT.format(R=R)).write_bytes(pickle.dumps(res["log"]))
    if res["mag"] is not None:
        Path(OUT_MAG_FMT.format(R=R)).write_bytes(pickle.dumps(res["mag"]))
    return {
        "R": R, "ok": True,
        "n_pos": res["n_pos"],
        "log_path": OUT_LOG_FMT.format(R=R),
        "mag_path": OUT_MAG_FMT.format(R=R) if res["mag"] is not None else None,
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
    log.info("HURDLE CAUSAL FOREST BUILD")
    log.info("  radii: %s  window: %dd  workers: %d", args.radii, args.window, args.workers)

    jobs = [{"R": R, "window_days": args.window} for R in args.radii]
    t0 = time.time()
    if args.workers <= 1:
        for j in jobs:
            res = _worker(j)
            if res["ok"]:
                log.info("✓ R=%dkm: log=%s mag=%s", res["R"], res["log_path"], res["mag_path"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, j): j for j in jobs}
            for fut in as_completed(futs):
                res = fut.result()
                if res["ok"]:
                    log.info("✓ R=%dkm: log=%s mag=%s", res["R"], res["log_path"], res["mag_path"])
    log.info("✅ total: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
