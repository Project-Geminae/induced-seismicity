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
    """A pickled bundle: the fitted Q + the metadata needed to apply it.

    The dashboard loads one of these per radius and uses .predict() with a
    feature dict to score per-well factual / counterfactual ML estimates.
    """
    radius_km:      int
    window_days:    int
    feature_cols:   list[str]      # column order for the design matrix
    formation_cols: list[str]      # the one-hot dummy column names
    top_formations: list[str]      # which formations got their own one-hot
    q:              tmle.HurdleSuperLearner
    n_train:        int
    n_pos:          int
    fit_time_sec:   float

    def predict(self, well_rows: pd.DataFrame) -> np.ndarray:
        """Score a DataFrame of per-well feature rows.

        Required columns in well_rows (case-sensitive, exactly as in the panel):
            cum_vol_365d_BBL, bhp_vw_avg_365d, perf_depth_ft, days_active,
            Nearest Fault Dist (km), Fault Segments <= R km, formation
        Missing values are median-imputed using the same global medians the
        Q was trained on (we re-impute on-the-fly here from the values
        present in well_rows; for a single-cell prediction the imputation is
        a no-op as long as the values are non-NaN).
        """
        df = well_rows.copy()
        # One-hot the formation column (top-K + OTHER), matching the train-time encoding
        df["_form"] = np.where(
            df[COL_FORMATION].isin(self.top_formations),
            df[COL_FORMATION],
            "OTHER",
        )
        for col in self.formation_cols:
            label = col[len("form_"):]  # strip the "form_" prefix
            df[col] = (df["_form"] == label).astype(float)
        # Build the design matrix in the train-time column order
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_cols].astype(float).to_numpy()
        # Impute any remaining NaN (defensive — well_rows should already be clean)
        if np.isnan(X).any():
            col_medians = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_medians, inds[1])
        return self.q.predict(X)


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

    log.info("[%2dkm] fitting HurdleSuperLearner: n=%d, n_pos=%d (%.2f%%)",
             R, len(X), n_pos, 100 * n_pos / max(len(X), 1))

    t0 = time.time()
    q = tmle.HurdleSuperLearner(random_state=42)
    q.fit(X, y)
    fit_time = time.time() - t0
    log.info("[%2dkm] ✅ fit in %.1fs", R, fit_time)

    return AttributionQ(
        radius_km      = R,
        window_days    = window_days,
        feature_cols   = feature_cols,
        formation_cols = formation_cols,
        top_formations = top_forms,
        q              = q,
        n_train        = len(X),
        n_pos          = n_pos,
        fit_time_sec   = fit_time,
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
