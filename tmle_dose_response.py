#!/usr/bin/env python3
"""
tmle_dose_response.py
─────────────────────
TMLE for the causal dose-response curve E[Y_a] at a grid of cumulative-volume
levels, for each radius.

This is the TMLE-driven replacement for the parametric Gaussian
exceedance-probability curves in causal_poe_curves.py. Where the PoE curves
assume a fixed-σ linear model and convert OLS slopes into Φ-CDF outputs, the
dose-response TMLE estimates E[Y_a] at each grid point directly via a
g-computation TMLE that uses Super Learner for Q.

Output: tmle_dose_response_<window>d_<timestamp>.csv with columns
  radius_km, window_days, a_star, psi, ci_low, ci_high, n_in_bin
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import causal_core as cc
import tmle_core as tmle


PANEL_FMT       = "panel_with_faults_{R}km.csv"
RADII           = list(range(1, 21))
DEFAULT_WINDOW  = 365


# Default grid: 1e3 to 1e8 BBL on a log scale, 8 points
def default_grid() -> np.ndarray:
    return np.logspace(3, 8, 8)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"Lookback window in days. Default {DEFAULT_WINDOW}.")
    p.add_argument("--radii", type=int, nargs="+", default=RADII,
                   help="Radii (km) to analyze.")
    p.add_argument("--grid", type=float, nargs="+", default=None,
                   help="Cumulative-volume grid points (BBL). "
                        "Default: log-spaced from 1e3 to 1e8.")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. Default: tmle_dose_response_<window>d_<timestamp>.csv")
    return p.parse_args()


def analyze_radius(R: int, window: int, grid: np.ndarray) -> pd.DataFrame | None:
    path = Path(PANEL_FMT.format(R=R))
    if not path.exists():
        log.warning("⚠️   %s missing — skipping", path)
        return None

    log.info("[%2dkm] loading + aggregating", R)
    panel = cc.load_panel(str(path), radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=window)

    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=window)
    data = data.copy()
    data["_cluster"] = cluster.values
    log.info("[%2dkm] design matrix: %d rows, %d clusters",
             R, len(data), int(pd.Series(cluster).nunique()))

    log.info("[%2dkm] fitting TMLE dose-response at %d grid points…", R, len(grid))
    t0 = time.time()
    df = tmle.tmle_dose_response(
        df=data, A_col=W, L_cols=confs, Y_col=S, cluster_col="_cluster",
        a_grid=grid,
    )
    elapsed = time.time() - t0
    log.info("[%2dkm] done in %.0fs", R, elapsed)

    df.insert(0, "radius_km", R)
    df.insert(1, "window_days", window)
    return df


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = Path(args.output or f"tmle_dose_response_{args.window}d_{timestamp}.csv")

    grid = np.array(args.grid) if args.grid is not None else default_grid()
    log.info("TMLE DOSE-RESPONSE ANALYSIS")
    log.info("  window: %d days  radii: %s  grid points: %d", args.window, args.radii, len(grid))
    log.info("  grid: %s", ", ".join(f"{g:.1e}" for g in grid))
    log.info("  output: %s", outfile)

    pipeline_t0 = time.time()
    all_dfs: list[pd.DataFrame] = []
    for R in args.radii:
        df = analyze_radius(R, args.window, grid)
        if df is not None:
            all_dfs.append(df)
            pd.concat(all_dfs, ignore_index=True).to_csv(outfile, index=False)

    pipeline_elapsed = time.time() - pipeline_t0
    log.info("✅  Wrote %s (%d rows) in %.1fs (%.1f min)",
             outfile, sum(len(d) for d in all_dfs), pipeline_elapsed, pipeline_elapsed / 60)

    if not all_dfs:
        return

    full = pd.concat(all_dfs, ignore_index=True)
    print("\n🎯  TMLE DOSE-RESPONSE CURVES — selected radii")
    for R in (3, 7, 15, 20):
        sub = full[full["radius_km"] == R]
        if sub.empty:
            continue
        print(f"\n  Radius {R} km:")
        print(f"    {'A (BBL)':>12} {'E[Y_a]':>12} {'95% CI':>30} {'n_bin':>8}")
        for _, r in sub.iterrows():
            ci = f"[{r['ci_low']:>+.3e}, {r['ci_high']:>+.3e}]"
            print(f"    {r['a_star']:>12.2e} {r['psi']:>+12.3e} {ci:>30} {int(r['n_in_bin']):>8}")


if __name__ == "__main__":
    main()
