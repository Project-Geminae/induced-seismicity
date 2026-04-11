#!/usr/bin/env python3
"""
tmle_shift_analysis.py
──────────────────────
Stochastic shift intervention TMLE for each radius.

For every radius R ∈ {1..20} km, aggregate the (well, day) panel to one row
per (date, ~5km cluster), then estimate the causal effect of a multiplicative
shift on cumulative 365-day injection volume:

    ψ_δ = E[Y_{A·(1+δ)}] − E[Y_A]

where Y = max ML on that cluster-day, A = cumulative volume at the chosen
lookback window. The default shift is +10%; pass --shifts 0.05 0.10 0.25 to
sweep multiple shift levels.

This is the TMLE analog of dowhy_ci_aggregated.py. Compared to the OLS-based
event-level analysis it:

  • Uses a hurdle Super Learner (logistic + GBM + XGBoost stack) for Q,
    appropriate for the heavily zero-inflated outcome.
  • Uses a histogram-via-XGBoost-multinomial conditional density for g.
  • Reports influence-function-based variance with a cluster-IF correction
    that respects the within-cluster correlation in the panel.
  • Is doubly robust: misspecification of Q OR g still yields a consistent
    point estimate.

Output: tmle_shift_<window>d_<timestamp>.csv with one row per (radius, shift).
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
DEFAULT_SHIFTS  = (0.10,)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"Lookback window in days (30/90/180/365). Default {DEFAULT_WINDOW}.")
    p.add_argument("--shifts", type=float, nargs="+", default=list(DEFAULT_SHIFTS),
                   help="Shift fractions, e.g. 0.05 0.10 0.25. Default 0.10.")
    p.add_argument("--radii", type=int, nargs="+", default=RADII,
                   help="Radii (km) to analyze.")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. Default: tmle_shift_<window>d_<timestamp>.csv")
    return p.parse_args()


def analyze_radius(R: int, window: int, shifts: list[float]) -> list[dict] | None:
    path = Path(PANEL_FMT.format(R=R))
    if not path.exists():
        log.warning("⚠️   %s missing — skipping", path)
        return None

    log.info("[%2dkm] loading + aggregating (window=%dd)", R, window)
    panel = cc.load_panel(str(path), radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=window)

    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=window)
    data = data.copy()
    data["_cluster"] = cluster.values
    log.info("[%2dkm] design matrix: %d rows × %d confounders, %d clusters",
             R, len(data), len(confs), int(pd.Series(cluster).nunique()))

    rows = []
    for shift in shifts:
        log.info("[%2dkm] running TMLE shift %+.0f%%…", R, shift * 100)
        t0 = time.time()
        result = tmle.tmle_shift(
            df=data, A_col=W, L_cols=confs, Y_col=S, cluster_col="_cluster",
            shift_pct=shift,
        )
        elapsed = time.time() - t0
        log.info("[%2dkm]   ψ=%+.3e  CI=[%+.2e, %+.2e]  p=%.3f  (%.0fs)",
                 R, result.psi, result.ci_low, result.ci_high, result.pval, elapsed)
        rows.append({
            "radius_km":          R,
            "window_days":        window,
            "shift_pct":          shift,
            "treatment_col":      W,
            "n":                  result.n,
            "n_clusters":         result.n_clusters,
            "psi":                result.psi,
            "se_iid":             result.se_iid,
            "se_cluster":         result.se_cluster,
            "ci_low":             result.ci_low,
            "ci_high":            result.ci_high,
            "pval":               result.pval,
            "epsilon":            result.epsilon,
            "psi_under_shift":    result.notes.get("psi_under_shift"),
            "psi_no_shift":       result.notes.get("psi_no_shift"),
            "mean_H":             result.notes.get("mean_H"),
            "max_H":              result.notes.get("max_H"),
            "elapsed_sec":        elapsed,
        })
    return rows


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = Path(args.output or f"tmle_shift_{args.window}d_{timestamp}.csv")

    log.info("TMLE SHIFT ANALYSIS")
    log.info("  window: %d days  shifts: %s  radii: %s", args.window, args.shifts, args.radii)
    log.info("  output: %s", outfile)

    pipeline_t0 = time.time()
    all_rows: list[dict] = []
    for R in args.radii:
        results = analyze_radius(R, args.window, args.shifts)
        if results:
            all_rows.extend(results)
            # Write incrementally so we don't lose progress on a long run
            pd.DataFrame(all_rows).to_csv(outfile, index=False)

    pipeline_elapsed = time.time() - pipeline_t0
    log.info("✅  Wrote %s (%d rows) in %.1fs (%.1f min)",
             outfile, len(all_rows), pipeline_elapsed, pipeline_elapsed / 60)

    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    print("\n🎯  TMLE SHIFT INTERVENTION RESULTS")
    print("─" * 110)
    print(f"{'Radius':>6} {'Shift':>7} {'ψ̂':>13} {'95% CI':>30} "
          f"{'p':>8} {'ε':>10} {'n':>10}")
    print("─" * 110)
    for _, r in df.iterrows():
        ci = f"[{r['ci_low']:>+.3e}, {r['ci_high']:>+.3e}]"
        print(f"{int(r['radius_km']):>4}km "
              f"{r['shift_pct']:>+6.0%}  "
              f"{r['psi']:>+13.3e} {ci:>30} "
              f"{r['pval']:>8.3f} {r['epsilon']:>+10.2e} "
              f"{int(r['n']):>10,}")


if __name__ == "__main__":
    main()
