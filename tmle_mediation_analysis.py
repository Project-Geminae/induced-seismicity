#!/usr/bin/env python3
"""
tmle_mediation_analysis.py
──────────────────────────
Mediational TMLE for natural direct and indirect effects (NDE / NIE) of
cumulative injection on max ML, with depth-corrected BHP as the mediator.

For each radius, the contrast is between two cumulative-volume levels:

  a_high = the 90th percentile of cum_vol_<window>d_BBL in the cluster-day data
  a_low  = the 10th percentile

The decomposition (under sequential ignorability + cross-world independence):

  Total effect (TE) = E[Y_{a_high}]              − E[Y_{a_low}]
  NIE              = E[Y_{a_high, M_{a_high}}]   − E[Y_{a_high, M_{a_low}}]
  NDE              = E[Y_{a_high, M_{a_low}}]    − E[Y_{a_low,  M_{a_low}}]
  TE = NIE + NDE

% mediated = NIE / TE

This replaces the linear Baron–Kenny path-product mediation in
dowhy_simple_all.py / dowhy_ci.py. The OLD pipeline assumed (a) linearity in
both the W → P path and the P → S path, and (b) no W × P interaction. The
hurdle Super Learner Q here handles the heavy zero mass in the outcome and
arbitrary nonlinearity in the mediator-outcome relationship; the cross-world
counterfactuals are estimated via plug-in g-computation.

Output: tmle_mediation_<window>d_<timestamp>.csv with one row per radius,
columns: TE, NDE, NIE, pct_mediated, and cluster-bootstrap CIs for each.
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
DEFAULT_HIGH_PCTL = 0.90
DEFAULT_LOW_PCTL  = 0.10


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
    p.add_argument("--high-pctl", type=float, default=DEFAULT_HIGH_PCTL,
                   help="High contrast quantile for cumulative volume.")
    p.add_argument("--low-pctl", type=float, default=DEFAULT_LOW_PCTL,
                   help="Low contrast quantile for cumulative volume.")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. Default: tmle_mediation_<window>d_<timestamp>.csv")
    return p.parse_args()


def analyze_radius(R: int, window: int, high_pctl: float, low_pctl: float) -> dict | None:
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

    a_high = float(np.quantile(data[W], high_pctl))
    a_low  = float(np.quantile(data[W], low_pctl))
    log.info("[%2dkm] design: %d rows; A high=%.2e, A low=%.2e",
             R, len(data), a_high, a_low)

    log.info("[%2dkm] fitting mediation TMLE…", R)
    t0 = time.time()
    res = tmle.tmle_mediation(
        df=data, A_col=W, M_col=P, L_cols=confs, Y_col=S, cluster_col="_cluster",
        a_high=a_high, a_low=a_low,
    )
    elapsed = time.time() - t0
    log.info("[%2dkm]   TE=%+.3e  NDE=%+.3e  NIE=%+.3e  %%med=%.1f%%  (%.0fs)",
             R, res["TE"], res["NDE"], res["NIE"], res["pct_mediated"], elapsed)

    return {
        "radius_km":      R,
        "window_days":    window,
        "n":              res["n"],
        "n_clusters":     res["n_clusters"],
        "a_high":         a_high,
        "a_low":          a_low,
        "TE":             res["TE"],
        "TE_ci_low":      res["TE_ci"][0],
        "TE_ci_high":     res["TE_ci"][1],
        "NDE":            res["NDE"],
        "NDE_ci_low":     res["NDE_ci"][0],
        "NDE_ci_high":    res["NDE_ci"][1],
        "NIE":            res["NIE"],
        "NIE_ci_low":     res["NIE_ci"][0],
        "NIE_ci_high":    res["NIE_ci"][1],
        "pct_mediated":   res["pct_mediated"],
        "n_iter_boot":    res["n_iter_boot"],
        "elapsed_sec":    elapsed,
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = Path(args.output or f"tmle_mediation_{args.window}d_{timestamp}.csv")

    log.info("TMLE MEDIATION ANALYSIS")
    log.info("  window: %d  radii: %s  contrast: p%d vs p%d",
             args.window, args.radii, int(args.high_pctl * 100), int(args.low_pctl * 100))
    log.info("  output: %s", outfile)

    pipeline_t0 = time.time()
    rows: list[dict] = []
    for R in args.radii:
        result = analyze_radius(R, args.window, args.high_pctl, args.low_pctl)
        if result is not None:
            rows.append(result)
            pd.DataFrame(rows).to_csv(outfile, index=False)

    pipeline_elapsed = time.time() - pipeline_t0
    log.info("✅  Wrote %s (%d rows) in %.1fs (%.1f min)",
             outfile, len(rows), pipeline_elapsed, pipeline_elapsed / 60)

    if not rows:
        return

    df = pd.DataFrame(rows)
    print("\n🎯  TMLE MEDIATION DECOMPOSITION (high-vs-low contrast)")
    print("─" * 110)
    print(f"{'Radius':>6} {'TE':>12} {'NDE':>12} {'NIE':>12} {'%med':>7} "
          f"{'TE 95% CI':>26}")
    print("─" * 110)
    for _, r in df.iterrows():
        ci = f"[{r['TE_ci_low']:>+.2e}, {r['TE_ci_high']:>+.2e}]"
        print(f"{int(r['radius_km']):>4}km "
              f"{r['TE']:>+12.3e} {r['NDE']:>+12.3e} {r['NIE']:>+12.3e} "
              f"{r['pct_mediated']:>6.1f}% {ci:>26}")


if __name__ == "__main__":
    main()
