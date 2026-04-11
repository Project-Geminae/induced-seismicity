#!/usr/bin/env python3
"""
dowhy_simple_all.py
───────────────────
Well-day causal analysis (no bootstrap).

For every radius R ∈ {1..20} km and every lookback window
W ∈ {30, 90, 180, 365} days, run a single OLS-based mediation analysis on
the (well, day) panel and write a summary CSV.

This is the "simple" / fast variant. The bootstrap CI version lives in
dowhy_ci.py.

Treatment   = cumulative_volume_<W>d_BBL
Mediator    = depth-corrected BHP from volume-weighted WHP over <W>d
Outcome     = max ML on that day within R km of the well, 0 if no event
Confounders = nearest fault distance, fault segment count within R, perf
              depth, days_active, formation (one-hot top-K + OTHER)

Output: causal_well_day_simple.csv with one row per (radius, window).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

import causal_core as cc
from column_maps import LOOKBACK_WINDOWS

PANEL_FMT = "panel_with_faults_{R}km.csv"
OUTFILE   = Path("causal_well_day_simple.csv")
RADII     = list(range(1, 21))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    rows = []
    t0 = time.time()
    for R in RADII:
        path = Path(PANEL_FMT.format(R=R))
        if not path.exists():
            log.warning("⚠️   %s missing — skipping", path)
            continue
        log.info("[%2dkm] loading %s", R, path.name)
        panel = cc.load_panel(str(path), radius_km=R)
        for window_days in LOOKBACK_WINDOWS:
            data, W, P, S, confs, cluster = cc.build_design_matrix(panel, R, window_days)
            fit = cc.fit_effects(data, W, P, S, confs, cluster_id=cluster)
            row = {
                "radius_km":          R,
                "window_days":        window_days,
                "treatment_col":      W,
                "mediator_col":       P,
                "n":                  fit.n,
                "n_wells":            data[cc.COL_API].nunique(),
                "total_effect":       fit.total_effect,
                "total_pval":         fit.total_pval,
                "direct_effect":      fit.direct_effect,
                "direct_pval":        fit.direct_pval,
                "indirect_diff":      fit.indirect_effect_diff,
                "indirect_product":   fit.indirect_effect_product,
                "path_a":             fit.path_a,
                "path_a_pval":        fit.path_a_pval,
                "path_b":             fit.path_b,
                "path_b_pval":        fit.path_b_pval,
                "causal_r2":          fit.causal_r2,
                "misspecified_flag":  fit.misspecified,
            }
            rows.append(row)
            log.info("       %3dd | total=%+.3e (p=%.2e) direct=%+.3e R²=%.4f%s",
                     window_days, fit.total_effect, fit.total_pval,
                     fit.direct_effect, fit.causal_r2,
                     "  MISSPEC" if fit.misspecified else "")

    out = pd.DataFrame(rows)
    out.to_csv(OUTFILE, index=False)
    log.info("✅  Wrote %s (%d rows) in %.1fs", OUTFILE, len(out), time.time() - t0)

    print("\n📊  Strongest total effect at each radius (across all windows):")
    if len(out):
        idx = out.groupby("radius_km")["total_effect"].apply(lambda s: s.abs().idxmax())
        for i in idx.values:
            r = out.iloc[i]
            print(f"   R={int(r['radius_km']):>2}km  win={int(r['window_days']):>3}d  "
                  f"total={r['total_effect']:+.3e}  p={r['total_pval']:.2e}")


if __name__ == "__main__":
    main()
