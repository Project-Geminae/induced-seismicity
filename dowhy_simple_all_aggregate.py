#!/usr/bin/env python3
"""
dowhy_simple_all_aggregate.py
─────────────────────────────
Event-level (cluster-day) causal analysis (no bootstrap).

Aggregates the (well, day) panel into one row per (date, ~5 km cluster) and
runs the same OLS-based mediation analysis as dowhy_simple_all.py. The unit
of analysis is "the cluster on a given day", which collapses cross-well
variation and (typically) gives a much higher signal-to-noise ratio than the
well-day analysis.

Aggregation rules used by causal_core.aggregate_panel_to_event_level():
  • Volume:               SUM
  • Pressure (mediator):  VOLUME-WEIGHTED MEAN  (Σ P_i V_i / Σ V_i)
  • Fault distance:       MIN  (closest well drives the geology)
  • Fault segment count:  MEAN (NOT sum — sum double-counts shared faults)
  • Depth:                VOLUME-WEIGHTED MEAN
  • Days active:          MEAN
  • Formation:            MODE
  • Outcome max ML:       MAX

The bootstrap CI version lives in dowhy_ci_aggregated.py.
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
OUTFILE   = Path("causal_event_level_simple.csv")
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
        log.info("       aggregating to (date, cluster) cells …")
        agg = cc.aggregate_panel_to_event_level(panel, R, window_days=30)
        n_cells = len(agg)
        n_event_cells = int((agg["outcome_max_ML"] > 0).sum())
        log.info("       %d cells, %d with outcome>0", n_cells, n_event_cells)

        for window_days in LOOKBACK_WINDOWS:
            data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days)
            fit = cc.fit_effects(data, W, P, S, confs, cluster_id=cluster)
            rows.append({
                "radius_km":          R,
                "window_days":        window_days,
                "treatment_col":      W,
                "mediator_col":       P,
                "n_cells":            n_cells,
                "n_event_cells":      n_event_cells,
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
            })
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
