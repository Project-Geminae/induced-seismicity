#!/usr/bin/env python3
"""
dowhy_ci.py
───────────
Well-day causal analysis WITH cluster-bootstrap CIs and substantive
refutation tests.

For every radius R ∈ {1..20} km and a single configurable lookback window
(default 30 days; override via DEFAULT_WINDOW), produce:

  • Total / direct / indirect effects from cluster-robust OLS
  • 95% CIs from cluster bootstrap (50 iters, resampling wells with
    replacement)
  • Placebo-treatment refutation (shuffle W → effect should be ≈ 0)
  • Random-common-cause refutation (mean over 20 iters of injecting an
    N(0,1) confounder — effect should not move)
  • Unobserved-confounder sensitivity sweep (effect strengths {0.1, 0.3, 0.5})
  • PASS / FLAG / FAIL classification (NOT the brittle 10% threshold from
    the old pipeline)
  • VIF for the design matrix
  • Misspecification flag when |indirect| > |total|

Output: dowhy_well_day_ci_<timestamp>.csv with one row per radius.

Note: this script REPLACES the old dowhy_ci.py which (a) silently changed
the working directory if no CSVs were found in cwd, (b) constructed three
DoWhy CausalModel objects per bootstrap iteration when statsmodels OLS
gives the same answer, (c) declared "PASS" if the 80% subset estimate was
within 20% of the original — a test that's guaranteed to pass on N=300k.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import causal_core as cc
from column_maps import LOOKBACK_WINDOWS

PANEL_FMT       = "panel_with_faults_{R}km.csv"
RADII           = list(range(1, 21))
DEFAULT_WINDOW  = 30          # days; bootstrap one window for runtime budget
N_BOOTSTRAP     = 50
TIMESTAMP       = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTFILE         = Path(f"dowhy_well_day_ci_{TIMESTAMP}.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def analyze_radius(R: int, window_days: int) -> dict | None:
    path = Path(PANEL_FMT.format(R=R))
    if not path.exists():
        log.warning("⚠️   %s missing — skipping", path)
        return None

    log.info("[%2dkm] loading %s", R, path.name)
    panel = cc.load_panel(str(path), radius_km=R)

    log.info("[%2dkm] building design matrix (window=%dd)…", R, window_days)
    data, W, P, S, confs, cluster = cc.build_design_matrix(panel, R, window_days)

    log.info("[%2dkm] fitting OLS effects…", R)
    fit = cc.fit_effects(data, W, P, S, confs, cluster_id=cluster)

    log.info("[%2dkm] cluster-bootstrapping (n_iter=%d)…", R, N_BOOTSTRAP)
    boot = cc.cluster_bootstrap_ci(data, W, P, S, confs, n_iter=N_BOOTSTRAP)

    log.info("[%2dkm] running refutations…", R)
    placebo = cc.placebo_treatment_refutation(data, W, P, S, confs)
    rcc_mean, rcc_std = cc.random_common_cause_refutation(data, W, P, S, confs, n_iter=20)
    sensitivity = cc.unobserved_confounder_sensitivity(data, W, P, S, confs)
    status = cc.refutation_status(fit.total_effect, placebo, rcc_mean)

    log.info("[%2dkm] computing VIFs…", R)
    vifs = cc.compute_vif(data, [W, P, *confs])
    avg_vif = sum(v for v in vifs.values() if v == v) / max(1, len(vifs))

    return {
        "radius_km":         R,
        "window_days":       window_days,
        "treatment_col":     W,
        "mediator_col":      P,
        "n":                 fit.n,
        "n_wells":           data[cc.COL_API].nunique(),
        "total_effect":      fit.total_effect,
        "total_pval":        fit.total_pval,
        "total_ci_low":      boot["total_ci"][0],
        "total_ci_high":     boot["total_ci"][1],
        "direct_effect":     fit.direct_effect,
        "direct_pval":       fit.direct_pval,
        "direct_ci_low":     boot["direct_ci"][0],
        "direct_ci_high":    boot["direct_ci"][1],
        "indirect_diff":     fit.indirect_effect_diff,
        "indirect_product":  fit.indirect_effect_product,
        "indirect_ci_low":   boot["indirect_ci"][0],
        "indirect_ci_high":  boot["indirect_ci"][1],
        "path_a":            fit.path_a,
        "path_a_pval":       fit.path_a_pval,
        "path_b":            fit.path_b,
        "path_b_pval":       fit.path_b_pval,
        "causal_r2":         fit.causal_r2,
        "misspecified_flag": fit.misspecified,
        "bootstrap_ok":      boot["n_iter_ok"],
        "placebo_effect":    placebo,
        "rcc_mean":          rcc_mean,
        "rcc_std":           rcc_std,
        **sensitivity,
        "refutation_status": status,
        "avg_vif":           avg_vif,
    }


def main() -> None:
    log.info("Well-day causal analysis (window=%dd, n_boot=%d)", DEFAULT_WINDOW, N_BOOTSTRAP)
    rows = []
    t0 = time.time()
    for R in RADII:
        result = analyze_radius(R, DEFAULT_WINDOW)
        if result is not None:
            rows.append(result)

    out = pd.DataFrame(rows)
    out.to_csv(OUTFILE, index=False)
    log.info("✅  Wrote %s (%d rows) in %.1fs", OUTFILE, len(out), time.time() - t0)

    print("\n🎯  WELL-DAY CAUSAL EFFECTS WITH CLUSTER-BOOTSTRAP 95% CIs")
    print("─" * 100)
    print(f"{'Radius':>6} {'Effect':>12} {'95% CI':>26} {'p':>10} {'R²':>7} {'Refute':>8}")
    print("─" * 100)
    for _, r in out.iterrows():
        ci = f"[{r['total_ci_low']:>+.2e}, {r['total_ci_high']:>+.2e}]"
        print(f"{int(r['radius_km']):>4}km {r['total_effect']:>+12.3e} {ci:>26} "
              f"{r['total_pval']:>10.2e} {r['causal_r2']:>7.4f} {r['refutation_status']:>8}")


if __name__ == "__main__":
    main()
