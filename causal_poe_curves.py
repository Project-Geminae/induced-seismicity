#!/usr/bin/env python3
"""
causal_poe_curves.py
────────────────────
Causal probability-of-exceedance (PoE) curves built on top of the new
panel-based causal pipeline.

For every radius R ∈ {1..20} km, fits the same OLS-based mediation model used
by dowhy_simple_all_aggregate.py at the (date, location-cluster) level, then
converts the fitted model into PoE curves:

    P( max ML ≥ M_thr | cumulative volume W ) = 1 − Φ((M_thr − μ(W)) / σ̂)

where μ(W) = α + τW + β·x̄ over the observed values of the confounders.

Inputs
------
  panel_with_faults_<R>km.csv  ← from add_geoscience_to_panel.py

Outputs
-------
  poe_radius_<R>km.csv          (M_thr × W grid of exceedance probabilities)
  poe_radius_<R>km.png          (per-radius log/log plot with 95% CI band)
  poe_all_radii.png             (combined comparison plot)

Note: this script REPLACES the old causal_poe_curves.py which used the brittle
COL_FRAGS substring matching, did its own per-script DoWhy CausalModel
construction (slow and duplicative), and consumed the OLD pipeline's
event_well_links_with_faults_<R>km.csv format. The new version delegates the
model fit to causal_core.fit_effects() and reads the new
panel_with_faults_<R>km.csv format produced by add_geoscience_to_panel.py.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import causal_core as cc
from column_maps import (
    COL_OUTCOME_MAX_ML,
    confounder_columns,
    treatment_column,
)


# ──────────────────── Configuration ─────────────────────────────
PANEL_FMT     = "panel_with_faults_{R}km.csv"
RADII         = list(range(1, 21))
WINDOW_DAYS   = 30
MAG_THRESH    = [2.0, 3.0, 4.0, 5.0]
VOL_GRID      = np.logspace(4, 6, 50)         # 10^4 .. 10^6 BBL cumulative
CONF_LEVEL    = 0.95


# ──────────────────── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


def fit_radius(R: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fit the OLS model for radius R and return (poe_df, design_data).

    Uses the cluster-day aggregation (same as dowhy_simple_all_aggregate.py)
    for higher signal-to-noise.
    """
    path = Path(PANEL_FMT.format(R=R))
    if not path.exists():
        log.warning("⚠️   %s missing — skipping", path)
        return None

    log.info("[%2dkm] loading + aggregating", R)
    panel = cc.load_panel(str(path), radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=WINDOW_DAYS)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, WINDOW_DAYS)

    if len(data) < 100:
        log.warning("[%2dkm] only %d rows — skipping", R, len(data))
        return None

    # Fit total-effect OLS (clustered on the cluster pseudo-id)
    X = sm.add_constant(data[[W, *confs]].astype(float), has_constant="add")
    y = data[S].astype(float)
    ols = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster.values},
    )

    alpha = float(ols.params["const"])
    tau   = float(ols.params[W])
    beta  = ols.params.drop(["const", W])
    sigma = float(ols.resid.std(ddof=len(ols.params)))
    x_mean = data[beta.index].astype(float).mean()

    # Build PoE surface
    z = norm.ppf(1 - (1 - CONF_LEVEL) / 2)
    cov_W = float(ols.cov_params().loc[W, W])

    recs = []
    for mthr in MAG_THRESH:
        for w in VOL_GRID:
            mu = alpha + tau * w + float(beta.values @ x_mean.values)
            p  = 1 - norm.cdf((mthr - mu) / sigma)
            # Variance of mu is dominated by Var(τ)·w² for large w
            var_mu = cov_W * w * w
            se_mu  = float(np.sqrt(max(var_mu, 0.0)))
            se_p   = norm.pdf((mthr - mu) / sigma) * se_mu / sigma
            recs.append({
                "radius_km": R,
                "M_thr": mthr,
                "W": w,
                "P_exceed": float(p),
                "P_lo":     float(max(p - z * se_p, 0.0)),
                "P_hi":     float(min(p + z * se_p, 1.0)),
            })

    poe = pd.DataFrame(recs)
    poe.to_csv(f"poe_radius_{R}km.csv", index=False)
    log.info("[%2dkm] wrote poe_radius_%dkm.csv (τ=%+.3e, σ=%.3f, n=%d)",
             R, R, tau, sigma, len(data))
    return poe, data


def plot_radius(poe: pd.DataFrame, R: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, mthr in enumerate(MAG_THRESH):
        sub = poe[poe["M_thr"] == mthr]
        color = palette[k % len(palette)]
        ax.plot(sub["W"], sub["P_exceed"], color=color, lw=2.5, label=f"M ≥ {mthr}")
        ax.fill_between(sub["W"], sub["P_lo"], sub["P_hi"], alpha=0.20, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative 30-day injection volume (BBL)")
    ax.set_ylabel("P(max ML ≥ threshold | volume)")
    ax.set_title(f"Probability of exceedance — radius {R} km (95% CI)")
    ax.grid(True, which="major", lw=0.5, alpha=0.7)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.5)
    ax.legend(title="Threshold", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"poe_radius_{R}km.png", dpi=180)
    plt.close(fig)


def plot_all_radii(all_results: dict[int, pd.DataFrame]) -> None:
    if not all_results:
        return
    n_thresh = len(MAG_THRESH)
    fig, axes = plt.subplots(1, n_thresh, figsize=(5 * n_thresh, 5), sharey=True)
    if n_thresh == 1:
        axes = [axes]
    cmap = plt.cm.viridis
    for ax, mthr in zip(axes, MAG_THRESH):
        for R in sorted(all_results.keys()):
            sub = all_results[R][all_results[R]["M_thr"] == mthr]
            color = cmap(R / 20.0)
            ax.plot(sub["W"], sub["P_exceed"], color=color, lw=1.5,
                    label=f"{R} km" if R in (1, 5, 10, 15, 20) else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cumulative volume (BBL)")
        ax.set_title(f"M ≥ {mthr}")
        ax.grid(True, which="major", lw=0.5, alpha=0.5)
    axes[0].set_ylabel("Exceedance probability")
    axes[-1].legend(title="Radius", fontsize=9)
    fig.suptitle(f"Probability of exceedance vs. radius ({CONF_LEVEL:.0%} omitted for clarity)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig("poe_all_radii.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote poe_all_radii.png")


def main() -> None:
    all_results: dict[int, pd.DataFrame] = {}
    for R in RADII:
        result = fit_radius(R)
        if result is None:
            continue
        poe, _ = result
        plot_radius(poe, R)
        all_results[R] = poe
    plot_all_radii(all_results)
    log.info("✅  PoE curves complete (%d radii)", len(all_results))


if __name__ == "__main__":
    main()
