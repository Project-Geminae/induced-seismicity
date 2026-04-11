#!/usr/bin/env python3
"""
induced_seismicity_scaling_plots.py
───────────────────────────────────
Publication-quality plots of induced-seismic effect estimates vs. distance,
loading the latest dowhy_well_day_ci_*.csv (or dowhy_event_level_ci_*.csv)
output instead of hardcoding numbers from a prior pipeline run.

Outputs (./plots/)
------------------
total_effect_vs_distance.png
direct_effect_vs_distance.png
indirect_effect_vs_distance.png

Note: the old version of this script also produced a "well-count multiplier"
plot whose underlying scaling assumption (direct ∝ 1/N) was hand-imposed
rather than learned from data. That plot has been removed because the new
analysis pipeline doesn't expose a per-well-count breakdown — if you want it
back, run the analysis at multiple well-density slices and concatenate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler


def latest(glob: str) -> Path | None:
    matches = sorted(Path(".").glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_results() -> pd.DataFrame:
    """Prefer event-level (higher SNR), fall back to well-day."""
    for glob in ("dowhy_event_level_ci_*.csv", "dowhy_well_day_ci_*.csv"):
        path = latest(glob)
        if path is not None:
            print(f"📄  Loading {path}")
            return pd.read_csv(path).sort_values("radius_km")
    sys.exit("❌  No dowhy_*_ci_*.csv files found — run dowhy_ci.py / dowhy_ci_aggregated.py")


def main() -> None:
    df = load_results()
    radii    = df["radius_km"].values
    total    = df["total_effect"].values
    total_lo = df["total_ci_low"].values
    total_hi = df["total_ci_high"].values
    direct   = df["direct_effect"].values
    direct_lo = df["direct_ci_low"].values
    direct_hi = df["direct_ci_high"].values
    indirect    = df["indirect_diff"].values
    indirect_lo = df["indirect_ci_low"].values
    indirect_hi = df["indirect_ci_high"].values

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize":  (9, 6),
        "savefig.dpi":     350,
        "axes.prop_cycle": cycler(color=plt.cm.tab10.colors),
        "axes.titlesize":  16,
        "axes.labelsize":  13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon":  False,
    })

    Path("plots").mkdir(exist_ok=True)

    # ── Total effect vs distance ────────────────────────────────────────
    fig, ax = plt.subplots()
    ax.plot(radii, total, "o-", lw=2.5, color="tab:blue", label="Total effect")
    ax.fill_between(radii, total_lo, total_hi, alpha=0.20, color="tab:blue",
                    label="95% cluster-bootstrap CI")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set(xlabel="Distance from injection well (km)",
           ylabel="ΔM per BBL of cumulative 30-day injection",
           title="Total causal effect vs. distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/total_effect_vs_distance.png")
    plt.close(fig)

    # ── Direct effect vs distance ───────────────────────────────────────
    fig, ax = plt.subplots()
    ax.plot(radii, direct, "s-", lw=2.5, color="tab:red", label="Direct effect")
    ax.fill_between(radii, direct_lo, direct_hi, alpha=0.20, color="tab:red")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set(xlabel="Distance from injection well (km)",
           ylabel="Direct ΔM per BBL (controlling for pressure)",
           title="Direct mechanical component vs. distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/direct_effect_vs_distance.png")
    plt.close(fig)

    # ── Indirect (mediated) effect vs distance ──────────────────────────
    fig, ax = plt.subplots()
    ax.plot(radii, indirect, "^-", lw=2.5, color="tab:green", label="Indirect effect (c − c′)")
    ax.fill_between(radii, indirect_lo, indirect_hi, alpha=0.20, color="tab:green")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set(xlabel="Distance from injection well (km)",
           ylabel="Indirect ΔM via depth-corrected BHP",
           title="Pressure-mediated component vs. distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/indirect_effect_vs_distance.png")
    plt.close(fig)

    print("✅  Plots written to ./plots/")


if __name__ == "__main__":
    main()
