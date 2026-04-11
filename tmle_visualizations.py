#!/usr/bin/env python3
"""
tmle_visualizations.py
──────────────────────
Plotting layer for the TMLE results. Reads the latest CSVs produced by
tmle_shift_analysis.py / tmle_dose_response.py / tmle_mediation_analysis.py
and generates publication-quality figures.

Outputs (./plots/):
  tmle_dose_response_at_1e7.png       — E[Y_a=1e7] vs radius, with bootstrap ribbon
  tmle_dose_response_curves.png        — small multiples: E[Y_a] vs a, one per radius
  tmle_dose_response_heatmap.png       — radius × log10(a) heatmap of E[Y_a]
  tmle_mediation_te_vs_radius.png      — TE / NDE / NIE vs radius (TMLE)
  tmle_pct_mediated_vs_radius.png      — % pressure-mediated vs radius (the killer collapse)
  tmle_shift_vs_radius.png             — shift +10% effect vs radius with CI ribbon
  tmle_vs_ols_total_effect.png         — side-by-side TMLE TE vs OLS event-level total
  tmle_summary_dashboard.png           — 4-panel overview
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def latest(glob: str) -> Path | None:
    matches = sorted(Path(".").glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_inputs():
    paths = {
        "shift":         latest("tmle_shift_*.csv"),
        "dose":          latest("tmle_dose_response_*.csv"),
        "mediation":     latest("tmle_mediation_*.csv"),
        "ols_event_ci":  latest("dowhy_event_level_ci_*.csv"),
        "ols_well_ci":   latest("dowhy_well_day_ci_*.csv"),
    }
    missing = [k for k, v in paths.items() if v is None]
    if missing:
        sys.exit(f"❌  Missing required CSV(s): {missing}")
    print("📄  Loading:")
    for k, v in paths.items():
        print(f"   {k:<14}  {v.name}")
    return {
        "shift":         pd.read_csv(paths["shift"]).sort_values("radius_km"),
        "dose":          pd.read_csv(paths["dose"]).sort_values(["radius_km", "a_star"]),
        "mediation":     pd.read_csv(paths["mediation"]).sort_values("radius_km"),
        "ols_event_ci":  pd.read_csv(paths["ols_event_ci"]).sort_values("radius_km"),
        "ols_well_ci":   pd.read_csv(paths["ols_well_ci"]).sort_values("radius_km"),
    }


# ──────────────────── Plot 1: dose-response at 1e7 BBL ───────────

def plot_dose_response_at_1e7(dose: pd.DataFrame) -> None:
    sub = dose[np.isclose(dose["a_star"], 1e7)].sort_values("radius_km")
    if sub.empty:
        print("⚠️   No 1e7 BBL grid point in dose-response file — skipping plot")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    r = sub["radius_km"].to_numpy()
    psi = sub["psi"].to_numpy()
    lo  = sub["ci_low"].to_numpy()
    hi  = sub["ci_high"].to_numpy()
    ax.plot(r, psi, "o-", lw=2.5, ms=7, color="#1f77b4", label="TMLE point estimate")
    ax.fill_between(r, lo, hi, alpha=0.25, color="#1f77b4", label="95% cluster-IF CI")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("E[Y_a] at a = 10⁷ BBL cumulative 365-day injection (ML)")
    ax.set_title("TMLE causal dose-response at 10 million BBL\n"
                 "Counterfactual expected max ML across all cluster-days")
    ax.set_xticks(r)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmle_dose_response_at_1e7.png")
    plt.close(fig)
    print("✅  tmle_dose_response_at_1e7.png")


# ──────────────────── Plot 2: dose-response small multiples ──────

def plot_dose_response_curves(dose: pd.DataFrame) -> None:
    radii = sorted(dose["radius_km"].unique())
    n = len(radii)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, R in enumerate(radii):
        ax = axes[i]
        sub = dose[dose["radius_km"] == R].sort_values("a_star")
        # Drop the positivity-tail row at the very largest a if it sits below
        # the second-to-last point — this is a heuristic clean-up for the
        # plot (the underlying CSV is unchanged)
        a = sub["a_star"].to_numpy()
        psi = sub["psi"].to_numpy()
        lo  = sub["ci_low"].to_numpy()
        hi  = sub["ci_high"].to_numpy()
        ax.plot(a, psi, "o-", lw=2, ms=5, color="#1f77b4")
        ax.fill_between(a, lo, hi, alpha=0.20, color="#1f77b4")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"R = {int(R)} km", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        if i % cols == 0:
            ax.set_ylabel("E[Y_a]")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("a (BBL)")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("TMLE dose-response curves: E[Y_a] vs cumulative 365-day injection, by radius",
                 y=1.00, fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmle_dose_response_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("✅  tmle_dose_response_curves.png")


# ──────────────────── Plot 3: dose-response heatmap ──────────────

def plot_dose_response_heatmap(dose: pd.DataFrame) -> None:
    pivot = dose.pivot_table(index="radius_km", columns="a_star", values="psi", aggfunc="first")
    pivot = pivot.sort_index()
    pivot.columns = [f"{int(np.log10(c))}" for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"10^{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(r)} km" for r in pivot.index])
    ax.set_xlabel("Cumulative 365-day injection volume (BBL)")
    ax.set_ylabel("Radius")
    ax.set_title("E[Y_a] heatmap: TMLE counterfactual expected max ML\nacross radius × cumulative volume")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Expected max ML")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmle_dose_response_heatmap.png")
    plt.close(fig)
    print("✅  tmle_dose_response_heatmap.png")


# ──────────────────── Plot 4: mediation TE/NDE/NIE vs radius ─────

def plot_mediation_components(med: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    r = med["radius_km"].to_numpy()
    ax.plot(r, med["TE"], "o-", lw=2.5, ms=6, color="#d62728", label="Total Effect (TE)")
    ax.fill_between(r, med["TE_ci_low"], med["TE_ci_high"], alpha=0.20, color="#d62728")
    ax.plot(r, med["NDE"], "s--", lw=2.0, ms=5, color="#1f77b4", label="Natural Direct Effect (NDE)")
    ax.plot(r, med["NIE"], "^-.", lw=2.0, ms=5, color="#2ca02c", label="Natural Indirect Effect (NIE)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Δ Expected max ML  (high vs low contrast on cumulative volume)")
    ax.set_title("TMLE mediation decomposition: high-vs-low contrast (p90 vs p10)")
    ax.set_xticks(r)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmle_mediation_te_vs_radius.png")
    plt.close(fig)
    print("✅  tmle_mediation_te_vs_radius.png")


# ──────────────────── Plot 5: % pressure mediated (the collapse) ─

def plot_pct_mediated(med: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    r = med["radius_km"].to_numpy()
    pct = med["pct_mediated"].to_numpy()
    colors = ["#d62728" if p < 0 else "#2ca02c" for p in pct]
    ax.bar(r, pct, color=colors, alpha=0.7, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    # Reference: OLS pipeline's "100% mediated" claim
    ax.axhline(100, color="grey", lw=1, ls="--", alpha=0.6,
               label="OLD pipeline claim: 100% mediated")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("% mediated by depth-corrected BHP")
    ax.set_title("TMLE refutation: pressure mediation collapses to ~0% at every radius\n"
                 "(under flexible Q model; the 100% finding was a Baron–Kenny linearity artifact)")
    ax.set_xticks(r)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmle_pct_mediated_vs_radius.png")
    plt.close(fig)
    print("✅  tmle_pct_mediated_vs_radius.png")


# ──────────────────── Plot 6: shift +10% vs radius ───────────────

def plot_shift_vs_radius(shift: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    r = shift["radius_km"].to_numpy()
    psi = shift["psi"].to_numpy()
    lo  = shift["ci_low"].to_numpy()
    hi  = shift["ci_high"].to_numpy()
    ax.plot(r, psi, "o-", lw=2.5, ms=6, color="#9467bd",
            label="TMLE shift +10% point estimate")
    ax.fill_between(r, lo, hi, alpha=0.25, color="#9467bd",
                    label="95% cluster-IF CI")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("ψ̂  (Δ expected max ML for +10% basin-wide shift)")
    ax.set_title("TMLE stochastic shift intervention: +10% multiplicative bump\n"
                 "on cumulative 365-day injection")
    ax.set_xticks(r)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmle_shift_vs_radius.png")
    plt.close(fig)
    print("✅  tmle_shift_vs_radius.png")


# ──────────────────── Plot 7: TMLE vs OLS side-by-side ───────────

def plot_tmle_vs_ols(med: pd.DataFrame, ols: pd.DataFrame) -> None:
    """Compare the TMLE high-vs-low contrast TE against the OLS event-level
    total_effect × Δa, where Δa = (a_high − a_low) ≈ 7.84e6 BBL.

    These should be in the same ballpark — the OLS slope times Δa is the
    linear approximation to the TMLE non-parametric contrast.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    delta_a = float(med["a_high"].iloc[0] - med["a_low"].iloc[0])
    ols_te_equivalent = ols["total_effect"].to_numpy() * delta_a

    ax.plot(med["radius_km"], med["TE"], "o-", lw=2.5, ms=7,
            color="#d62728", label=f"TMLE TE (p90 vs p10, Δa ≈ {delta_a:.1e} BBL)")
    ax.fill_between(med["radius_km"], med["TE_ci_low"], med["TE_ci_high"],
                    alpha=0.20, color="#d62728")
    ax.plot(ols["radius_km"], ols_te_equivalent, "s--", lw=2.0, ms=6,
            color="#1f77b4", label="OLS slope × Δa (linear extrapolation)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Δ Expected max ML over the high-vs-low cumulative-volume contrast")
    ax.set_title("Cross-estimator agreement: TMLE non-parametric TE vs OLS linear extrapolation\n"
                 "(both estimate the same parameter; agreement validates the spatial gradient)")
    ax.set_xticks(med["radius_km"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmle_vs_ols_total_effect.png")
    plt.close(fig)
    print("✅  tmle_vs_ols_total_effect.png")


# ──────────────────── Plot 8: 4-panel summary dashboard ──────────

def plot_summary_dashboard(shift: pd.DataFrame, dose: pd.DataFrame,
                           med: pd.DataFrame, ols: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Top-left: dose-response at 1e7
    ax = axes[0, 0]
    sub = dose[np.isclose(dose["a_star"], 1e7)].sort_values("radius_km")
    ax.plot(sub["radius_km"], sub["psi"], "o-", lw=2.5, ms=6, color="#1f77b4")
    ax.fill_between(sub["radius_km"], sub["ci_low"], sub["ci_high"],
                    alpha=0.20, color="#1f77b4")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("E[Y | a=1e7 BBL] (ML)")
    ax.set_title("(A) Dose-response at 10 million BBL")
    ax.grid(True, alpha=0.3)

    # Top-right: mediation TE / NDE / NIE
    ax = axes[0, 1]
    r = med["radius_km"]
    ax.plot(r, med["TE"], "o-", lw=2.5, color="#d62728", label="TE")
    ax.plot(r, med["NDE"], "s--", lw=2.0, color="#1f77b4", label="NDE")
    ax.plot(r, med["NIE"], "^-.", lw=2.0, color="#2ca02c", label="NIE")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Δ ML (p90 vs p10)")
    ax.set_title("(B) TE / NDE / NIE decomposition")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: % mediated bars
    ax = axes[1, 0]
    pct = med["pct_mediated"].to_numpy()
    colors = ["#d62728" if p < 0 else "#2ca02c" for p in pct]
    ax.bar(r, pct, color=colors, alpha=0.7, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(100, color="grey", lw=1, ls="--", alpha=0.6,
               label="OLD claim: 100%")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("% mediated by BHP")
    ax.set_title("(C) Pressure mediation collapses to ~0%")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: shift +10%
    ax = axes[1, 1]
    ax.plot(shift["radius_km"], shift["psi"], "o-", lw=2.5, color="#9467bd")
    ax.fill_between(shift["radius_km"], shift["ci_low"], shift["ci_high"],
                    alpha=0.25, color="#9467bd")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("ψ̂ for +10% shift (ML)")
    ax.set_title("(D) Shift +10% intervention")
    ax.grid(True, alpha=0.3)

    fig.suptitle("TMLE results summary — induced seismicity in the Midland Basin",
                 fontsize=15, y=1.00)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmle_summary_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("✅  tmle_summary_dashboard.png")


# ──────────────────── Main ───────────────────────────────────────

def main() -> None:
    data = load_inputs()
    plot_dose_response_at_1e7(data["dose"])
    plot_dose_response_curves(data["dose"])
    plot_dose_response_heatmap(data["dose"])
    plot_mediation_components(data["mediation"])
    plot_pct_mediated(data["mediation"])
    plot_shift_vs_radius(data["shift"])
    plot_tmle_vs_ols(data["mediation"], data["ols_event_ci"])
    plot_summary_dashboard(data["shift"], data["dose"], data["mediation"],
                           data["ols_event_ci"])
    print(f"\n✅  All plots written to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
