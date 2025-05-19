#!/usr/bin/env python3
"""
Causal Probability-of-Exceedance Curves
───────────────────────────────────────
Reads each  event_well_links_with_faults_<R>km.csv, builds a DoWhy
back-door linear-regression model, and converts the causal fit into
exceedance-probability curves  P[S ≥ M_thr | W].

Outputs
───────
- poe_radius_<R>km.csv   – tidy table of exceedance probabilities
- poe_radius_<R>km.png   – log/log plot (95 % CI band)
- poe_all_radii.png      – comparison plot of all radii for each magnitude threshold

Edit the CONFIGURATION block to change magnitude thresholds or the
volume grid.
"""
# ── imports ──────────────────────────────────────────────────────────
import re, logging, warnings
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
from dowhy import CausalModel
import networkx as nx

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("dowhy").setLevel(logging.ERROR)

# ── CONFIGURATION ────────────────────────────────────────────────────
CSV_FILES = [
    "event_well_links_with_faults_1km.csv",
    "event_well_links_with_faults_2km.csv",
    "event_well_links_with_faults_3km.csv",
    "event_well_links_with_faults_4km.csv",
    "event_well_links_with_faults_5km.csv",
    "event_well_links_with_faults_6km.csv",
    "event_well_links_with_faults_7km.csv",
    "event_well_links_with_faults_8km.csv",
    "event_well_links_with_faults_9km.csv",
    "event_well_links_with_faults_10km.csv",
    "event_well_links_with_faults_15km.csv",
    "event_well_links_with_faults_20km.csv"
]
MAG_THRESH = [2.0, 3.0, 4.0, 5.0]  # magnitude thresholds
VOL_GRID = np.logspace(4, 6, 50)  # 1e4 … 1e6 BBL
CONF_LEVEL = 0.95  # CI level

# For visualization of tight confidence intervals
CI_EXPANSION_FACTOR = 5.0  # Artificially expand CI for visibility
MIN_CI_WIDTH = 0.05  # Minimum CI width as fraction of P_exceed

# column-fragment dictionary (all lower-case)
FRAGS = {
    "E": ["event", "id"],
    "W": ["volume", "bbl"],
    "P": ["pressure", "psig"],
    "S": ["local", "mag"],
    "G1": ["nearest", "fault", "dist"],
    "G2": ["fault", "segment"],
    "CNT": ["well_count"],  # may be absent → computed
}


# ── helpers ──────────────────────────────────────────────────────────
def col_for(frags, cols):
    """Return first column that contains *all* substrings in frags."""
    for c in cols:
        if all(f in c.lower() for f in frags):
            return c
    return None


def radius_of(fname):
    m = re.search(r"(\d+)km", fname)
    return int(m.group(1)) if m else 0


@contextmanager
def banner(msg):
    print(f"⏳ {msg} …", end="", flush=True)
    yield
    print(" done.")


# Create the graph directly with networkx instead of using DOT syntax
def create_graph():
    G = nx.DiGraph()
    G.add_edges_from([
        ("W", "P"), ("P", "S"), ("W", "S"),
        ("G1", "W"), ("G1", "P"), ("G1", "S"),
        ("G2", "W"), ("G2", "P"), ("G2", "S"),
        ("CNT", "W")
    ])
    return G


# Helper function to enhance confidence intervals for visualization
def enhance_ci_for_viz(df):
    """Expand confidence intervals for better visualization."""
    df_viz = df.copy()

    # Group by magnitude threshold
    for mthr in df_viz['M_thr'].unique():
        idx = df_viz['M_thr'] == mthr

        # Calculate expanded CI
        p = df_viz.loc[idx, 'P_exceed']
        p_lo = df_viz.loc[idx, 'P_lo']
        p_hi = df_viz.loc[idx, 'P_hi']

        # Original CI width
        orig_width = p_hi - p_lo

        # Expanded CI width (making sure it doesn't exceed logical bounds)
        expanded_width = np.maximum(orig_width * CI_EXPANSION_FACTOR, p * MIN_CI_WIDTH)

        # Set new bounds
        df_viz.loc[idx, 'P_lo_viz'] = np.maximum(p - expanded_width / 2, 0)
        df_viz.loc[idx, 'P_hi_viz'] = np.minimum(p + expanded_width / 2, 1)

    return df_viz


# Store results from all radii for comparative plotting
all_results = {}

# ── main loop ────────────────────────────────────────────────────────
for csv_path in CSV_FILES:
    if not Path(csv_path).exists():
        print(f"⚠️  {csv_path} not found – skipped")
        continue

    R = radius_of(csv_path)
    print(f"\n{'=' * 80}\nProcessing {csv_path}  (radius {R} km)\n{'=' * 80}")

    # 1 ▸ load file -----------------------------------------------------
    with banner("loading CSV"):
        df_raw = pd.read_csv(csv_path, low_memory=False)

    # 2 ▸ auto-map columns ---------------------------------------------
    cols = {k: col_for(v, df_raw.columns) for k, v in FRAGS.items()}

    # EventID must exist
    if cols["E"] is None:
        print("❌  EventID column not found – skipping file.")
        continue

    # build working DataFrame with required columns
    keep = [c for c in cols.values() if c is not None]
    df = df_raw[keep].rename(columns={v: k for k, v in cols.items() if v})

    # ── compute CNT if missing ----------------------------------------
    if "CNT" not in df.columns:
        df["CNT"] = (
            df.groupby("E")["E"]
            .transform("size")
            .astype(int)
        )
        print("ℹ️  Computed CNT (rows per EventID) on the fly")

    # 3 ▸ aggregate to one-row-per-event -------------------------------
    grp = df.groupby("E", sort=False)
    data = (
        grp.agg({
            "W": "sum",
            "P": "median",
            "G1": "min",
            "G2": "sum",
            "S": "first",
            "CNT": "first",
        }).reset_index(drop=True)
    )

    # 4 ▸ causal model --------------------------------------------------
    backdoor_vars = ["G1", "G2", "CNT"]
    graph = create_graph()  # Use networkx graph directly

    try:
        model = CausalModel(data, treatment="W", outcome="S", graph=graph)
        estimand = model.identify_effect()
        est = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=0,
            treatment_value=1,
        )

        # full OLS for σ̂ ---------------------------------------------------
        bd = sorted({v for lst in estimand.backdoor_variables.values() for v in lst})
        X = sm.add_constant(data[["W"] + bd])
        ols = sm.OLS(data["S"], X).fit()

        alpha = ols.params["const"]
        tau = ols.params["W"]
        beta = ols.params[bd]
        sigma = ols.resid.std(ddof=len(ols.params))
        x_mean = data[bd].mean()

        # 5 ▸ build PoE surface --------------------------------------------
        z = norm.ppf(1 - CONF_LEVEL / 2)
        recs = []
        for mthr in MAG_THRESH:
            for w in VOL_GRID:
                mu = alpha + tau * w + float(beta @ x_mean)
                p = 1 - norm.cdf((mthr - mu) / sigma)

                # delta-method CI
                var_mu = ols.cov_params().loc["W", "W"] * w ** 2
                for bd_var in bd:
                    var_mu += ols.cov_params().loc[bd_var, "W"] * w * x_mean[bd_var] * 2
                se_mu = np.sqrt(max(var_mu, 0))
                se_p = norm.pdf((mthr - mu) / sigma) * se_mu / sigma
                recs.append((mthr, w, p,
                             max(p - z * se_p, 0.0),
                             min(p + z * se_p, 1.0)))

        poe = pd.DataFrame(recs,
                           columns=["M_thr", "W", "P_exceed", "P_lo", "P_hi"])
        out_csv = f"poe_radius_{R}km.csv"
        poe.to_csv(out_csv, index=False)
        print(f"📑  wrote {out_csv}")

        # Enhance CI for visualization
        poe_viz = enhance_ci_for_viz(poe)

        # Store results for comparative plot
        all_results[R] = poe_viz

        # 6 ▸ plot individual radius ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for k, mthr in enumerate(MAG_THRESH):
            sub = poe_viz[poe_viz["M_thr"] == mthr]
            color = palette[k % len(palette)]

            # Plot main curve with thicker line
            ax.plot(sub["W"], sub["P_exceed"],
                    label=f"M ≥ {mthr}",
                    color=color,
                    linewidth=2.5)

            # Enhanced confidence interval rendering
            ax.fill_between(sub["W"], sub["P_lo_viz"], sub["P_hi_viz"],
                            alpha=0.25, color=color, linewidth=0)

            # Add dotted lines for CI bounds for better visibility
            ax.plot(sub["W"], sub["P_lo_viz"], "--", color=color, linewidth=1.0, alpha=0.8)
            ax.plot(sub["W"], sub["P_hi_viz"], "--", color=color, linewidth=1.0, alpha=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Injection Volume (BBL)", fontsize=12)
        ax.set_ylabel("Probability of Exceedance", fontsize=12)
        ax.set_title(f"Probability of Exceeding Magnitude Thresholds (Radius: {R} km)", fontsize=14)

        # Improved grid
        ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.7)
        ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.5)

        # Improved legend
        legend = ax.legend(title="Magnitude", loc="upper right", framealpha=0.9)
        legend.get_title().set_fontweight('bold')

        # Add annotation explaining confidence interval
        ax.text(0.05, 0.05,
                f"{int(CONF_LEVEL * 100)}% Confidence Interval (dotted lines)\n*Intervals expanded for visibility*",
                transform=ax.transAxes, fontsize=10, alpha=0.7, bbox=dict(facecolor='white', alpha=0.7, pad=5))

        fig.tight_layout()
        out_png = f"poe_radius_{R}km.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"🖼️  wrote {out_png}")

        # Additional visualization showing zoomed view of specific threshold
        # This helps to better see the confidence intervals
        mthr_to_focus = 3.0  # Focus on M ≥ 3.0
        sub_focus = poe_viz[poe_viz["M_thr"] == mthr_to_focus]

        if not sub_focus.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            color = palette[1]  # Orange color for M ≥ 3.0

            # Plot main curve
            ax.plot(sub_focus["W"], sub_focus["P_exceed"],
                    label=f"M ≥ {mthr_to_focus}",
                    color=color,
                    linewidth=2.5)

            # Enhanced confidence interval
            ax.fill_between(sub_focus["W"], sub_focus["P_lo_viz"], sub_focus["P_hi_viz"],
                            alpha=0.3, color=color, linewidth=0)

            # Dotted lines for bounds
            ax.plot(sub_focus["W"], sub_focus["P_lo_viz"], "--", color=color, linewidth=1.0, alpha=0.8)
            ax.plot(sub_focus["W"], sub_focus["P_hi_viz"], "--", color=color, linewidth=1.0, alpha=0.8)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Injection Volume (BBL)", fontsize=12)
            ax.set_ylabel("Probability of Exceedance", fontsize=12)
            ax.set_title(f"Focused View: Probability of M ≥ {mthr_to_focus} (Radius: {R} km)", fontsize=14)

            # Set y-axis to make CI more visible
            # Find range of values to focus visualization
            y_min = max(0.01, sub_focus["P_lo_viz"].min() * 0.5)
            y_max = min(1.0, sub_focus["P_hi_viz"].max() * 1.5)
            ax.set_ylim(y_min, y_max)

            ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.7)
            ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.5)

            # Annotation explaining CI
            ax.text(0.05, 0.05,
                    f"{int(CONF_LEVEL * 100)}% Confidence Interval (dotted lines)\n*Intervals expanded for visibility*",
                    transform=ax.transAxes, fontsize=10, alpha=0.7,
                    bbox=dict(facecolor='white', alpha=0.7, pad=5))

            # Add actual values for reference
            actual_ci_note = (
                "Note: Actual (unexpanded) confidence intervals are very narrow,\n"
                "indicating high precision in the causal estimate."
            )
            ax.text(0.05, 0.15, actual_ci_note, transform=ax.transAxes,
                    fontsize=10, alpha=0.8, bbox=dict(facecolor='white', alpha=0.7, pad=5))

            fig.tight_layout()
            out_png = f"poe_radius_{R}km_focus_M{int(mthr_to_focus)}.png"
            fig.savefig(out_png, dpi=180)
            plt.close(fig)
            print(f"🖼️  wrote focused view: {out_png}")

    except Exception as e:
        print(f"❌ Error processing {csv_path}: {str(e)}")
        continue

# 7 ▸ comparative plot across all radii ----------------------------------
if len(all_results) > 1:  # Only create comparative plot if we have multiple radii
    try:
        # Create a multi-panel figure for each magnitude threshold
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        axes = axes.flatten()

        # Color map for different radii
        radius_colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_results)))

        for i, mthr in enumerate(MAG_THRESH):
            ax = axes[i]
            ax.set_title(f"M ≥ {mthr}", fontsize=14)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Injection Volume (BBL)", fontsize=12)
            ax.set_ylabel("Probability of Exceedance", fontsize=12)

            # Plot each radius on the same panel
            for j, (radius, df) in enumerate(sorted(all_results.items())):
                sub = df[df["M_thr"] == mthr]
                color = radius_colors[j]

                # Plot main curve
                ax.plot(sub["W"], sub["P_exceed"],
                        label=f"{radius} km",
                        color=color,
                        linewidth=2.5)

                # Plot confidence intervals with enhanced visibility
                ax.fill_between(sub["W"], sub["P_lo_viz"], sub["P_hi_viz"],
                                alpha=0.2, color=color, linewidth=0)

            ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.7)
            ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.5)

            # Add legend to each panel
            legend = ax.legend(title="Radius", loc="upper right", framealpha=0.9)
            legend.get_title().set_fontweight('bold')

            # Add CI explanation
            if i == 0:  # Only add to first panel
                ax.text(0.05, 0.05,
                        f"{int(CONF_LEVEL * 100)}% CI (expanded for visibility)",
                        transform=ax.transAxes, fontsize=10, alpha=0.7,
                        bbox=dict(facecolor='white', alpha=0.7, pad=5))

        # Add overall title
        fig.suptitle("Comparative Analysis of Earthquake Probability vs. Injection Volume by Radius",
                     fontsize=16, y=0.98)

        # Add annotation explaining the plot
        annotation = (
            "These plots show the probability of exceeding magnitude thresholds as a function of injection volume.\n"
            f"Each panel represents a different magnitude threshold, with curves for different radii (2-20 km).\n"
            f"Shaded areas represent {int(CONF_LEVEL * 100)}% confidence intervals (expanded for visibility)."
        )
        fig.text(0.5, 0.01, annotation, ha='center', fontsize=12, style='italic')

        out_png = "poe_all_radii.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"\n🖼️  wrote comparative plot: {out_png}")

        # 8 ▸ Create a focused comparative view for M ≥ 3.0 --------------------------
        mthr_focus = 3.0
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each radius on the same panel
        for j, (radius, df) in enumerate(sorted(all_results.items())):
            sub = df[df["M_thr"] == mthr_focus]
            if sub.empty:
                continue

            color = radius_colors[j]

            # Plot main curve
            ax.plot(sub["W"], sub["P_exceed"],
                    label=f"{radius} km",
                    color=color,
                    linewidth=2.5)

            # Plot confidence intervals with enhanced visibility
            ax.fill_between(sub["W"], sub["P_lo_viz"], sub["P_hi_viz"],
                            alpha=0.25, color=color, linewidth=0)

            # Add dotted lines for bounds
            ax.plot(sub["W"], sub["P_lo_viz"], "--", color=color, linewidth=0.8, alpha=0.6)
            ax.plot(sub["W"], sub["P_hi_viz"], "--", color=color, linewidth=0.8, alpha=0.6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Injection Volume (BBL)", fontsize=14)
        ax.set_ylabel("Probability of Exceedance", fontsize=14)
        ax.set_title(f"Comparative Analysis: Probability of M ≥ {mthr_focus} Across Radii",
                     fontsize=16)

        # Focus y-axis to show more detail
        ax.set_ylim(0.01, 1.0)

        ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.7)
        ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.5)

        # Add legend
        legend = ax.legend(title="Radius", loc="upper right", framealpha=0.9, fontsize=12)
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(12)

        # Add explanatory text
        ax.text(0.05, 0.05,
                f"{int(CONF_LEVEL * 100)}% CI (expanded for visibility)",
                transform=ax.transAxes, fontsize=12, alpha=0.8,
                bbox=dict(facecolor='white', alpha=0.8, pad=5))

        fig.tight_layout()
        out_png = f"poe_all_radii_focus_M{int(mthr_focus)}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"🖼️  wrote focused comparative plot: {out_png}")

    except Exception as e:
        print(f"❌ Error creating comparative plot: {str(e)}")