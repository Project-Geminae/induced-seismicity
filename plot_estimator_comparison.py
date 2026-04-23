#!/usr/bin/env python3
"""Single-figure comparison of estimators across all 20 radii.

Produces estimator_comparison.png: a vertical strip chart where each
radius has up to 5 horizontal error bars (one per estimator), letting
the reader see at a glance which estimators cluster together and which
are outliers. If the full-n plug-in methods (OLS, XGBoost, undersmoothed
HAL) cluster at +4-8e-4 while the targeting estimators (CV-TMLE v3 TMLE)
stack at +2-5e-3, the methodological story is immediately legible.

Input files (all from the repo root):
  cv_tmle_sweep_summary.csv                        — CV-TMLE subsampled (20 radii)
  tmle_shift_365d_<timestamp>.csv                  — Standard TMLE v3 (20 radii)
  hal_shift_{R}km.csv (optional)                   — Undersmoothed HAL per radius
  benchmark_shift_7km.csv                          — OLS/XGBoost at R=7 only
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(".")


def load_cv_tmle():
    p = ROOT / "cv_tmle_sweep_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)[["radius_km", "psi", "ci_low", "ci_high"]]
    df["estimator"] = "CV-TMLE + haldensify + HAL (n=50k)"
    return df


def load_v3_tmle():
    files = sorted(ROOT.glob("tmle_shift_365d_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[-1])[["radius_km", "psi", "ci_low", "ci_high"]]
    df["estimator"] = "Standard TMLE v3 (full n)"
    return df


def load_hal():
    frames = []
    for f in ROOT.glob("hal_shift_*km.csv"):
        d = pd.read_csv(f)
        if "ci_low" not in d.columns and "se_boot" in d.columns:
            d["ci_low"] = d["psi"] - 1.96 * d["se_boot"]
            d["ci_high"] = d["psi"] + 1.96 * d["se_boot"]
        frames.append(d[["radius_km", "psi", "ci_low", "ci_high"]])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["estimator"] = "Undersmoothed HAL (full n)"
    return df


def load_benchmark_plugins():
    """Load the OLS and XGBoost benchmark points (radius 7 only for now)."""
    p = ROOT / "benchmark_shift_7km.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        method = r["method"]
        if "OLS" in method:
            rows.append({"radius_km": 7, "psi": float(r["psi"]),
                         "ci_low": np.nan, "ci_high": np.nan,
                         "estimator": "OLS plug-in (full n)"})
        elif "XGBoost" in method:
            rows.append({"radius_km": 7, "psi": float(r["psi"]),
                         "ci_low": np.nan, "ci_high": np.nan,
                         "estimator": "XGBoost plug-in (full n)"})
    return pd.DataFrame(rows)


def make_figure(df_all: pd.DataFrame, outpath: Path = Path("estimator_comparison.png")):
    estimators = [
        "Standard TMLE v3 (full n)",
        "CV-TMLE + haldensify + HAL (n=50k)",
        "Undersmoothed HAL (full n)",
        "XGBoost plug-in (full n)",
        "OLS plug-in (full n)",
    ]
    colors = {
        "Standard TMLE v3 (full n)":            "#D62728",  # red (the prior headline)
        "CV-TMLE + haldensify + HAL (n=50k)":   "#FF7F0E",  # orange
        "Undersmoothed HAL (full n)":           "#2CA02C",  # green (van der Laan-faithful)
        "XGBoost plug-in (full n)":             "#1F77B4",  # blue
        "OLS plug-in (full n)":                 "#9467BD",  # purple
    }

    radii = sorted(df_all["radius_km"].unique().tolist())
    fig, ax = plt.subplots(figsize=(10, 12))

    for i_est, est in enumerate(estimators):
        sub = df_all[df_all["estimator"] == est]
        for _, r in sub.iterrows():
            y = r["radius_km"] + 0.12 * (i_est - (len(estimators) - 1) / 2)
            x = r["psi"]
            if np.isfinite(r.get("ci_low", np.nan)) and np.isfinite(r.get("ci_high", np.nan)):
                ax.errorbar(x, y,
                            xerr=[[x - r["ci_low"]], [r["ci_high"] - x]],
                            fmt="o", color=colors[est], capsize=2, markersize=5,
                            label=est if r["radius_km"] == radii[0] else None,
                            alpha=0.85)
            else:
                ax.plot(x, y, "D", color=colors[est], markersize=6,
                        label=est if r["radius_km"] == radii[0] else None,
                        alpha=0.85)

    ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.set_yticks(radii)
    ax.set_ylabel("Radius (km)", fontsize=12)
    ax.set_xlabel("ψ(δ = −0.10): change in expected M_L under 10% volume reduction", fontsize=11)
    ax.set_title("Shift-intervention estimates across methods and radii\nMidland Basin, 365-day cumulative volume, 2017–2026", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower right", fontsize=9, framealpha=0.95)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    print(f"Wrote {outpath}")


def main():
    frames = [load_v3_tmle(), load_cv_tmle(), load_hal(), load_benchmark_plugins()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        print("No estimator data found in cwd.")
        return
    df_all = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df_all)} rows across {df_all['estimator'].nunique()} estimators")
    print(df_all.groupby("estimator")["radius_km"].count())
    make_figure(df_all)


if __name__ == "__main__":
    main()
