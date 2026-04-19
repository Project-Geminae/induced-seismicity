#!/usr/bin/env python3
"""
formation_sensitivity.py — Does dropping operator-reported formation bias the results?

Fits OLS mediation (total, direct, indirect effects) TWICE for each radius:
  1. Depth-class proxy only (current pipeline)
  2. Depth-class + formation one-hots (top-K most common formations)

Reports side-by-side comparison:
  - Total effect (c)
  - Direct effect (c')
  - Indirect effect (c - c')
  - % mediated

If estimates change substantially, the depth-proxy substitution is biasing results.

Usage:
    python formation_sensitivity.py [--top-k 5] [--radii 3 5 7 10 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import column_maps as cm

COL_TREATMENT = "cum_vol_365d_BBL"
COL_MEDIATOR = "bhp_vw_avg_365d"
COL_OUTCOME = "outcome_max_ML"
COL_CLUSTER = "API Number"


def build_design_matrices(df: pd.DataFrame, with_formation: bool, top_k: int = 5, radius_km: int = 7):
    """Construct X matrices for total/direct regressions, with or without formation."""
    # Fault segment column is per-radius: "Fault Segments <= 7 km"
    fault_seg_col = cm.fault_segment_col(radius_km)
    if fault_seg_col not in df.columns:
        # Try to find any fault-segment column
        candidates = [c for c in df.columns if "fault segments" in c.lower()]
        fault_seg_col = candidates[0] if candidates else None

    # Base confounders: fault dist, fault count, depth, days active, depth-class
    base = pd.DataFrame({
        "fault_dist": df["Nearest Fault Dist (km)"].fillna(df["Nearest Fault Dist (km)"].median()),
        "fault_segs": df[fault_seg_col].fillna(0) if fault_seg_col else 0,
        "perf_depth_ft": df["perf_depth_ft"].fillna(df["perf_depth_ft"].median()),
        "days_active": df["days_active"].fillna(df["days_active"].median()),
    })
    # Depth class (shallow < 6000, mid 6000-10000, deep >= 10000)
    d = df["perf_depth_ft"].fillna(df["perf_depth_ft"].median())
    base["depth_shallow"] = (d < 6000).astype(int)
    base["depth_mid"] = ((d >= 6000) & (d < 10000)).astype(int)

    if with_formation:
        # Get top-K most common formations
        fm_col = None
        for candidate in ["formation", "Current Injection Formations"]:
            if candidate in df.columns:
                fm_col = candidate
                break
        if fm_col is None:
            return None
        fm = df[fm_col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
        top = fm.value_counts().head(top_k).index.tolist()
        for formation in top:
            base[f"fm_{formation}"] = (fm == formation).astype(int)

    return base


def fit_mediation(df: pd.DataFrame, X_confounders: pd.DataFrame) -> dict:
    """Fit total, direct, and indirect effects via OLS.

    total:    Y ~ W + L          → β_W = c
    direct:   Y ~ W + M + L      → β_W = c'
    path a:   M ~ W + L          → β_W = a
    path b:   Y ~ W + M + L      → β_M = b  (direct equation)
    indirect: a * b  or  c - c'
    """
    W = df[COL_TREATMENT].values
    M = df[COL_MEDIATOR].values
    Y = df[COL_OUTCOME].values
    L = X_confounders.values
    clusters = df[COL_CLUSTER].values

    # Standardize treatment for numerical stability
    W_mean = W.mean()
    W_std = W.std() if W.std() > 0 else 1
    W_std_ = (W - W_mean) / W_std

    # Total: Y ~ W + L
    X_total = sm.add_constant(np.column_stack([W_std_, L]))
    m1 = sm.OLS(Y, X_total).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    c_std = m1.params[1]
    c = c_std / W_std  # de-standardize to per-BBL

    # Direct: Y ~ W + M + L
    X_direct = sm.add_constant(np.column_stack([W_std_, M, L]))
    m2 = sm.OLS(Y, X_direct).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    c_prime_std = m2.params[1]
    c_prime = c_prime_std / W_std
    b = m2.params[2]

    # Path a: M ~ W + L
    X_a = sm.add_constant(np.column_stack([W_std_, L]))
    m3 = sm.OLS(M, X_a).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    a_std = m3.params[1]
    a = a_std / W_std

    # Indirect effect
    indirect_ab = a * b
    indirect_diff = c - c_prime

    return {
        "c_total": c,
        "c_prime_direct": c_prime,
        "a": a,
        "b": b,
        "indirect_ab": indirect_ab,
        "indirect_diff": indirect_diff,
        "pct_mediated": 100 * indirect_ab / c if abs(c) > 1e-15 else np.nan,
        "n": len(df),
        "n_clusters": df[COL_CLUSTER].nunique(),
    }


def run_radius(radius_km: int, top_k: int) -> dict:
    """Run both models for one radius and return side-by-side results."""
    panel_path = REPO / f"panel_with_faults_{radius_km}km.csv"
    if not panel_path.exists():
        return None

    df = pd.read_csv(panel_path, low_memory=False)
    # Drop rows missing treatment or outcome
    df = df.dropna(subset=[COL_TREATMENT, COL_OUTCOME, COL_MEDIATOR, COL_CLUSTER])
    if len(df) < 100:
        return None

    # Model 1: depth-class only (current pipeline)
    X1 = build_design_matrices(df, with_formation=False, radius_km=radius_km)
    r1 = fit_mediation(df, X1)

    # Model 2: depth-class + formation
    X2 = build_design_matrices(df, with_formation=True, top_k=top_k, radius_km=radius_km)
    if X2 is None:
        return None
    r2 = fit_mediation(df, X2)

    # Compute deltas
    def pct_diff(a, b):
        if abs(b) < 1e-15:
            return np.nan
        return 100 * (a - b) / abs(b)

    return {
        "radius_km": radius_km,
        "n": r1["n"],
        "depth_only_c": r1["c_total"],
        "depth_only_c_prime": r1["c_prime_direct"],
        "depth_only_indirect": r1["indirect_ab"],
        "depth_only_pct_med": r1["pct_mediated"],
        "with_formation_c": r2["c_total"],
        "with_formation_c_prime": r2["c_prime_direct"],
        "with_formation_indirect": r2["indirect_ab"],
        "with_formation_pct_med": r2["pct_mediated"],
        "delta_c_pct": pct_diff(r2["c_total"], r1["c_total"]),
        "delta_c_prime_pct": pct_diff(r2["c_prime_direct"], r1["c_prime_direct"]),
        "delta_indirect_pct": pct_diff(r2["indirect_ab"], r1["indirect_ab"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of top formations to include (default: 5)")
    ap.add_argument("--radii", type=int, nargs="+", default=[3, 5, 7, 10, 15, 20],
                    help="Radii to run (default: 3 5 7 10 15 20)")
    args = ap.parse_args()

    print(f"Formation sensitivity analysis · top-{args.top_k} formations · radii {args.radii}")
    print("=" * 80)

    results = []
    for radius_km in args.radii:
        print(f"\n--- R={radius_km} km ---")
        res = run_radius(radius_km, args.top_k)
        if res is None:
            print(f"  (no data)")
            continue
        results.append(res)
        print(f"  n = {res['n']:,}")
        print(f"                      Depth-only    With-formation   Δ (%)")
        print(f"  Total (c):       {res['depth_only_c']:>12.4e}  {res['with_formation_c']:>14.4e}  {res['delta_c_pct']:>+7.1f}%")
        print(f"  Direct (c'):     {res['depth_only_c_prime']:>12.4e}  {res['with_formation_c_prime']:>14.4e}  {res['delta_c_prime_pct']:>+7.1f}%")
        print(f"  Indirect (a·b):  {res['depth_only_indirect']:>12.4e}  {res['with_formation_indirect']:>14.4e}  {res['delta_indirect_pct']:>+7.1f}%")
        print(f"  % mediated:      {res['depth_only_pct_med']:>11.1f}%  {res['with_formation_pct_med']:>13.1f}%")

    df = pd.DataFrame(results)
    out_path = REPO / "formation_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path.name}")

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    if len(df) == 0:
        print("No radii had data.")
        return

    max_abs_delta_c = df["delta_c_pct"].abs().max()
    max_abs_delta_ind = df["delta_indirect_pct"].abs().max()

    if max_abs_delta_c < 15 and max_abs_delta_ind < 30:
        print(f"✅ Formation substitution is LOW IMPACT.")
        print(f"   Max |Δ total effect|:    {max_abs_delta_c:.1f}%")
        print(f"   Max |Δ indirect effect|: {max_abs_delta_ind:.1f}%")
        print(f"   Depth-class proxy captures most of the formation-level variation.")
    elif max_abs_delta_c < 30 and max_abs_delta_ind < 60:
        print(f"⚠  Formation substitution is MODERATE IMPACT.")
        print(f"   Max |Δ total effect|:    {max_abs_delta_c:.1f}%")
        print(f"   Max |Δ indirect effect|: {max_abs_delta_ind:.1f}%")
        print(f"   Consider reporting formation-adjusted results as sensitivity.")
    else:
        print(f"🚨 Formation substitution is HIGH IMPACT.")
        print(f"   Max |Δ total effect|:    {max_abs_delta_c:.1f}%")
        print(f"   Max |Δ indirect effect|: {max_abs_delta_ind:.1f}%")
        print(f"   Results are sensitive to formation — reconsider depth-only approach.")


if __name__ == "__main__":
    main()
