#!/usr/bin/env python3
"""
model_improvements.py — Run four modeling improvements at primary radii (7, 10 km)
and report impact on OLS mediation estimates.

Improvements tested:
  1. INJECTION RATE: add avg daily BBL/day as confounder
  2. EVENT CLUSTERING: cluster bootstrap by (date, grid5km) instead of well
  3. H TRIMMING: trim observations at extreme propensity scores (for TMLE)
  4. W×M INTERACTION: test for violated mediation assumption

For each, compare to the baseline (current pipeline) estimates side-by-side.
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
COL_CLUSTER_WELL = "API Number"


def load_panel(radius_km: int) -> pd.DataFrame:
    path = REPO / f"panel_with_faults_{radius_km}km.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=[COL_TREATMENT, COL_OUTCOME, COL_MEDIATOR, COL_CLUSTER_WELL])
    return df


def add_injection_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute avg daily injection rate over the 365-day window.
    avg_rate_365d = cum_vol_365d / min(365, days_active)
    """
    days = df["days_active"].fillna(365).clip(upper=365).clip(lower=1)
    df = df.copy()
    df["avg_rate_365d"] = df[COL_TREATMENT] / days
    return df


def add_event_cluster_id(df: pd.DataFrame, grid_deg: float = 0.05) -> pd.DataFrame:
    """Assign cluster ID by (event_date, 5km spatial grid).

    Only meaningful for rows with outcome > 0 (an actual event occurred).
    For non-event rows, cluster by well (keep independence assumption).
    """
    df = df.copy()
    # Date column
    date_col = None
    for c in ["Date of Injection", "event_date", "Event Date"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None or "Latitude (WGS84)" not in df.columns:
        # Fallback: use well-based clustering
        df["event_cluster"] = df[COL_CLUSTER_WELL].astype(str)
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    lat_bin = (df["Latitude (WGS84)"].fillna(0) / grid_deg).round().astype(int)
    lon_bin = (df["Longitude (WGS84)"].fillna(0) / grid_deg).round().astype(int)
    date_str = df[date_col].dt.strftime("%Y-%m-%d").fillna("1900-01-01")

    # For non-event rows, keep per-well clustering to preserve independence
    has_event = df[COL_OUTCOME] > 0
    df["event_cluster"] = np.where(
        has_event,
        date_str + "_" + lat_bin.astype(str) + "_" + lon_bin.astype(str),
        df[COL_CLUSTER_WELL].astype(str)
    )
    return df


def build_X(df: pd.DataFrame, radius_km: int, include_rate: bool = False) -> pd.DataFrame:
    """Base confounder design matrix. Optionally include injection rate."""
    fault_seg_col = cm.fault_segment_col(radius_km)
    if fault_seg_col not in df.columns:
        candidates = [c for c in df.columns if "fault segments" in c.lower()]
        fault_seg_col = candidates[0] if candidates else None

    d = df["perf_depth_ft"].fillna(df["perf_depth_ft"].median())
    X = pd.DataFrame({
        "fault_dist": df["Nearest Fault Dist (km)"].fillna(df["Nearest Fault Dist (km)"].median()),
        "fault_segs": df[fault_seg_col].fillna(0) if fault_seg_col else 0,
        "perf_depth_ft": d,
        "days_active": df["days_active"].fillna(df["days_active"].median()),
        "depth_shallow": (d < 6000).astype(int),
        "depth_mid": ((d >= 6000) & (d < 10000)).astype(int),
    })

    if include_rate and "avg_rate_365d" in df.columns:
        X["avg_rate_365d"] = df["avg_rate_365d"].fillna(df["avg_rate_365d"].median())

    return X


def fit_mediation(df: pd.DataFrame, X: pd.DataFrame, cluster_col: str = COL_CLUSTER_WELL,
                  with_interaction: bool = False) -> dict:
    """Fit 3-equation mediation (total, mediator-on-treatment, direct) with clustered SEs.

    If with_interaction=True, adds W*M term to the direct equation to test
    the sequential ignorability assumption.
    """
    W = df[COL_TREATMENT].values
    M = df[COL_MEDIATOR].values
    Y = df[COL_OUTCOME].values
    L = X.values
    clusters = df[cluster_col].values

    W_mean, W_std = W.mean(), max(W.std(), 1.0)
    W_z = (W - W_mean) / W_std

    # Total
    X1 = sm.add_constant(np.column_stack([W_z, L]))
    m1 = sm.OLS(Y, X1).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    c_z = m1.params[1]
    c = c_z / W_std
    c_se = m1.bse[1] / W_std

    # Path a: M ~ W + L
    Xa = sm.add_constant(np.column_stack([W_z, L]))
    ma = sm.OLS(M, Xa).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    a_z = ma.params[1]
    a = a_z / W_std

    # Direct: Y ~ W + M + L [+ W*M]
    if with_interaction:
        WM = W_z * M
        Xd = sm.add_constant(np.column_stack([W_z, M, WM, L]))
        md = sm.OLS(Y, Xd).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        c_prime_z = md.params[1]
        b = md.params[2]
        interaction = md.params[3]
        interaction_se = md.bse[3]
        interaction_pval = md.pvalues[3]
    else:
        Xd = sm.add_constant(np.column_stack([W_z, M, L]))
        md = sm.OLS(Y, Xd).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        c_prime_z = md.params[1]
        b = md.params[2]
        interaction = None
        interaction_se = None
        interaction_pval = None

    c_prime = c_prime_z / W_std
    c_prime_se = md.bse[1] / W_std

    return {
        "c": c, "c_se": c_se,
        "c_prime": c_prime, "c_prime_se": c_prime_se,
        "a": a, "b": b,
        "indirect_ab": a * b,
        "indirect_diff": c - c_prime,
        "pct_mediated": 100 * (a * b) / c if abs(c) > 1e-18 else np.nan,
        "interaction": interaction,
        "interaction_se": interaction_se,
        "interaction_pval": interaction_pval,
        "n": len(df),
        "n_clusters": pd.Series(clusters).nunique(),
    }


def simulate_h_trimming(df: pd.DataFrame, radius_km: int, trim_pct: float = 0.01) -> dict:
    """Simulate the impact of trimming extreme treatment values.

    In the current TMLE, H = g(a-δ)/g(a) can blow up when g(a|l) is tiny at the
    tails of the treatment distribution. We don't have the actual g model here,
    so we approximate by: flag rows in the top 1% of the treatment distribution
    and the bottom 1% of the Volume Injected distribution.

    Refit the OLS mediation with trimming, compare estimates.
    """
    W = df[COL_TREATMENT]
    p_low = W.quantile(trim_pct)
    p_high = W.quantile(1 - trim_pct)

    trimmed = df[(W >= p_low) & (W <= p_high)].copy()
    X_full = build_X(df, radius_km)
    X_trim = build_X(trimmed, radius_km)

    full = fit_mediation(df, X_full)
    trim_res = fit_mediation(trimmed, X_trim)

    return {
        "full": full,
        "trimmed": trim_res,
        "pct_trimmed": 100 * (len(df) - len(trimmed)) / len(df),
        "trim_low": p_low,
        "trim_high": p_high,
    }


def run_all_for_radius(radius_km: int) -> dict:
    print(f"\n{'='*70}")
    print(f"RADIUS = {radius_km} km")
    print(f"{'='*70}")

    df = load_panel(radius_km)
    if df is None:
        print(f"  No panel data")
        return None

    print(f"  n = {len(df):,}  unique wells = {df[COL_CLUSTER_WELL].nunique():,}")

    # Baseline (current pipeline)
    X_baseline = build_X(df, radius_km)
    baseline = fit_mediation(df, X_baseline, cluster_col=COL_CLUSTER_WELL)

    # Improvement 1: Add injection rate
    df_with_rate = add_injection_rate(df)
    X_with_rate = build_X(df_with_rate, radius_km, include_rate=True)
    with_rate = fit_mediation(df_with_rate, X_with_rate, cluster_col=COL_CLUSTER_WELL)

    # Improvement 2: Cluster by event
    df_with_clusters = add_event_cluster_id(df)
    by_event = fit_mediation(df_with_clusters, X_baseline, cluster_col="event_cluster")

    # Improvement 3: H-trimming (approximated as treatment trimming)
    trim_result = simulate_h_trimming(df, radius_km, trim_pct=0.01)

    # Improvement 4: W × M interaction test
    with_interaction = fit_mediation(df, X_baseline, cluster_col=COL_CLUSTER_WELL,
                                      with_interaction=True)

    # Combined: all four improvements
    df_combined = add_injection_rate(df)
    df_combined = add_event_cluster_id(df_combined)
    W = df_combined[COL_TREATMENT]
    p_low = W.quantile(0.01); p_high = W.quantile(0.99)
    df_combined = df_combined[(W >= p_low) & (W <= p_high)].copy()
    X_combined = build_X(df_combined, radius_km, include_rate=True)
    combined = fit_mediation(df_combined, X_combined, cluster_col="event_cluster",
                              with_interaction=True)

    # Print comparison
    def fmt(v, e=None):
        if v is None or np.isnan(v): return "—"
        s = f"{v:+.3e}"
        if e is not None and not np.isnan(e):
            s += f" (±{e:.2e})"
        return s

    print(f"\n  MEDIATION ESTIMATES (Total / Direct / Indirect / % mediated):")
    print(f"  {'':<30}  {'Total c':>18}  {'Direct c′':>18}  {'Indirect a·b':>15}  {'% med':>8}")
    for label, r in [
        ("1. Baseline (current)", baseline),
        ("2. + injection rate", with_rate),
        ("3. + cluster by event", by_event),
        ("4. + trim 1% tails", trim_result["trimmed"]),
        ("5. ALL improvements", combined),
    ]:
        pct = f"{r['pct_mediated']:+6.1f}%" if r['pct_mediated'] is not None and not np.isnan(r['pct_mediated']) else "—"
        print(f"  {label:<30}  {fmt(r['c'], r['c_se']):>18}  {fmt(r['c_prime'], r['c_prime_se']):>18}  {fmt(r['indirect_ab']):>15}  {pct:>8}")

    print(f"\n  W×M INTERACTION (tests mediation additivity assumption):")
    print(f"  Baseline interaction β = {fmt(with_interaction['interaction'], with_interaction['interaction_se'])}")
    pval = with_interaction.get('interaction_pval')
    print(f"  p-value = {pval:.3e}" if pval is not None else "  p-value = —")
    if pval is not None and pval < 0.05:
        print(f"  ⚠ SIGNIFICANT W×M interaction — additive NDE+NIE decomposition may be invalid")
    else:
        print(f"  ✓ No significant interaction — mediation decomposition is defensible")

    print(f"\n  H-TRIMMING: dropped {trim_result['pct_trimmed']:.1f}% of rows")
    print(f"    (cum_vol_365d outside [{trim_result['trim_low']:,.0f}, {trim_result['trim_high']:,.0f}] BBL)")

    return {
        "radius_km": radius_km,
        "n": len(df),
        "baseline": baseline,
        "with_rate": with_rate,
        "by_event": by_event,
        "trimmed": trim_result["trimmed"],
        "interaction": with_interaction,
        "combined": combined,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=int, nargs="+", default=[7, 10],
                    help="Radii to test (default: 7 10)")
    args = ap.parse_args()

    print(f"Model Improvements Impact Assessment")
    print(f"Radii: {args.radii}")

    all_results = []
    for r in args.radii:
        res = run_all_for_radius(r)
        if res:
            all_results.append(res)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: Does each improvement meaningfully change estimates?")
    print(f"{'='*70}")

    rows = []
    for res in all_results:
        r = res["radius_km"]
        b = res["baseline"]
        for name, alt in [
            ("inj_rate", res["with_rate"]),
            ("cluster_event", res["by_event"]),
            ("trim_1pct", res["trimmed"]),
            ("all_combined", res["combined"]),
        ]:
            d_c = 100 * (alt["c"] - b["c"]) / abs(b["c"]) if abs(b["c"]) > 1e-18 else np.nan
            d_c_se = 100 * (alt["c_se"] - b["c_se"]) / b["c_se"] if b["c_se"] > 0 else np.nan
            d_ind = 100 * (alt["indirect_ab"] - b["indirect_ab"]) / abs(b["indirect_ab"]) if abs(b["indirect_ab"]) > 1e-18 else np.nan
            rows.append({
                "radius_km": r,
                "improvement": name,
                "delta_c_pct": d_c,
                "delta_c_se_pct": d_c_se,
                "delta_indirect_pct": d_ind,
                "new_c": alt["c"],
                "new_c_se": alt["c_se"],
            })

    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))

    out_path = REPO / "model_improvements_summary.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    main()
