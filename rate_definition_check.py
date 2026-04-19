#!/usr/bin/env python3
"""
rate_definition_check.py — Compare rate confounder definitions for multicollinearity.

Tests three definitions:
  A. avg_rate_365d = cum_vol_365d / clip(days_active, 1, 365)  [CURRENT — collinear with treatment]
  B. avg_rate_90d  = cum_vol_90d / 90                          [PROPOSED — different window]
  C. recent_to_annual_ratio = (cum_vol_90d/90) / (cum_vol_365d/365)  [orthogonal by construction]

For each:
  - correlation with treatment (cum_vol_365d) — lower is better for identification
  - VIF when included alongside treatment in a regression — < 10 is OK
  - Impact on OLS total effect estimate at 7km

Then runs the OLS mediation with whichever definition has the cleanest properties.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import column_maps as cm

COL_TREATMENT = "cum_vol_365d_BBL"
COL_OUTCOME = "outcome_max_ML"
COL_MEDIATOR = "bhp_vw_avg_365d"
COL_CLUSTER = "API Number"


def load_panel(radius_km: int) -> pd.DataFrame:
    path = REPO / f"panel_with_faults_{radius_km}km.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=[COL_TREATMENT, COL_OUTCOME, COL_MEDIATOR, COL_CLUSTER])
    return df


def add_rate_definitions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the three candidate rate confounders."""
    df = df.copy()
    days = df["days_active"].fillna(365).clip(lower=1, upper=365)
    df["rate_365d"] = df[COL_TREATMENT] / days

    if "cum_vol_90d_BBL" in df.columns:
        df["rate_90d"] = df["cum_vol_90d_BBL"].fillna(0) / 90

        # Annual avg rate = cum_vol_365d / 365 (NOT clipped — divisor is constant)
        annual_avg = df[COL_TREATMENT] / 365
        # Avoid division by zero
        df["recent_to_annual"] = np.where(
            annual_avg > 0,
            df["rate_90d"] / annual_avg,
            1.0
        )
        # Cap at reasonable values (well that's been off and just turned on can have huge ratios)
        df["recent_to_annual"] = df["recent_to_annual"].clip(0, 10)
    else:
        df["rate_90d"] = np.nan
        df["recent_to_annual"] = np.nan
    return df


def diagnose_collinearity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation and VIF for each rate definition vs treatment."""
    print("\n" + "=" * 70)
    print("MULTICOLLINEARITY DIAGNOSIS")
    print("=" * 70)
    print(f"  n = {len(df):,}")

    cols = ["rate_365d", "rate_90d", "recent_to_annual"]
    rows = []
    for col in cols:
        if col not in df.columns or df[col].isna().all():
            continue
        ok = df[[COL_TREATMENT, col]].dropna()
        # Pearson correlation with treatment
        corr = ok.corr().iloc[0, 1]

        # Spearman (rank) correlation
        sp = ok.corr(method="spearman").iloc[0, 1]

        # VIF when included alongside treatment + a few other predictors
        try:
            X = df[[COL_TREATMENT, col, "perf_depth_ft", "days_active",
                    "Nearest Fault Dist (km)"]].dropna()
            X_arr = sm.add_constant(X.values)
            vifs = [variance_inflation_factor(X_arr, i) for i in range(1, X_arr.shape[1])]
            vif_treatment = vifs[0]
            vif_rate = vifs[1]
        except Exception as e:
            vif_treatment = np.nan
            vif_rate = np.nan

        rows.append({
            "rate_def": col,
            "corr_with_treatment": corr,
            "spearman_with_treatment": sp,
            "vif_treatment": vif_treatment,
            "vif_rate": vif_rate,
            "verdict": "OK" if abs(corr) < 0.7 and vif_treatment < 10 and vif_rate < 10 else "COLLINEAR",
        })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    return df_out


def fit_total_effect(df: pd.DataFrame, rate_col: str, radius_km: int) -> dict:
    """Fit OLS total effect: Y ~ W + L. Optionally include rate_col as confounder."""
    fault_seg_col = cm.fault_segment_col(radius_km)
    if fault_seg_col not in df.columns:
        candidates = [c for c in df.columns if "fault segments" in c.lower()]
        fault_seg_col = candidates[0] if candidates else None

    cols = [COL_TREATMENT, "perf_depth_ft", "days_active",
            "Nearest Fault Dist (km)", fault_seg_col]
    if rate_col is not None:
        cols.append(rate_col)
    cols = [c for c in cols if c is not None and c in df.columns]

    sub = df[cols + [COL_OUTCOME, COL_CLUSTER]].dropna()
    W = sub[COL_TREATMENT].values
    Y = sub[COL_OUTCOME].values
    clusters = sub[COL_CLUSTER].values
    L_cols = [c for c in cols if c != COL_TREATMENT]
    L = sub[L_cols].values

    W_std = max(W.std(), 1.0)
    W_z = (W - W.mean()) / W_std
    X = sm.add_constant(np.column_stack([W_z, L]))
    m = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    c = m.params[1] / W_std
    c_se = m.bse[1] / W_std
    return {
        "rate_def": rate_col or "none",
        "n": len(sub),
        "c": c,
        "c_se": c_se,
        "c_t": c / c_se if c_se > 0 else np.nan,
    }


def main():
    for radius_km in [7, 10]:
        print("\n" + "#" * 70)
        print(f"# RADIUS {radius_km} KM")
        print("#" * 70)

        df = load_panel(radius_km)
        df = add_rate_definitions(df)

        # Step 1: diagnose collinearity for each rate definition
        diag = diagnose_collinearity(df)

        # Step 2: fit OLS total effect with each rate definition
        print("\n" + "=" * 70)
        print("OLS TOTAL EFFECT WITH EACH RATE DEFINITION")
        print("=" * 70)
        results = []
        for rate_col in [None, "rate_365d", "rate_90d", "recent_to_annual"]:
            if rate_col is not None and (rate_col not in df.columns or df[rate_col].isna().all()):
                continue
            r = fit_total_effect(df, rate_col, radius_km)
            results.append(r)
        out = pd.DataFrame(results)
        # Compute % change from baseline (no rate)
        baseline = out.iloc[0]["c"]
        out["delta_pct"] = (out["c"] - baseline) / abs(baseline) * 100
        out["delta_pct"] = out["delta_pct"].apply(lambda x: f"{x:+.1f}%" if not np.isnan(x) else "—")
        out["c"] = out["c"].apply(lambda x: f"{x:+.3e}")
        out["c_se"] = out["c_se"].apply(lambda x: f"{x:.2e}")
        out["c_t"] = out["c_t"].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "—")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
