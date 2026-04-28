#!/usr/bin/env python3
"""Cross-method comparison of the shift-intervention ψ at radius 7.

Benchmarks undersmoothed HAL (still running) against:
  1. OLS: linear regression of Y on W + L_confounders. ψ_OLS = β_W × 0.1 × mean(W).
  2. GBM plug-in: XGBoost on Y vs (W, L), ψ = mean[Q(1.1W, L) - Q(W, L)].
  3. Standard TMLE v3: pulled from existing CSV.
  4. CV-TMLE (subsampled): pulled from OG run result.

Outputs a comparison table — if HAL agrees with OLS/GBM in
order-of-magnitude, the subsampled estimators were the outliers.
"""
import glob
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb

import causal_core as cc


R = 7
print(f"Benchmark comparison at R={R} km, shift_pct=0.10\n")

# Load panel
panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
print(f"Panel: n={len(data)}, n_clusters={data[cluster.name].nunique() if hasattr(cluster, 'name') else cluster.nunique()}")

A = data[W].to_numpy(dtype=float)
L = data[confs].to_numpy(dtype=float)
Y = data[S].to_numpy(dtype=float)
AL = np.column_stack([A, L])
A_post = A * 1.10
AL_post = np.column_stack([A_post, L])

results = []

# ─── 1. OLS ───────────────────────────────────────────────────────────
t0 = time.time()
ols = LinearRegression().fit(AL, Y)
psi_ols = float(np.mean(ols.predict(AL_post) - ols.predict(AL)))
elapsed = time.time() - t0
results.append(("OLS (plug-in)", psi_ols, elapsed))
print(f"  OLS:       ψ = {psi_ols:+.4e}   ({elapsed:.1f}s)")

# ─── 2. XGBoost plug-in ──────────────────────────────────────────────
t0 = time.time()
gbm = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                       tree_method="hist", verbosity=0, random_state=42)
gbm.fit(AL, Y)
psi_gbm = float(np.mean(gbm.predict(AL_post) - gbm.predict(AL)))
elapsed = time.time() - t0
results.append(("XGBoost (plug-in)", psi_gbm, elapsed))
print(f"  XGB:       ψ = {psi_gbm:+.4e}   ({elapsed:.1f}s)")

# ─── 3. Standard TMLE v3 ─────────────────────────────────────────────
# Look for any of the v3 TMLE result files
v3_files = sorted(glob.glob("tmle_shift_365d_*.csv"))
if v3_files:
    v3_df = pd.read_csv(v3_files[-1])
    v3_7km = v3_df[v3_df["radius_km"] == R]
    if not v3_7km.empty:
        psi_v3 = float(v3_7km.iloc[0]["psi"])
        ci_v3 = (float(v3_7km.iloc[0]["ci_low"]), float(v3_7km.iloc[0]["ci_high"]))
        results.append(("Standard TMLE v3 (full panel)", psi_v3, 0))
        print(f"  TMLE v3:   ψ = {psi_v3:+.4e}   CI=[{ci_v3[0]:+.3e}, {ci_v3[1]:+.3e}]")
    else:
        print(f"  TMLE v3:   (no row for R={R} in {v3_files[-1]})")
else:
    print("  TMLE v3:   (no tmle_shift_365d_*.csv found)")

# ─── 4. CV-TMLE subsampled (from OG log) ─────────────────────────────
psi_cv = 2.156e-3  # from OG 7km run
ci_cv = (1.39e-4, 4.17e-3)
results.append(("CV-TMLE + haldensify + HAL (n=50k subsampled)", psi_cv, 0))
print(f"  CV-TMLE:   ψ = {psi_cv:+.4e}   CI=[{ci_cv[0]:+.3e}, {ci_cv[1]:+.3e}]")

# ─── Summary table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON SUMMARY (radius 7 km, 10% shift)")
print("=" * 70)
for name, psi, elapsed in results:
    print(f"  {name:<50}  ψ = {psi:+.4e}")
print("=" * 70)
print("\nUndersmoothed HAL: still running (check hal_validate_7km*.log)")

# Save
pd.DataFrame(
    [{"method": n, "psi": p, "elapsed_sec": e} for n, p, e in results]
).to_csv(f"benchmark_shift_{R}km.csv", index=False)
print(f"\nWrote benchmark_shift_{R}km.csv")
