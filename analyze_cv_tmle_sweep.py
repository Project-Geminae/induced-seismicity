#!/usr/bin/env python3
"""Analyze the CV-TMLE seed=42 sweep with multiplicity correction.

Produces:
  - Summary table with raw, Bonferroni, and BH-FDR-adjusted p-values
  - Combined test: mean psi over 7-20 km band (where shift effect is
    hypothesized to be present) with cluster-IF SE via simple average
    of per-radius IFs under independence assumption
  - Combined test: unweighted mean psi across all 20 radii
"""
import glob
import numpy as np
import pandas as pd
from scipy import stats


# ─── Load all seed=42 (default) CSVs ─────────────────────────────────
files = sorted(glob.glob("tmle_shift_cvtmle_*km_*.csv"))
# Exclude seed!=42 files (named seed{n}.csv)
files = [f for f in files if "seed" not in f.split("_")[-1].replace(".csv", "")
         or "seed42" in f]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates("radius_km", keep="first").sort_values("radius_km").reset_index(drop=True)

n_tests = len(df)
alpha = 0.05
print(f"Loaded {n_tests} radii (seed=42)")

# ─── Adjust p-values ──────────────────────────────────────────────────
# Bonferroni
df["pval_bonferroni"] = np.minimum(df["pval"] * n_tests, 1.0)

# Benjamini-Hochberg FDR
pvals = df["pval"].values
order = np.argsort(pvals)
ranks = np.empty_like(order)
ranks[order] = np.arange(1, len(pvals) + 1)
bh_adjusted = pvals * n_tests / ranks
# Enforce monotonicity
bh_sorted = np.sort(bh_adjusted)[::-1]
bh_monotone = np.minimum.accumulate(bh_sorted)[::-1]
df["pval_bh"] = np.empty_like(pvals)
df.loc[order, "pval_bh"] = np.minimum(bh_monotone, 1.0)

# Decisions
df["sig_raw"] = (df["pval"] < alpha).map({True: "✓", False: ""})
df["sig_bonf"] = (df["pval_bonferroni"] < alpha).map({True: "✓", False: ""})
df["sig_bh"] = (df["pval_bh"] < alpha).map({True: "✓", False: ""})

# ─── Print summary table ──────────────────────────────────────────────
print("\n" + "=" * 100)
print("MULTIPLICITY-CORRECTED SIGNIFICANCE (α = 0.05)")
print("=" * 100)
show = df[["radius_km", "psi", "ci_low", "ci_high",
           "pval", "pval_bonferroni", "pval_bh",
           "sig_raw", "sig_bonf", "sig_bh"]].copy()
show.columns = ["R", "psi", "CI lo", "CI hi", "p", "p_bonf", "p_bh", "raw", "bonf", "bh"]
print(show.to_string(index=False, float_format=lambda x: f"{x:+.2e}" if abs(x) < 1 else f"{x:.3f}"))

n_raw = (df["pval"] < alpha).sum()
n_bonf = (df["pval_bonferroni"] < alpha).sum()
n_bh = (df["pval_bh"] < alpha).sum()
print(f"\nSignificant at α=0.05:  raw={n_raw}/{n_tests}  Bonferroni={n_bonf}/{n_tests}  BH-FDR={n_bh}/{n_tests}")

# ─── Combined tests ───────────────────────────────────────────────────
print("\n" + "=" * 100)
print("COMBINED TESTS (single hypothesis)")
print("=" * 100)

def combined_test(df_subset, label):
    """Z-test on the weighted mean psi. Weights = 1/var (inverse-variance)."""
    psi = df_subset["psi"].values
    se = df_subset["se_cluster"].values
    var = se ** 2
    w = 1.0 / var
    psi_bar = float(np.sum(w * psi) / np.sum(w))
    se_bar = float(np.sqrt(1.0 / np.sum(w)))
    z = psi_bar / se_bar
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    ci_lo = psi_bar - 1.96 * se_bar
    ci_hi = psi_bar + 1.96 * se_bar
    print(f"\n{label}:")
    print(f"  n_radii      = {len(df_subset)}")
    print(f"  psi_pooled   = {psi_bar:+.3e}")
    print(f"  se_pooled    = {se_bar:.3e}")
    print(f"  95% CI       = [{ci_lo:+.3e}, {ci_hi:+.3e}]")
    print(f"  z            = {z:.2f}")
    print(f"  p            = {p:.2e}")
    print(f"  Significant  = {p < 0.05}")
    return psi_bar, se_bar, ci_lo, ci_hi, p

combined_test(df, "All 20 radii (pooled)")
combined_test(df[(df["radius_km"] >= 7) & (df["radius_km"] <= 19)], "Pressure band (7-19 km)")
combined_test(df[df["radius_km"] <= 6], "Near-field (1-6 km)")

# ─── Save enriched CSV ────────────────────────────────────────────────
out = df[["radius_km", "n", "n_clusters", "psi", "se_cluster",
          "ci_low", "ci_high", "pval", "pval_bonferroni", "pval_bh",
          "mean_H", "max_H", "elapsed_sec"]]
out.to_csv("cv_tmle_sweep_summary.csv", index=False)
print("\nWrote cv_tmle_sweep_summary.csv")
