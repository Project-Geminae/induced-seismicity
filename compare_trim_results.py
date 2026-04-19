#!/usr/bin/env python3
"""
compare_trim_results.py — Compare TMLE shift + dose-response with vs without
1% tail trimming.

Inputs:
  Baseline (no trim, current production):
    - tmle_shift_365d_<latest>.csv
    - tmle_dose_response_365d_<latest>.csv
  Trimmed:
    - tmle_shift_365d_trim01.csv
    - tmle_dose_response_365d_trim01.csv

Reports per-radius comparison: psi, CI width, max H.
"""
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent


def latest(pattern: str) -> Path:
    files = [p for p in REPO.glob(pattern)
             if "trim01" not in p.name]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def compare_shift():
    base_path = latest("tmle_shift_365d_*.csv")
    trim_path = REPO / "tmle_shift_365d_trim01.csv"
    if not (base_path and trim_path.exists()):
        print(f"Missing inputs: base={base_path}, trim={trim_path}")
        return
    print(f"Shift baseline: {base_path.name}")
    print(f"Shift trimmed:  {trim_path.name}")

    base = pd.read_csv(base_path)
    trim = pd.read_csv(trim_path)
    merged = base.merge(trim, on="radius_km", suffixes=("_base", "_trim"))

    print(f"\n{'R':>3}  {'psi base':>12}  {'psi trim':>12}  {'ΔΨ %':>8}  "
          f"{'CI base':>10}  {'CI trim':>10}  {'maxH base':>10}  {'maxH trim':>10}")
    for _, r in merged.iterrows():
        ci_base = r["ci_high_base"] - r["ci_low_base"]
        ci_trim = r["ci_high_trim"] - r["ci_low_trim"]
        d_pct = 100 * (r["psi_trim"] - r["psi_base"]) / abs(r["psi_base"]) if abs(r["psi_base"]) > 1e-15 else float("nan")
        print(f"{int(r['radius_km']):>3}  {r['psi_base']:>+12.4e}  {r['psi_trim']:>+12.4e}  "
              f"{d_pct:>+7.1f}%  {ci_base:>10.4e}  {ci_trim:>10.4e}  "
              f"{r['max_H_base']:>10.1f}  {r['max_H_trim']:>10.1f}")

    # Summary
    print(f"\nSummary across all radii:")
    print(f"  Mean |Δψ|:           {merged.apply(lambda r: 100*abs(r['psi_trim']-r['psi_base'])/abs(r['psi_base']) if abs(r['psi_base'])>1e-15 else 0, axis=1).mean():.1f}%")
    print(f"  Max |Δψ|:            {merged.apply(lambda r: 100*abs(r['psi_trim']-r['psi_base'])/abs(r['psi_base']) if abs(r['psi_base'])>1e-15 else 0, axis=1).max():.1f}%")
    print(f"  Mean max H baseline: {merged['max_H_base'].mean():.0f}")
    print(f"  Mean max H trimmed:  {merged['max_H_trim'].mean():.0f}")


def compare_dose():
    base_path = latest("tmle_dose_response_365d_*.csv")
    trim_path = REPO / "tmle_dose_response_365d_trim01.csv"
    if not (base_path and trim_path.exists()):
        print(f"Missing inputs: base={base_path}, trim={trim_path}")
        return
    print(f"\nDose-response baseline: {base_path.name}")
    print(f"Dose-response trimmed:  {trim_path.name}")

    base = pd.read_csv(base_path)
    trim = pd.read_csv(trim_path)

    # Compare at the 7 km and 10 km radii at all grid points
    for R in [7, 10]:
        b = base[base["radius_km"] == R].sort_values("a_star")
        t = trim[trim["radius_km"] == R].sort_values("a_star")
        if len(b) == 0 or len(t) == 0:
            print(f"\nR={R}km: no data")
            continue
        m = b.merge(t, on=["radius_km", "a_star"], suffixes=("_base", "_trim"))
        print(f"\nR={R}km:")
        print(f"  {'a_star':>12}  {'psi base':>12}  {'psi trim':>12}  {'Δ %':>8}  {'n_in_bin':>10}")
        for _, r in m.iterrows():
            d_pct = 100 * (r["psi_trim"] - r["psi_base"]) / abs(r["psi_base"]) if abs(r["psi_base"]) > 1e-15 else float("nan")
            print(f"  {r['a_star']:>12,.0f}  {r['psi_base']:>+12.4e}  {r['psi_trim']:>+12.4e}  "
                  f"{d_pct:>+7.1f}%  {int(r.get('n_in_bin_trim', 0)):>10}")


if __name__ == "__main__":
    compare_shift()
    compare_dose()
