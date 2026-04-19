#!/usr/bin/env python3
"""
positivity_diagnostics.py — Report positivity/overlap for the TMLE dose-response.

For each radius and grid point:
  - Empirical percentile of the grid point in the observed treatment distribution
  - Fraction of observations with cum_vol_365d >= grid point
  - Clever covariate H statistics from the TMLE run (already in output CSV)
  - Flag: "SUPPORT" if grid point is within P95 of observed, "EXTRAPOLATION" if beyond P99

Usage:
    python positivity_diagnostics.py

Outputs:
    positivity_diagnostics.csv — one row per (radius, grid point)
"""
import argparse
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent


def latest(pattern: str) -> Path:
    files = sorted(REPO.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_panel_treatment_distribution() -> dict[int, np.ndarray]:
    """Load cum_vol_365d for each radius's event-well panel."""
    panels = {}
    for radius_km in range(1, 21):
        path = REPO / f"panel_with_faults_{radius_km}km.csv"
        if not path.exists():
            continue
        # Read only the treatment column and outcome presence
        df = pd.read_csv(path, usecols=["cum_vol_365d_BBL", "outcome_max_ML"], low_memory=False)
        # Restrict to rows with positive treatment (support for log-scale inference)
        vals = df["cum_vol_365d_BBL"].dropna()
        vals = vals[vals > 0]
        panels[radius_km] = vals.values
    return panels


def analyze_dose_response(dose_csv: Path, panels: dict[int, np.ndarray]) -> pd.DataFrame:
    """For each (radius, a_star) grid point in the dose-response output, report positivity."""
    dr = pd.read_csv(dose_csv)
    rows = []
    for _, r in dr.iterrows():
        radius = int(r["radius_km"])
        a_star = float(r["a_star"])
        panel_vals = panels.get(radius, np.array([]))

        if len(panel_vals) == 0:
            pct = None
            frac_ge = None
            n_above = 0
        else:
            # What percentile is a_star in the observed treatment distribution?
            pct = float((panel_vals < a_star).mean() * 100)
            frac_ge = float((panel_vals >= a_star).mean())
            n_above = int((panel_vals >= a_star).sum())

        # Extrapolation flag
        if pct is None:
            flag = "NO_DATA"
        elif pct <= 95:
            flag = "SUPPORT"
        elif pct <= 99:
            flag = "SPARSE"
        else:
            flag = "EXTRAPOLATION"

        rows.append({
            "radius_km": radius,
            "a_star": a_star,
            "psi": r.get("psi", np.nan),
            "ci_low": r.get("ci_low", np.nan),
            "ci_high": r.get("ci_high", np.nan),
            "n_in_bin": r.get("n_in_bin", np.nan),
            "panel_n": len(panel_vals),
            "pct_of_observed": pct,
            "frac_wells_at_or_above": frac_ge,
            "n_wells_at_or_above": n_above,
            "flag": flag,
        })

    return pd.DataFrame(rows)


def analyze_shift(shift_csv: Path, panels: dict[int, np.ndarray]) -> pd.DataFrame:
    """Positivity diagnostics for the shift intervention (clever covariate H)."""
    sh = pd.read_csv(shift_csv)
    rows = []
    for _, r in sh.iterrows():
        radius = int(r["radius_km"])
        mean_H = float(r.get("mean_H", np.nan))
        max_H = float(r.get("max_H", np.nan))

        # Clever covariate should be close to 1; large H indicates extrapolation
        if np.isnan(max_H):
            h_flag = "NO_H_REPORTED"
        elif max_H < 5:
            h_flag = "STABLE"
        elif max_H < 20:
            h_flag = "BORDERLINE"
        else:
            h_flag = "UNSTABLE"

        panel_vals = panels.get(radius, np.array([]))
        rows.append({
            "radius_km": radius,
            "shift_pct": r.get("shift_pct", np.nan),
            "psi": r.get("psi", np.nan),
            "se_cluster": r.get("se_cluster", np.nan),
            "mean_H": mean_H,
            "max_H": max_H,
            "h_flag": h_flag,
            "panel_n": len(panel_vals),
            "obs_p95_BBL": float(np.percentile(panel_vals, 95)) if len(panel_vals) else None,
            "obs_p99_BBL": float(np.percentile(panel_vals, 99)) if len(panel_vals) else None,
            "obs_max_BBL": float(panel_vals.max()) if len(panel_vals) else None,
        })
    return pd.DataFrame(rows)


def main():
    print(f"Positivity diagnostics · {datetime.now().isoformat()}")

    dose_csv = latest("tmle_dose_response_365d_*.csv")
    shift_csv = latest("tmle_shift_365d_*.csv")

    if not dose_csv:
        print("No dose-response CSV found. Run tmle_dose_response.py first.")
        return

    print(f"Dose-response: {dose_csv.name}")
    print(f"Shift:         {shift_csv.name if shift_csv else 'none'}")

    print("Loading panel treatment distributions...")
    panels = load_panel_treatment_distribution()
    print(f"  Loaded {len(panels)} radii")
    for r in sorted(panels.keys()):
        vals = panels[r]
        print(f"  R={r:2d} km: n={len(vals):>7,}  "
              f"p50={np.percentile(vals,50):>10,.0f}  "
              f"p95={np.percentile(vals,95):>10,.0f}  "
              f"p99={np.percentile(vals,99):>10,.0f}  "
              f"max={vals.max():>10,.0f}")

    # Dose-response analysis
    dr_out = analyze_dose_response(dose_csv, panels)
    dr_out_path = REPO / "positivity_dose_response.csv"
    dr_out.to_csv(dr_out_path, index=False)
    print(f"\nDose-response positivity → {dr_out_path.name}")

    # Summary by radius
    print("\nDose-response extrapolation summary:")
    print(f"{'R':>4}  {'a_star':>12}  {'psi':>10}  {'pct obs':>8}  {'n ≥':>8}  {'flag':>14}")
    for _, r in dr_out.iterrows():
        pct = r['pct_of_observed']
        pct_str = f"{pct:5.1f}%" if pct is not None else "—"
        n_str = f"{r['n_wells_at_or_above']:,}" if r['n_wells_at_or_above'] else "0"
        print(f"{int(r['radius_km']):>4}  {r['a_star']:>12,.0f}  "
              f"{r['psi']:>10.4f}  {pct_str:>8}  {n_str:>8}  {r['flag']:>14}")

    # Shift analysis
    if shift_csv:
        sh_out = analyze_shift(shift_csv, panels)
        sh_out_path = REPO / "positivity_shift.csv"
        sh_out.to_csv(sh_out_path, index=False)
        print(f"\nShift positivity → {sh_out_path.name}")

        print("\nShift H-statistic summary:")
        print(f"{'R':>4}  {'shift':>6}  {'mean H':>8}  {'max H':>8}  {'flag':>12}")
        for _, r in sh_out.iterrows():
            print(f"{int(r['radius_km']):>4}  {r['shift_pct']:>6.2f}  "
                  f"{r['mean_H']:>8.2f}  {r['max_H']:>8.2f}  {r['h_flag']:>12}")


if __name__ == "__main__":
    main()
