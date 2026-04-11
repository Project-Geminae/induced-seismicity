#!/usr/bin/env python3
"""
evalue_sensitivity.py
─────────────────────
E-value sensitivity analysis layered on top of the TMLE point estimates.

For an observational study, the E-value is the minimum strength of association
on the risk-ratio scale that an unmeasured confounder would need to have
with BOTH the treatment AND the outcome (above and beyond the measured
confounders) to fully explain away an observed effect estimate. It's the
clean, single-number quantitative answer to the question "how robust is
this finding to unmeasured confounding?"

References:
  - VanderWeele & Ding (2017), Sensitivity Analysis in Observational
    Research: Introducing the E-Value. Annals of Internal Medicine.
  - https://www.evalue-calculator.com/

For continuous outcomes (which is our case — Y = max ML), VanderWeele &
Ding's adaptation gives:

    RR_obs ≈ exp(0.91 · d)

where d = (effect / SD_outcome) is the standardized mean difference. The
E-value is then:

    E = RR_obs + sqrt(RR_obs · (RR_obs - 1))

A second E-value is computed for the lower confidence limit (the relevant
quantity for asking "how strong would a confounder need to be to push the
CI to include the null"):

    RR_lower = exp(0.91 · d_lower)
    E_lower  = RR_lower + sqrt(RR_lower · (RR_lower - 1))   if RR_lower > 1
             = 1                                              if RR_lower ≤ 1

This script reads the latest TMLE result CSVs and writes
`evalue_sensitivity.csv` with one row per (driver, radius) combination.

Outputs:
    evalue_sensitivity.csv      tabular results
    plots/evalue_vs_radius.png  E-values plotted against radius
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)


def latest(glob: str) -> Path | None:
    matches = sorted(Path(".").glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def evalue_continuous(effect: float, sd_outcome: float) -> float:
    """E-value for a continuous outcome (VanderWeele & Ding 2017).

    Standardized mean difference d = effect / sd_outcome
    Approximate risk-ratio analog: RR ≈ exp(0.91 · d)
    E-value = RR + sqrt(RR · (RR - 1))
    """
    if sd_outcome <= 0 or not np.isfinite(effect):
        return float("nan")
    d = abs(effect / sd_outcome)
    rr = float(np.exp(0.91 * d))
    if rr <= 1.0:
        return 1.0
    return rr + float(np.sqrt(rr * (rr - 1.0)))


def evalue_for_lower_ci(ci_low: float, sd_outcome: float) -> float:
    """E-value for the lower confidence limit (the more conservative number).

    If the CI already includes 0 (or the wrong sign), the lower-CI E-value
    is 1 — meaning even a trivial unmeasured confounder could explain away
    the lower bound, since the bound is consistent with no effect.
    """
    if not np.isfinite(ci_low):
        return float("nan")
    if ci_low <= 0:
        return 1.0
    return evalue_continuous(ci_low, sd_outcome)


def get_outcome_sd_per_radius(panel_glob: str = "panel_with_faults_{R}km.csv") -> dict[int, float]:
    """Compute the SD of outcome_max_ML at the cluster-day level for each radius.

    Used as the denominator for the standardized mean difference. Falls back
    to a global SD across all panels if a per-radius file isn't available.
    """
    sds = {}
    for R in range(1, 21):
        path = Path(panel_glob.format(R=R))
        if not path.exists():
            continue
        # Sample to keep memory low
        df = pd.read_csv(path, usecols=["outcome_max_ML"])
        sds[R] = float(df["outcome_max_ML"].std())
    return sds


def evalue_table_from_shift(shift_df: pd.DataFrame, sds: dict[int, float]) -> pd.DataFrame:
    rows = []
    for _, r in shift_df.iterrows():
        R = int(r["radius_km"])
        sd = sds.get(R, float("nan"))
        rows.append({
            "driver":     "shift_+10pct",
            "radius_km":  R,
            "estimate":   float(r["psi"]),
            "ci_low":     float(r["ci_low"]),
            "ci_high":    float(r["ci_high"]),
            "sd_outcome": sd,
            "evalue":         evalue_continuous(r["psi"], sd),
            "evalue_ci_low":  evalue_for_lower_ci(r["ci_low"], sd),
        })
    return pd.DataFrame(rows)


def evalue_table_from_mediation(med_df: pd.DataFrame, sds: dict[int, float]) -> pd.DataFrame:
    rows = []
    for _, r in med_df.iterrows():
        R = int(r["radius_km"])
        sd = sds.get(R, float("nan"))
        rows.append({
            "driver":     "TE_p90_vs_p10",
            "radius_km":  R,
            "estimate":   float(r["TE"]),
            "ci_low":     float(r["TE_ci_low"]),
            "ci_high":    float(r["TE_ci_high"]),
            "sd_outcome": sd,
            "evalue":         evalue_continuous(r["TE"], sd),
            "evalue_ci_low":  evalue_for_lower_ci(r["TE_ci_low"], sd),
        })
    return pd.DataFrame(rows)


def evalue_table_from_dose(dose_df: pd.DataFrame, sds: dict[int, float],
                            target_a: float = 1e7) -> pd.DataFrame:
    """E-values for the dose-response point estimate at a single grid point."""
    sub = dose_df[np.isclose(dose_df["a_star"], target_a)].copy()
    rows = []
    for _, r in sub.iterrows():
        R = int(r["radius_km"])
        sd = sds.get(R, float("nan"))
        rows.append({
            "driver":     f"dose_at_{int(np.log10(target_a))}",
            "radius_km":  R,
            "estimate":   float(r["psi"]),
            "ci_low":     float(r["ci_low"]),
            "ci_high":    float(r["ci_high"]),
            "sd_outcome": sd,
            "evalue":         evalue_continuous(r["psi"], sd),
            "evalue_ci_low":  evalue_for_lower_ci(r["ci_low"], sd),
        })
    return pd.DataFrame(rows)


def plot_evalues(combined: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    palette = {
        "shift_+10pct":   "#9467bd",
        "TE_p90_vs_p10":  "#d62728",
        "dose_at_7":      "#1f77b4",
    }
    for driver, color in palette.items():
        sub = combined[combined["driver"] == driver].sort_values("radius_km")
        if sub.empty:
            continue
        ax.plot(sub["radius_km"], sub["evalue"], "o-", lw=2.5, ms=6, color=color,
                label=f"{driver} (point estimate)")
        ax.plot(sub["radius_km"], sub["evalue_ci_low"], "s--", lw=1.5, ms=4,
                color=color, alpha=0.6,
                label=f"{driver} (lower CI)")
    ax.axhline(1, color="black", lw=0.6, ls=":", label="E = 1 (no robustness)")
    ax.axhline(2, color="grey", lw=0.6, ls=":", alpha=0.6,
               label="E = 2 (modest robustness)")
    ax.set_yscale("log")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("E-value")
    ax.set_title("VanderWeele–Ding E-values for TMLE estimates\n"
                 "(higher = more robust to unmeasured confounding)")
    ax.set_xticks(sorted(combined["radius_km"].unique()))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    out = PLOT_DIR / "evalue_vs_radius.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✅  {out}")


def main() -> None:
    shift_path = latest("tmle_shift_*.csv")
    med_path   = latest("tmle_mediation_*.csv")
    dose_path  = latest("tmle_dose_response_*.csv")
    if not (shift_path and med_path and dose_path):
        sys.exit("❌  Need tmle_shift_*.csv, tmle_mediation_*.csv, tmle_dose_response_*.csv")

    print(f"📄  shift:     {shift_path.name}")
    print(f"📄  mediation: {med_path.name}")
    print(f"📄  dose:      {dose_path.name}")

    print("📊  Computing per-radius outcome SDs from panel files…")
    sds = get_outcome_sd_per_radius()
    if not sds:
        sys.exit("❌  No panel_with_faults_*.csv files found — need them to compute SD")
    print(f"   {len(sds)} radii available, SD range: {min(sds.values()):.3f}–{max(sds.values()):.3f}")

    shift = pd.read_csv(shift_path)
    med   = pd.read_csv(med_path)
    dose  = pd.read_csv(dose_path)

    e_shift = evalue_table_from_shift(shift, sds)
    e_med   = evalue_table_from_mediation(med, sds)
    e_dose  = evalue_table_from_dose(dose, sds, target_a=1e7)

    combined = pd.concat([e_shift, e_med, e_dose], ignore_index=True)
    combined.to_csv("evalue_sensitivity.csv", index=False)
    print(f"\n✅  Wrote evalue_sensitivity.csv ({len(combined)} rows)")

    print("\n🎯  E-VALUES BY DRIVER AND RADIUS")
    print("─" * 80)
    print(f"{'Driver':<18} {'Radius':>7} {'Estimate':>12} {'E-value':>9} {'E (lower CI)':>13}")
    print("─" * 80)
    for _, r in combined.sort_values(["driver", "radius_km"]).iterrows():
        print(f"{r['driver']:<18} {int(r['radius_km']):>4}km "
              f"{r['estimate']:>+12.3e} {r['evalue']:>9.2f} {r['evalue_ci_low']:>13.2f}")

    plot_evalues(combined)
    print("\nDone.")


if __name__ == "__main__":
    main()
