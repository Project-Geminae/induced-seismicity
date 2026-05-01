"""Inverse-variance-pooled combined test across pressure-band radii (7-19 km)
at full n.

Reads `hurdle_full_n_cv_R<R>km.csv` for R in {7, 8, ..., 19} (13 correlated
radii in the pressure-diffusion band) and computes:

  ψ_pooled = Σ_R w_R · ψ_R / Σ_R w_R         w_R = 1 / SE_cluster_R²
  SE_pooled = sqrt(1 / Σ_R w_R)
  z = ψ_pooled / SE_pooled
  p = 2 · (1 - Φ(|z|))

This is the full-n, full-CV analogue of §5.1's combined test (which
currently uses regHAL-TMLE Delta-method on n=50k subsamples).

Output:
  - hurdle_full_n_combined_test.csv: one-row pooled result with all
    13 per-radius point estimates and SEs.
  - Console summary.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radii", type=int, nargs="+",
                    default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    help="Pressure-band radii to pool (default 7..19).")
    ap.add_argument("--out", type=str, default="hurdle_full_n_combined_test.csv")
    args = ap.parse_args()

    rows = []
    missing = []
    for R in args.radii:
        path = Path(f"hurdle_full_n_cv_R{R}km.csv")
        if not path.exists():
            missing.append(R)
            continue
        d = pd.read_csv(path).iloc[0].to_dict()
        rows.append({
            "radius_km": R,
            "psi_total": float(d["psi_total"]),
            "psi_freq":  float(d["psi_freq"]),
            "psi_mag":   float(d["psi_mag"]),
            "psi_cross": float(d["psi_cross"]),
            "se_cluster_mean": float(d["se_cluster_mean"]),
            "n":            int(d["n"]),
            "n_clusters":   int(d["n_clusters"]),
            "n_active_pos": int(d["n_active_pos"]),
            "n_active_mag": int(d["n_active_mag"]),
            "lambda_pos_cv": float(d["lambda_pos_cv"]),
            "lambda_mag_cv": float(d["lambda_mag_cv"]),
        })

    if missing:
        print(f"⚠ Missing per-radius CSVs for R = {missing}; "
              f"computing pooled estimate from {len(rows)} available radii.")

    if not rows:
        print("✗ No per-radius CSVs available; cannot compute pooled estimate.")
        return

    df = pd.DataFrame(rows).sort_values("radius_km").reset_index(drop=True)

    # Inverse-variance pool on ψ_total
    se = df["se_cluster_mean"].to_numpy()
    psi = df["psi_total"].to_numpy()
    w = 1.0 / np.maximum(se ** 2, 1e-30)
    psi_pooled = float(np.sum(w * psi) / np.sum(w))
    se_pooled = float(np.sqrt(1.0 / np.sum(w)))
    z = psi_pooled / max(se_pooled, 1e-30)
    pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    z95 = 1.959963984540054
    ci_low = psi_pooled - z95 * se_pooled
    ci_high = psi_pooled + z95 * se_pooled

    # Per-radius display
    print(f"=== Full-n hurdle CV combined test (radii {args.radii[0]}–{args.radii[-1]} km) ===\n")
    print(f"{'R(km)':>5}  {'ψ_total':>12}  {'SE_cluster':>12}  {'z':>7}  "
          f"{'active_pos/mag':>14}  {'wall_min':>8}")
    for _, r in df.iterrows():
        z_r = r["psi_total"] / max(r["se_cluster_mean"], 1e-30)
        print(f"{int(r['radius_km']):>5}  {r['psi_total']:>+12.4e}  "
              f"{r['se_cluster_mean']:>12.4e}  {z_r:>+7.2f}  "
              f"{r['n_active_pos']:>5}/{r['n_active_mag']:<5}")

    print(f"\n=== Pooled (inverse variance) ===")
    print(f"  n_radii      = {len(df)}")
    print(f"  ψ_pooled     = {psi_pooled:+.4e}")
    print(f"  SE_pooled    = {se_pooled:.4e}")
    print(f"  CI95         = [{ci_low:+.4e}, {ci_high:+.4e}]")
    print(f"  z            = {z:+.3f}")
    print(f"  p            = {pval:.4e}")

    # Pool channels separately too (informative for the paper)
    for ch in ["psi_freq", "psi_mag", "psi_cross"]:
        # Use the same weights w (from total SE) as a heuristic — channels
        # don't have their own SE in the per-radius CSV. This is an
        # approximate channel pooling; cite as such.
        chv = df[ch].to_numpy()
        ch_pool = float(np.sum(w * chv) / np.sum(w))
        print(f"  {ch:>10} = {ch_pool:+.4e}")

    # Output
    out = {
        "n_radii":     len(df),
        "radii":       ",".join(str(int(r)) for r in df["radius_km"]),
        "psi_pooled":  psi_pooled,
        "se_pooled":   se_pooled,
        "ci_low":      ci_low,
        "ci_high":     ci_high,
        "z":           z,
        "pval":        pval,
        "psi_freq_pooled":  float(np.sum(w * df["psi_freq"])  / np.sum(w)),
        "psi_mag_pooled":   float(np.sum(w * df["psi_mag"])   / np.sum(w)),
        "psi_cross_pooled": float(np.sum(w * df["psi_cross"]) / np.sum(w)),
        "estimator":   "hurdle_gpu_hal_active_set_full_n_cv_pooled",
    }
    pd.DataFrame([out]).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    # Also write the per-radius detail CSV for the paper table
    detail_path = Path(args.out).with_suffix(".per_radius.csv")
    df.to_csv(detail_path, index=False)
    print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()
