#!/usr/bin/env python3
"""Run undersmoothed-HAL for the shift intervention at a given radius.

First pass: small B (e.g., 20) to measure compute cost per bootstrap.
If that finishes in reasonable time, scale up to B=100.

Usage:
    python run_undersmoothed_hal.py --radius 7 --B 20
"""
import argparse
import time
import sys

import pandas as pd

import causal_core as cc
import undersmoothed_hal as uhal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--B", type=int, default=100, help="Bootstrap iterations.")
    p.add_argument("--shift-pct", type=float, default=0.10)
    p.add_argument("--max-degree", type=int, default=2)
    p.add_argument("--num-knots", type=str, default="25,10",
                   help="Comma-separated knots per degree, e.g. '25,10'.")
    p.add_argument("--undersmoothing", type=float, default=None,
                   help="Multiplier on CV-selected lambda. Default = 1/sqrt(log(n)).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    R = args.radius
    num_knots = tuple(int(x) for x in args.num_knots.split(","))
    print(f"Undersmoothed HAL at R={R}km, B={args.B}, max_degree={args.max_degree}, num_knots={num_knots}", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values
    print(f"Panel: {len(data)} rows, {data[S].gt(0).sum()} positive outcomes, confounders: {confs}", flush=True)

    t0 = time.time()
    result = uhal.undersmoothed_hal_shift(
        df=data, A_col=W, L_cols=confs, Y_col=S, cluster_col="_cluster",
        shift_pct=args.shift_pct,
        B=args.B,
        hal_kwargs={
            "max_degree": args.max_degree,
            "num_knots": num_knots,
            "smoothness_orders": 1,
            "undersmoothing_factor": args.undersmoothing,
        },
        seed=args.seed,
    )
    total = time.time() - t0

    print(f"\n═══════════════════════════════════════════════", flush=True)
    print(f"RESULT  (total {total:.0f}s)", flush=True)
    print(f"═══════════════════════════════════════════════", flush=True)
    print(f"  radius       = {R} km", flush=True)
    print(f"  n            = {result.n}", flush=True)
    print(f"  n_clusters   = {result.n_clusters}", flush=True)
    print(f"  psi          = {result.psi:+.4e}", flush=True)
    print(f"  CI 95%       = [{result.ci_low:+.4e}, {result.ci_high:+.4e}]", flush=True)
    print(f"  SE (boot)    = {result.se_boot:.4e}", flush=True)
    print(f"  B            = {result.B}", flush=True)
    print(f"  lambda_cv    = {result.lambda_cv:.4e}", flush=True)
    print(f"  lambda_used  = {result.lambda_used:.4e}", flush=True)
    print(f"  fit time     = {result.fit_time_sec:.0f}s", flush=True)
    print(f"  boot time    = {result.boot_time_sec:.0f}s", flush=True)
    print(f"  Significant  = {result.ci_low > 0 or result.ci_high < 0}", flush=True)

    # Save CSV
    out = {
        "radius_km":      R,
        "window_days":    365,
        "shift_pct":      args.shift_pct,
        "n":              result.n,
        "n_clusters":     result.n_clusters,
        "psi":            result.psi,
        "ci_low":         result.ci_low,
        "ci_high":        result.ci_high,
        "se_boot":        result.se_boot,
        "B":              result.B,
        "lambda_cv":      result.lambda_cv,
        "lambda_used":    result.lambda_used,
        "fit_time_sec":   result.fit_time_sec,
        "boot_time_sec":  result.boot_time_sec,
        "total_time_sec": total,
        "estimator":      "undersmoothed_hal",
    }
    outfile = f"hal_shift_{R}km.csv"
    pd.DataFrame([out]).to_csv(outfile, index=False)
    print(f"\nWrote {outfile}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
