#!/usr/bin/env python3
"""Run CV-TMLE + haldensify + HAL at a given radius with a given seed.

Used for seed-sensitivity checks: same pipeline, different cluster
subsample draw, to verify results aren't driven by the seed=42 subsample.

Usage:
    python run_cv_tmle_seed.py --radius 7 --seed 1
"""
import argparse
import time
import sys

import causal_core as cc
import tmle_core as tmle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-n", type=int, default=50000)
    p.add_argument("--shift-pct", type=float, default=0.10)
    args = p.parse_args()

    R = args.radius
    print(f"CV_DENSITY = {tmle.CV_DENSITY}", flush=True)
    print(f"USE_HAL    = {tmle.USE_HAL}", flush=True)
    print(f"seed       = {args.seed}", flush=True)
    print(f"max_n      = {args.max_n}", flush=True)
    print(f"radius     = {R} km", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values
    print(f"Full panel: {len(data)} rows, {data[S].gt(0).sum()} positive outcomes", flush=True)
    print(f"Starting cv_tmle_shift (seed={args.seed})...", flush=True)

    t0 = time.time()
    result = tmle.cv_tmle_shift(
        df=data, A_col=W, L_cols=confs, Y_col=S, cluster_col="_cluster",
        shift_pct=args.shift_pct, n_splits=5,
        seed=args.seed,
        max_n=args.max_n,
    )
    elapsed = time.time() - t0

    print(f"\nResult ({elapsed:.0f}s):", flush=True)
    print(f"  n            = {result.n}", flush=True)
    print(f"  n_clusters   = {result.n_clusters}", flush=True)
    print(f"  psi          = {result.psi:.6e}", flush=True)
    print(f"  se_cluster   = {result.se_cluster:.6e}", flush=True)
    print(f"  CI           = [{result.ci_low:.6e}, {result.ci_high:.6e}]", flush=True)
    print(f"  p-value      = {result.pval:.4f}", flush=True)
    print(f"  max_H        = {result.notes['max_H']:.4f}", flush=True)
    print(f"  mean_H       = {result.notes['mean_H']:.4f}", flush=True)
    print(f"  epsilon      = {result.epsilon:.6e}", flush=True)
    print(f"  Significant  = {result.ci_low > 0 or result.ci_high < 0}", flush=True)

    # Write a CSV matching the parallel driver's schema
    import pandas as pd
    row = {
        "radius_km":      R,
        "window_days":    365,
        "shift_pct":      args.shift_pct,
        "treatment_col":  W,
        "seed":           args.seed,
        "n":              result.n,
        "n_clusters":     result.n_clusters,
        "psi":            result.psi,
        "se_iid":         result.se_iid,
        "se_cluster":     result.se_cluster,
        "ci_low":         result.ci_low,
        "ci_high":        result.ci_high,
        "pval":           result.pval,
        "epsilon":        result.epsilon,
        "psi_under_shift": result.notes.get("psi_under_shift", float("nan")),
        "psi_no_shift":    result.notes.get("psi_no_shift", float("nan")),
        "mean_H":          result.notes["mean_H"],
        "max_H":           result.notes["max_H"],
        "elapsed_sec":     elapsed,
    }
    outfile = f"tmle_shift_cvtmle_{R}km_seed{args.seed}.csv"
    pd.DataFrame([row]).to_csv(outfile, index=False)
    print(f"\nWrote {outfile}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
