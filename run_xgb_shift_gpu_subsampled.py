#!/usr/bin/env python3
"""Run XGBoost-GPU shift at a cluster-aware subsampled panel.

Purpose: isolate sample-size from methodology in the CI-width comparison
between regHAL-TMLE (n=50k, 42 clusters) and XGBoost-GPU B=500 (n=451k,
389 clusters). If XGBoost-GPU at n=50k gives CI widths similar to
regHAL-TMLE at n=50k, then the 100× CI-width gap we observed is
entirely sample-size-driven (not methodology-driven).

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_xgb_shift_gpu_subsampled.py \\
        --radius 7 --max-n 50000 --B 500 --seed 42
"""
import argparse
import time
import numpy as np
import pandas as pd

import causal_core as cc
from run_xgb_shift_gpu import xgb_shift


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--max-n", type=int, default=50000)
    p.add_argument("--B", type=int, default=500)
    p.add_argument("--shift-pct", type=float, default=0.10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    R = args.radius
    print(f"XGBoost-GPU shift at R={R}km, subsampled to n<={args.max_n}, B={args.B}", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    # Cluster-aware subsample matching regHAL-TMLE's subsampling
    if args.max_n and len(data) > args.max_n:
        rng = np.random.default_rng(args.seed)
        clusters_all = data["_cluster"].values
        unique = np.unique(clusters_all)
        rng.shuffle(unique)
        kept, rows = [], 0
        for c in unique:
            nc = int((clusters_all == c).sum())
            if rows + nc > args.max_n and rows > 0:
                break
            kept.append(c)
            rows += nc
        mask = np.isin(clusters_all, kept)
        data = data.loc[mask].reset_index(drop=True)
        print(f"Subsampled to {len(data)} rows ({len(kept)} clusters)", flush=True)

    t0 = time.time()
    result = xgb_shift(
        data, W, confs, S, "_cluster",
        shift_pct=args.shift_pct, B=args.B,
        device=args.device, seed=args.seed,
    )
    print(f"\n=== RESULT ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"  psi          = {result.psi:+.4e}", flush=True)
    print(f"  95% CI       = [{result.ci_low:+.4e}, {result.ci_high:+.4e}]", flush=True)
    print(f"  SE_boot      = {result.se_boot:.4e}", flush=True)
    print(f"  n            = {result.n}", flush=True)
    print(f"  n_clusters   = {result.n_clusters}", flush=True)

    out = {
        "radius_km": R,
        "max_n": args.max_n,
        "n": result.n,
        "n_clusters": result.n_clusters,
        "B": result.B,
        "psi": result.psi,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "se_boot": result.se_boot,
        "ci_width": result.ci_high - result.ci_low,
        "estimator": "xgb_gpu_subsampled",
        "seed": args.seed,
    }
    outfile = f"xgb_shift_{R}km_n{args.max_n}.csv"
    pd.DataFrame([out]).to_csv(outfile, index=False)
    print(f"\nWrote {outfile}", flush=True)


if __name__ == "__main__":
    main()
