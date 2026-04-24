#!/usr/bin/env python3
"""Delta-method regHAL-TMLE runner. Validates + runs at a single radius.

Per Li, Qiu, Wang & van der Laan (arXiv:2506.17214, June 2025).

Uses hurdle structure pragmatically: fits HAL-gaussian on raw Y (so the
regHAL-TMLE machinery operates on a single model, not a composition).
This is a simplification of the proper hurdle regHAL-TMLE and serves as
a first cut. If the targeted psi lands in a sensible range, we have
evidence that the approach works on our data; proper hurdle extension
can follow.

Usage:
    python run_reghal_tmle.py --radius 7 --max-n 50000
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

import causal_core as cc
import reghal_tmle as rht


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--max-n", type=int, default=50000,
                   help="Cluster-aware subsample size. 0 = no subsampling.")
    p.add_argument("--shift-pct", type=float, default=0.10)
    p.add_argument("--max-degree", type=int, default=2)
    p.add_argument("--num-knots", type=str, default="25,10")
    p.add_argument("--ridge-eta", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    R = args.radius
    num_knots = tuple(int(x) for x in args.num_knots.split(","))
    print(f"regHAL-TMLE Delta-method at R={R}km (max_n={args.max_n})", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values
    print(f"Full panel: {len(data)} rows, {data[S].gt(0).sum()} positive outcomes", flush=True)

    # Optional cluster-aware subsample for HAL tractability
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
    result = rht.reghal_tmle_shift(
        df=data, A_col=W, L_cols=confs, Y_col=S, cluster_col="_cluster",
        shift_pct=args.shift_pct,
        ridge_eta=args.ridge_eta,
        max_iter=args.max_iter,
        hal_kwargs={
            "max_degree":        args.max_degree,
            "num_knots":         num_knots,
            "smoothness_orders": 1,
        },
        verbose=True,
    )
    total = time.time() - t0

    print(f"\n═══════════════════════════════════════════════", flush=True)
    print(f"regHAL-TMLE RESULT ({total:.0f}s)", flush=True)
    print(f"═══════════════════════════════════════════════", flush=True)
    print(f"  radius        = {R} km", flush=True)
    print(f"  n             = {result.n}", flush=True)
    print(f"  n_clusters    = {result.n_clusters}", flush=True)
    print(f"  n_basis       = {result.n_basis}", flush=True)
    print(f"  psi_plugin    = {result.psi_plugin:+.4e}", flush=True)
    print(f"  psi_targeted  = {result.psi_targeted:+.4e}", flush=True)
    print(f"  SE (IF)       = {result.se_if:.4e}", flush=True)
    print(f"  95% CI        = [{result.ci_low:+.4e}, {result.ci_high:+.4e}]", flush=True)
    print(f"  p             = {result.pval:.4e}", flush=True)
    print(f"  targeting iter = {result.n_iter} {'(converged)' if result.converged else '(max)'}", flush=True)
    print(f"  Significant   = {result.ci_low > 0 or result.ci_high < 0}", flush=True)

    # Write CSV
    out = {
        "radius_km":      R,
        "window_days":    365,
        "shift_pct":      args.shift_pct,
        "n":              result.n,
        "n_clusters":     result.n_clusters,
        "n_basis":        result.n_basis,
        "psi_plugin":     result.psi_plugin,
        "psi_targeted":   result.psi_targeted,
        "se_cluster":     result.se_if,
        "ci_low":         result.ci_low,
        "ci_high":        result.ci_high,
        "pval":           result.pval,
        "n_iter":         result.n_iter,
        "converged":      result.converged,
        "elapsed_sec":    result.elapsed_sec,
        "estimator":      "reghal_tmle_delta",
    }
    outfile = f"reghal_shift_{R}km.csv"
    pd.DataFrame([out]).to_csv(outfile, index=False)
    print(f"\nWrote {outfile}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
