#!/usr/bin/env python3
"""Run GPU HAL fit on full n=451k panel at a single radius.

This is the validation milestone for the GPU HAL backend. Compare:
  - GPU HAL fit_hal_gpu(X, y) at full n
  - regHAL-TMLE plug-in at n=50k subsample

If GPU HAL converges and the lambda_cv is in a reasonable range, we
have the core estimator working at full sample size.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_gpu_hal_full_n.py --radius 7
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import causal_core as cc
from gpu_hal import hal_fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-lambdas", type=int, default=20)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--shift-pct", type=float, default=0.10)
    p.add_argument("--undersmoothing", type=float, default=None,
                   help="Factor < 1 to shrink lambda_cv. Default: None (use lambda_cv).")
    args = p.parse_args()

    R = args.radius
    print(f"=== GPU HAL full-n fit at R={R} km ===", flush=True)
    print(f"  n_folds={args.n_folds}, n_lambdas={args.n_lambdas}, max_iter={args.max_iter}", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    n_full = len(data)
    X = data[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = data[S].to_numpy(dtype=np.float64)
    print(f"  Full panel: n={n_full}, d={X.shape[1]}, positives={int((y>0).sum())}", flush=True)

    t0 = time.time()
    fit = hal_fit.fit_hal_gpu(
        X, y,
        max_degree=2, num_knots=(25, 10), smoothness_orders=1,
        n_folds=args.n_folds, n_lambdas=args.n_lambdas,
        max_iter=args.max_iter,
        undersmoothing=args.undersmoothing,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Plug-in psi for the shift target
    A_post = data[W].to_numpy() * (1.0 + args.shift_pct)
    X_post = data[[W] + list(confs)].to_numpy(dtype=np.float64).copy()
    X_post[:, 0] = A_post
    y_post_pred = fit.predict(X_post)
    y_obs_pred = fit.predict(X)
    psi_plugin = float(np.mean(y_post_pred - y_obs_pred))

    print(f"\n=== RESULT ({elapsed:.0f}s) ===", flush=True)
    print(f"  n             = {fit.n}", flush=True)
    print(f"  p (basis)     = {fit.p}", flush=True)
    print(f"  active bases  = {len(fit.active_idx)}", flush=True)
    print(f"  lambda_cv     = {fit.lambda_cv:.4e}", flush=True)
    print(f"  lambda_used   = {fit.lambda_used:.4e}", flush=True)
    print(f"  psi_plugin    = {psi_plugin:+.4e}", flush=True)

    out = {
        "radius_km":      R,
        "n":              fit.n,
        "n_basis":        fit.p,
        "n_active":       len(fit.active_idx),
        "lambda_cv":      fit.lambda_cv,
        "lambda_used":    fit.lambda_used,
        "psi_plugin":     psi_plugin,
        "elapsed_sec":    elapsed,
        "estimator":      "gpu_hal_fit_full_n",
        "shift_pct":      args.shift_pct,
        "undersmoothing": args.undersmoothing if args.undersmoothing is not None else 1.0,
    }
    outfile = f"gpu_hal_fit_{R}km_fullN.csv"
    pd.DataFrame([out]).to_csv(outfile, index=False)
    print(f"  wrote {outfile}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
