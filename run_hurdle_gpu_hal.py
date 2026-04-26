#!/usr/bin/env python3
"""Run hurdle GPU HAL at a single radius (validation + benchmark)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import causal_core as cc
from gpu_hal import hurdle_hal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--max-n", type=int, default=50000,
                   help="Cluster-aware subsample (0 = no subsampling).")
    p.add_argument("--shift-pct", type=float, default=0.10)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-lambdas", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    R = args.radius
    print(f"=== Hurdle-GPU-HAL at R={R} km, max_n={args.max_n} ===", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    if args.max_n and len(data) > args.max_n:
        rng = np.random.default_rng(args.seed)
        clusters_all = data["_cluster"].values
        unique_clusters = np.unique(clusters_all)
        rng.shuffle(unique_clusters)
        kept, rows = [], 0
        for c in unique_clusters:
            nc = int((clusters_all == c).sum())
            if rows + nc > args.max_n and rows > 0:
                break
            kept.append(c); rows += nc
        mask = np.isin(clusters_all, kept)
        data = data.loc[mask].reset_index(drop=True)
        print(f"Subsampled to n={len(data)} ({len(kept)} clusters)", flush=True)

    X = data[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = data[S].to_numpy(dtype=np.float64)
    print(f"n={len(X)}, positives={int((y>0).sum())}", flush=True)

    t0 = time.time()
    fit = hurdle_hal.fit_hurdle_hal_gpu(
        X, y,
        max_degree=2, num_knots=(25, 10), smoothness_orders=1,
        n_folds=args.n_folds, n_lambdas=args.n_lambdas,
        max_iter=args.max_iter, verbose=True,
    )
    elapsed = time.time() - t0

    # Compute psi via shift intervention
    A_post = X[:, 0] * (1.0 + args.shift_pct)
    X_post = X.copy(); X_post[:, 0] = A_post
    Q_obs = fit.predict(X)
    Q_post = fit.predict(X_post)
    psi_total = float(np.mean(Q_post - Q_obs))

    # Decompose: classifier-only (P shift) and magnitude-only effects
    from scipy.sparse import diags
    from gpu_hal import backend
    phi = backend.apply_basis(X, fit.basis_list).tocsr()
    phi_scaled = (phi @ diags(1.0/fit.col_norms)).tocsr()
    phi_post = backend.apply_basis(X_post, fit.basis_list).tocsr()
    phi_post_scaled = (phi_post @ diags(1.0/fit.col_norms)).tocsr()

    eta_pos = phi_scaled @ fit.beta_pos + fit.intercept_pos
    p_pos = 1.0 / (1.0 + np.exp(-np.clip(eta_pos, -50.0, 50.0)))
    eta_pos_post = phi_post_scaled @ fit.beta_pos + fit.intercept_pos
    p_pos_post = 1.0 / (1.0 + np.exp(-np.clip(eta_pos_post, -50.0, 50.0)))

    log_mag = phi_scaled @ fit.beta_mag + fit.intercept_mag
    log_mag_post = phi_post_scaled @ fit.beta_mag + fit.intercept_mag
    mag = np.expm1(log_mag); mag_post = np.expm1(log_mag_post)

    # Frequency channel: hold magnitude constant at observed, shift only P(Y>0)
    psi_freq = float(np.mean((p_pos_post - p_pos) * mag))
    # Magnitude channel: hold P(Y>0) constant at observed, shift only magnitude
    psi_mag = float(np.mean(p_pos * (mag_post - mag)))
    # Cross-term
    psi_cross = float(np.mean((p_pos_post - p_pos) * (mag_post - mag)))

    print(f"\n=== Hurdle GPU HAL RESULT ({elapsed:.0f}s) ===", flush=True)
    print(f"  n={fit.n}, p={fit.p}", flush=True)
    print(f"  Stage 1 (logistic): λ={fit.lambda_pos:.4e}, active={fit.n_active_pos}", flush=True)
    print(f"  Stage 2 (gaussian): λ={fit.lambda_mag:.4e}, active={fit.n_active_mag}", flush=True)
    print(f"  ψ_total      = {psi_total:+.4e}", flush=True)
    print(f"  ψ_frequency  = {psi_freq:+.4e}  (volume → P(Y>0))", flush=True)
    print(f"  ψ_magnitude  = {psi_mag:+.4e}  (volume → E[Y|Y>0])", flush=True)
    print(f"  ψ_cross      = {psi_cross:+.4e}  (interaction)", flush=True)
    print(f"  CPU regHAL n=50k baseline (for comparison): psi_plugin = +4.02e-3", flush=True)

    out = {
        "radius_km":      R,
        "max_n":          args.max_n,
        "n":              fit.n,
        "p":              fit.p,
        "n_active_pos":   fit.n_active_pos,
        "n_active_mag":   fit.n_active_mag,
        "lambda_pos":     fit.lambda_pos,
        "lambda_mag":     fit.lambda_mag,
        "psi_total":      psi_total,
        "psi_frequency":  psi_freq,
        "psi_magnitude":  psi_mag,
        "psi_cross":      psi_cross,
        "elapsed_sec":    elapsed,
        "estimator":      "hurdle_gpu_hal",
    }
    outfile = f"hurdle_gpu_hal_{R}km.csv"
    pd.DataFrame([out]).to_csv(outfile, index=False)
    print(f"\nWrote {outfile}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
