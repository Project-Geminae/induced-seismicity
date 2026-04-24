#!/usr/bin/env python3
"""GPU-accelerated XGBoost dose-response curve E[Y_a] at a grid of A values.

Fits a hurdle XGBoost outcome model on full n and evaluates the
counterfactual mean at each point on a log-spaced injection-volume
grid. Cluster bootstrap for CIs at each grid point.

Grid: [1e3, 1e4, 1e5, 1e6, 1e7] BBL — spans 99th-pct support down to
low-volume wells. Results beyond 1e7 BBL are extrapolation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_xgb_dose_response_gpu.py --radius 7 --B 500
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import causal_core as cc


# Log-spaced grid in BBL. Covers observed distribution (median ~1e5, p99 ~1e7).
DOSE_GRID = np.array([1e3, 1e4, 1e5, 1e6, 1e7], dtype=float)


def fit_hurdle_xgb(X, y, device="cuda", seed=42):
    is_pos = (y > 0).astype(float)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        tree_method="hist", device=device, verbosity=0,
        eval_metric="logloss", random_state=seed,
    )
    clf.fit(X, is_pos)
    mask = y > 0
    if mask.sum() < 50:
        mean_log = float(np.log1p(y[mask]).mean()) if mask.any() else 0.0
        def predict(Xq):
            p = clf.predict_proba(Xq)[:, 1]
            return p * np.expm1(mean_log)
        return predict
    reg = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        tree_method="hist", device=device, verbosity=0, random_state=seed,
    )
    reg.fit(X[mask], np.log1p(y[mask]))
    def predict(Xq):
        p = clf.predict_proba(Xq)[:, 1]
        log_mag = reg.predict(Xq)
        return p * np.expm1(log_mag)
    return predict


def dose_response(df, A_col, L_cols, Y_col, cluster_col,
                  grid=DOSE_GRID, B=500, device="cuda", seed=42, verbose=True):
    """Fit HurdleXGBoost, evaluate E[Y_a] at each a in grid, bootstrap CI per point."""
    A = df[A_col].to_numpy(dtype=np.float32)
    L = df[L_cols].to_numpy(dtype=np.float32)
    Y = df[Y_col].to_numpy(dtype=np.float32)
    clusters = df[cluster_col].to_numpy()
    n = len(df)
    n_clusters = int(pd.Series(clusters).nunique())

    AL = np.column_stack([A, L])

    # ── Point estimate ───────────────────────────────────────────────
    t0 = time.time()
    if verbose:
        print(f"  Fitting hurdle XGBoost on {device} (n={n}, grid={list(grid)})...", flush=True)
    predict = fit_hurdle_xgb(AL, Y, device=device, seed=seed)
    psi_grid = []
    for a in grid:
        AL_a = np.column_stack([np.full(n, float(a), dtype=np.float32), L])
        psi_grid.append(float(np.mean(predict(AL_a))))
    psi_grid = np.asarray(psi_grid)
    fit_time = time.time() - t0
    if verbose:
        print(f"  Fit + grid eval complete ({fit_time:.1f}s)", flush=True)
        for a, p in zip(grid, psi_grid):
            print(f"    E[Y | A={a:.0e}] = {p:+.4e}", flush=True)

    # ── Cluster bootstrap ────────────────────────────────────────────
    t0 = time.time()
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    cluster_to_idx = {c: np.where(clusters == c)[0] for c in unique_clusters}
    boot = np.zeros((B, len(grid)))
    n_succ = 0
    for b in range(B):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        try:
            A_b = A[idx]; L_b = L[idx]; Y_b = Y[idx]
            AL_b = np.column_stack([A_b, L_b])
            pred_b = fit_hurdle_xgb(AL_b, Y_b, device=device, seed=seed)
            for j, a in enumerate(grid):
                AL_ba = np.column_stack([np.full(len(idx), float(a), dtype=np.float32), L_b])
                boot[n_succ, j] = float(np.mean(pred_b(AL_ba)))
            n_succ += 1
            if verbose and ((b + 1) % max(1, B // 10) == 0):
                print(f"    boot {b+1}/{B}  elapsed={time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            if verbose:
                print(f"    boot {b+1} failed: {e}", flush=True)
            continue
    boot = boot[:n_succ]
    boot_time = time.time() - t0

    if verbose:
        print(f"  Bootstrap done ({boot_time:.0f}s, B_successful={n_succ})", flush=True)

    rows = []
    for j, a in enumerate(grid):
        psi = psi_grid[j]
        ci_low = float(np.quantile(boot[:, j], 0.025))
        ci_high = float(np.quantile(boot[:, j], 0.975))
        se_boot = float(np.std(boot[:, j], ddof=1))
        rows.append({
            "a_star":     float(a),
            "psi":        psi,
            "ci_low":     ci_low,
            "ci_high":    ci_high,
            "se_boot":    se_boot,
            "B":          n_succ,
            "n":          n,
            "n_clusters": n_clusters,
            "device":     device,
        })
    return pd.DataFrame(rows), fit_time + boot_time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--B", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    R = args.radius
    print(f"XGBoost-GPU dose-response at R={R}km, B={args.B}, device={args.device}", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values
    print(f"Panel: {len(data)} rows, {data[S].gt(0).sum()} positive outcomes", flush=True)

    df_dr, elapsed = dose_response(
        data, W, confs, S, "_cluster",
        B=args.B, device=args.device, seed=args.seed,
    )
    df_dr["radius_km"] = R
    df_dr["estimator"] = "xgb_hurdle_gpu_dose_response"
    outfile = f"xgb_dose_{R}km.csv"
    df_dr.to_csv(outfile, index=False)

    print(f"\n=== RESULT ({elapsed:.0f}s) ===", flush=True)
    print(df_dr[["a_star", "psi", "ci_low", "ci_high", "se_boot"]].to_string(
        index=False, float_format=lambda x: f"{x:+.3e}" if abs(x) < 1 else f"{x:.0f}"))
    print(f"\nWrote {outfile}", flush=True)


if __name__ == "__main__":
    main()
