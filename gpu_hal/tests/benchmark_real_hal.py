"""Benchmark GPU HAL vs CPU hal9001 on a realistic induced-seismicity basis.

Builds HAL basis at R=7 (or caller-specified radius), n determined by
panel load. Times: basis construction (R, CPU) + GPU Lasso CV vs the
existing regHAL pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_real_hal.py --radius 7 --max-n 10000
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import causal_core as cc

from gpu_hal import backend, hal_fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, default=7)
    p.add_argument("--max-n", type=int, default=10000)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-lambdas", type=int, default=30)
    args = p.parse_args()

    R = args.radius
    print(f"GPU HAL benchmark at R={R}km, max_n={args.max_n}", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    # Subsample for benchmark
    if args.max_n and len(data) > args.max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), size=args.max_n, replace=False)
        data = data.iloc[idx].reset_index(drop=True)

    n = len(data)
    X = data[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = data[S].to_numpy(dtype=np.float64)
    print(f"Data: n={n}, d={X.shape[1]}, positives={int((y>0).sum())}", flush=True)

    # ── Phase A: basis construction ───────────────────────────────────
    t0 = time.time()
    phi, basis_list = backend.build_hal_basis(X, max_degree=2, num_knots=(25, 10))
    t_basis = time.time() - t0
    print(f"[A] Basis: ({phi.shape[0]} × {phi.shape[1]}), nnz={phi.nnz}, "
          f"density={phi.nnz/(phi.shape[0]*phi.shape[1]):.2%}  ({t_basis:.1f}s)", flush=True)

    # ── Phase B: full GPU HAL fit with CV ─────────────────────────────
    t0 = time.time()
    fit = hal_fit.fit_hal_gpu(
        X, y,
        max_degree=2, num_knots=(25, 10),
        n_folds=args.n_folds, n_lambdas=args.n_lambdas,
        verbose=True,
    )
    t_gpu = time.time() - t0
    print(f"[B] GPU fit: {t_gpu:.1f}s total", flush=True)
    print(f"    lambda_cv = {fit.lambda_cv:.4e}", flush=True)
    print(f"    active bases: {len(fit.active_idx)} / {fit.p}", flush=True)

    # Sanity: predict at observed data, compute R²-like fit
    yhat = fit.predict(X)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"    in-sample R²: {r2:.4f}", flush=True)

    print(f"\nTiming summary:", flush=True)
    print(f"  Basis construction (R, CPU): {t_basis:.1f}s", flush=True)
    print(f"  GPU HAL CV+fit:              {t_gpu:.1f}s", flush=True)
    print(f"  Total:                       {t_basis + t_gpu:.1f}s", flush=True)


if __name__ == "__main__":
    main()
