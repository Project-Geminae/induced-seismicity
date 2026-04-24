#!/usr/bin/env python3
"""GPU-accelerated XGBoost plug-in for the shift-intervention estimand.

Mirrors the undersmoothed-HAL API (same target parameter, same cluster
bootstrap for inference) but uses a hurdle XGBoost outcome model fit on
the GPU instead of HAL. For the Midland Basin panel at n=451k with 7-8
features, a single XGBoost GPU fit takes ~3-5 seconds, so B=500 bootstraps
finish in ~30-45 minutes per radius. At ~3 hours of compute per GPU this
lets a single worker sweep all 20 radii in one shift, or we parallelize
across GPUs for the full sweep in under an hour.

Reports the plug-in ψ = E[Q(1.1·A, L) - Q(A, L)] with percentile-bootstrap CI.
This is not TMLE — it's the honest XGBoost plug-in, reported alongside
HAL plug-in and regHAL-TMLE as a cross-method sanity check.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_xgb_shift_gpu.py --radius 7 --B 500
"""
import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

import causal_core as cc


@dataclass
class XGBShiftResult:
    psi:        float
    ci_low:     float
    ci_high:    float
    se_boot:    float
    n:          int
    n_clusters: int
    B:          int
    elapsed_sec: float
    device:     str


def fit_hurdle_xgb(X, y, device="cuda", seed=42):
    """Two-part XGBoost: logistic(Y>0) + gaussian(log1p(Y) | Y>0)."""
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


def xgb_shift(df, A_col, L_cols, Y_col, cluster_col,
              shift_pct=0.10, B=500, device="cuda", seed=42, verbose=True):
    A = df[A_col].to_numpy(dtype=np.float32)
    L = df[L_cols].to_numpy(dtype=np.float32)
    Y = df[Y_col].to_numpy(dtype=np.float32)
    clusters = df[cluster_col].to_numpy()
    n = len(df)
    n_clusters = int(pd.Series(clusters).nunique())

    AL = np.column_stack([A, L])
    AL_post = np.column_stack([A * (1.0 + shift_pct), L])

    t0 = time.time()
    if verbose:
        print(f"  Fitting hurdle XGBoost on {device} (n={n}, features={AL.shape[1]})...", flush=True)
    predict = fit_hurdle_xgb(AL, Y, device=device, seed=seed)
    psi = float(np.mean(predict(AL_post) - predict(AL)))
    fit_time = time.time() - t0
    if verbose:
        print(f"  Fit complete ({fit_time:.1f}s). psi_plugin = {psi:+.4e}", flush=True)

    t0 = time.time()
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    cluster_to_idx = {c: np.where(clusters == c)[0] for c in unique_clusters}
    boot = []
    for b in range(B):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        try:
            A_b = A[idx]; L_b = L[idx]; Y_b = Y[idx]
            AL_b = np.column_stack([A_b, L_b])
            AL_b_post = np.column_stack([A_b * (1.0 + shift_pct), L_b])
            pred_b = fit_hurdle_xgb(AL_b, Y_b, device=device, seed=seed)
            psi_b = float(np.mean(pred_b(AL_b_post) - pred_b(AL_b)))
            boot.append(psi_b)
            if verbose and ((b + 1) % max(1, B // 10) == 0):
                print(f"    boot {b+1}/{B}  psi*={psi_b:+.3e}  elapsed={time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"    boot {b+1} failed: {e}", flush=True)
            continue
    boot_time = time.time() - t0

    boot = np.asarray(boot)
    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))
    se_boot = float(np.std(boot, ddof=1))
    if verbose:
        print(f"  Bootstrap done ({boot_time:.0f}s). CI = [{ci_low:+.3e}, {ci_high:+.3e}]", flush=True)

    return XGBShiftResult(
        psi=psi, ci_low=ci_low, ci_high=ci_high, se_boot=se_boot,
        n=n, n_clusters=n_clusters, B=len(boot),
        elapsed_sec=fit_time + boot_time, device=device,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--B", type=int, default=500)
    p.add_argument("--shift-pct", type=float, default=0.10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    R = args.radius
    print(f"XGBoost-GPU shift estimator at R={R}km, B={args.B}, device={args.device}", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values
    print(f"Panel: {len(data)} rows, {data[S].gt(0).sum()} positive outcomes", flush=True)

    result = xgb_shift(
        data, W, confs, S, "_cluster",
        shift_pct=args.shift_pct, B=args.B,
        device=args.device, seed=args.seed,
    )

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
        "device":         result.device,
        "elapsed_sec":    result.elapsed_sec,
        "estimator":      "xgb_hurdle_gpu",
    }
    outfile = f"xgb_shift_{R}km.csv"
    pd.DataFrame([out]).to_csv(outfile, index=False)
    print(f"\n=== RESULT ===", flush=True)
    print(f"  psi          = {result.psi:+.4e}", flush=True)
    print(f"  95% CI       = [{result.ci_low:+.4e}, {result.ci_high:+.4e}]", flush=True)
    print(f"  SE (boot)    = {result.se_boot:.4e}", flush=True)
    print(f"  Significant  = {result.ci_low > 0 or result.ci_high < 0}", flush=True)
    print(f"  Wrote {outfile}", flush=True)


if __name__ == "__main__":
    main()
