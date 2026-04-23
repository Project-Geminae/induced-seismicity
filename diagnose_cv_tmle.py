#!/usr/bin/env python3
"""Diagnostic instrumentation for CV-TMLE + haldensify + HAL.

Runs a single radius with seed=42 (matching the production sweep) but
captures per-fold:
  - SuperLearner NNLS meta-weights for Q classifier stack (logit/xgb/gbm/rf/hal)
  - SuperLearner NNLS meta-weights for Q regressor stack
  - H distribution (min, quantiles, max, mean, std)
  - Fold-specific epsilon
  - Q initial vs targeted prediction summary

Used to verify: (a) HAL is contributing non-trivially to the SuperLearner
meta-weights, (b) the clever covariate H is non-trivial so targeting is
actually doing work.

Usage:
    python diagnose_cv_tmle.py --radius 7
"""
import argparse
import time
import numpy as np
import pandas as pd

import causal_core as cc
import tmle_core as tmle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--max-n", type=int, default=50000)
    p.add_argument("--shift-pct", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    R = args.radius
    print(f"DIAGNOSTIC CV-TMLE at R={R}km, seed={args.seed}", flush=True)
    print(f"  CV_DENSITY = {tmle.CV_DENSITY}  USE_HAL = {tmle.USE_HAL}", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    # Inline a copy of cv_tmle_shift with diagnostic capture
    from sklearn.model_selection import KFold

    # Subsample exactly as cv_tmle_shift does
    rng = np.random.default_rng(args.seed)
    clusters_all = data["_cluster"].values
    unique_clusters = np.unique(clusters_all)
    rng.shuffle(unique_clusters)
    kept = []
    rows = 0
    for c in unique_clusters:
        nc = int((clusters_all == c).sum())
        if rows + nc > args.max_n and rows > 0:
            break
        kept.append(c)
        rows += nc
    mask = np.isin(clusters_all, kept)
    sub = data.loc[mask].reset_index(drop=True)
    print(f"Subsample: {len(sub)} rows, {len(kept)} clusters", flush=True)

    A = sub[W].to_numpy(dtype=float)
    L = sub[confs].to_numpy(dtype=float)
    Y = sub[S].to_numpy(dtype=float)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_diagnostics = []

    for fold_idx, (tr, va) in enumerate(kf.split(np.arange(len(sub)))):
        print(f"\n═══════════ Fold {fold_idx+1} ═══════════", flush=True)
        t0 = time.time()

        # Q fit
        Q_model = tmle.HurdleSuperLearner(random_state=args.seed + fold_idx)
        AL_tr = np.column_stack([A[tr], L[tr]])
        Q_model.fit(AL_tr, Y[tr])

        # Get meta-weights
        clf_weights = Q_model.clf_meta_.coef_ if hasattr(Q_model.clf_meta_, "coef_") else None
        reg_weights = Q_model.reg_meta_.coef_ if Q_model.reg_meta_ is not None else None
        clf_names = [n for n, _ in Q_model._build_classifier_stack()]
        reg_names = [n for n, _ in Q_model._build_regressor_stack()]
        print(f"  Q classifier learners: {clf_names}", flush=True)
        if clf_weights is not None:
            print(f"  Q classifier NNLS weights: {dict(zip(clf_names, [f'{w:.3f}' for w in clf_weights]))}", flush=True)
        print(f"  Q regressor learners:  {reg_names}", flush=True)
        if reg_weights is not None:
            print(f"  Q regressor NNLS weights:  {dict(zip(reg_names, [f'{w:.3f}' for w in reg_weights]))}", flush=True)

        # g fit
        if tmle.CV_DENSITY == "haldensify":
            g_model = tmle.HALDensifyConditionalDensity(random_state=args.seed + fold_idx)
        elif tmle.CV_DENSITY == "kde":
            g_model = tmle.KDEConditionalDensity(random_state=args.seed + fold_idx)
        else:
            g_model = tmle.HistogramConditionalDensity(random_state=args.seed + fold_idx)
        g_model.fit(A[tr], L[tr])

        # Compute H on validation
        g_obs_val = g_model.density(A[va], L[va])
        A_pre_val = A[va] / (1.0 + args.shift_pct)
        g_pre_val = g_model.density(A_pre_val, L[va])
        H_val = (g_pre_val / g_obs_val) / (1.0 + args.shift_pct)

        # H distribution
        H_stats = {
            "min": float(np.min(H_val)),
            "q025": float(np.quantile(H_val, 0.025)),
            "q25":  float(np.quantile(H_val, 0.25)),
            "median": float(np.median(H_val)),
            "mean": float(np.mean(H_val)),
            "q75":  float(np.quantile(H_val, 0.75)),
            "q975": float(np.quantile(H_val, 0.975)),
            "max":  float(np.max(H_val)),
            "std":  float(np.std(H_val)),
        }
        print(f"  H distribution: {H_stats}", flush=True)

        # Target
        AL_val = np.column_stack([A[va], L[va]])
        Q_hat_val = Q_model.predict(AL_val)
        residual_val = Y[va] - Q_hat_val
        eps = float(np.dot(H_val, residual_val) / max(np.dot(H_val, H_val), 1e-12))
        print(f"  epsilon        = {eps:.6e}", flush=True)

        # Impact of targeting
        Q_star = Q_hat_val + eps * H_val
        delta_Q = Q_star - Q_hat_val
        print(f"  |Q_star - Q_hat| mean={np.abs(delta_Q).mean():.3e}  max={np.abs(delta_Q).max():.3e}", flush=True)
        print(f"  Q_hat mean={Q_hat_val.mean():.3e}  Q_star mean={Q_star.mean():.3e}", flush=True)

        fold_diagnostics.append({
            "fold": fold_idx + 1,
            "clf_weights": dict(zip(clf_names, clf_weights.tolist())) if clf_weights is not None else None,
            "reg_weights": dict(zip(reg_names, reg_weights.tolist())) if reg_weights is not None else None,
            "H_stats": H_stats,
            "epsilon": eps,
            "delta_Q_mean_abs": float(np.abs(delta_Q).mean()),
            "delta_Q_max_abs": float(np.abs(delta_Q).max()),
            "Q_hat_mean": float(Q_hat_val.mean()),
            "Q_star_mean": float(Q_star.mean()),
            "fold_time_sec": time.time() - t0,
        })

    # Write summary
    import json
    outfile = f"cv_tmle_diagnostic_{R}km_seed{args.seed}.json"
    with open(outfile, "w") as f:
        json.dump(fold_diagnostics, f, indent=2)
    print(f"\nWrote {outfile}", flush=True)

    # Summary across folds
    print("\n═══════════ Summary across folds ═══════════", flush=True)
    print(f"  epsilon range:  min={min(d['epsilon'] for d in fold_diagnostics):.3e}  max={max(d['epsilon'] for d in fold_diagnostics):.3e}", flush=True)
    print(f"  max_H range:    min={min(d['H_stats']['max'] for d in fold_diagnostics):.3f}  max={max(d['H_stats']['max'] for d in fold_diagnostics):.3f}", flush=True)
    # HAL weight across folds
    if fold_diagnostics[0].get("reg_weights"):
        hal_weights = [d["reg_weights"].get("hal", 0.0) for d in fold_diagnostics]
        print(f"  HAL (regressor) weights per fold: {[f'{w:.3f}' for w in hal_weights]}", flush=True)
        print(f"  HAL mean weight: {np.mean(hal_weights):.3f}", flush=True)


if __name__ == "__main__":
    main()
