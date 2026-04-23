#!/usr/bin/env python3
"""Verify the CV-TMLE influence function has mean ≈ 0 at convergence.

TMLE's entire inference machinery assumes the targeting step solves the
efficient IF equation, so E[IF(O)] = 0 at the final estimate. If the
global sample mean of the IF is materially different from zero, the
targeting didn't converge and reported CIs are invalid.

This script mimics cv_tmle_shift's internal loop, captures per-row IF,
and reports mean(IF), SE(IF), and mean(IF) / SE(IF) as a diagnostic.

Under proper convergence: |mean(IF)| < 0.1 × SE(IF).
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import causal_core as cc
import tmle_core as tmle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-n", type=int, default=50000)
    p.add_argument("--shift-pct", type=float, default=0.10)
    args = p.parse_args()

    R = args.radius
    print(f"IF-mean diagnostic at R={R}km, seed={args.seed}", flush=True)

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    # Same subsampling as cv_tmle_shift
    rng = np.random.default_rng(args.seed)
    clusters_all = data["_cluster"].values
    unique_clusters = np.unique(clusters_all)
    rng.shuffle(unique_clusters)
    kept, rows = [], 0
    for c in unique_clusters:
        nc = int((clusters_all == c).sum())
        if rows + nc > args.max_n and rows > 0:
            break
        kept.append(c)
        rows += nc
    mask = np.isin(clusters_all, kept)
    sub = data.loc[mask].reset_index(drop=True)

    n = len(sub)
    A = sub[W].to_numpy(dtype=float)
    L = sub[confs].to_numpy(dtype=float)
    Y = sub[S].to_numpy(dtype=float)
    clusters = sub["_cluster"].values
    print(f"Subsample: n={n}, clusters={len(kept)}", flush=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    Q_hat_cv = np.zeros(n)
    Q_post_cv = np.zeros(n)
    H_cv = np.zeros(n)
    H_post_cv = np.zeros(n)
    eps_per_fold = np.zeros(n)

    for fold_idx, (tr, va) in enumerate(kf.split(np.arange(n))):
        print(f"  Fold {fold_idx+1}/5 ...", flush=True)
        Q_model = tmle.HurdleSuperLearner(random_state=args.seed + fold_idx)
        Q_model.fit(np.column_stack([A[tr], L[tr]]), Y[tr])
        Q_hat_cv[va] = Q_model.predict(np.column_stack([A[va], L[va]]))

        if tmle.CV_DENSITY == "haldensify":
            g_model = tmle.HALDensifyConditionalDensity(random_state=args.seed + fold_idx)
        elif tmle.CV_DENSITY == "kde":
            g_model = tmle.KDEConditionalDensity(random_state=args.seed + fold_idx)
        else:
            g_model = tmle.HistogramConditionalDensity(random_state=args.seed + fold_idx)
        g_model.fit(A[tr], L[tr])

        g_obs_val = g_model.density(A[va], L[va])
        A_pre_val = A[va] / (1.0 + args.shift_pct)
        g_pre_val = g_model.density(A_pre_val, L[va])
        H_val = (g_pre_val / g_obs_val) / (1.0 + args.shift_pct)
        H_cv[va] = H_val

        residual = Y[va] - Q_hat_cv[va]
        eps = float(np.dot(H_val, residual) / max(np.dot(H_val, H_val), 1e-12))
        eps_per_fold[va] = eps

        A_post = A[va] * (1.0 + args.shift_pct)
        Q_post_cv[va] = Q_model.predict(np.column_stack([A_post, L[va]]))
        g_post = g_model.density(A_post, L[va])
        H_post_cv[va] = (g_obs_val / g_post) / (1.0 + args.shift_pct)

    # Compute IF
    Q_star_obs  = Q_hat_cv + eps_per_fold * H_cv
    Q_star_post = Q_post_cv + eps_per_fold * H_post_cv
    psi_n = float(np.mean(Q_star_post))
    psi_0 = float(np.mean(Q_star_obs))

    if_psi  = H_cv * (Y - Q_star_obs) + Q_star_post - psi_n
    if_psi0 = (Y - Q_star_obs)        + Q_star_obs  - psi_0
    if_diff = if_psi - if_psi0

    # Diagnostic stats
    print("\n" + "=" * 60)
    print("IF MEAN DIAGNOSTIC")
    print("=" * 60)
    for name, IF in [("if_psi  (shifted)", if_psi),
                     ("if_psi0 (baseline)", if_psi0),
                     ("if_diff (shift effect)", if_diff)]:
        mean = np.mean(IF)
        se = np.sqrt(np.var(IF, ddof=1) / n)
        ratio = abs(mean) / max(se, 1e-15)
        flag = "OK" if ratio < 0.1 else ("MARGINAL" if ratio < 0.5 else "FAIL")
        print(f"{name}:")
        print(f"  mean = {mean:+.3e}   SE = {se:.3e}   |mean|/SE = {ratio:.3f}   [{flag}]")

    print(f"\npsi = {psi_n - psi_0:+.3e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
