"""Capture the actual λ values that CPU HurdleHAL picks at each stage.

This closes the loop on the +4.02e-3 vs +1.76e-3 gap. Both pipelines use
the same two-stage hurdle structure but different λ-grids:

  GPU hurdle (Apr 25):
    Stage 1 logistic: λ_cv = 1.77e-7, active = 227
    Stage 2 gaussian: λ_cv = 1.99e-6, active = 134
    psi_total = +1.76e-3

  CPU HurdleHAL (this run, expected to print):
    Stage 1 logistic: λ_cv = ?, active = ?
    Stage 2 gaussian: λ_cv = ?, active = ?
    psi_total ≈ +4.02e-3 (or whatever HurdleHAL produces today)

Compare λ_cv across both stages. If CPU λ_cv is much larger (more
regularized) than GPU's, CPU underfits → smaller active set → predictions
hew closer to mean → potentially LARGER psi (because the hurdle
multiplication smooths differently).

Or, CPU might pick a DIFFERENT λ that isn't degenerate because binomial
deviance + small-positive-rate behaves better in glmnet's path than the
gaussian-on-zero-inflated case.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import causal_core as cc


def main():
    R = 7
    print(f"=== CPU HurdleHAL λ-stage capture, R={R} km, n=50k ===\n")

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    rng = np.random.default_rng(42)
    clusters_all = data["_cluster"].values
    unique_clusters = np.unique(clusters_all)
    rng.shuffle(unique_clusters)
    kept, rows = [], 0
    for c in unique_clusters:
        nc = int((clusters_all == c).sum())
        if rows + nc > 50_000 and rows > 0:
            break
        kept.append(c); rows += nc
    mask = np.isin(clusters_all, kept)
    sub = data.loc[mask].reset_index(drop=True)
    X = sub[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = sub[S].to_numpy(dtype=np.float64)
    print(f"n={len(X)}, clusters={len(kept)}, positives={int((y>0).sum())}\n")

    from undersmoothed_hal import HurdleHAL
    print("Fitting HurdleHAL (CPU regHAL pipeline) …")
    t0 = time.time()
    hal = HurdleHAL(max_degree=2, num_knots=(25, 10), smoothness_orders=1)
    hal.fit(X, y)
    elapsed = time.time() - t0
    print(f"  HurdleHAL.fit complete in {elapsed:.0f}s\n")

    # Inspect each stage's CV λ
    clf = hal.clf_  # UndersmoothedHAL family=binomial
    reg = hal.reg_  # UndersmoothedHAL family=gaussian (only on Y>0)

    import rpy2.robjects as ro

    def stage_summary(stage, label):
        if stage is None or stage._hal_fit is None:
            print(f"  {label}: not fit")
            return None, None
        lam_star = float(ro.r("function(f) f$lambda_star")(stage._hal_fit)[0])
        # Extract beta at lambda_star to count active
        beta_raw = ro.r("function(f) as.numeric(stats::coef(f$lasso_fit, s=f$lambda_star))")(stage._hal_fit)
        beta = np.asarray(beta_raw, dtype=np.float64)
        n_active = int(np.sum(np.abs(beta[1:]) > 1e-10))
        # Also check lambda path range
        lambdas_r = ro.r("function(f) f$lasso_fit$lambda")(stage._hal_fit)
        lambdas_arr = np.asarray(lambdas_r, dtype=np.float64)
        # Active count at the smallest lambda in the path (deepest into signal)
        beta_min_raw = ro.r(f"function(f) as.numeric(stats::coef(f$lasso_fit, s=f$lasso_fit$lambda[length(f$lasso_fit$lambda)]))")(stage._hal_fit)
        beta_min = np.asarray(beta_min_raw, dtype=np.float64)
        n_active_at_min = int(np.sum(np.abs(beta_min[1:]) > 1e-10))
        print(f"  {label}:")
        print(f"    family   = {stage.family}")
        print(f"    λ_star   = {lam_star:.4e}")
        print(f"    λ_path   = [{lambdas_arr.min():.4e}, {lambdas_arr.max():.4e}], length={len(lambdas_arr)}")
        print(f"    active@λ_star = {n_active}")
        print(f"    active@λ_min  = {n_active_at_min}")
        return lam_star, n_active

    lam1, na1 = stage_summary(clf, "Stage 1 (logistic, P(Y>0))")
    lam2, na2 = stage_summary(reg, "Stage 2 (gaussian, log(1+Y)|Y>0)")

    # Compute psi to confirm we reproduce +4.02e-3
    A_post = X[:, 0] * 1.10
    X_post = X.copy(); X_post[:, 0] = A_post
    psi = float(np.mean(hal.predict(X_post) - hal.predict(X)))
    print(f"\n  HurdleHAL psi (shift +10%) = {psi:+.4e}")
    print(f"  Apr 25 GPU hurdle baseline = +1.76e-3 (for comparison)")
    print(f"  Apr ?? CPU baseline note   = +4.02e-3 (per run_hurdle_gpu_hal.py comment)")


if __name__ == "__main__":
    main()
