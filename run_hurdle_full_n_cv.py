"""Full-n hurdle HAL-TMLE with proper cross-validation at n=451,212.

Stage 1 (logistic on Y > 0): cluster-aware 5-fold CV across a 15-point
log-spaced λ-grid using active-set IRLS with warm-starts across the
path. ~4-6 hours wall.

Stage 2 (gaussian on log(1+Y) | Y > 0): cluster-aware 5-fold CV on the
~18.7k positives. ~30 min wall.

Final fits on full data at CV-selected λ values. ψ + freq/mag/cross
decomposition + cluster-IF SE. Output: hurdle_full_n_cv_R<R>km.csv.

Logs intermediate state per fold per lambda so we can recover from
crashes.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import causal_core as cc
from gpu_hal import backend, cd_gram, fista_gram
from gpu_hal.cd_logistic_active_set import logistic_lasso_active_set


def cluster_foldid(cluster_vec: np.ndarray, n_folds: int, seed: int = 42) -> np.ndarray:
    """1-indexed fold assignment; rows of the same cluster keep the same fold."""
    rng = np.random.default_rng(seed)
    unique = np.unique(cluster_vec)
    rng.shuffle(unique)
    cluster_to_fold = {c: int(i % n_folds) + 1 for i, c in enumerate(unique)}
    return np.array([cluster_to_fold[c] for c in cluster_vec], dtype=np.int64)


def cluster_if_se_total(IF_per_row: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """SE of the mean parameter ψ̂ = (1/n) Σ IF_i + ψ̂."""
    n = len(IF_per_row)
    se_iid_mean = float(np.std(IF_per_row, ddof=1) / np.sqrt(n))
    df = pd.DataFrame({"if": IF_per_row, "c": clusters})
    by_c = df.groupby("c", sort=False)["if"].sum().to_numpy()
    n_c = len(by_c)
    centered = by_c - by_c.mean()
    var_cluster_sum = (n_c / max(n_c - 1, 1)) * float(np.sum(centered ** 2))
    se_cluster_mean = float(np.sqrt(var_cluster_sum) / n)
    return se_iid_mean, se_cluster_mean


def cv_stage1_logistic(phi_scaled, is_pos, foldid, lambdas, n_folds,
                        max_irls=15, log_path=None, single_fold=None):
    """5-fold CV for logistic stage 1. Warm-start across decreasing λ
    within each fold for ≥2× speedup vs cold-start path.

    Returns: (cv_dev: array shape (n_folds, n_lambdas),
              path_results: list[list[LogisticResult]])
    """
    cv_dev = np.zeros((n_folds, len(lambdas)))
    path_results = [[None] * len(lambdas) for _ in range(n_folds)]

    folds_iter = [single_fold] if single_fold is not None else range(1, n_folds + 1)
    for f in folds_iter:
        ho_mask = (foldid == f)
        tr_mask = ~ho_mask
        n_tr = int(tr_mask.sum()); n_ho = int(ho_mask.sum())
        n_pos_tr = int(is_pos[tr_mask].sum()); n_pos_ho = int(is_pos[ho_mask].sum())
        print(f"\n━━━━ Stage 1 CV Fold {f}/{n_folds}: "
              f"n_tr={n_tr:,} (positives {n_pos_tr:,}), "
              f"n_ho={n_ho:,} (positives {n_pos_ho:,}) ━━━━", flush=True)

        beta_warm = None
        intercept_warm = 0.0
        # Sub-slice once per fold (avoids re-slicing inside the lambda loop)
        phi_tr = phi_scaled[tr_mask]
        y_tr = is_pos[tr_mask]
        phi_ho = phi_scaled[ho_mask]
        y_ho = is_pos[ho_mask]

        for i, lam in enumerate(lambdas):  # decreasing λ → grows |β|
            t_lam = time.time()
            res = logistic_lasso_active_set(
                phi_tr, y_tr, lam,
                max_irls=max_irls, irls_tol=1e-6,
                cd_max_sweeps=200, cd_tol=1e-7, kkt_tol=1e-3,
                initial_full_irls=1 if beta_warm is None else 0,  # only warmup on first λ
                beta0=beta_warm, intercept0=intercept_warm,
                fit_intercept=True, verbose=False,
            )
            # Hold-out deviance
            eta_ho = phi_ho @ res.beta + res.intercept
            p_ho = 1.0 / (1.0 + np.exp(-np.clip(eta_ho, -50.0, 50.0)))
            dev = -2.0 * float(np.mean(
                y_ho * np.log(np.clip(p_ho, 1e-12, 1.0))
                + (1 - y_ho) * np.log(np.clip(1 - p_ho, 1e-12, 1.0))
            ))
            cv_dev[f - 1, i] = dev
            path_results[f - 1][i] = res
            n_act = int(np.sum(np.abs(res.beta) > 1e-10))
            print(f"  fold {f} λ {i+1:2d}/{len(lambdas)} = {lam:.4e}: "
                  f"active={n_act:4d}, IRLS={res.n_irls:2d}, "
                  f"ho_dev={dev:.4e} ({time.time()-t_lam:.0f}s)", flush=True)
            beta_warm = res.beta
            intercept_warm = res.intercept

            if log_path is not None:
                # Persist progress per fold per lambda
                pd.DataFrame({
                    "fold":   list(range(1, n_folds + 1)) * len(lambdas),
                    "lambda": np.repeat(lambdas, n_folds),
                    "dev":    cv_dev.T.flatten(),
                }).to_csv(log_path, index=False)

    return cv_dev, path_results


def cv_stage2_gaussian(phi_pos, y_pos_log, foldid_pos, lambdas, n_folds,
                        log_path=None, single_fold=None):
    """5-fold CV for gaussian stage 2 on positives. Full-Gram CD is fine
    here — n_pos is small (~18.7k)."""
    cv_mse = np.zeros((n_folds, len(lambdas)))
    import jax.numpy as jnp

    folds_iter = [single_fold] if single_fold is not None else range(1, n_folds + 1)
    for f in folds_iter:
        ho_mask = (foldid_pos == f)
        tr_mask = ~ho_mask
        n_tr = int(tr_mask.sum()); n_ho = int(ho_mask.sum())
        print(f"\n━━━━ Stage 2 CV Fold {f}/{n_folds}: "
              f"n_tr={n_tr:,}, n_ho={n_ho:,} ━━━━", flush=True)

        phi_tr = phi_pos[tr_mask]
        y_tr = y_pos_log[tr_mask]
        phi_ho = phi_pos[ho_mask]
        y_ho = y_pos_log[ho_mask]

        # Build Gram once per fold
        G_full, Xty_full = fista_gram.compute_gram(phi_tr, y_tr - y_tr.mean())
        G_j = jnp.asarray(G_full)
        Xty_j = jnp.asarray(Xty_full)

        for i, lam in enumerate(lambdas):
            t_lam = time.time()
            res = cd_gram.cd_lasso_gram(
                G=G_j, Xty=Xty_j, lam=float(lam),
                max_sweeps=2000, tol=1e-8, verbose=False,
            )
            beta = np.asarray(res.beta)
            intercept = float(y_tr.mean())
            pred_ho = phi_ho @ beta + intercept
            mse = float(np.mean((y_ho - pred_ho) ** 2))
            cv_mse[f - 1, i] = mse
            n_act = int(np.sum(np.abs(beta) > 1e-10))
            print(f"  fold {f} λ {i+1:2d}/{len(lambdas)} = {lam:.4e}: "
                  f"active={n_act:4d}, sweeps={res.n_sweeps}, "
                  f"ho_mse={mse:.4e} ({time.time()-t_lam:.1f}s)", flush=True)

            if log_path is not None:
                pd.DataFrame({
                    "fold":   list(range(1, n_folds + 1)) * len(lambdas),
                    "lambda": np.repeat(lambdas, n_folds),
                    "mse":    cv_mse.T.flatten(),
                }).to_csv(log_path, index=False)

    return cv_mse


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--max-n", type=int, default=0,
                    help="Cluster-aware subsample size; 0 = full panel.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-lambdas", type=int, default=15)
    ap.add_argument("--lambda-ratio", type=float, default=1e-3,
                    help="λ_min / λ_max ratio (default 1e-3).")
    ap.add_argument("--shift-pct", type=float, default=0.10)
    ap.add_argument("--max-irls", type=int, default=15,
                    help="Max IRLS iters per fit; lower for CV speed.")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--fold-only", type=int, default=None,
                    help="Run only fold N (1-indexed). Used by the parallel "
                         "fan-out runner — each fold writes its own CV CSV. "
                         "After all folds complete, run --aggregate to "
                         "compute the final ψ.")
    ap.add_argument("--aggregate", action="store_true",
                    help="Read per-fold CV CSVs, pick CV-best λ for both "
                         "stages, run final fits + ψ + decomposition.")
    args = ap.parse_args()

    R = args.radius
    out_path = Path(args.out or f"hurdle_full_n_cv_R{R}km.csv")
    s1_log = Path(f"hurdle_full_n_cv_R{R}km_stage1_dev.csv")
    s2_log = Path(f"hurdle_full_n_cv_R{R}km_stage2_mse.csv")

    print(f"=== Full-n Hurdle HAL-TMLE with CV at R={R} km ===")
    print(f"  n_folds={args.n_folds}, n_lambdas={args.n_lambdas}, "
          f"λ-ratio={args.lambda_ratio}, max_irls={args.max_irls}\n", flush=True)

    print("Loading panel + design matrix …", flush=True)
    t0 = time.time()
    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P_col, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    if args.max_n > 0 and len(data) > args.max_n:
        rng = np.random.default_rng(42)
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

    X = data[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = data[S].to_numpy(dtype=np.float64)
    cluster_vec = data["_cluster"].to_numpy()
    n = len(X)
    n_clusters = int(data["_cluster"].nunique())
    print(f"  n={n:,}, clusters={n_clusters}, "
          f"positives={int((y>0).sum()):,} "
          f"({100*(y>0).mean():.2f}%) in {time.time()-t0:.0f}s\n", flush=True)

    print("Building HAL basis …", flush=True)
    t0 = time.time()
    phi_csr, basis_list = backend.build_hal_basis(
        X, max_degree=2, num_knots=(25, 10), smoothness_orders=1,
    )
    p = phi_csr.shape[1]
    print(f"  basis ({n} × {p}), nnz={phi_csr.nnz}, "
          f"density={phi_csr.nnz/(n*p):.2%} in {time.time()-t0:.0f}s\n", flush=True)

    from scipy.sparse import diags
    col_sumsq = np.asarray(phi_csr.multiply(phi_csr).sum(axis=0)).ravel()
    col_norms = np.sqrt(np.maximum(col_sumsq, 1e-24))
    D_inv = diags(1.0 / col_norms)
    phi_scaled = (phi_csr @ D_inv).tocsr()
    print("  basis pre-scaled to unit column L2 norm\n", flush=True)

    is_pos = (y > 0).astype(np.float64)
    foldid_full = cluster_foldid(cluster_vec, n_folds=args.n_folds, seed=42)
    print(f"  cluster-aware folds at full n: sizes "
          f"{np.bincount(foldid_full)[1:].tolist()}\n", flush=True)

    # ── Stage 1 λ-grid (data-driven) ──
    Xty1 = (phi_scaled.T @ (is_pos - is_pos.mean())) / float(n)
    lam_max1 = float(np.max(np.abs(Xty1)))
    lambdas1 = lam_max1 * np.logspace(0, np.log10(args.lambda_ratio), args.n_lambdas)
    print(f"Stage 1 λ-grid: λ_max={lam_max1:.4e}, λ_min={lambdas1[-1]:.4e}, "
          f"n={len(lambdas1)}", flush=True)

    # ── Stage 1 CV (full or single-fold) ──
    if args.fold_only is not None:
        f = args.fold_only
        print(f"\n=== Stage 1 CV: FOLD-ONLY mode, running fold {f}/{args.n_folds} ===", flush=True)
        t_s1_cv = time.time()
        # Run only the requested fold by masking foldid
        cv_dev, _ = cv_stage1_logistic(
            phi_scaled, is_pos, foldid_full, lambdas1, args.n_folds,
            max_irls=args.max_irls, log_path=None,
            single_fold=f,
        )
        # cv_dev is (n_folds, n_lambdas) but only row f-1 is filled
        s1_cv_time = time.time() - t_s1_cv
        # Write fold-specific CSV
        fold_csv = Path(f"hurdle_full_n_cv_R{R}km_stage1_fold{f}.csv")
        pd.DataFrame({
            "fold": [f] * len(lambdas1),
            "lambda_idx": list(range(len(lambdas1))),
            "lambda": lambdas1,
            "dev": cv_dev[f - 1],
        }).to_csv(fold_csv, index=False)
        print(f"\nStage 1 fold {f} done in {s1_cv_time/60:.1f} min, "
              f"wrote {fold_csv}", flush=True)

        # Also do stage 2 single-fold for symmetric parallelism
        pos_mask = y > 0
        n_pos = int(pos_mask.sum())
        phi_pos = phi_scaled[pos_mask]
        y_pos_log = np.log1p(y[pos_mask])
        cluster_pos = cluster_vec[pos_mask]
        foldid_pos = cluster_foldid(cluster_pos, n_folds=args.n_folds, seed=42)
        Xty2 = (phi_pos.T @ (y_pos_log - y_pos_log.mean())) / float(n_pos)
        lam_max2 = float(np.max(np.abs(Xty2)))
        lambdas2 = lam_max2 * np.logspace(0, np.log10(args.lambda_ratio), args.n_lambdas)
        print(f"\n=== Stage 2 CV: FOLD-ONLY mode, fold {f} on positives n={n_pos:,} ===", flush=True)
        t_s2_cv = time.time()
        cv_mse = cv_stage2_gaussian(
            phi_pos, y_pos_log, foldid_pos, lambdas2, args.n_folds,
            log_path=None, single_fold=f,
        )
        s2_cv_time = time.time() - t_s2_cv
        fold_csv2 = Path(f"hurdle_full_n_cv_R{R}km_stage2_fold{f}.csv")
        pd.DataFrame({
            "fold": [f] * len(lambdas2),
            "lambda_idx": list(range(len(lambdas2))),
            "lambda": lambdas2,
            "mse": cv_mse[f - 1],
        }).to_csv(fold_csv2, index=False)
        print(f"\nStage 2 fold {f} done in {s2_cv_time/60:.1f} min, "
              f"wrote {fold_csv2}", flush=True)

        print(f"\n=== FOLD {f} COMPLETE ===", flush=True)
        print(f"  total time: {(s1_cv_time + s2_cv_time) / 60:.1f} min", flush=True)
        return

    if args.aggregate:
        print(f"\n=== AGGREGATE mode: reading per-fold CSVs and computing final ψ ===", flush=True)
        # Read all fold CSVs and concatenate
        s1_dfs = []; s2_dfs = []
        for f in range(1, args.n_folds + 1):
            s1_dfs.append(pd.read_csv(f"hurdle_full_n_cv_R{R}km_stage1_fold{f}.csv"))
            s2_dfs.append(pd.read_csv(f"hurdle_full_n_cv_R{R}km_stage2_fold{f}.csv"))
        s1_all = pd.concat(s1_dfs, ignore_index=True)
        s2_all = pd.concat(s2_dfs, ignore_index=True)
        # Pivot to (n_folds, n_lambdas)
        cv_dev = s1_all.pivot(index="fold", columns="lambda_idx", values="dev").to_numpy()
        cv_mse = s2_all.pivot(index="fold", columns="lambda_idx", values="mse").to_numpy()
        # Skip stage 1 CV; jump directly to aggregation
        s1_cv_time = 0.0  # already done in parallel
    else:
        print(f"\n=== Stage 1: logistic CV (active-set IRLS, warm-starts, sequential folds) ===", flush=True)
        t_s1_cv = time.time()
        cv_dev, _ = cv_stage1_logistic(
            phi_scaled, is_pos, foldid_full, lambdas1, args.n_folds,
            max_irls=args.max_irls, log_path=s1_log,
        )
        s1_cv_time = time.time() - t_s1_cv

    mean_dev = cv_dev.mean(axis=0)
    min_idx1 = int(np.argmin(mean_dev))
    lambda_pos = float(lambdas1[min_idx1])
    print(f"\nStage 1 CV: λ_pos = {lambda_pos:.4e} (idx {min_idx1}), "
          f"min mean dev = {mean_dev[min_idx1]:.4e}\n", flush=True)

    # ── Stage 1 final fit on full data ──
    print(f"=== Stage 1 final fit at λ_pos = {lambda_pos:.4e} ===", flush=True)
    t_s1_final = time.time()
    s1_final = logistic_lasso_active_set(
        phi_scaled, is_pos, lambda_pos,
        max_irls=30, irls_tol=1e-6,
        cd_max_sweeps=200, cd_tol=1e-7, kkt_tol=1e-3,
        initial_full_irls=1, fit_intercept=True, verbose=True,
    )
    s1_final_time = time.time() - t_s1_final
    n_active_pos = int(np.sum(np.abs(s1_final.beta) > 1e-10))
    print(f"  Stage 1 final: active={n_active_pos}, "
          f"time={s1_final_time:.0f}s\n", flush=True)

    # ── Stage 2 CV ──
    pos_mask = y > 0
    n_pos = int(pos_mask.sum())
    phi_pos = phi_scaled[pos_mask]
    y_pos_log = np.log1p(y[pos_mask])
    cluster_pos = cluster_vec[pos_mask]
    foldid_pos = cluster_foldid(cluster_pos, n_folds=args.n_folds, seed=42)
    print(f"=== Stage 2: gaussian CV on positives n={n_pos:,} ===", flush=True)

    Xty2 = (phi_pos.T @ (y_pos_log - y_pos_log.mean())) / float(n_pos)
    lam_max2 = float(np.max(np.abs(Xty2)))
    lambdas2 = lam_max2 * np.logspace(0, np.log10(args.lambda_ratio), args.n_lambdas)
    print(f"Stage 2 λ-grid: λ_max={lam_max2:.4e}, λ_min={lambdas2[-1]:.4e}", flush=True)

    t_s2_cv = time.time()
    cv_mse = cv_stage2_gaussian(
        phi_pos, y_pos_log, foldid_pos, lambdas2, args.n_folds, log_path=s2_log,
    )
    s2_cv_time = time.time() - t_s2_cv
    mean_mse = cv_mse.mean(axis=0)
    min_idx2 = int(np.argmin(mean_mse))
    lambda_mag = float(lambdas2[min_idx2])
    print(f"\nStage 2 CV done in {s2_cv_time/60:.1f} min", flush=True)
    print(f"  λ_mag (CV) = {lambda_mag:.4e} (idx {min_idx2}), "
          f"min mean mse = {mean_mse[min_idx2]:.4e}\n", flush=True)

    # ── Stage 2 final fit on positives ──
    print(f"=== Stage 2 final fit at λ_mag = {lambda_mag:.4e} ===", flush=True)
    t_s2_final = time.time()
    import jax.numpy as jnp
    G_full, Xty_full = fista_gram.compute_gram(phi_pos, y_pos_log - y_pos_log.mean())
    G_j = jnp.asarray(G_full); Xty_j = jnp.asarray(Xty_full)
    s2_final = cd_gram.cd_lasso_gram(
        G=G_j, Xty=Xty_j, lam=lambda_mag,
        max_sweeps=2000, tol=1e-8, verbose=False,
    )
    beta_mag = np.asarray(s2_final.beta)
    intercept_mag = float(y_pos_log.mean())
    n_active_mag = int(np.sum(np.abs(beta_mag) > 1e-10))
    s2_final_time = time.time() - t_s2_final
    print(f"  Stage 2 final: active={n_active_mag}, "
          f"time={s2_final_time:.0f}s\n", flush=True)

    # ── Compose hurdle, ψ, decomposition, cluster-IF SE ──
    print(f"=== Compose Q + shift ψ at +{args.shift_pct*100:.0f}% ===", flush=True)
    t_comp = time.time()
    A = X[:, 0]
    A_post = A * (1.0 + args.shift_pct)
    X_post = X.copy(); X_post[:, 0] = A_post
    phi_post_raw = backend.apply_basis(X_post, basis_list).tocsr()
    phi_post_scaled = (phi_post_raw @ D_inv).tocsr()

    eta_pos_obs  = phi_scaled      @ s1_final.beta + s1_final.intercept
    eta_pos_post = phi_post_scaled @ s1_final.beta + s1_final.intercept
    p_obs  = 1.0 / (1.0 + np.exp(-np.clip(eta_pos_obs,  -50, 50)))
    p_post = 1.0 / (1.0 + np.exp(-np.clip(eta_pos_post, -50, 50)))

    log_mag_obs  = phi_scaled      @ beta_mag + intercept_mag
    log_mag_post = phi_post_scaled @ beta_mag + intercept_mag
    log_mag_obs  = np.clip(log_mag_obs,  -30, 30)
    log_mag_post = np.clip(log_mag_post, -30, 30)
    mag_obs  = np.expm1(log_mag_obs)
    mag_post = np.expm1(log_mag_post)

    Q_obs  = p_obs  * mag_obs
    Q_post = p_post * mag_post
    Q_diff = Q_post - Q_obs
    psi_total = float(Q_diff.mean())

    dP = p_post - p_obs
    dM = mag_post - mag_obs
    psi_freq  = float(np.mean(dP * mag_obs))
    psi_mag   = float(np.mean(p_obs * dM))
    psi_cross = float(np.mean(dP * dM))

    IF_per_row = Q_diff - psi_total
    se_iid_mean, se_cluster_mean = cluster_if_se_total(IF_per_row, cluster_vec)
    z = psi_total / max(se_cluster_mean, 1e-15)
    pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    ci_low  = psi_total - 1.959963984540054 * se_cluster_mean
    ci_high = psi_total + 1.959963984540054 * se_cluster_mean
    comp_time = time.time() - t_comp
    print(f"  compose done in {comp_time:.0f}s\n", flush=True)

    total_wall = s1_cv_time + s1_final_time + s2_cv_time + s2_final_time + comp_time
    print(f"=== RESULT (full-n CV, R={R} km) ===")
    print(f"  n = {n:,}, p = {p}, n_clusters = {n_clusters}, "
          f"positives = {int((y>0).sum()):,}")
    print(f"  Stage 1 CV:        time = {s1_cv_time/60:.1f} min, "
          f"λ_pos = {lambda_pos:.4e}, n_active_final = {n_active_pos}")
    print(f"  Stage 2 CV:        time = {s2_cv_time/60:.1f} min, "
          f"λ_mag = {lambda_mag:.4e}, n_active_final = {n_active_mag}")
    print(f"  Final fits + ψ:    time = "
          f"{(s1_final_time+s2_final_time+comp_time)/60:.1f} min")
    print(f"  Total wall time:   {total_wall/60:.1f} min "
          f"({total_wall/3600:.2f} hrs)")
    print()
    print(f"  ψ_total      = {psi_total:+.4e}")
    print(f"  ψ_frequency  = {psi_freq:+.4e}  ({100*psi_freq/max(abs(psi_total),1e-30):+.0f}%)")
    print(f"  ψ_magnitude  = {psi_mag:+.4e}  ({100*psi_mag/max(abs(psi_total),1e-30):+.0f}%)")
    print(f"  ψ_cross      = {psi_cross:+.4e}  ({100*psi_cross/max(abs(psi_total),1e-30):+.0f}%)")
    print()
    print(f"  se_iid (mean):     {se_iid_mean:.4e}")
    print(f"  se_cluster (mean): {se_cluster_mean:.4e}")
    print(f"  Design effect:     {se_cluster_mean / max(se_iid_mean, 1e-30):.2f}×")
    print(f"  CI95 (cluster):    [{ci_low:+.4e}, {ci_high:+.4e}]")
    print(f"  z = {z:.3f}, p = {pval:.4e}")

    # Output CSV
    out_row = {
        "radius_km":     R,
        "n":             n,
        "p":             p,
        "n_clusters":    n_clusters,
        "n_positives":   int((y>0).sum()),
        "n_folds":       args.n_folds,
        "n_lambdas":     args.n_lambdas,
        "lambda_pos_cv": lambda_pos,
        "lambda_mag_cv": lambda_mag,
        "n_active_pos":  n_active_pos,
        "n_active_mag":  n_active_mag,
        "shift_pct":     args.shift_pct,
        "psi_total":     psi_total,
        "psi_freq":      psi_freq,
        "psi_mag":       psi_mag,
        "psi_cross":     psi_cross,
        "se_iid_mean":   se_iid_mean,
        "se_cluster_mean": se_cluster_mean,
        "design_effect": se_cluster_mean / max(se_iid_mean, 1e-30),
        "ci_low":        ci_low,
        "ci_high":       ci_high,
        "z":             z,
        "pval":          pval,
        "stage1_cv_min": s1_cv_time / 60,
        "stage2_cv_min": s2_cv_time / 60,
        "total_wall_min": total_wall / 60,
        "estimator":     "hurdle_gpu_hal_active_set_full_n_cv",
    }
    pd.DataFrame([out_row]).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
