"""Full-n hurdle HAL-TMLE on the induced-seismicity panel using
active-set IRLS for the logistic stage.

Skips CV (uses λ values already CV-selected at n=50k). Runs end-to-end
at n=451,212. Reports ψ_total, ψ_freq, ψ_mag, ψ_cross with cluster-IF
standard errors, plus per-stage timing.

Output: hurdle_full_n_R<R>km.csv + console summary.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import causal_core as cc
from gpu_hal import backend, cd_gram, fista_gram
from gpu_hal.cd_logistic_active_set import logistic_lasso_active_set


def cluster_if_se_total(IF_per_row: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Cluster-robust influence-function SE for the *mean* parameter
    ψ = mean(IF + ψ_hat) (i.e. the per-row IF variance scaled to a mean).

    For a mean ψ̂ = (1/n) Σ IF_i + ψ̂:
       Var(ψ̂)_iid     = Var(IF) / n
       Var(ψ̂)_cluster = (n_c/(n_c-1)) · Σ_c (Σ_{i∈c} IF_i)² / n²

    Returns (se_iid_mean, se_cluster_mean).
    """
    n = len(IF_per_row)
    se_iid_mean = float(np.std(IF_per_row, ddof=1) / np.sqrt(n))
    df = pd.DataFrame({"if": IF_per_row, "c": clusters})
    by_c = df.groupby("c", sort=False)["if"].sum().to_numpy()
    n_c = len(by_c)
    centered = by_c - by_c.mean()
    var_cluster_sum = (n_c / max(n_c - 1, 1)) * float(np.sum(centered ** 2))
    se_cluster_mean = float(np.sqrt(var_cluster_sum) / n)
    return se_iid_mean, se_cluster_mean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--max-n", type=int, default=0,
                    help="Cluster-aware subsample size; 0 = full panel.")
    ap.add_argument("--lambda-pos", type=float, default=1.77e-7,
                    help="L1 penalty for stage 1 (logistic). "
                         "Default 1.77e-7 = CV pick from n=50k Apr 25 hurdle log.")
    ap.add_argument("--lambda-mag", type=float, default=1.99e-6,
                    help="L1 penalty for stage 2 (gaussian on positives). "
                         "Default 1.99e-6 = CV pick from n=50k Apr 25 hurdle log.")
    ap.add_argument("--shift-pct", type=float, default=0.10,
                    help="Multiplicative shift: A_post = (1 + shift_pct) · A. "
                         "Default 0.10 = 10%% volume increase.")
    ap.add_argument("--max-irls", type=int, default=30)
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV path (default: hurdle_full_n_R<R>km.csv)")
    args = ap.parse_args()

    R = args.radius
    out_path = Path(args.out or f"hurdle_full_n_R{R}km.csv")

    print(f"=== Full-n Hurdle HAL-TMLE on R={R} km ===")
    print(f"  λ_pos = {args.lambda_pos:.4e}, λ_mag = {args.lambda_mag:.4e}, "
          f"shift = +{args.shift_pct*100:.0f}%, max_irls = {args.max_irls}\n")

    # ── Load data ──
    print("Loading panel + design matrix …")
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
        print(f"  cluster-aware subsample: n = {len(data)}")

    X = data[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = data[S].to_numpy(dtype=np.float64)
    cluster_vec = data["_cluster"].to_numpy()
    n = len(X)
    n_clusters = int(data["_cluster"].nunique())
    print(f"  loaded n = {n:,}, n_clusters = {n_clusters}, "
          f"positives = {int((y>0).sum()):,} ({100*(y>0).mean():.2f}%) "
          f"in {time.time()-t0:.1f}s\n")

    # ── Build basis once ──
    print("Building HAL basis (max_degree=2, num_knots=(25,10), smoothness_orders=1) …")
    t0 = time.time()
    phi_csr, basis_list = backend.build_hal_basis(
        X, max_degree=2, num_knots=(25, 10), smoothness_orders=1,
    )
    p = phi_csr.shape[1]
    print(f"  basis ({n} × {p}), nnz = {phi_csr.nnz}, "
          f"density = {phi_csr.nnz/(n*p):.2%} in {time.time()-t0:.1f}s\n")

    # Pre-scale basis to unit column L2 norm
    from scipy.sparse import diags
    col_sumsq = np.asarray(phi_csr.multiply(phi_csr).sum(axis=0)).ravel()
    col_norms = np.sqrt(np.maximum(col_sumsq, 1e-24))
    D_inv = diags(1.0 / col_norms)
    phi_scaled = (phi_csr @ D_inv).tocsr()

    # ── Stage 1: logistic on Y > 0 via active-set IRLS ──
    print(f"━━━━ Stage 1: logistic on P(Y > 0) at n = {n:,} ━━━━")
    is_pos = (y > 0).astype(np.float64)
    t0 = time.time()
    s1 = logistic_lasso_active_set(
        phi_scaled, is_pos, args.lambda_pos,
        max_irls=args.max_irls, irls_tol=1e-6,
        cd_max_sweeps=200, cd_tol=1e-7, kkt_tol=1e-3,
        initial_full_irls=1,
        fit_intercept=True, verbose=True,
    )
    s1_time = time.time() - t0
    n_active_pos = int(np.sum(np.abs(s1.beta) > 1e-10))
    print(f"  Stage 1 done: {s1.n_irls} IRLS iters, "
          f"n_active = {n_active_pos}, total = {s1_time:.1f}s\n")

    # ── Stage 2: gaussian on log(1+Y) | Y > 0 ──
    pos_mask = y > 0
    n_pos = int(pos_mask.sum())
    print(f"━━━━ Stage 2: gaussian on log(1+Y) | Y > 0 at n = {n_pos:,} ━━━━")
    phi_pos = phi_scaled[pos_mask]
    y_pos_log = np.log1p(y[pos_mask])

    t0 = time.time()
    G_full, Xty_full = fista_gram.compute_gram(phi_pos, y_pos_log - y_pos_log.mean())
    import jax.numpy as jnp
    G_j = jnp.asarray(G_full); Xty_j = jnp.asarray(Xty_full)
    s2 = cd_gram.cd_lasso_gram(
        G=G_j, Xty=Xty_j, lam=args.lambda_mag,
        max_sweeps=2000, tol=1e-8, verbose=False,
    )
    beta_mag = np.asarray(s2.beta)
    intercept_mag = float(y_pos_log.mean())
    n_active_mag = int(np.sum(np.abs(beta_mag) > 1e-10))
    s2_time = time.time() - t0
    print(f"  Stage 2 done: {s2.n_sweeps} CD sweeps, "
          f"n_active = {n_active_mag}, total = {s2_time:.1f}s\n")

    # ── Compose hurdle prediction + shift ψ ──
    print(f"━━━━ Compose Q(X) and shift ψ at +{args.shift_pct*100:.0f}% ━━━━")
    t0 = time.time()
    A = X[:, 0]
    A_post = A * (1.0 + args.shift_pct)
    X_post = X.copy(); X_post[:, 0] = A_post
    phi_post_raw = backend.apply_basis(X_post, basis_list).tocsr()
    phi_post_scaled = (phi_post_raw @ D_inv).tocsr()

    eta_pos_obs  = phi_scaled      @ s1.beta + s1.intercept
    eta_pos_post = phi_post_scaled @ s1.beta + s1.intercept
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
    Q_diff_per_row = Q_post - Q_obs
    psi_total = float(Q_diff_per_row.mean())

    dP = p_post - p_obs
    dM = mag_post - mag_obs
    psi_freq  = float(np.mean(dP * mag_obs))
    psi_mag   = float(np.mean(p_obs * dM))
    psi_cross = float(np.mean(dP * dM))

    # Cluster-IF SE for ψ_total (per-row IF = Q_post - Q_obs - ψ_total)
    IF_per_row = Q_diff_per_row - psi_total
    se_iid_mean, se_cluster_mean = cluster_if_se_total(IF_per_row, cluster_vec)

    z = psi_total / max(se_cluster_mean, 1e-15)
    import math
    pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    ci_low  = psi_total - 1.959963984540054 * se_cluster_mean
    ci_high = psi_total + 1.959963984540054 * se_cluster_mean
    compose_time = time.time() - t0

    print(f"  Compose done in {compose_time:.1f}s\n")

    # ── Summary ──
    total_wall = s1_time + s2_time + compose_time
    print(f"=== RESULT ===")
    print(f"  n = {n:,}, p = {p}, n_clusters = {n_clusters}, "
          f"positives = {int((y>0).sum()):,}")
    print(f"  Stage 1 (logistic): λ = {args.lambda_pos:.4e}, "
          f"n_active = {n_active_pos}, time = {s1_time:.1f}s")
    print(f"  Stage 2 (gaussian): λ = {args.lambda_mag:.4e}, "
          f"n_active = {n_active_mag}, time = {s2_time:.1f}s")
    print(f"  Compose + IF SE:    time = {compose_time:.1f}s")
    print(f"  Total wall time:    {total_wall:.1f}s ({total_wall/60:.1f} min)")
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
    print()
    print(f"  Apr 25 GPU hurdle baseline (n=49,519): ψ_total = +1.76e-3, "
          f"channels 54/34/12")

    # Write CSV
    out_row = {
        "radius_km":     R,
        "n":             n,
        "p":             p,
        "n_clusters":    n_clusters,
        "n_positives":   int((y>0).sum()),
        "lambda_pos":    args.lambda_pos,
        "lambda_mag":    args.lambda_mag,
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
        "stage1_time_s": s1_time,
        "stage2_time_s": s2_time,
        "compose_time_s": compose_time,
        "total_wall_s":  total_wall,
        "estimator":     "hurdle_gpu_hal_active_set",
    }
    pd.DataFrame([out_row]).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
