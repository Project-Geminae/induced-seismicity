"""Full-n benchmark for active-set IRLS vs full-Gram logistic Lasso.

The full-Gram solver in cd_logistic.py rebuilds X^T diag(w) X / n every
IRLS iter — at n=451k, p=1564, density 28% this is ~25-30 min/iter and
makes the full-n hurdle fit infeasible.

This script:
  1. Builds the real HAL basis on the induced-seismicity panel at R=7
     for a chosen n (cluster-aware subsample).
  2. Runs both solvers (or just active-set if n is too large for full-
     Gram) on the binary outcome 1{Y>0} (matching the logistic stage of
     the hurdle).
  3. Reports per-iter wall time, active-set size, total time, and
     coefficient agreement.

Usage:
    python gpu_hal/tests/benchmark_active_set_full_n.py --max-n 50000
    python gpu_hal/tests/benchmark_active_set_full_n.py --max-n 200000 --skip-full-gram
    python gpu_hal/tests/benchmark_active_set_full_n.py --max-n 0  --skip-full-gram   # full panel
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

import causal_core as cc
from gpu_hal import backend
from gpu_hal.cd_logistic import logistic_lasso, _build_weighted_gram_cpu
from gpu_hal.cd_logistic_active_set import logistic_lasso_active_set


def load_subsample(R: int, max_n: int, seed: int = 42):
    """Cluster-aware subsample of the panel at radius R. max_n=0 → full panel."""
    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    if max_n > 0 and len(data) > max_n:
        rng = np.random.default_rng(seed)
        clusters_all = data["_cluster"].values
        unique_clusters = np.unique(clusters_all)
        rng.shuffle(unique_clusters)
        kept, rows = [], 0
        for c in unique_clusters:
            nc = int((clusters_all == c).sum())
            if rows + nc > max_n and rows > 0:
                break
            kept.append(c); rows += nc
        mask = np.isin(clusters_all, kept)
        data = data.loc[mask].reset_index(drop=True)

    X = data[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = data[S].to_numpy(dtype=np.float64)
    n_clusters = int(data["_cluster"].nunique())
    return X, y, n_clusters


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--max-n", type=int, default=50_000,
                    help="Cluster-aware subsample size. 0 = full panel.")
    ap.add_argument("--lambda", type=float, default=1e-7, dest="lam",
                    help="L1 penalty (default 1e-7, matches GPU hurdle "
                         "logistic stage's CV pick at n=50k).")
    ap.add_argument("--skip-full-gram", action="store_true",
                    help="Skip the full-Gram comparator (use for n > 100k "
                         "where full-Gram is infeasible).")
    ap.add_argument("--max-irls", type=int, default=15)
    args = ap.parse_args()

    R = args.radius
    print(f"=== Active-set IRLS full-n benchmark ===")
    print(f"  R = {R} km, max_n = {args.max_n}, λ = {args.lam:.4e}, "
          f"max_irls = {args.max_irls}\n")

    print("Loading subsample …")
    t0 = time.time()
    X, y, n_clusters = load_subsample(R, args.max_n)
    print(f"  loaded n = {len(X)}, p_X = {X.shape[1]}, "
          f"n_clusters = {n_clusters}, positives = {int((y>0).sum())} "
          f"({100*(y>0).mean():.2f}%) in {time.time()-t0:.1f}s\n")

    # Hurdle stage 1 outcome: 1{Y > 0}
    is_pos = (y > 0).astype(float)

    print("Building HAL basis (max_degree=2, num_knots=(25,10), smoothness_orders=1) …")
    t0 = time.time()
    phi_csr, basis_list = backend.build_hal_basis(
        X, max_degree=2, num_knots=(25, 10), smoothness_orders=1,
    )
    n, p = phi_csr.shape
    print(f"  basis ({n} × {p}), nnz = {phi_csr.nnz}, "
          f"density = {phi_csr.nnz/(n*p):.2%} in {time.time()-t0:.1f}s\n")

    # Pre-scale to unit column L2-norm (matches GPU pipeline + custom-grid CPU baseline)
    from scipy.sparse import diags
    col_sumsq = np.asarray(phi_csr.multiply(phi_csr).sum(axis=0)).ravel()
    col_norms = np.sqrt(np.maximum(col_sumsq, 1e-24))
    D_inv = diags(1.0 / col_norms)
    phi_scaled_csr = (phi_csr @ D_inv).tocsr()
    print(f"  basis pre-scaled: col_norms ∈ [{col_norms.min():.2e}, "
          f"{col_norms.max():.2e}]\n")

    # ── Active-set IRLS ──
    print(f"━━━━ Active-set IRLS (n = {n}) ━━━━")
    t0 = time.time()
    as_result = logistic_lasso_active_set(
        phi_scaled_csr, is_pos, args.lam,
        max_irls=args.max_irls, irls_tol=1e-6,
        cd_max_sweeps=200, cd_tol=1e-7, kkt_tol=1e-3,
        initial_full_irls=1,  # 1 warmup full-Gram iter for active-set seeding
        fit_intercept=True, verbose=True,
    )
    as_total = time.time() - t0
    n_active_as = int(np.sum(np.abs(as_result.beta) > 1e-10))
    print(f"  RESULT: n_irls = {as_result.n_irls}, n_active = {n_active_as}, "
          f"converged = {as_result.converged}, total = {as_total:.1f}s "
          f"({as_total/60:.1f} min)\n")

    # ── Full-Gram baseline (if n is small enough) ──
    fg_total = None
    n_active_fg = None
    rel_l2 = None
    if not args.skip_full_gram:
        print(f"━━━━ Full-Gram IRLS (n = {n}) ━━━━")
        t0 = time.time()
        fg_result = logistic_lasso(
            phi_scaled_csr, is_pos, args.lam,
            max_irls=args.max_irls, irls_tol=1e-6,
            cd_max_sweeps=200, cd_tol=1e-7,
            fit_intercept=True, verbose=True,
        )
        fg_total = time.time() - t0
        n_active_fg = int(np.sum(np.abs(fg_result.beta) > 1e-10))
        rel_l2 = float(np.linalg.norm(fg_result.beta - as_result.beta)
                       / max(np.linalg.norm(fg_result.beta), 1e-12))
        print(f"  RESULT: n_irls = {fg_result.n_irls}, n_active = {n_active_fg}, "
              f"converged = {fg_result.converged}, total = {fg_total:.1f}s "
              f"({fg_total/60:.1f} min)\n")

    print("━━━━ Summary ━━━━")
    print(f"  n              = {n:,}")
    print(f"  p (basis)      = {p}")
    print(f"  positives      = {int(is_pos.sum())} ({100*is_pos.mean():.2f}%)")
    print(f"  λ              = {args.lam:.4e}")
    print()
    print(f"  Active-set IRLS:  total = {as_total:.1f}s ({as_total/60:.1f} min), "
          f"n_active = {n_active_as}")
    if fg_total is not None:
        speedup = fg_total / max(as_total, 1e-9)
        print(f"  Full-Gram IRLS:   total = {fg_total:.1f}s ({fg_total/60:.1f} min), "
              f"n_active = {n_active_fg}")
        print(f"  Speedup:          {speedup:.2f}×")
        print(f"  Coefficient rel-L2 (vs full-Gram): {rel_l2:.3e}")
        print(f"  Active-set agreement: |Δ| = "
              f"{abs(n_active_as - n_active_fg)}")
    else:
        print(f"  Full-Gram IRLS:   SKIPPED (--skip-full-gram set)")

    # Estimate one-iter cost for projection
    print(f"\n  Per-iter timing (active-set, average): "
          f"{as_total / max(as_result.n_irls, 1):.1f}s")
    if fg_total is not None:
        print(f"  Per-iter timing (full-Gram, average):  "
              f"{fg_total / max(fg_result.n_irls, 1):.1f}s")


if __name__ == "__main__":
    main()
