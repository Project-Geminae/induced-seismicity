"""GPU CD vs hal9001 along the full regularization path.

The previous diagnostic (diagnose_vs_hal9001.py) tested at hal9001's
CV-chosen λ — but on the n=50k zero-inflated subsample, hal9001 picks
λ_max with 0 active bases. Comparing GPU CD to hal9001 at λ_max is
vacuous (both produce null models that agree on floating-point noise).

This diagnostic instead picks intermediate λ values from hal9001's full
path where the model has *some* signal:
  - λ giving ~10 active bases
  - λ giving ~50 active bases
  - λ giving ~100 active bases (if path goes that low)

For each, runs GPU CD at the same λ and compares betas.

Strategy:
  1. fit_hal -> extract lasso_fit$lambda (full grid) and beta path
  2. Compute active count at each λ on the path
  3. For target active counts {10, 50, 100, 200}, find the closest λ
  4. Run GPU CD at that λ with all three scaling options
  5. Compare betas (rel L2, active-set overlap)

If GPU CD matches hal9001 within ~5% rel-L2 at intermediate λ values,
the solver is correct and the dashboard / hurdle gap is purely about
λ-selection (not standardization or design-matrix construction).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import causal_core as cc
from gpu_hal import fista_gram, cd_gram

import jax.numpy as jnp


def fit_hal_with_path(X, y, max_degree=2, num_knots=(25, 10), smoothness_orders=1,
                       lambda_min_ratio=None, n_lambdas=100):
    """Run hal9001::fit_hal and extract the full λ-path beta matrix.

    Parameters
    ----------
    lambda_min_ratio : float | None
        Override glmnet's default `lambda.min.ratio` (1e-4 when n>p, 1e-2 when
        n<p). On zero-inflated outcomes hal9001's default truncates the λ-path
        14 orders of magnitude above where any basis becomes active. Pass
        e.g. 1e-15 to extend the path far enough to find non-trivial signal.
    n_lambdas : int
        Length of the λ-grid. Default 100 (glmnet default).

    Returns
    -------
    basis_csr : scipy.sparse.csr_matrix
    lambdas : np.ndarray, shape (L,)
    betas : np.ndarray, shape (p, L)
    intercepts : np.ndarray, shape (L,)
    lambda_star : float
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    importr("hal9001")

    n, d = X.shape
    flat = ro.FloatVector(X.T.reshape(-1).tolist())
    r_X = ro.r["matrix"](flat, nrow=n, ncol=d)
    r_Y = ro.FloatVector(y.tolist())
    r_knots = ro.IntVector(list(num_knots))

    if lambda_min_ratio is not None:
        # Pass lambda.min.ratio + nlambda through fit_control to glmnet
        ctrl = ro.r(f"""
        list(cv_select = TRUE,
             use_min   = TRUE,
             lambda.min.ratio = {lambda_min_ratio},
             nlambda          = {n_lambdas})
        """)
    else:
        ctrl = ro.r("list(cv_select = TRUE)")

    fit_fn = ro.r("""
    function(X, Y, max_degree, num_knots, smoothness_orders, ctrl) {
        hal9001::fit_hal(X = X, Y = Y, family = "gaussian",
                         max_degree = max_degree,
                         num_knots = num_knots,
                         smoothness_orders = smoothness_orders,
                         fit_control = ctrl)
    }
    """)
    print(f"Running hal9001::fit_hal "
          f"(lambda_min_ratio={lambda_min_ratio}, nlambda={n_lambdas}) …")
    t0 = time.time()
    fit = fit_fn(r_X, r_Y, max_degree, r_knots, smoothness_orders, ctrl)
    print(f"  fit_hal complete in {time.time()-t0:.1f}s")

    lambda_star = float(ro.r("function(f) f$lambda_star")(fit)[0])

    # Extract full path of betas
    extract_path = ro.r("""
    function(f) {
        L <- f$lasso_fit$lambda
        # coef returns a sparse matrix (p+1) x length(L); densify
        B <- as.matrix(stats::coef(f$lasso_fit, s = L))
        list(lambdas = L, B = B)
    }
    """)
    parts = extract_path(fit)
    lambdas = np.asarray(parts.rx2("lambdas"), dtype=np.float64)
    B = np.asarray(parts.rx2("B"), dtype=np.float64)  # shape (p+1, L)
    intercepts = B[0, :]
    betas = B[1:, :]

    print(f"  λ-path: {len(lambdas)} values, range [{lambdas.min():.3e}, {lambdas.max():.3e}]")
    print(f"  λ_star (CV) = {lambda_star:.4e}")
    print(f"  active count at λ_star = {int(np.sum(np.abs(betas[:, np.argmin(np.abs(lambdas - lambda_star))]) > 1e-10))}")

    # Extract the basis matrix (CSC slots, no dense coercion)
    extract_basis = ro.r("""
    function(f, X) {
        m <- hal9001::make_design_matrix(X, f$basis_list)
        list(i = m@i, p = m@p, x = m@x,
             nrow = m@Dim[1], ncol = m@Dim[2])
    }
    """)
    parts = extract_basis(fit, r_X)
    indptr  = np.asarray(parts.rx2("p"), dtype=np.int64)
    indices = np.asarray(parts.rx2("i"), dtype=np.int64)
    data    = np.asarray(parts.rx2("x"), dtype=np.float64)
    nrow = int(parts.rx2("nrow")[0])
    ncol = int(parts.rx2("ncol")[0])
    from scipy.sparse import csc_matrix
    basis_csr = csc_matrix((data, indices, indptr), shape=(nrow, ncol)).tocsr()
    print(f"  basis: ({nrow} × {ncol}), nnz={basis_csr.nnz}, density={basis_csr.nnz/(nrow*ncol):.2%}")

    return basis_csr, lambdas, betas, intercepts, lambda_star


def gpu_cd_at_lambda(basis_csr, y, lam, scaling="unit_var", max_sweeps=2000, tol=1e-9):
    """Run GPU CD at a specified λ using the given scaling. Returns
    beta in the original (unscaled) basis."""
    n, p = basis_csr.shape
    col_sumsq = np.asarray(basis_csr.multiply(basis_csr).sum(axis=0)).ravel()
    if scaling == "unit_l2":
        s = np.sqrt(col_sumsq)
    elif scaling == "unit_var":
        s = np.sqrt(col_sumsq / n)
    elif scaling == "none":
        s = np.ones(p)
    else:
        raise ValueError(scaling)
    s = np.maximum(s, 1e-12)

    from scipy.sparse import diags
    D_inv = diags(1.0 / s)
    X_scaled = (basis_csr @ D_inv).tocsr()

    y_c = y - y.mean()
    G_np, Xty_np = fista_gram.compute_gram(X_scaled, y_c)
    G_j = jnp.asarray(G_np)
    Xty_j = jnp.asarray(Xty_np)

    t0 = time.time()
    res = cd_gram.cd_lasso_gram(G=G_j, Xty=Xty_j, lam=float(lam),
                                 max_sweeps=max_sweeps, tol=tol, verbose=False)
    elapsed = time.time() - t0

    beta_unscaled = np.asarray(res.beta) / s
    return beta_unscaled, res, elapsed


def compare_betas(beta_hal, beta_gpu, label):
    """Print rel-L2, max-abs-diff, and active-set overlap."""
    nz_hal = np.abs(beta_hal) > 1e-10
    nz_gpu = np.abs(beta_gpu) > 1e-10
    n_hal = int(nz_hal.sum())
    n_gpu = int(nz_gpu.sum())
    common = int((nz_hal & nz_gpu).sum())

    rel_l2 = np.linalg.norm(beta_gpu - beta_hal) / (np.linalg.norm(beta_hal) + 1e-30)
    max_diff = np.max(np.abs(beta_gpu - beta_hal))
    print(f"  [{label}]  active hal={n_hal:4d}  gpu={n_gpu:4d}  common={common:4d}  "
          f"rel-L2={rel_l2:.3e}  max-diff={max_diff:.3e}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives-only", action="store_true",
                    help="Restrict to Y>0 subset (matches hurdle stage 2). "
                         "Fixes the zero-inflation issue that breaks hal9001's "
                         "λ-grid construction.")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--max-n", type=int, default=50_000)
    ap.add_argument("--lambda-min-ratio", type=float, default=None,
                    help="Override glmnet's default λ_min/λ_max ratio (1e-4 when "
                         "n>p). Pass 1e-15 to force the path to extend far enough "
                         "for non-trivial signal on zero-inflated outcomes.")
    ap.add_argument("--n-lambdas", type=int, default=100)
    args = ap.parse_args()

    R = args.radius
    print(f"=== GPU CD vs hal9001 along λ-path (R={R} km, "
          f"max_n={args.max_n}, positives_only={args.positives_only}) ===\n")

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
        if rows + nc > args.max_n and rows > 0:
            break
        kept.append(c)
        rows += nc
    mask = np.isin(clusters_all, kept)
    sub = data.loc[mask].reset_index(drop=True)

    if args.positives_only:
        # Magnitude stage data: log(1+Y) on Y>0 subset
        sub = sub.loc[sub[S] > 0].reset_index(drop=True)
        X = sub[[W] + list(confs)].to_numpy(dtype=np.float64)
        y = np.log1p(sub[S].to_numpy(dtype=np.float64))
        print(f"  Y>0 subset: n={len(X)} (was {int((data[S]>0).sum())} positives in full data)")
    else:
        X = sub[[W] + list(confs)].to_numpy(dtype=np.float64)
        y = sub[S].to_numpy(dtype=np.float64)
    print(f"n={len(X)}, clusters={len(kept)}, positives={int((y>0).sum())}\n")

    basis_csr, lambdas, betas, intercepts, lambda_star = fit_hal_with_path(
        X, y, max_degree=2, num_knots=(25, 10), smoothness_orders=1,
        lambda_min_ratio=args.lambda_min_ratio,
        n_lambdas=args.n_lambdas,
    )

    # Active counts along the path
    active_counts = (np.abs(betas) > 1e-10).sum(axis=0)
    print(f"\n  active-count summary along path: min={active_counts.min()}, "
          f"max={active_counts.max()}, median={int(np.median(active_counts))}")

    # Pick target points: the λ on the path closest to having 10 / 50 / 100 active
    targets = [10, 50, 100, 200, 500]
    pick_idx = []
    for tgt in targets:
        if active_counts.max() < tgt:
            continue
        # Find smallest λ (i.e. largest active count up to tgt) that has >= tgt active
        idx = int(np.argmin(np.abs(active_counts - tgt)))
        pick_idx.append((tgt, idx))

    if not pick_idx:
        print("\n  ⚠ hal9001 path never reaches a non-trivial active count — diagnostic still vacuous.")
        print(f"  max active along path = {active_counts.max()}")
        # Fall back to whatever non-zero point we have
        nonzero_idx = np.where(active_counts > 0)[0]
        if len(nonzero_idx) > 0:
            pick_idx = [(int(active_counts[nonzero_idx[-1]]), int(nonzero_idx[-1]))]
            print(f"  falling back to λ index {pick_idx[0][1]} (active={pick_idx[0][0]})")
        else:
            print("  ⚠ entire path is null. Exiting.")
            return

    print(f"\n  testing at {len(pick_idx)} λ values along the path:")
    for tgt, idx in pick_idx:
        actual = int(active_counts[idx])
        print(f"    target {tgt:4d} active → λ[{idx:3d}] = {lambdas[idx]:.4e}, "
              f"actual active = {actual}")

    print()
    for tgt, idx in pick_idx:
        lam = float(lambdas[idx])
        beta_hal = betas[:, idx]
        n_active_hal = int((np.abs(beta_hal) > 1e-10).sum())
        print(f"\n--- λ index {idx} (target ~{tgt}, hal active = {n_active_hal}, λ = {lam:.4e}) ---")

        for scaling in ["unit_var", "unit_l2", "none"]:
            try:
                beta_gpu, res, elapsed = gpu_cd_at_lambda(
                    basis_csr, y, lam, scaling=scaling, max_sweeps=2000,
                )
                converged = "✓" if res.converged else "✗"
                print(f"  scaling={scaling:8s}  sweeps={res.n_sweeps:4d} "
                      f"gap={res.final_gap:.2e} {converged} ({elapsed:.1f}s)")
                compare_betas(beta_hal, beta_gpu, scaling)
            except Exception as e:
                print(f"  scaling={scaling}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
