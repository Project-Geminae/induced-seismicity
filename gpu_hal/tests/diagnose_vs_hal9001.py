"""Isolate solver bug from CV bug.

Strategy:
  1. Call hal9001::fit_hal on the same basis matrix our pipeline builds.
  2. Extract hal9001's chosen lambda and the resulting beta.
  3. Call our GPU CD at THAT exact lambda on the same basis.
  4. Compare betas.

Outcomes:
  - If betas match (within ~1%): our solver is fine, the discrepancy is
    in CV/lambda selection. Fix is to match hal9001's CV.
  - If betas differ substantially: our solver is solving a different
    problem than glmnet. Likely standardization mismatch.

Also run with both our standardization (unit-L2-norm) and glmnet-style
(unit-variance) to see which matches.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import causal_core as cc
from gpu_hal import backend, fista_gram, cd_gram

import jax.numpy as jnp


def get_hal9001_fit(X, y, max_degree=2, num_knots=(25, 10), smoothness_orders=1):
    """Run hal9001::fit_hal on (X, y) and return (basis_csr, basis_list, lambda_cv, beta_hal)."""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    hal9001 = importr("hal9001")
    base = importr("base")

    n, d = X.shape
    flat = ro.FloatVector(X.T.reshape(-1).tolist())
    r_X = ro.r["matrix"](flat, nrow=n, ncol=d)
    r_Y = ro.FloatVector(y.tolist())
    r_knots = ro.IntVector(list(num_knots))

    # Run fit_hal — has its own internal CV
    fit_fn = ro.r("""
    function(X, Y, max_degree, num_knots, smoothness_orders) {
        hal9001::fit_hal(X = X, Y = Y, family = "gaussian",
                         max_degree = max_degree,
                         num_knots = num_knots,
                         smoothness_orders = smoothness_orders,
                         fit_control = list(cv_select = TRUE))
    }
    """)
    print("Running hal9001::fit_hal on full pipeline (this is our CPU baseline)...")
    t0 = time.time()
    fit = fit_fn(r_X, r_Y, max_degree, r_knots, smoothness_orders)
    print(f"  hal9001 fit_hal complete in {time.time()-t0:.1f}s")

    # Extract lambda_cv
    lambda_cv = float(ro.r("function(f) f$lambda_star")(fit)[0])
    print(f"  hal9001 lambda_star = {lambda_cv:.4e}")

    # Extract beta at lambda_star
    beta_hal_raw = ro.r("function(f) as.numeric(stats::coef(f$lasso_fit, s=f$lambda_star))")(fit)
    beta_hal = np.asarray(beta_hal_raw)
    # Position 0 is intercept; positions 1..p are coefficients
    intercept_hal = float(beta_hal[0])
    beta_hal_coef = beta_hal[1:]
    n_active_hal = int(np.sum(np.abs(beta_hal_coef) > 1e-10))
    print(f"  hal9001 active bases: {n_active_hal}/{len(beta_hal_coef)}")
    print(f"  hal9001 intercept: {intercept_hal:.4e}")

    # Get the basis matrix
    extract_fn = ro.r("""
    function(f, X) {
        m <- hal9001::make_design_matrix(X, f$basis_list)
        list(i = m@i, p = m@p, x = m@x,
             nrow = m@Dim[1], ncol = m@Dim[2])
    }
    """)
    parts = extract_fn(fit, r_X)
    indptr = np.asarray(parts.rx2("p"), dtype=np.int64)
    indices = np.asarray(parts.rx2("i"), dtype=np.int64)
    data = np.asarray(parts.rx2("x"), dtype=np.float64)
    nrow = int(parts.rx2("nrow")[0])
    ncol = int(parts.rx2("ncol")[0])
    from scipy.sparse import csc_matrix
    basis_csc = csc_matrix((data, indices, indptr), shape=(nrow, ncol))
    basis_csr = basis_csc.tocsr()
    print(f"  hal9001 basis: ({nrow} × {ncol}), nnz={basis_csr.nnz}")

    return basis_csr, fit, lambda_cv, beta_hal_coef, intercept_hal


def gpu_cd_at_lambda(basis_csr, y, lam, scaling="unit_l2", max_sweeps=2000, tol=1e-9):
    """Run GPU CD at a specified lambda using the given scaling."""
    n, p = basis_csr.shape

    # Compute column scaling
    col_sumsq = np.asarray(basis_csr.multiply(basis_csr).sum(axis=0)).ravel()
    if scaling == "unit_l2":
        s = np.sqrt(col_sumsq)            # ||col||_2 = 1 after scaling
    elif scaling == "unit_var":
        s = np.sqrt(col_sumsq / n)        # std without centering = 1 (glmnet default)
    elif scaling == "none":
        s = np.ones(p)
    else:
        raise ValueError(scaling)
    s = np.maximum(s, 1e-12)

    from scipy.sparse import diags
    D_inv = diags(1.0 / s)
    X_scaled = (basis_csr @ D_inv).tocsr()

    # Center y, build Gram
    y_c = y - y.mean()
    G_np, Xty_np = fista_gram.compute_gram(X_scaled, y_c)
    G_j = jnp.asarray(G_np)
    Xty_j = jnp.asarray(Xty_np)

    print(f"  CD with scaling={scaling}, lam={lam:.4e} ...")
    t0 = time.time()
    res = cd_gram.cd_lasso_gram(G=G_j, Xty=Xty_j, lam=float(lam),
                                 max_sweeps=max_sweeps, tol=tol, verbose=False)
    print(f"    {res.n_sweeps} sweeps, gap={res.final_gap:.2e}, "
          f"converged={res.converged}, {time.time()-t0:.1f}s")

    # Unscale back to original basis
    beta_unscaled = np.asarray(res.beta) / s
    n_active = int(np.sum(np.abs(beta_unscaled) > 1e-10))
    return beta_unscaled, n_active, res


def main():
    R = 7
    print(f"=== Isolating GPU CD vs hal9001 mismatch at R={R}km, n=50k ===\n")

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

    # Cluster-aware subsample to 50k
    rng = np.random.default_rng(42)
    clusters_all = data["_cluster"].values
    unique_clusters = np.unique(clusters_all)
    rng.shuffle(unique_clusters)
    kept, rows = [], 0
    for c in unique_clusters:
        nc = int((clusters_all == c).sum())
        if rows + nc > 50000 and rows > 0:
            break
        kept.append(c)
        rows += nc
    mask = np.isin(clusters_all, kept)
    sub = data.loc[mask].reset_index(drop=True)
    X = sub[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = sub[S].to_numpy(dtype=np.float64)
    print(f"n={len(X)}, clusters={len(kept)}, positives={int((y>0).sum())}\n")

    # Step 1: Run hal9001 to get the canonical solution
    basis_csr, hal_fit, lambda_cv, beta_hal, intercept_hal = get_hal9001_fit(
        X, y, max_degree=2, num_knots=(25, 10), smoothness_orders=1,
    )

    print(f"\nhal9001 baseline:")
    print(f"  λ_cv     = {lambda_cv:.4e}")
    print(f"  ||β||_1  = {np.sum(np.abs(beta_hal)):.4e}")
    print(f"  ||β||_2  = {np.linalg.norm(beta_hal):.4e}")
    print(f"  active   = {int(np.sum(np.abs(beta_hal) > 1e-10))}")

    # Step 2: Run GPU CD at hal9001's lambda with each scaling option
    for scaling in ["unit_l2", "unit_var", "none"]:
        print(f"\n--- Scaling: {scaling} ---")
        beta_gpu, n_active, _ = gpu_cd_at_lambda(
            basis_csr, y, lambda_cv, scaling=scaling, max_sweeps=2000,
        )
        # Compare betas
        rel_l2 = np.linalg.norm(beta_gpu - beta_hal) / (np.linalg.norm(beta_hal) + 1e-30)
        max_diff = np.max(np.abs(beta_gpu - beta_hal))
        print(f"  GPU CD: active={n_active}, ||β||_1={np.sum(np.abs(beta_gpu)):.4e}")
        print(f"  vs hal9001: rel L2 = {rel_l2:.3e}, max diff = {max_diff:.3e}")
        # Active set agreement
        active_hal = set(np.where(np.abs(beta_hal) > 1e-10)[0])
        active_gpu = set(np.where(np.abs(beta_gpu) > 1e-10)[0])
        common = len(active_hal & active_gpu)
        print(f"  active set: {common} common, {len(active_hal-active_gpu)} hal-only, {len(active_gpu-active_hal)} gpu-only")


if __name__ == "__main__":
    main()
