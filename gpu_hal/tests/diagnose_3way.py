"""Three-way diagnostic to isolate where GPU hurdle HAL diverges from CPU regHAL.

Builds the same basis matrix three ways and runs Lasso three ways to
isolate basis-construction vs lambda-selection vs solver bugs:

  Pipeline A: hal9001::fit_hal end-to-end (CPU baseline)
  Pipeline B: our basis + cv.glmnet via rpy2 (basis check)
  Pipeline C: our basis + our CV + our GPU CD (current implementation)

If ψ(A) ≈ ψ(B): our basis matches hal9001's basis — the gap is in CV/lambda
   ψ(B) ≈ ψ(C): our solver matches glmnet at the same lambda — no solver bug

If ψ(A) ≠ ψ(B): basis or hal9001-specific preprocessing differs
If ψ(B) ≠ ψ(C): our solver disagrees with glmnet at same lambda
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import causal_core as cc

import jax
import jax.numpy as jnp
from gpu_hal import backend, fista_gram, cd_gram, cd_logistic


def get_data(R=7, max_n=50000):
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
        if rows + nc > max_n and rows > 0:
            break
        kept.append(c); rows += nc
    mask = np.isin(clusters_all, kept)
    sub = data.loc[mask].reset_index(drop=True)
    X = sub[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = sub[S].to_numpy(dtype=np.float64)
    return X, y


def pipeline_A_hal9001_full(X, y):
    """Full hal9001 pipeline using HurdleHAL (matches CPU regHAL)."""
    print("\n--- Pipeline A: hal9001 hurdle (CPU regHAL baseline) ---")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from undersmoothed_hal import HurdleHAL
    t0 = time.time()
    hal = HurdleHAL(max_degree=2, num_knots=(25, 10), smoothness_orders=1)
    hal.fit(X, y)
    elapsed = time.time() - t0
    A_post = X[:, 0] * 1.10
    X_post = X.copy(); X_post[:, 0] = A_post
    psi = float(np.mean(hal.predict(X_post) - hal.predict(X)))
    print(f"  hal9001 hurdle psi = {psi:+.4e}  ({elapsed:.0f}s)")
    return psi


def pipeline_B_ourbasis_glmnet(X, y):
    """Our basis enumeration + cv.glmnet via rpy2 for both stages.

    Tests: does our basis agree with hal9001's, given glmnet does the Lasso?
    """
    print("\n--- Pipeline B: our basis + cv.glmnet ---")
    t0 = time.time()
    phi_csr, basis_list = backend.build_hal_basis(
        X, max_degree=2, num_knots=(25, 10), smoothness_orders=1,
    )
    print(f"  basis: {phi_csr.shape}, nnz={phi_csr.nnz}")

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    glmnet = importr("glmnet")
    Matrix = importr("Matrix")
    n, p = phi_csr.shape

    # Convert scipy CSR to R sparseMatrix (dgCMatrix)
    phi_csc = phi_csr.tocsc()
    new_sparse = ro.r("function(i, p, x, dims) Matrix::sparseMatrix(i=i, p=p, x=x, dims=dims, index1=FALSE)")
    r_X = new_sparse(
        ro.IntVector(phi_csc.indices.astype(int)),
        ro.IntVector(phi_csc.indptr.astype(int)),
        ro.FloatVector(phi_csc.data),
        ro.IntVector(list(phi_csc.shape)),
    )

    # Stage 1: logistic via cv.glmnet
    is_pos = (y > 0).astype(float)
    cv_log = ro.r("""
    function(X, Y) {
        glmnet::cv.glmnet(X, Y, family="binomial", standardize=TRUE,
                          intercept=TRUE, nfolds=10)
    }
    """)
    print(f"  Stage 1 (logistic) cv.glmnet on n={n}, p={p}, positives={int(is_pos.sum())} ...")
    fit1 = cv_log(r_X, ro.FloatVector(is_pos.tolist()))
    lam1 = float(ro.r("function(f) f$lambda.min")(fit1)[0])
    coef1 = np.asarray(ro.r("function(f) as.numeric(coef(f, s='lambda.min'))")(fit1))
    int_pos = float(coef1[0])
    beta_pos = coef1[1:]
    print(f"    λ_min = {lam1:.4e}, intercept = {int_pos:.4e}, active = {int(np.sum(np.abs(beta_pos)>1e-10))}")

    # Stage 2: gaussian on positives via cv.glmnet
    pos_mask = y > 0
    if pos_mask.sum() < 50:
        print("  Stage 2 skipped (too few positives).")
        return float("nan")
    phi_pos = phi_csc[pos_mask]
    new_sparse_pos = new_sparse
    r_X_pos = new_sparse(
        ro.IntVector(phi_pos.indices.astype(int)),
        ro.IntVector(phi_pos.indptr.astype(int)),
        ro.FloatVector(phi_pos.data),
        ro.IntVector(list(phi_pos.shape)),
    )
    cv_gauss = ro.r("""
    function(X, Y) {
        glmnet::cv.glmnet(X, Y, family="gaussian", standardize=TRUE,
                          intercept=TRUE, nfolds=10)
    }
    """)
    y_pos = np.log1p(y[pos_mask])
    print(f"  Stage 2 (gaussian) cv.glmnet on positives n={int(pos_mask.sum())} ...")
    fit2 = cv_gauss(r_X_pos, ro.FloatVector(y_pos.tolist()))
    lam2 = float(ro.r("function(f) f$lambda.min")(fit2)[0])
    coef2 = np.asarray(ro.r("function(f) as.numeric(coef(f, s='lambda.min'))")(fit2))
    int_mag = float(coef2[0])
    beta_mag = coef2[1:]
    print(f"    λ_min = {lam2:.4e}, intercept = {int_mag:.4e}, active = {int(np.sum(np.abs(beta_mag)>1e-10))}")

    # Predict using these betas (no rescaling — glmnet already absorbs it)
    A_post = X[:, 0] * 1.10
    X_post = X.copy(); X_post[:, 0] = A_post
    phi_obs = phi_csr
    phi_post = backend.apply_basis(X_post, basis_list).tocsr()

    eta_pos_obs = phi_obs @ beta_pos + int_pos
    p_obs = 1.0 / (1.0 + np.exp(-np.clip(eta_pos_obs, -50, 50)))
    eta_pos_post = phi_post @ beta_pos + int_pos
    p_post = 1.0 / (1.0 + np.exp(-np.clip(eta_pos_post, -50, 50)))
    log_mag_obs = phi_obs @ beta_mag + int_mag
    log_mag_post = phi_post @ beta_mag + int_mag
    Q_obs = p_obs * np.expm1(log_mag_obs)
    Q_post = p_post * np.expm1(log_mag_post)
    psi = float(np.mean(Q_post - Q_obs))
    elapsed = time.time() - t0
    print(f"  Pipeline B psi = {psi:+.4e}  ({elapsed:.0f}s)")
    return psi


def main():
    print("=== Three-way diagnostic at R=7, n=50k ===")
    X, y = get_data(R=7, max_n=50000)
    print(f"n={len(X)}, positives={int((y>0).sum())}\n")

    psi_A = pipeline_A_hal9001_full(X, y)
    psi_B = pipeline_B_ourbasis_glmnet(X, y)

    print("\n=== Summary ===")
    print(f"  A (hal9001 hurdle, CPU regHAL):              psi = {psi_A:+.4e}")
    print(f"  B (our basis + cv.glmnet):                   psi = {psi_B:+.4e}")
    print(f"  C (our basis + our CV + GPU CD, prior run):  psi = +1.76e-3")
    print(f"\n  A vs B ratio: {psi_B / psi_A if abs(psi_A) > 0 else float('nan'):.3f}")
    print(f"  A vs C ratio: {1.76e-3 / psi_A if abs(psi_A) > 0 else float('nan'):.3f}")


if __name__ == "__main__":
    main()
