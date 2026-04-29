"""CPU regHAL baseline with explicit glmnet λ-grid.

Closes the +4.02e-3 vs +1.76e-3 gap by feeding glmnet an explicit
log-spaced λ-vector that extends below where its dev.ratio early-stop
would normally truncate the path.

Pipeline:
  1. Build basis via gpu_hal.backend.build_hal_basis (same as GPU pipeline)
  2. Stage 1 (binomial on 1{Y>0}): cv.glmnet with explicit λ-grid
  3. Stage 2 (gaussian on log(1+Y)|Y>0): cv.glmnet with explicit λ-grid
  4. Compose hurdle prediction P̂(X)·expm1(M̂(X))
  5. Compute ψ = E[Q(X·1.10) − Q(X)] for shift +10%

Compare to:
  - hal9001 default-grid HurdleHAL: ~ +3-4e-3 (today's diagnostic gave +3.08e-3)
  - GPU hurdle CPU pipeline (Apr 25): +1.76e-3 (227+134 active bases)

Expected outcome: with the same λ-region as GPU, glmnet will pick
similar λ values and produce a similar ψ. If yes → both pipelines
agree, the gap was λ-grid; gpu_hal is fully validated.

If no (CPU-with-explicit-grid still differs from GPU) → there's a
real difference in CV scoring or basis enumeration between the two
pipelines that needs further investigation.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import causal_core as cc
from gpu_hal import backend


def get_data(R=7, max_n=50_000, seed=42):
    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=365)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=365)
    data = data.copy()
    data["_cluster"] = cluster.values

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
    sub = data.loc[mask].reset_index(drop=True)
    X = sub[[W] + list(confs)].to_numpy(dtype=np.float64)
    y = sub[S].to_numpy(dtype=np.float64)
    cluster_vec = sub["_cluster"].to_numpy()
    return X, y, len(kept), cluster_vec


def cluster_foldid(cluster_vec: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each row to a CV fold based on its cluster. Returns 1-indexed
    integer vector of length len(cluster_vec). All rows of a given cluster
    are assigned the same fold."""
    rng = np.random.default_rng(seed)
    unique = np.unique(cluster_vec)
    rng.shuffle(unique)
    cluster_to_fold = {c: int(i % n_folds) + 1 for i, c in enumerate(unique)}
    return np.array([cluster_to_fold[c] for c in cluster_vec], dtype=np.int64)


def run_glmnet_cv(r_X, y, family, lam_grid, ro, foldid=None, nfolds=5):
    """Fit cv.glmnet with explicit λ-grid; standardize=FALSE because we
    pre-scale the basis to unit column-norm (matches GPU convention).

    Pass `foldid` (1-indexed integer vector) to use cluster-aware folds.
    """
    if foldid is not None:
        cv_fn = ro.r("""
        function(X, Y, lam, family, fid) {
            glmnet::cv.glmnet(X, Y,
                              family      = family,
                              standardize = FALSE,
                              intercept   = TRUE,
                              foldid      = fid,
                              lambda      = lam)
        }
        """)
        r_lam = ro.FloatVector(lam_grid.tolist())
        r_y   = ro.FloatVector(np.asarray(y, dtype=float).tolist())
        r_fid = ro.IntVector(np.asarray(foldid, dtype=int).tolist())
        fit = cv_fn(r_X, r_y, r_lam, family, r_fid)
    else:
        cv_fn = ro.r("""
        function(X, Y, lam, family, nf) {
            glmnet::cv.glmnet(X, Y,
                              family      = family,
                              standardize = FALSE,
                              intercept   = TRUE,
                              nfolds      = nf,
                              lambda      = lam)
        }
        """)
        r_lam = ro.FloatVector(lam_grid.tolist())
        r_y   = ro.FloatVector(np.asarray(y, dtype=float).tolist())
        r_nf  = ro.IntVector([int(nfolds)])
        fit = cv_fn(r_X, r_y, r_lam, family, r_nf)
    lam_min = float(ro.r("function(f) f$lambda.min")(fit)[0])
    coef = np.asarray(ro.r("function(f) as.numeric(coef(f, s='lambda.min'))")(fit))
    intercept, beta = float(coef[0]), coef[1:]
    n_active = int(np.sum(np.abs(beta) > 1e-10))
    return lam_min, intercept, beta, n_active, fit


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--max-n", type=int, default=50_000)
    ap.add_argument("--lambda-min", type=float, default=1e-8,
                    help="Smallest λ in the explicit grid (default 1e-8)")
    ap.add_argument("--lambda-max", type=float, default=1e+2,
                    help="Largest λ in the explicit grid (default 1e+2)")
    ap.add_argument("--n-lambdas", type=int, default=100)
    ap.add_argument("--n-folds", type=int, default=5,
                    help="CV folds in cv.glmnet. GPU pipeline uses 3.")
    ap.add_argument("--folds-mode", choices=["cluster", "random"], default="cluster",
                    help="cluster: cluster_foldid assigns each well to a fold "
                         "(rows of the same well are kept together). "
                         "random: cv.glmnet's default random row-level folds.")
    ap.add_argument("--shift-pct", type=float, default=0.10)
    ap.add_argument("--shift-mode", choices=["multiplicative", "calibrated"],
                    default="multiplicative",
                    help="multiplicative: A_post = (1-shift_pct)·A. "
                         "calibrated: A_post = max((1-shift_pct)·A, q_α(A)); "
                         "implied-interventions framework, García Meixide & vdL "
                         "2025 (arXiv:2506.21501).")
    ap.add_argument("--percentile-floor", type=float, default=0.05,
                    help="α for calibrated-shift floor q_α(A) (default 0.05).")
    args = ap.parse_args()

    R = args.radius
    print(f"=== Custom-grid CPU baseline at R={R}km, max_n={args.max_n} ===")
    print(f"    λ-grid: log-spaced [{args.lambda_min:.1e}, {args.lambda_max:.1e}], "
          f"{args.n_lambdas} points\n")

    X, y, n_clusters, cluster_vec = get_data(R, args.max_n)
    print(f"n={len(X)}, clusters={n_clusters}, positives={int((y>0).sum())}")
    if args.folds_mode == "cluster":
        foldid_full = cluster_foldid(cluster_vec, n_folds=args.n_folds, seed=42)
        print(f"  cluster-aware foldid: {n_clusters} clusters → {args.n_folds} folds, "
              f"sizes per fold: "
              f"{np.bincount(foldid_full)[1:].tolist()}\n")
    else:
        foldid_full = None  # fall back to cv.glmnet's random row-level folds
        print(f"  random row-level folds (n_folds={args.n_folds})\n")

    print("Building HAL basis (same as GPU pipeline) …")
    t0 = time.time()
    phi_csr, basis_list = backend.build_hal_basis(
        X, max_degree=2, num_knots=(25, 10), smoothness_orders=1,
    )
    print(f"  basis: {phi_csr.shape}, nnz={phi_csr.nnz}, "
          f"density={phi_csr.nnz/(phi_csr.shape[0]*phi_csr.shape[1]):.2%} "
          f"({time.time()-t0:.0f}s)\n")

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    importr("glmnet"); importr("Matrix")

    # ── Pre-scale basis to unit column L2-norm (matches GPU pipeline) ──
    # GPU: phi_scaled = phi @ diag(1/col_norms), CD on phi_scaled with
    #      no internal standardization. We reproduce that exactly so the
    #      same λ values give comparable solutions across implementations.
    from scipy.sparse import diags
    col_sumsq = np.asarray(phi_csr.multiply(phi_csr).sum(axis=0)).ravel()
    col_norms = np.sqrt(np.maximum(col_sumsq, 1e-24))
    D_inv = diags(1.0 / col_norms)
    phi_scaled_csr = (phi_csr @ D_inv).tocsr()
    phi_scaled_csc = phi_scaled_csr.tocsc()
    print(f"  basis scaled: col_norms range [{col_norms.min():.2e}, "
          f"{col_norms.max():.2e}], median {np.median(col_norms):.2e}\n")

    # Convert scaled basis to dgCMatrix for glmnet
    new_sparse = ro.r(
        "function(i, p, x, dims) Matrix::sparseMatrix(i=i, p=p, x=x, dims=dims, index1=FALSE)"
    )
    r_X = new_sparse(
        ro.IntVector(phi_scaled_csc.indices.astype(int)),
        ro.IntVector(phi_scaled_csc.indptr.astype(int)),
        ro.FloatVector(phi_scaled_csc.data),
        ro.IntVector(list(phi_scaled_csc.shape)),
    )

    # Explicit log-spaced λ-grid (decreasing, glmnet convention)
    lam_grid = np.logspace(
        np.log10(args.lambda_max), np.log10(args.lambda_min), args.n_lambdas,
    )

    # ── Stage 1: binomial on 1{Y>0} ──
    is_pos = (y > 0).astype(float)
    fold_label = "cluster-aware" if args.folds_mode == "cluster" else "random"
    print(f"Stage 1 (logistic, n={len(y)}, positives={int(is_pos.sum())}) "
          f"with {fold_label} folds (nfolds={args.n_folds}) …")
    t0 = time.time()
    lam1, int_pos, beta_pos, na1, fit1 = run_glmnet_cv(
        r_X, is_pos, "binomial", lam_grid, ro,
        foldid=foldid_full, nfolds=args.n_folds,
    )
    print(f"  λ_min = {lam1:.4e}, intercept = {int_pos:.4e}, active = {na1} "
          f"({time.time()-t0:.0f}s)")
    print(f"    GPU baseline (Apr 25):  λ = 1.77e-7, active = 227")

    # ── Stage 2: gaussian on log(1+Y)|Y>0 ──
    pos_mask = y > 0
    phi_pos = phi_scaled_csc[pos_mask]
    r_X_pos = new_sparse(
        ro.IntVector(phi_pos.indices.astype(int)),
        ro.IntVector(phi_pos.indptr.astype(int)),
        ro.FloatVector(phi_pos.data),
        ro.IntVector(list(phi_pos.shape)),
    )
    y_pos = np.log1p(y[pos_mask])
    if args.folds_mode == "cluster":
        foldid_pos = cluster_foldid(cluster_vec[pos_mask], n_folds=args.n_folds, seed=42)
        print(f"\nStage 2 (gaussian, n={int(pos_mask.sum())}) with cluster-aware folds "
              f"(sizes: {np.bincount(foldid_pos)[1:].tolist()}) …")
    else:
        foldid_pos = None
        print(f"\nStage 2 (gaussian, n={int(pos_mask.sum())}) with random folds …")
    t0 = time.time()
    lam2, int_mag, beta_mag, na2, fit2 = run_glmnet_cv(
        r_X_pos, y_pos, "gaussian", lam_grid, ro,
        foldid=foldid_pos, nfolds=args.n_folds,
    )
    print(f"  λ_min = {lam2:.4e}, intercept = {int_mag:.4e}, active = {na2} "
          f"({time.time()-t0:.0f}s)")
    print(f"    GPU baseline (Apr 25):  λ = 1.99e-6, active = 134")

    # ── Compose hurdle and compute ψ ──
    base_factor = 1.0 + args.shift_pct
    if args.shift_mode == "multiplicative":
        print(f"\nComposing hurdle prediction (shift_mode=multiplicative, "
              f"factor={base_factor:.3f}) …")
        A_post = X[:, 0] * base_factor
        n_clipped = 0
    else:
        a_floor = float(np.quantile(X[:, 0], args.percentile_floor))
        A_unrestricted = X[:, 0] * base_factor
        A_post = np.maximum(A_unrestricted, a_floor)
        n_clipped = int((A_post != A_unrestricted).sum())
        print(f"\nComposing hurdle prediction (shift_mode=calibrated, "
              f"factor={base_factor:.3f}, q_{args.percentile_floor:.2f}={a_floor:.3e}) …")
        print(f"  clipped {n_clipped}/{len(A_post)} "
              f"({100*n_clipped/len(A_post):.1f}%) rows at the calibrated floor")
    X_post = X.copy(); X_post[:, 0] = A_post
    phi_obs_scaled  = phi_scaled_csr
    phi_post_raw    = backend.apply_basis(X_post, basis_list).tocsr()
    phi_post_scaled = (phi_post_raw @ D_inv).tocsr()

    eta_pos_obs  = phi_obs_scaled  @ beta_pos + int_pos
    eta_pos_post = phi_post_scaled @ beta_pos + int_pos
    p_obs  = 1.0 / (1.0 + np.exp(-np.clip(eta_pos_obs,  -50, 50)))
    p_post = 1.0 / (1.0 + np.exp(-np.clip(eta_pos_post, -50, 50)))

    log_mag_obs  = phi_obs_scaled  @ beta_mag + int_mag
    log_mag_post = phi_post_scaled @ beta_mag + int_mag
    print(f"  log_mag range: obs=[{log_mag_obs.min():+.2f}, {log_mag_obs.max():+.2f}], "
          f"post=[{log_mag_post.min():+.2f}, {log_mag_post.max():+.2f}]")
    # Clip the linear predictor before expm1 to prevent OOD blow-up.
    # 30 corresponds to expm1(30) ≈ 1.07e+13, well above any reasonable Y in ML.
    log_mag_obs  = np.clip(log_mag_obs,  -30, 30)
    log_mag_post = np.clip(log_mag_post, -30, 30)
    mag_obs  = np.expm1(log_mag_obs)
    mag_post = np.expm1(log_mag_post)

    Q_obs  = p_obs  * mag_obs
    Q_post = p_post * mag_post
    psi_total = float(np.mean(Q_post - Q_obs))

    psi_freq  = float(np.mean((p_post - p_obs) * mag_obs))
    psi_mag   = float(np.mean(p_obs * (mag_post - mag_obs)))
    psi_cross = float(np.mean((p_post - p_obs) * (mag_post - mag_obs)))

    print(f"\n=== RESULT (custom-grid CPU baseline, shift_mode={args.shift_mode}) ===")
    print(f"  n_folds={args.n_folds}, shift_pct={args.shift_pct}, "
          f"percentile_floor={args.percentile_floor if args.shift_mode == 'calibrated' else 'n/a'}")
    if args.shift_mode == "calibrated":
        print(f"  n_clipped at floor: {n_clipped}/{len(X)} "
              f"({100*n_clipped/len(X):.1f}%)")
    print(f"  Stage 1 (logistic): λ={lam1:.4e}, active={na1}")
    print(f"  Stage 2 (gaussian): λ={lam2:.4e}, active={na2}")
    print(f"  ψ_total      = {psi_total:+.4e}")
    print(f"  ψ_frequency  = {psi_freq:+.4e}  ({psi_freq/psi_total*100:+.0f}%)")
    print(f"  ψ_magnitude  = {psi_mag:+.4e}  ({psi_mag/psi_total*100:+.0f}%)")
    print(f"  ψ_cross      = {psi_cross:+.4e}  ({psi_cross/psi_total*100:+.0f}%)")
    print(f"\n  Comparison:")
    print(f"    GPU hurdle (Apr 25):              ψ = +1.76e-3, channels 54/34/12")
    print(f"    CPU HurdleHAL default grid (today): ψ = +3.08e-3, channels not captured")
    print(f"    CPU custom grid (this run):         ψ = {psi_total:+.4e}, "
          f"channels {psi_freq/psi_total*100:.0f}/{psi_mag/psi_total*100:.0f}/{psi_cross/psi_total*100:.0f}")


if __name__ == "__main__":
    main()
