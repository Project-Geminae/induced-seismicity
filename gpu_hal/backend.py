"""
hal9001 bridge: extract a HAL basis matrix from R and expose it as a
scipy sparse CSR for the GPU solver. Uses rpy2 (same pattern as
reghal_tmle.py).

Function `build_hal_basis(X, max_degree, num_knots, smoothness_orders)`
fits a 1-iteration hal9001 that only constructs the basis (doesn't solve
the Lasso), extracts the basis matrix and basis list, and returns
(csr_matrix, basis_list) for downstream use.

After GPU Lasso solves for β, call `apply_basis(X_new, basis_list)` to
evaluate at new data for predictions.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def _init_r():
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    hal9001 = importr("hal9001")
    base = importr("base")
    return ro, hal9001, base


def _to_r_mat(ro, arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    flat = ro.FloatVector(arr.T.reshape(-1).tolist())
    return ro.r["matrix"](flat, nrow=arr.shape[0], ncol=arr.shape[1])


def build_hal_basis(
    X: np.ndarray,
    max_degree: int = 2,
    num_knots: tuple = (25, 10),
    smoothness_orders: int = 1,
):
    """Build HAL basis matrix for X using hal9001's enumerate + make_basis.

    Returns:
        basis_csr: scipy.sparse.csr_matrix of shape (n, p)
        basis_list: R object (keep for prediction at new data)
    """
    from scipy.sparse import csr_matrix

    ro, hal9001, base = _init_r()
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    r_X = _to_r_mat(ro, X)
    r_knots = ro.IntVector(list(num_knots))

    # enumerate_basis + make_design_matrix — get the basis matrix without
    # running the Lasso. This is the hal9001 idiom for building basis only.
    enumerate_fn = ro.r("""
    function(X, max_degree, num_knots, smoothness_orders) {
        hal9001::enumerate_basis(X, max_degree = max_degree,
                                 num_knots = num_knots,
                                 smoothness_orders = smoothness_orders)
    }
    """)
    basis_list = enumerate_fn(r_X, max_degree, r_knots, smoothness_orders)

    design_fn = ro.r("function(X, blist) as.matrix(hal9001::make_design_matrix(X, blist))")
    r_design = design_fn(r_X, basis_list)
    design = np.asarray(r_design)
    basis_csr = csr_matrix(design)

    return basis_csr, basis_list


def apply_basis(
    X_new: np.ndarray,
    basis_list,
):
    """Evaluate the HAL basis at new data points using the stored basis_list.

    Returns scipy.sparse.csr_matrix of shape (n_new, p).
    """
    from scipy.sparse import csr_matrix

    ro, hal9001, base = _init_r()
    X_new = np.asarray(X_new, dtype=float)
    r_X = _to_r_mat(ro, X_new)
    design_fn = ro.r("function(X, blist) as.matrix(hal9001::make_design_matrix(X, blist))")
    r_design = design_fn(r_X, basis_list)
    return csr_matrix(np.asarray(r_design))
