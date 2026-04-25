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

    # Build the sparse design matrix in R, then extract its dgCMatrix
    # slots WITHOUT a dense coercion. as.matrix() on a sparse Matrix at
    # n=451k can need 5+ GB; we avoid that by reading the CSC triple
    # (i = row indices, p = column pointers, x = values, Dim = shape).
    extract_fn = ro.r("""
    function(X, blist) {
        m <- hal9001::make_design_matrix(X, blist)
        list(i = m@i, p = m@p, x = m@x,
             nrow = m@Dim[1], ncol = m@Dim[2])
    }
    """)
    parts = extract_fn(r_X, basis_list)
    # rpy2 returns these as R vectors; numpy.asarray copies into Python.
    indptr  = np.asarray(parts.rx2("p"), dtype=np.int64)
    indices = np.asarray(parts.rx2("i"), dtype=np.int64)
    data    = np.asarray(parts.rx2("x"), dtype=np.float64)
    nrow    = int(parts.rx2("nrow")[0])
    ncol    = int(parts.rx2("ncol")[0])
    # dgCMatrix is column-compressed (CSC). Build CSC then convert to CSR.
    from scipy.sparse import csc_matrix
    basis_csc = csc_matrix((data, indices, indptr), shape=(nrow, ncol))
    basis_csr = basis_csc.tocsr()
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
    # Same dense-allocation avoidance as build_hal_basis.
    extract_fn = ro.r("""
    function(X, blist) {
        m <- hal9001::make_design_matrix(X, blist)
        list(i = m@i, p = m@p, x = m@x,
             nrow = m@Dim[1], ncol = m@Dim[2])
    }
    """)
    parts = extract_fn(r_X, basis_list)
    indptr  = np.asarray(parts.rx2("p"), dtype=np.int64)
    indices = np.asarray(parts.rx2("i"), dtype=np.int64)
    data    = np.asarray(parts.rx2("x"), dtype=np.float64)
    nrow    = int(parts.rx2("nrow")[0])
    ncol    = int(parts.rx2("ncol")[0])
    from scipy.sparse import csc_matrix
    return csc_matrix((data, indices, indptr), shape=(nrow, ncol)).tocsr()
