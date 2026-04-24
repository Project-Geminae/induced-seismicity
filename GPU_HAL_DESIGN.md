# GPU HAL Backend — Scope & Design

## Objective

Build a GPU-accelerated Highly Adaptive Lasso (HAL) Lasso solver that
plugs into the existing `hal9001`-based pipeline. Goal: turn a 40-minute
CPU fit into a 30-second GPU fit, enabling full-panel regHAL-TMLE with
cluster bootstrap inference in hours instead of weeks.

## What HAL computes

Given covariates X (n × d) and outcome Y (n), HAL:

1. **Builds a sparse basis matrix Φ (n × p)** with one column per
   indicator basis function φ_j(x) = 1{x_j1 ≥ k_j1, x_j2 ≥ k_j2, ...}
   for each subset of covariates up to `max_degree` and each knot
   combination. For our panel: n=451,212, d=8, max_degree=2,
   num_knots=(25, 10) → p ≈ 2,000 active bases.

2. **Solves L1-penalized regression:**
        β̂(λ) = argmin_β (1/2n) ||Y − Φβ||² + λ ||β||₁

3. **Selects λ via k-fold CV**, typically 10-fold.

4. **Optionally undersmooths** by using λ_used = λ_cv · δ, δ ∈ (0, 1].

5. For prediction: Q̂(x) = Σ_j β̂_j φ_j(x).

## What to replace on GPU

**Current CPU bottleneck:** step 2, glmnet's coordinate descent along a
λ-path. ~40 minutes per fit on full n.

**Leave on CPU/R:**
- Step 1 (basis construction) — already fast in hal9001 (~1-2 min),
  GPU port not necessary
- Evaluating predictions at new data — cheap matrix-vector product

**Port to GPU:**
- Step 2 (Lasso solve along λ-path with warm starts)
- Step 3 (CV evaluation — each fold's solve is independent, can also parallelize across folds on GPU)

## Algorithm: FISTA with sparse tensors

Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle 2009)
is a natural choice for sparse Lasso on GPU:

    x_0 = y_0 = 0
    t_0 = 1
    for k = 1, 2, ...:
        g = ∇f(y_{k-1}) = -Φᵀ(Y − Φy_{k-1}) / n
        x_k = soft_threshold(y_{k-1} − L⁻¹ g, λ/L)
        t_k = (1 + sqrt(1 + 4 t_{k-1}²)) / 2
        y_k = x_k + ((t_{k-1} − 1) / t_k) (x_k − x_{k-1})

where L is the Lipschitz constant (spectral norm of ΦᵀΦ/n) and
soft_threshold(u, τ) = sign(u) max(|u| − τ, 0).

Key GPU operations:
- Φᵀ(Y − Φβ): two sparse matrix-vector products. cuSPARSE or
  jax.experimental.sparse.
- soft_threshold: elementwise, embarrassingly parallel
- Lipschitz constant L: one-time power iteration, ~1s

Convergence: FISTA is O(1/k²), meaning ~500 iterations for 1e-6 duality
gap on well-conditioned problems. Per iter: 2 SpMVs + 2 vector ops
≈ 10ms on RTX 3080 at p=2000, n=451k.

Projected total runtime:
- λ-path fit (50 λ values, warm-started): 50 × 500 iters × 10ms = 250 s
- 10-fold CV for λ selection: 10 × λ-path = 2500 s on one GPU, or
  parallel across GPUs for ~800 s = 13 min per HAL fit

That's down from 40 min CPU to 13 min GPU — ~3x speedup, not the 200x
initially projected. The gap is because FISTA is slower than coordinate
descent for sparse problems (CD converges in O(log 1/ε), FISTA in
O(1/sqrt(ε))). To get more speedup we'd need GPU CD, which is harder.

**Revised approach:** use FISTA as v1, measure real speed, optimize if
needed. Path to 200x speedup: GPU coordinate descent (1-2 weeks additional).

## Dependencies

- JAX with CUDA 12 support (`pip install -U "jax[cuda12_pip]"`)
- numpy, scipy.sparse (to bridge from hal9001)
- rpy2 (existing) to call hal9001 for basis construction
- R package hal9001 (existing)

## Module layout

    induced-seismicity/
    └── gpu_hal/
        ├── __init__.py
        ├── fista.py          # Core FISTA Lasso on JAX sparse arrays
        ├── cv.py             # k-fold CV with warm-started λ path
        ├── backend.py        # hal9001 basis extraction → JAX sparse
        ├── hal_fit.py        # End-to-end API: fit_hal_gpu(X, y, ...)
        └── tests/
            └── test_fista_vs_sklearn.py

## API contract

    from gpu_hal import fit_hal_gpu

    fit = fit_hal_gpu(
        X=AL,                    # (n, d) numpy array
        y=Y,                     # (n,) numpy array
        family="gaussian",       # "gaussian" or "binomial"
        max_degree=2,
        num_knots=(25, 10),
        smoothness_orders=1,
        lambda_grid=None,        # auto: log-spaced 100-point grid
        n_folds=10,
        undersmoothing=None,     # None or factor in (0, 1]
        device="cuda",
    )

    # Match hal9001 API where possible
    preds = fit.predict(X_new)
    lambda_cv = fit.lambda_cv
    lambda_used = fit.lambda_used
    coef_active = fit.coef_active  # only non-zero basis coefficients
    basis_list = fit.basis_list    # for downstream reuse

## Validation plan

**Phase 1 — synthetic data (p=100, n=1000):** compare FISTA output to
`sklearn.linear_model.LassoCV` point-for-point across 20 lambdas. Target:
|β_FISTA - β_sklearn|_∞ < 1e-4, |MSE_FISTA - MSE_sklearn|/MSE < 1%.

**Phase 2 — hal9001 basis matrix, moderate n=10k:** fit both CPU
(hal9001) and GPU (our code) on the same basis. Compare psi estimates
for the shift target parameter. Target: within 5% relative difference
on a representative synthetic problem.

**Phase 3 — full n=451k panel at r=7:** run once, verify total runtime
is in the 10-15 minute range and the point estimate matches the earlier
CPU-based regHAL-TMLE result (+4-5e-3). Then bootstrap B=100 via JAX
vmap across bootstrap samples on GPU.

**Phase 4 — full 20-radius regHAL-TMLE sweep with full-n GPU HAL**:
end-to-end timing and result comparison against the n=50k CPU regHAL.

## Risks & mitigation

1. **JAX sparse ops may be immature**. Mitigation: benchmark
   `jax.experimental.sparse.BCOO` vs scipy sparse on a small problem
   first. If slow, fall back to dense for moderate p.

2. **rpy2 basis extraction may be lossy**. We've used it successfully in
   the existing `reghal_tmle.py`; reuse that pattern.

3. **FISTA may not match glmnet numerically for very small λ.** FISTA
   is an approximation; glmnet has exact KKT conditions. Accept 1-2%
   relative error at the CV-selected λ as "close enough" since the
   TMLE ψ estimate is relatively insensitive to which λ is picked.

4. **GPU memory.** At n=451k and p=5k, Φ dense is 9 GB. Must use sparse.
   Our basis has ~3-5% density, so sparse COO is 0.5-1 GB — fine.

5. **Undersmoothing factor shifts the active set.** When we shift to a
   smaller λ, the coefficient vector becomes less sparse. The FISTA
   iteration count grows roughly linearly with the number of active
   coefficients; expect 1.5-2x slowdown at undersmoothed λ relative to
   CV-optimal λ.

## Success criteria

**Phase 1 pass:** FISTA output matches sklearn on synthetic Lasso.
**Phase 2 pass:** GPU HAL fit matches hal9001 on moderate-n HAL problem.
**Phase 3 pass:** Full-n r=7 fit in <30 min, psi within 10% of CPU regHAL
result.
**Phase 4 pass:** All 20 radii regHAL-TMLE with B=100 bootstrap complete
in <24 h total GPU time.

## Deferred scope

- GPU coordinate descent solver (alternative to FISTA, potentially 10x faster)
- Logistic HAL on GPU (needed for full hurdle; for now fall back to CPU
  for the logistic piece)
- Elastic net (α-parameter). Not used in HAL.
- Group Lasso. Not used in HAL.
- Python-native basis construction (stay in R via hal9001).

## Timeline

- Hour 0-2: FISTA core + dense synthetic validation
- Hour 2-4: Sparse support + hal9001 bridge
- Hour 4-6: CV λ-path with warm starts
- Hour 6-8: Undersmoothing factor + end-to-end API
- Hour 8-10: Benchmark on induced-seismicity panel
- Hour 10-12: Documentation + validation on r=7

Realistic: 1-2 days of focused work for a working v1. Polish (tests,
docs, open-source release) another 1 week.
