# gpu-hal

**GPU-accelerated Highly Adaptive Lasso (HAL) for population-level causal inference at scale.**

Built on JAX. Validates against `sklearn.linear_model.Lasso` to 7-digit
precision. Tested at n=50,000 with p ≈ 1,500 active basis functions on
a single RTX 3080 (10 GB).

This is companion software for the induced-seismicity causal-inference
paper (Matthews 2025, SPE-228051). It is **not** a full replacement for
[hal9001](https://github.com/tlverse/hal9001) — it integrates with
hal9001 for basis enumeration and targets the bottleneck (the L1
regression) on GPU.

## Why

`hal9001` is the production HAL implementation. Its CPU coordinate
descent (via `glmnet`) becomes impractical at n ≳ 10⁵ on dense bases,
which is common for public-policy applications.

`gpu-hal` provides:

- **Gram-form Coordinate Descent on GPU** (`cd_gram.py`) — the inner
  Lasso solver. Cyclic Gauss-Seidel CD on a precomputed `G = X^T X / n`
  matrix held on the GPU. Validates against sklearn.Lasso to 1e-7
  relative L2 error.
- **IRLS-wrapped CD for logistic Lasso** (`cd_logistic.py`) — necessary
  for the hurdle classifier on zero-inflated outcomes.
- **HurdleGPUHAL composer** (`hurdle_hal.py`) — combines logistic
  + Gaussian stages into a single hurdle prediction
  `Q(x) = P(Y > 0 | x) · expm1(E[log(1+Y) | Y > 0, x])`.
- **rpy2 backend** (`backend.py`) — extracts hal9001's basis matrix as a
  scipy sparse CSR (no dense coercion, no GB-scale allocations).
- **Frequency / magnitude decomposition** of the shift-intervention
  target parameter — direct policy artifact for hurdle outcomes.

## Install

```bash
pip install gpu-hal
```

For GPU support:

```bash
pip install "gpu-hal[gpu]"
```

For the rpy2 hal9001 backend (needed for real HAL bases):

```bash
pip install "gpu-hal[backend]"
# In R:
#   install.packages("hal9001")
```

## Quickstart — Gaussian HAL

```python
import numpy as np
from gpu_hal import fit_hal_gpu

# Synthetic data
rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 5))
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(1000)

fit = fit_hal_gpu(
    X, y,
    max_degree=2,
    num_knots=(25, 10),
    smoothness_orders=1,
    n_folds=5,
    n_lambdas=30,
)

print(f"Active bases: {len(fit.active_idx)} / {fit.p}")
print(f"λ_cv = {fit.lambda_cv:.4e}")
y_pred = fit.predict(X)
```

## Quickstart — Hurdle HAL

For zero-inflated outcomes:

```python
from gpu_hal.hurdle_hal import fit_hurdle_hal_gpu
import numpy as np

# Y has many zeros (e.g., earthquake counts)
fit = fit_hurdle_hal_gpu(
    X, y,
    max_degree=2,
    num_knots=(25, 10),
    n_folds=5,
)

# Predict
preds = fit.predict(X_new)

# Inspect each stage
print(f"Stage 1 (logistic): λ={fit.lambda_pos:.3e}, "
      f"active={fit.n_active_pos}")
print(f"Stage 2 (gaussian): λ={fit.lambda_mag:.3e}, "
      f"active={fit.n_active_mag}")

# Shift intervention with frequency/magnitude decomposition
A_post = X[:, 0] * 1.10
X_post = X.copy()
X_post[:, 0] = A_post

# (See run_hurdle_gpu_hal.py for the full decomposition formula)
```

## Validation

`gpu_hal/tests/test_cd_vs_sklearn.py` runs synthetic Lasso problems and
compares the GPU CD solution to `sklearn.linear_model.Lasso`:

| λ regime | Sweeps | Active set match | Relative L2 vs sklearn |
|---|---|---|---|
| Heavy regularization | 12 | 100% | 2.1e-7 |
| Moderate (λ_max × 0.05) | 200 | 100% | 4.2e-7 |
| Near-OLS (λ = 1e-4) | 46 | 100% | 5.5e-7 |

Run the suite:

```bash
pytest gpu_hal/tests/
```

## Performance

At n = 50,000, p ≈ 1,500 (real HAL basis from induced-seismicity panel,
density 28%):

| Stage | Time |
|---|---|
| Build sparse basis (R hal9001 → scipy) | ~5 sec |
| Compute Gram once (CPU sparse-sparse mult) | ~5 sec |
| 5-fold CV path × 30 lambdas, GPU CD | ~3 min |
| Final fit at λ_cv | ~2 sec |
| Hurdle (logistic + gaussian) | ~30 min total |

Compare to `hal9001::fit_hal` hurdle on the same problem on CPU: ~2 hours.

## Known limitations

- **Full-n logistic IRLS is infeasible** with the current Gram-rebuild
  approach. At n = 451,212 the weighted Gram rebuild takes ~25-30 min
  per IRLS iteration → days of wall time. Active-set IRLS would address
  this; not yet implemented (see `FUTURE_WORK/README.md`).
- **Gram approach scales poorly when p ≳ 10,000** — the dense Gram is
  O(p²) memory.
- **No `hal9001::screen_basis` port** — we use the raw enumerated basis
  matrix. This contributes to a documented finite-sample λ-selection
  gap with `hal9001` on zero-inflated outcomes (~factor 2 on the
  induced-seismicity application).

## License

Apache 2.0. See `LICENSE`.

## Citation

If you use this software, please cite the companion paper:

```bibtex
@inproceedings{matthews2025causal,
  author = {Matthews, Lewis},
  title = {A Causal Inference Pipeline for Injection Open-Source
           Methodology and Implementation},
  booktitle = {SPE Annual Technical Conference and Exhibition},
  year = {2025},
  publisher = {Society of Petroleum Engineers},
  doi = {10.2118/228051-MS}
}
```
