# Future Work

Items deferred from the main paper-writing track. The induced-seismicity
science paper (in `PAPER_*.md` at repo root) is the primary deliverable.
The items below are legitimate follow-up but should not gate the main paper.

## Methods paper (deferred — see `METHODS_PAPER_DEFERRED.md`)

Originally drafted as a JCI submission on GPU-accelerated hurdle HAL.
Withdrawn after analysis of van der Laan's lab norms: his community
treats software as software (PyPI/GitHub release), not as paper material.
JCI publishes new estimator theory, not engineering papers.

`gpu_hal` will be released as a Python package with a short arXiv
software descriptor; the substantive methods contribution lives in the
science paper's application section.

## Active-set IRLS for full-n logistic HAL

The original `cd_logistic.py` rebuilds the full weighted Gram every
IRLS iteration. At n=451k, p=1564, density=28%, this is ~25-30 min per
build — the full-n hurdle fit infeasible at ~3-4 weeks of projected
wall-time.

**Status (2026-04-28):** A first working cut is committed at
`gpu_hal/cd_logistic_active_set.py`. The implementation:

- Phase 1: K_warm full-Gram IRLS iterations to seed an initial
  active set S = {j : |β_j| > 0}.
- Phase 2: For each subsequent IRLS iter, rebuild G_S = X_S^T diag(w)
  X_S / n on the active columns only (|S| × |S|), run CD on the
  smaller Gram, and KKT-check inactive coordinates by computing
  c_j = (X^T diag(w) (z − X_S β_S))_j / n. Coordinates with
  |c_j| > λ + kkt_tol are promoted to S; the iter is re-solved.
- Phase 3: Final full-coordinate KKT check at convergence guarantees
  equivalence to the full-Gram fixed point.

Validated (`gpu_hal/tests/test_active_set_vs_full.py`) against the
full-Gram baseline at λ ∈ {0.05, 0.01, 0.005} on synthetic data: 100%
active-set agreement, rel-L2 ≤ 1.8 × 10⁻⁷, intercept agreement to
1 × 10⁻⁷. Empirically matches full-Gram at convergence.

**What still needs to be done:**

1. ~~**Full-n empirical benchmark.**~~ ✅ **Done 2026-04-29.** Measured
   on the n = 451,212, p = 1,564 induced-seismicity panel:
   **7.7 s/iter** average for active-set IRLS (vs the projected
   25-30 min/iter for full-Gram), |S| stable at ~133 from iter 8
   onward. End-to-end full-n hurdle (logistic + Gaussian on positives
   + ψ-decomposition + cluster-IF SE) completes in **5.0 minutes**.
   Empirical confirmation that active-set IRLS unlocks the full-n
   hurdle. Logs: `bench_active_set_50k.log`, `bench_active_set_451k.log`,
   `hurdle_full_n_R7.log`.
2. **Path solver for `logistic_lasso_active_set_path`.** Warm-start
   across decreasing λ — the natural way to fit a regularization path
   without paying the warmup cost at every λ. Estimated: 0.5 day.
3. **Hurdle pipeline plumbing.** Wire the active-set solver into
   `gpu_hal/hurdle_hal.py` behind a flag so users can choose
   `solver={"full_gram", "active_set"}`. Estimated: 0.5 day.
4. **JOSS submission writeup.** With active-set IRLS validated and
   benchmarked, `gpu_hal` becomes "first full-n hurdle HAL-TMLE on
   n = 451k well-days" — a concrete new-scale-estimand contribution
   suitable for a JOSS short paper. Estimated: 1 day.

Total remaining: ~2-3 days. The big risk item — solver correctness —
is now closed.

## hal9001 basis screening

`hal9001::screen_basis` filters basis functions by minimum activation
fraction. We don't replicate this in the GPU pipeline, which contributes
to the CV-procedure-driven gap with `hal9001` reported in the diagnostic.

## CV-stable channel-decomposition baseline

The 2026-04-27 diagnostic round (`gpu_hal/tests/diagnose_*.py`)
established that the CPU/GPU ψ_total gap at R = 7, n = 50k is dominated
by CV configuration, not by solver behaviour. Five pipelines on
identical (X, y, basis) inputs span ψ_total = +2.7 × 10⁻⁴ to
+4.0 × 10⁻³ (a 14× range), and the channel decomposition's frequency
component sign-flips between random-fold and cluster-fold CV. This is
written up in PAPER_DRAFT.md §5.2 (sensitivity panel) and §6.5
(λ-selection gap).

The follow-up that would close this is a **CV-stable baseline** that
mirrors the GPU pipeline's exact CV setup (3-fold cluster-aware folds,
explicit log-spaced λ-grid covering the active region, no
`dev.ratio` early stop). The bypass-`hal9001` driver in
`gpu_hal/tests/diagnose_custom_grid.py` is a working scaffold; finishing
it requires (a) cluster-fold construction at the *positives subset*
matching GPU's stage-2 fold structure, (b) refit with explicit λ at
the GPU-selected value to give a single-λ point comparison free of CV
noise, and (c) a multi-seed sensitivity panel showing the spread
across 10 random fold-assignment seeds. ≈ 1 day of work; not on the
science paper critical path.

## LTMLE for time-varying confounding

The well-day panel has operator-feedback structure (operators reduce
injection in response to seismicity). Cross-sectional TMLE doesn't
capture this. LTMLE (van der Laan & Gruber 2012) would address it.

## Channel-decomposition TMLE with proper IF

The frequency/magnitude decomposition reported in the science paper is
a plug-in calculation, not a properly-targeted estimator. Deriving the
efficient influence function for each channel separately (and
multiplicity-corrected joint inference) is a real estimator-development
project, ~3-6 months. Could be a follow-up paper.
