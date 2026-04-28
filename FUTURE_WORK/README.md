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

The current `cd_logistic.py` rebuilds the full weighted Gram every IRLS
iteration. At n=451k, p=1564, density=28%, this is ~25-30 min per build,
making the full-n hurdle fit infeasible (~3-4 weeks projected).

The standard fix is **active-set IRLS**: only rebuild Gram on the active
basis (~80-200 columns instead of 1564), with periodic KKT checks for
violator coordinates. Expected speedup: 10-20×, putting full-n hurdle
within range.

This is ~2-3 days of careful implementation. Worth doing as a separate
methodological contribution — but not for the science paper.

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
