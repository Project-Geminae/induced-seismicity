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
