# Follow-up paper outline — Longitudinal regHAL-TMLE for induced seismicity

**Working title:** A longitudinal regHAL-TMLE for time-varying
operator-feedback in induced-seismicity causal inference: integrating
sequential ignorability with Highly Adaptive Lasso outcome models

**Target venue:** *Journal of Causal Inference* (methodology paper) or
*Annals of Applied Statistics* (applied-with-novel-methods).

---

## 1. The gap

Two papers in 2025 attacked induced-seismicity causal inference from
methodologically orthogonal directions:

- **Xiao, Zigler, Hennings & Savvaidis (2025; arXiv:2510.16360)** —
  Fort-Worth Basin, MSM-IPTW (Robins 2000) on a quarterly Poisson
  count. *Strength:* explicit time-varying confounding via operator-
  feedback (L(t) = quarterly seismicity indicator). *Weakness:*
  25-year-old Robins MSM-IPTW estimator; no magnitude modelling; small
  sample (30 cluster centroids × 7 quarters).

- **Matthews (this PAPER_DRAFT.md, 2026)** — Permian Basin, regHAL-TMLE
  (Li/Qiu/Wang/vdL 2025) on a hurdle outcome with frequency/magnitude
  decomposition. *Strength:* doubly-robust targeted estimator,
  hurdle channels, basin-scale n = 451,212 well-days. *Weakness:*
  cross-sectional only; doesn't address operator-feedback.

The follow-up paper completes the matrix: a longitudinal TMLE
(LTMLE; van der Laan & Gruber 2012) on a HAL-implied working model
with hurdle outcome and stochastic shift intervention. Both axes —
longitudinal identification AND estimator efficiency — under one
roof, applied at Permian basin scale.

## 2. Estimand

For each well-day t and well i, define A(i, t) = cumulative 365-day
volume up to t. The time-varying confounder L(i, t) includes the
cross-sectional confounders (G₁–G₆) plus a lagged seismicity
indicator: L_seis(i, t) = 1{event within R km of well i in days
t−30 to t−1}. This captures the operator-feedback loop in which
TexNet/SSN alerts trigger pre-emptive volume reductions.

Target: the longitudinal stochastic shift parameter

  ψ_long(δ) = E[Y_{d̄(Ā, L̄)}] − E[Y_{Ā}]

where d̄ applies a 10 % volume reduction sequentially across all
time points, with hurdle outcome Y_{i,t} = max ML within R km
of well i on day t.

## 3. Estimator

regHAL-TMLE Delta-method (Li et al. 2025) extended to the
sequential setting via the LTMLE recursion:

- t = T (last time point): standard cross-sectional regHAL-TMLE
  on Q_T = E[Y | L̄_T, Ā_T] → Q*_T after fluctuation.
- t = T−1, …, 1 (recursive backward induction):
    - Q_t(L̄_t, Ā_t) = E[Q*_{t+1}(L̄_{t+1}, Ā_{t+1}) | L̄_t, Ā_t]
      via a HAL fit on Q*_{t+1} regressed on (L̄_t, Ā_t, A(t+1) shifted).
    - Fluctuation step using the time-t clever covariate.
- ψ̂ = Σ_i Q*_1(L̄_1(i), Ā_1(i)) − Q_1^{plugin}(L̄_1(i), Ā_1(i))

Inference: longitudinal influence function → cluster-robust IF SE
at the well level (Two-Stage TMLE framing, Nugent et al. 2024).

## 4. Hurdle decomposition under longitudinal confounding

The cross-sectional channel split (ψ_freq, ψ_mag, ψ_cross) extends
naturally: at each time point t, decompose Q_t into P_t × M_t and
propagate the multiplicative structure through the LTMLE recursion.
Open methodological question (worth a §): is the longitudinal
channel decomposition still calibration-invariant the way the
cross-sectional one is? Plausibly yes if Q_t factorises at every
recursion step, but needs proof.

## 5. Computational requirements

Bottleneck: T LTMLE recursion steps × full-n HAL fit per step. With
the active-set IRLS solver landed in `gpu_hal` (FUTURE_WORK item)
this becomes feasible at n = 451k. Estimated wall time on minitim:
~6 hours per CV fold × 5 folds = ~30 hours. Acceptable.

## 6. Application result preview

Two leading questions for the Permian Basin application:

- Does the longitudinal estimate exceed or fall below the cross-
  sectional one? (Operator-feedback typically biases cross-sectional
  estimates *downward* — Xiao et al.'s simulation shows naive
  estimators are biased downward by ~40 %.)
- Does the channel decomposition shift? Specifically: under
  longitudinal adjustment, is the frequency channel STILL dominant,
  or does the magnitude channel grow when we account for operator
  pre-emption (which selectively targets high-likely-event wells)?

Both questions are publishable regardless of direction.

## 7. Comparison to Xiao et al. 2025

Apples-to-apples comparison on the same Permian Basin data, with
their MSM-IPTW estimator and our LTMLE-regHAL on the same panel.
This is the cleanest possible methodological cross-check between
the longitudinal-identification axis (Robins 2000) and the
LTMLE-with-HAL axis (van der Laan 2012 + Li et al. 2025).

## 8. Software deliverable

`gpu_hal.ltmle.LongitudinalRegHAL` — a new sub-module in the existing
`gpu-hal` package, exposing:

```python
fit = LongitudinalRegHAL(max_degree=2, num_knots=(25,10),
                         smoothness_orders=1).fit(
    panel, time_col="Date", treatment_col="cum_vol_365d_BBL",
    L_cols=[...], outcome_col="outcome_max_ML",
    shift_pct=-0.10, n_folds=5,
)
print(fit.psi_total, fit.psi_freq, fit.psi_mag, fit.psi_cross)
```

JOSS submission concurrent with the methods paper.

## 9. Timeline

- Active-set IRLS benchmark + CV-path solver (FUTURE_WORK, ~3 days)
- LTMLE recursion implementation in gpu_hal (~5 days)
- Application + paper writeup (~3 weeks)
- **Total: ~6 weeks from "go"**

## 10. Risks

- **The recursion may not converge cleanly under HAL nuisance fits.**
  Backup plan: initial cut with parametric Q_t models, gradually
  upgrade.
- **Operator-feedback signal may be weaker in Permian than DFW.**
  In which case the longitudinal estimate ≈ cross-sectional estimate
  and the paper's contribution is the negative result + the
  methodology + the LTMLE-regHAL software. Still publishable.
- **Computational ceiling.** If active-set IRLS doesn't deliver
  10× speedup, longitudinal at n=451k is infeasible and we fall
  back to a cluster-aware-subsampled n=50k analysis (still useful;
  matches Xiao et al.'s scale).
