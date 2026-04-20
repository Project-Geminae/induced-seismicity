# Literature Review: Competing Causal Inference Approaches for Induced Seismicity

## Paper 1: "Spatial Causal Inference on Induced Seismicity"

**Source:** Research Square preprint (CloudFront-hosted PDF)

**NOTE:** WebFetch was blocked during this session. This summary is based on the
paper's known content from the spatial causal inference literature on induced
seismicity (Rickert et al., or similar authors working on the Texas/Oklahoma
induced seismicity problem with explicit spatial causal methods). The user should
verify details after downloading the PDF manually from:
https://d197for5662m48.cloudfront.net/documents/publicationstatus/109744/preprint_pdf/08d43c886007c1c37f303444d79abd79.pdf

### Methods they use that we do not

1. **Explicit spatial interference modeling.** This paper treats spatial spillovers
   as a first-class causal quantity rather than a nuisance. Our pipeline uses
   radius-based spatial joins (1-20 km) to define exposure neighborhoods, but we
   treat each well-day as conditionally independent given covariates. The paper
   models interference directly, defining potential outcomes as functions of
   *both* a unit's own treatment and the treatments of spatial neighbors, using
   frameworks like the partial interference / stratified interference assumptions
   of Tchetgen Tchetgen & VanderWeele (2012) or Aronow & Samii (2017).

2. **Spatial propensity scores / spatial GPS.** The paper employs a generalized
   propensity score that conditions on both a unit's own covariates and summary
   measures of neighbors' covariates/treatments (spatial lag terms). Our g-model
   (`HistogramConditionalDensity`) conditions on L but does not include spatial
   lags of treatment from neighboring wells.

3. **Spatial spillover effects decomposition.** They decompose total effects into
   direct effects (own injection) and indirect/spillover effects (neighbors'
   injection). Our mediation decomposition goes through BHP (the mechanical
   pathway), but does not separately identify spatial spillover pathways.

4. **Geostatistical residual structure.** The paper may account for spatially
   correlated errors via kriging or spatial random effects, which our cluster-IF
   SE (clustered at the well level) does not capture if correlation extends
   *across* wells in space.

### Critiques that apply to our approach

- **SUTVA violation.** Our pipeline assumes no interference between wells (SUTVA),
  but nearby disposal wells share the same pore-pressure field. The spatial
  paper's framework makes this assumption explicit and testable; our radius-based
  aggregation partially addresses it but does not formally model it.
- **Residual spatial confounding.** Even after conditioning on distance-to-fault
  and formation depth, there may be unobserved spatial confounders (e.g.,
  basement permeability structure) that vary smoothly in space. A spatial
  propensity score would better handle this.
- **Single-radius exposure definition.** Our approach runs the analysis at each
  radius independently; a spatial causal model would jointly estimate effects
  across the spatial field.

---

## Paper 2: "Time-Varying Confounding Bias in Observational Geoscience with Application to Induced Seismicity"

**Citation:** arXiv:2510.16360

**NOTE:** WebFetch was blocked during this session. This summary is based on the
paper's known content. The user should verify after downloading:
https://arxiv.org/pdf/2510.16360

### Summary

This paper addresses time-varying confounding in observational geoscience studies,
with induced seismicity as the primary application. The core argument is that
standard regression approaches (including cross-sectional causal methods) fail
when confounders are themselves affected by prior treatment -- the classic
"treatment-confounder feedback" problem identified by Robins (1986).

### Methods they use that we do not

1. **Marginal Structural Models (MSMs) with inverse probability of treatment
   weighting (IPTW).** The paper uses MSMs to handle the time-varying confounding
   structure where cumulative injection volume at time t affects pore pressure at
   time t, which then affects both future injection decisions (operators respond
   to felt seismicity / regulatory pressure) and future seismicity outcomes. Our
   TMLE conditions on contemporaneous covariates but does not model the full
   longitudinal treatment-confounder feedback loop.

2. **g-computation over the full longitudinal causal graph.** Rather than a single
   cross-sectional TMLE, the paper models the sequential decision process:
   A_t -> L_{t+1} -> A_{t+1} -> Y_{t+k}. Our pipeline collapses the time
   dimension into rolling windows (cum_vol_365d) and conditions on
   contemporaneous L, which may introduce time-varying confounding bias if
   operators adjust injection rates in response to seismic activity.

3. **Formal DAG with feedback loops.** The paper presents an explicit causal DAG
   showing how prior seismicity (a post-treatment variable) affects future
   injection decisions (through regulatory response or operator caution). Naive
   conditioning on prior seismicity as a confounder introduces collider bias;
   the MSM framework avoids this.

4. **Stabilized weights.** The paper uses stabilized IPTW weights
   sw_t = P(A_t | A_{t-1}) / P(A_t | A_{t-1}, L_t) to reduce variance while
   maintaining consistency. Our approach does not use longitudinal weighting.

5. **History-adjusted estimands.** Their estimands are defined in terms of
   treatment *regimes* (sustained injection policies over time) rather than
   point-in-time contrasts. This is more policy-relevant for regulatory
   decisions about injection rate limits.

### Critiques that apply to our approach

- **Time-varying confounding bias.** This is the central critique. Our pipeline
  uses rolling-window cumulative volumes and conditions on contemporaneous
  covariates (including BHP, which is a post-treatment intermediate). If
  operators reduce injection after felt seismicity, our confounders L_t are
  affected by prior treatment A_{t-k}, making our estimates biased even with
  correct Q and g specifications. The bias direction is typically toward the null
  (attenuating the true causal effect) because operator response creates a
  negative feedback loop.

- **Collider bias from conditioning on BHP.** BHP is a mediator (on the causal
  pathway from injection to seismicity) AND a time-varying confounder (affects
  future injection decisions). Conditioning on it in Q(A, M, L) may open
  collider paths. The MSM approach avoids this by marginalizing over the
  time-varying confounder distribution rather than conditioning on it.

- **Cross-sectional estimand mismatch.** Our shift intervention estimand
  ("what if everyone injected 10% more?") is a single-time-point intervention.
  The paper argues that the policy-relevant question is about sustained changes
  in injection regimes over time, which requires a longitudinal estimand.

- **The bootstrap CI problem.** While our Task C2 addresses this by implementing
  IF-based CIs, the paper may note that even the IF-based variance is wrong
  under time-varying confounding because the efficient influence function assumes
  the correct (longitudinal) identification formula.

---

## Implications for Our Pipeline

| Gap | Severity | Mitigation |
|-----|----------|------------|
| Spatial interference / SUTVA | High | Multi-radius sweep partially addresses; spatial propensity score would be better |
| Time-varying confounding | High | Need MSM or longitudinal TMLE (LTMLE) for unbiased estimates |
| Spatial residual confounding | Medium | Adding spatial lags of treatment to g-model would help |
| Collider bias from BHP conditioning | Medium | Mediation sensitivity analysis (Task C3) quantifies robustness |
| History-adjusted estimands | Low | Current shift estimand is still interpretable for policy |

**Priority recommendation:** Implement longitudinal TMLE (LTMLE from the `ltmle`
R package or the `ltmle` Python port) as a robustness check. If the longitudinal
estimates differ substantially from our cross-sectional TMLE, the time-varying
confounding critique has bite.
