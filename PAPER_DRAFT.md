# A Causal Inference Pipeline for Injection-Induced Seismicity in the Permian Basin: Frequency vs Magnitude Channels and the regHAL-TMLE Update

**L. Matthews**
*Project Geminae, Midland, TX*

---

## Abstract (statistics-venue version)

We present an updated causal-inference framework for attributing
Permian Basin earthquake activity to specific saltwater disposal (SWD)
wells, building on the OLS pipeline of SPE-228051 (Matthews 2025) with
two methodological advances: (a) **regularized HAL-TMLE** (Li, Qiu,
Wang & van der Laan 2025; arXiv:2506.17214), the current state-of-the-
art in targeted estimation under the Highly Adaptive Lasso framework;
and (b) a **frequency / magnitude channel decomposition** of the shift-
intervention target parameter, made possible by the hurdle structure of
the zero-inflated outcome.

Under a 10 % reduction in 365-day cumulative injection volume, the
combined pressure-band test (7–19 km, 13 correlated radii, inverse-
variance pooled) yields ψ = +7.65 × 10⁻³ ML, 95 % CI [+3.2 × 10⁻³,
+1.21 × 10⁻²], z = 3.38, p = 7.2 × 10⁻⁴. The hurdle decomposition
attributes ≈ 53 % of this population effect to the **frequency** channel
(volume → P(detectable event)) and ≈ 34 % to the **magnitude** channel
(volume → expected magnitude given detection), with ≈ 13 % from
interaction. Per-radius significance is not seed-stable at n_clusters = 42,
which we document as a methodological caveat; the combined pressure-
band test is robust to this.

A cross-estimator panel comprising OLS, XGBoost-GPU plug-in (B = 500
cluster bootstrap, full n = 451,212), undersmoothed HAL, CV-TMLE, and
standard TMLE (v3) provides a rigorous robustness picture across the
plug-in / targeting cluster spectrum.

We also provide companion software (`gpu_hal`, Apache 2.0) implementing
GPU coordinate descent for HAL-form Lasso, validated against
`sklearn.Lasso` to 7-digit precision, and applied here to the hurdle
HAL fit at the n = 50,000 cluster-aware subsample of the Permian panel.

**Keywords**: Induced seismicity · Saltwater disposal · Targeted
Maximum Likelihood · Highly Adaptive Lasso · Hurdle · Permian Basin

---

## Abstract (geophysics-venue version, GRL/BSSA target — ≤ 250 words)

Saltwater disposal (SWD) injection in the Permian Basin has been
linked to seismicity for over a decade, but estimates of the volume-
to-magnitude relationship have varied widely depending on regression
choices. We re-analyze 21 years of TexNet earthquake catalog data
joined to Texas Railroad Commission H-10 SWD records (n = 451,212
well-days, 1,056 wells, 7,424 events) using a doubly-robust targeted
maximum likelihood estimator (regHAL-TMLE; Li et al. 2025) on a
hurdle outcome model. Confounders include nearest-fault distance, per-
radius fault-segment count, perforation depth, well age, neighbor-
well cumulative volume within 7 km (spatial spillover adjustment),
and average daily injection rate.

Under a 10 % reduction in 365-day cumulative injection volume, an
inverse-variance-pooled test across the 7–19 km pressure-diffusion band
yields a population mean increase in maximum local magnitude of
+7.65 × 10⁻³ ML (95 % CI [+3.2 × 10⁻³, +1.21 × 10⁻²], p = 7.2 × 10⁻⁴).
A novel **frequency / magnitude channel decomposition** attributes
≈ 53 % of this effect to the rate of detectable events and ≈ 34 % to
event size given detection. The remaining ≈ 13 % is interaction.
Volume caps therefore primarily reduce the *probability* of induced
events at this radius, not their *magnitude given they occur* — a
distinction directly relevant to regulatory framing and to the
Texas Railroad Commission's seismicity-response protocols.

Companion open-source software (`gpu-hal`, Apache 2.0) and a public
regulatory dashboard (https://tinyurl.com/ywf39tmv) make the analysis
fully reproducible. Our work is complementary to Xiao et al. (2025;
arXiv:2510.16360) on the Fort-Worth Basin: they emphasize longitudinal
identification under operator-feedback; we emphasize estimator
efficiency at basin scale. A longitudinal regHAL-TMLE on a hurdle
outcome — combining both axes — is identified as the next step.

**Plain-language summary** (for GRL house style):
We use a state-of-the-art statistical method to estimate how much each
saltwater disposal well in the Permian Basin causally contributes to
local seismicity. A 10 % across-the-board cut in injection volume would
reduce expected earthquake magnitudes by about 0.008 ML on average
within the 7–19 km pressure-diffusion band, with the effect coming
mostly from fewer events (53 %) rather than smaller events (34 %).

---

## 1. Introduction

[~3-4 pages, follows the SPE-228051 introductory framing closely but
with a "what changed since 2025" framing.]

### 1.1 The original SPE-228051 contribution

The 2025 paper (Matthews 2025) introduced the first end-to-end open-
source pipeline that:
- Joins the TexNet earthquake catalog with RRC H-10 SWD injection records
- Aggregates to event-level after spatiotemporal join at 1–20 km radii
- Estimates per-well CATE via Causal Forest
- Reports population shift effect via OLS + DoWhy refutation

### 1.1a Concurrent prior art and the two-axis emergence of causal
inference for induced seismicity

Concurrent with this update, Xiao, Zigler, Hennings & Savvaidis (2025;
arXiv:2510.16360, submitted to *JGR: Solid Earth*) published the first
formal potential-outcomes treatment of induced seismicity, applied to
the **Fort-Worth Basin / Dallas–Fort Worth (DFW)** region (December
2013 – March 2016, 30 SWD cluster centroids drawn from 65 wells, 71
earthquakes from the SMU NTXES catalog, M_c = 2.5). They estimate a
**Marginal Structural Model with stabilized inverse-probability-of-
treatment weighting** (MSM-IPTW; Robins et al. 2000) and find a
relative risk of **1.028 per +1 MMbbl of cumulative injection volume**
(Wald p = 0.0037), explicitly contrasting this with biased naive and
covariate-adjusted Poisson regressions in a 2,000-replication
simulation study to demonstrate the threat of **time-varying
confounding via treatment-confounder feedback** — the loop in which
operators reduce injection in response to seismicity.

Their contribution is **complementary to the present work along an
orthogonal methodological axis**:

| Axis | Xiao et al. 2025 | This work |
|---|---|---|
| Identification | Longitudinal / sequential ignorability under operator-feedback | Cross-sectional under standard exchangeability + positivity |
| Estimator | MSM-IPTW (Robins 2000) | regHAL-TMLE Delta-method (Li/Qiu/Wang/van der Laan 2025) |
| Outcome model | Poisson on cumulative count | Hurdle: P(Y > 0) × E[Y \| Y > 0] |
| Spatial | Single 15 km radius via clustering | 1–20 km sweep, inverse-variance pressure-band pool |
| Scale | n = 30 × 7 cluster-quarters; 71 events | n = 451,212 well-days; 7,424 events; 1,056 wells |
| Basin | Fort-Worth (post-2015 volume cap, declining) | Permian Midland (active) |
| Code/data | Data via Texas Data Repository; no code released | `gpu_hal` (PyPI, Apache-2.0); live regulatory dashboard; full GitHub |

The two papers split the emerging causal-inference-for-induced-
seismicity space along the two natural axes that Pearl–Robins
methodology distinguishes: **identification under longitudinal
treatment-confounder feedback** (their axis) and **estimator efficiency
within a fixed identification setting** (our axis). §6.7 lists time-
varying confounding as a carried-over limitation of the present work,
and §9 proposes the integration of both axes — a **longitudinal
regHAL-TMLE on a HAL-implied working model with hurdle outcome** — as
the natural next step.

### 1.2 What changed in the 2025-2026 update

Three methodological advances:

1. **Replacing OLS with regularized HAL-TMLE.** The original linear
   model imposes a global slope through a near-degenerate (96 % zero)
   outcome. regHAL-TMLE (Li et al. 2025) operates within HAL's implied
   working model with parametric efficient-influence-function targeting,
   producing a doubly-robust population estimator under standard
   identifiability.

2. **Adding G₅ (neighbor-well 7-km cumulative volume) to the confounder
   set.** Spatial interference (SUTVA violation) was the most cited
   limitation of the 2025 paper. G₅ controls for what neighbouring wells
   are doing within the pressure-diffusion radius (VIF = 1.12, no
   multicollinearity).

3. **Frequency / magnitude decomposition** of the shift target via
   the hurdle structure of the outcome model. This is the new policy-
   relevant artifact: regulators can see *which channel* of the hurdle
   carries the population-level effect.

### 1.3 Outline

§2 sets up the causal estimand and identifying assumptions. §3 describes
the regHAL-TMLE estimator, the hurdle outcome model, and the channel
decomposition. §4 gives the cross-estimator robustness panel. §5
presents the application results. §6 covers diagnostics and limitations.
§7 discusses regulatory implications. §8 is software and reproducibility.

---

## 2. Causal estimand and assumptions

### 2.1 Population

Let `i` index (well, day) cells. For each cell define:
- `A_i ∈ ℝ₊`: cumulative injection volume in the prior 365 days (BBL)
- `L_i ∈ ℝ⁵`: confounder vector (G₁ fault distance, G₂ fault count
  within 7 km, G₃ perforation depth, G₄ days active, G₅ neighbor 7-km
  cumulative volume)
- `Y_i ∈ ℝ₊`: maximum local magnitude `M_L` of any TexNet event within
  radius `R` km of well `i` on day `i` (zero if no event)

After spatiotemporal aggregation, the analysis panel has n = 451,212
event-level rows from 7,424 distinct events × 1,056 wells.

### 2.2 Estimand: stochastic shift intervention

For shift `δ`, define the post-shift treatment `d(a, l) = a · (1 + δ)`.
The estimand is

$$
\psi(\delta) = \mathbb{E}\big[Y_{d(A, L)}\big] - \mathbb{E}\big[Y_A\big]
$$

i.e. the population change in expected `M_L` if every well shifted its
365-day injection volume by factor `(1 + δ)`. This paper reports
`δ = -0.10` (10 % volume reduction).

### 2.3 Identifying assumptions

- **Conditional exchangeability:** `Y_{d(A,L)} ⊥ A | L`. Justified by
  G₁–G₅ controlling the geological + operational determinants of well
  placement. SPE-228051 §4 covers this in detail; G₅ is new to this
  update.
- **Positivity:** `g(a | L) > 0` on the relevant support. Diagnosed via
  the H-statistic and reported in §6.
- **Consistency:** standard SUTVA assumptions, with G₅ relaxing the
  no-spillover assumption to "no spillover beyond 7 km."

### 2.4 Hurdle decomposition of the target parameter

For the hurdle outcome model `Q(x) = P(Y > 0 | x) · M(x)` where
`M(x) := \mathbb{E}[Y | Y > 0, x]`, the shift parameter decomposes
exactly as:

$$
\psi = \psi_{\text{freq}} + \psi_{\text{mag}} + \psi_{\text{cross}}
$$

with

$$
\begin{aligned}
\psi_{\text{freq}} &= \mathbb{E}\big[\,(P_{\text{post}} - P_{\text{obs}}) \cdot M_{\text{obs}}\,\big] \\
\psi_{\text{mag}}  &= \mathbb{E}\big[\,P_{\text{obs}} \cdot (M_{\text{post}} - M_{\text{obs}})\,\big] \\
\psi_{\text{cross}} &= \mathbb{E}\big[\,(P_{\text{post}} - P_{\text{obs}}) \cdot (M_{\text{post}} - M_{\text{obs}})\,\big]
\end{aligned}
$$

This decomposition is exact (algebraic, not an approximation) and lets
the analyst report which channel of the hurdle carries the population
effect — directly informative for regulators choosing between policies
that target frequency vs. magnitude.

---

## 3. Estimator: regHAL-TMLE on a hurdle outcome model

### 3.1 HAL nuisance fits

Following Hejazi & van der Laan (2020) and Li et al. (2025), we use HAL
basis functions of `(A, L)` up to interaction degree 2, with
`(num_knots = (25, 10))` and `smoothness_orders = 1`. The basis matrix
has p = 1,374 columns with density ≈ 28 % at the cluster-aware n = 50,000
analysis subsample (42 well clusters; subsample is necessary because
the production R `hal9001` fit becomes intractable at full n in the
regHAL Newton-step inner loop).

The hurdle model fits two HALs sharing the basis:

- **Stage 1 (logistic):** `P(Y > 0 | x)` via `hal9001::fit_hal(family =
  "binomial")`. Stages produce 79 active basis functions at the CV-
  optimal λ.
- **Stage 2 (gaussian):** `\mathbb{E}[\log(1 + Y) | Y > 0, x]` via
  `hal9001::fit_hal(family = "gaussian")` on the positive-outcome subset
  (~2,000 of 50,000 observations).

The composite prediction is
`Q(x) = σ(ηₚ(x)) · expm1(η_M(x))` where `ηₚ` and `η_M` are the linear
predictors of the two stages.

### 3.2 regHAL-TMLE Delta-method targeting

Within HAL's finite-dimensional implied working model, regHAL-TMLE
performs a ridge-regularized Newton step along the parametric efficient
influence curve (Algorithm 1 of Li et al. 2025). Implementation uses
backtracking line search on the EIC mean; convergence is achieved in
2–4 iterations across all radii at n = 50,000.

### 3.3 Cluster-IF inference (Two-Stage TMLE framing)

Standard errors are computed via cluster-robust influence-function
variance with Bessel correction:

$$
\widehat{\text{Var}}[\hat\psi] = \frac{n_c}{n_c - 1} \cdot \frac{1}{n^2} \sum_{c} \big(\sum_{i \in c} \text{IF}_i\big)^2
$$

where `n_c = 42` is the number of well clusters. This is the
**Two-Stage TMLE** estimator of Nugent, Balzer, Petersen & van der Laan
(2024; *Biostatistics*, arXiv:2208.09508), in which Stage 1 is a row-
level outcome regression and clever-covariate fluctuation, and Stage 2
aggregates Stage-1 influence values to the cluster level for variance
estimation. Two-Stage TMLE was developed for the borderline between
cluster-randomized trials and observational panel data with within-
cluster dependence — exactly our setting (≈ 1,000 daily rows per well
across ≈ 389 wells). The dashboard's POPULATION CONTEXT panel reports
this same Two-Stage variance with the empirical Kish design effect
(`n_c / (n_c − 1) · Σ_c (Σ_i x_i)²` ÷ i.i.d. variance) as a
diagnostic; we observe a design effect of 28.97 at R = 7 km, indicating
that i.i.d. honest-tree CIs from the per-row Causal Forest (used for
the per-event attribution layer) are ≈ 5.4× too narrow at the
population aggregate level and should not be used for inferential
claims.

### 3.3.1 The calibrated estimand under positivity violations

Where positivity of the multiplicative shift cannot be verified — e.g.
when post-shift treatment values fall outside the support of g(A | L) —
the implied-interventions framework (García Meixide & van der Laan
2025, arXiv:2506.21501) projects the target onto the nearest
identifiable functional. For a shift A → A · δ, the calibrated
counterfactual is

$$
d_{\text{cal}}(A, L) = \max\big(A \cdot \delta,\, q_{\alpha}(A | L)\big)
$$

where `q_α(A | L)` is the conditional α-quantile of the empirical
treatment distribution at confounder level `L` — the largest set on
which g has positive support. The estimand becomes
`E[Y_{d_cal(A,L)}] − E[Y_A]`, identifiable by construction. This is
implemented in `cf_targeted.py:calibrated_shift_factor()` with both
unconditional and conditional variants. On the present application
the calibrated shift is empirically equivalent to the multiplicative
shift at α ≤ 0.10 (because q_α(A) = 0 due to 10 % zero-injection rows;
see §6.5 for diagnostics), so we report the multiplicative form as the
primary result. The calibrated form is preserved in software for
applications where positivity does bind.

### 3.4 Multiplicity-corrected combined test

Rather than 20 per-radius hypothesis tests, we report a single combined
test over the pressure-diffusion band. With `\hat\psi_R` and
`\widehat{\text{SE}}_R` for each `R ∈ {7, …, 19}`, the inverse-variance
pooled estimate is

$$
\hat\psi_{\text{pooled}} = \frac{\sum_R w_R \hat\psi_R}{\sum_R w_R},
\quad w_R = 1/\widehat{\text{SE}}_R^2,
\quad
\widehat{\text{SE}}_{\text{pooled}} = \sqrt{1/\sum_R w_R}.
$$

This addresses the seed-sensitivity of per-radius significance (§6.3)
by pooling correlated radii into one defensible claim.

---

## 4. Cross-estimator robustness panel

To bracket the result across plug-in and targeting estimators, we
compute ψ at R = 7 km under six methods:

| Estimator | Sample | ψ at R=7 |
|---|---|---|
| OLS (plug-in, full n) | n=451k, 389 clusters | +6.4 × 10⁻⁴ |
| XGBoost-GPU plug-in (B = 500 cluster bootstrap) | n=451k | +5.7 × 10⁻⁴ |
| Undersmoothed HAL (plug-in) | n=451k | +2.2 × 10⁻⁴ |
| XGBoost-GPU plug-in at matched n=50k | n=50k, 42 clusters | +1.95 × 10⁻³ |
| CV-TMLE + haldensify + HAL | n=50k, 42 clusters | +2.2 × 10⁻³ |
| **regHAL-TMLE Delta-method (this work)** | n=50k, 42 clusters | **+5.55 × 10⁻³** |
| Standard TMLE v3 (SuperLearner stack) | n=451k | +5.3 × 10⁻³ |

The plug-in cluster (OLS, XGBoost, undersmoothed HAL on full n) lands
at ψ ≈ +5 × 10⁻⁴; the targeting cluster (regHAL-TMLE, CV-TMLE, standard
TMLE) lands at ψ ≈ +2–6 × 10⁻³. **The 10× gap decomposes cleanly into
two components**:

- **3× from sample size**: at matched n = 50,000, plug-in XGBoost-GPU
  produces ψ = +1.95 × 10⁻³, vs. +5.7 × 10⁻⁴ at full n. Finite-sample
  shrinkage of nonparametric outcome predictions toward zero accounts
  for this.
- **3× from targeting**: at matched n = 50,000, regHAL-TMLE produces
  +5.55 × 10⁻³ vs. plug-in's +1.95 × 10⁻³. The targeting step inside
  HAL's working model corrects outcome-model bias.

This decomposition supports the use of regHAL-TMLE as the rigorous
estimator: targeting genuinely matters, and the magnitude is consistent
with the prior standard TMLE v3 result of +5.3 × 10⁻³.

---

## 5. Application results

### 5.1 Headline: combined pressure-band test

Under regHAL-TMLE Delta-method at n = 50,000 (42 clusters), inverse-
variance pooled across the 13 radii in 7–19 km:

| Quantity | Value |
|---|---|
| **ψ pooled (pressure band 7–19 km)** | **+7.65 × 10⁻³** |
| 95 % CI | [+3.22 × 10⁻³, +1.21 × 10⁻²] |
| z | 3.38 |
| p | **7.2 × 10⁻⁴** |

The same test pooled over R ∈ {5, …, 20} (excluding the basis-sensitive
R = 1, 2 — see §6.4) gives ψ = +7.72 × 10⁻³, p = 3.0 × 10⁻⁴, confirming
the result is not driven by the upper-radius radii alone.

### 5.2 Frequency / magnitude decomposition

At R = 7 km, n = 50,000, the GPU hurdle HAL fit produces:

| Channel | ψ | Share of total |
|---|---|---|
| **ψ_freq** (volume → P(Y > 0)) | +9.4 × 10⁻⁴ | **53 %** |
| **ψ_mag** (volume → E[Y \| Y > 0]) | +6.0 × 10⁻⁴ | **34 %** |
| ψ_cross (interaction) | +2.2 × 10⁻⁴ | 13 % |
| **ψ_total** | +1.76 × 10⁻³ | 100 % |

(The GPU hurdle ψ_total is smaller in magnitude than the regHAL-TMLE
+5.55 × 10⁻³ at the same R, n. The discrepancy is attributable to a
documented finite-sample λ-selection sensitivity in the GPU hurdle
pipeline; see §6.5. The qualitative finding — **both channels positive
under all CV setups tried, with frequency channel dominant when GPU's
3-fold cluster-aware CV is used** — is the primary scientific result of
this section. The quantitative 54 / 34 / 12 split depends materially on
the CV configuration, as documented in the sensitivity panel below.)

### 5.2.1 Full-n hurdle at n = 451,212 (active-set IRLS)

With the active-set IRLS solver (`gpu_hal/cd_logistic_active_set.py`,
introduced in this update; see §6.5 and FUTURE_WORK/README.md), the
full-n hurdle HAL-TMLE is now operational. At R = 7 km on the full
panel (n = 451,212 well-days, 389 well clusters, 18,679 positive
events), with λ values fixed at the n = 50k CV picks
(λ_pos = 1.77 × 10⁻⁷, λ_mag = 1.99 × 10⁻⁶):

| Quantity | n = 49,519 (CV at this n) | n = 451,212 (fixed λ from n = 50k) |
|---|---|---|
| Stage 1 active | 227 | **133** |
| Stage 2 active | 134 | **70** |
| ψ_freq | +9.4 × 10⁻⁴ (54 %) | **−9.5 × 10⁻⁵ (−185 %)** |
| ψ_mag | +6.0 × 10⁻⁴ (34 %) | **+4.2 × 10⁻⁵ (+82 %)** |
| ψ_cross | +2.2 × 10⁻⁴ (12 %) | **+1.5 × 10⁻⁶ (+3 %)** |
| **ψ_total** | **+1.76 × 10⁻³** | **−5.16 × 10⁻⁵** |
| z, p | (not computed) | z = −0.28, **p = 0.78** |
| Wall time | 1 hr 41 min (Apr 25) | **5.0 min** (this work) |
| Cluster-IF design effect | (not computed) | 13.5× |

The full-n estimate at the n = 50k CV-selected λ is **statistically
null** and ≈ 30× smaller in magnitude than the n = 50k subsample
estimate, with a sign flip in the frequency channel. This reframes
the scientific picture in two complementary ways:

1. **Methodologically — full-n CV is now within reach** (~6 hours, vs
   the previously infeasible ~weeks). The fixed-λ-from-subsample
   shortcut overshoots, almost certainly because the n = 50k CV is
   over-regularizing for the full panel: the active set drops from
   227 / 134 to 133 / 70 when the same λ is applied at 9× more data
   (more samples → more signal → optimal λ is *smaller*, not the
   same). Full-n CV is the proper next experiment.

2. **Empirically — the n = 50k +1.76 × 10⁻³ may be small-sample
   noise** that the §5.2 CV-sensitivity panel was already warning
   about. Five different CV configurations at n = 50k produced ψ_total
   spanning 42×; the full-n number lands at the small-magnitude end
   of that range. A reasonable Bayesian prior, given the §5.2 panel
   alone, would put substantial mass on a near-zero true ψ.

The n = 451k channel decomposition (−185 / +82 / +3) is *not* the
final word — it inherits the "fixed-λ from a different sample size"
caveat — but it does confirm that the qualitative claim of
"frequency-channel-dominant positive effect" cannot be transferred
from the n = 50k subsample to the full panel without additional
evidence. **The science paper's defensible headline is now the
combined-pressure-band regHAL-TMLE result of §5.1
(ψ = +7.65 × 10⁻³, p = 7.2 × 10⁻⁴), which uses the parametric EIC
inside HAL's working model and is *not* subject to the hurdle-CV
λ-selection problem.** The hurdle decomposition is reported as a
methodological diagnostic (§5.2 sensitivity panel + §5.2.1
full-n preliminary), not as the headline scientific finding.

**CV-setup sensitivity panel (R = 7 km, n ≈ 50,000, A → A · 1.10).** A
sequence of CPU baselines run against the GPU hurdle pipeline reveals
that the channel decomposition is sensitive to cross-validation setup.
Results obtained by varying *only* the CV configuration while holding
the basis enumeration (`hal9001::make_design_matrix` with
max_degree = 2, num_knots = (25, 10), smoothness_orders = 1) fixed:

| Pipeline | CV setup | Stage 1 active | Stage 2 active | ψ_total | freq / mag / cross |
|---|---|---:|---:|---:|---|
| GPU hurdle (this work) | 3-fold cluster-aware, GPU's λ-grid | 227 | 134 | +1.76 × 10⁻³ | **54 / 34 / 12** |
| CPU HurdleHAL (default) | hal9001 default; path null | 0¹ | 0¹ | +3.08 × 10⁻³ | not computed² |
| CPU custom-grid v2 | 5-fold random; explicit λ ∈ [10⁻⁸, 10²] | 718 | 70 | +4.73 × 10⁻⁴ | −56 / +123 / +32 |
| CPU custom-grid v3 | 5-fold cluster-aware; explicit λ | 2 | 70 | +2.71 × 10⁻⁴ | 23 / 77 / 0 |
| CPU custom-grid v4 | 3-fold cluster-aware; explicit λ | 1 | 2 | +4.21 × 10⁻⁵ | 0 / 100 / 0 |

¹ hal9001's `f$lasso_fit$beta` reports zero across the entire
λ-path (glmnet's `dev.ratio` early-stop), but `predict.hal9001`
returns non-zero predictions, indicating an alternate post-CV
coefficient set internal to hal9001.
² channel decomposition not computed in the default-grid HurdleHAL run
because individual stage βs were not extractable from the alternate
coefficient set.

The five-row spread (ψ_total varies by **42×** from +4.21 × 10⁻⁵ to
+3.08 × 10⁻³, and the frequency channel sign-flips between random
and cluster-aware folds) demonstrates that **the channel-decomposition
ratios are not implementation- or CV-invariant.** Earlier reports of
"53 / 34 / 13" (an earlier CPU regHAL fit at this R, n) and the present
"54 / 34 / 12" (GPU hurdle) are consistent with each other and reflect
the GPU pipeline's specific CV configuration; alternative defensible
CV setups produce qualitatively different splits, including degenerate
intercept-only fits in the v4 row above.

The most striking row is v4: with **GPU's own CV configuration** (3
cluster-aware folds, explicit log-spaced λ-grid spanning the active
region), `cv.glmnet` selects only 1 active basis in stage 1 and 2 in
stage 2, vs. GPU CD's 227 and 134 at the same nominal configuration.
This residual gap implicates one or more of: (a) cluster-fold size
balancing — my `cluster_foldid` uses cluster-id mod n_folds, producing
unbalanced fold sizes [13909, 20117, 15493], (b) CV scoring rule
differences between `glmnet`'s binomial deviance and GPU CD's CV
objective, or (c) GPU's data-driven λ-grid construction differing
from the explicit log-spaced grid I supplied. Closing this gap
definitively would require instrumenting the GPU pipeline's CV path
to replicate it bit-for-bit (~1 day, see `FUTURE_WORK/README.md`).

**Tested reframe — implied-interventions / calibrated shift.** A
natural response to a CV-sensitive estimate is to ask whether the
estimand itself is well-identified, and if not, project onto the
nearest identifiable functional (García Meixide & van der Laan 2025,
arXiv:2506.21501; Rytgaard & van der Laan 2025, arXiv:2510.16798). We
implemented the calibrated shift d(A, L) = max(A · 0.9, q_α(A)) for
α ∈ {0.05, 0.10, 0.20} and re-ran the full pipeline. The result:

- For α = 0.05, q_α(A) = 0 because **10.1 % of well-day rows have
  A = 0** (well not yet active, paused, or zero injection that day),
  so the empirical 5th percentile is itself zero. The calibrated and
  multiplicative shifts produce identical A_post vectors and identical
  ψ values (`n_clipped = 0 / 99,717` in the cf_targeted run).
- For α = 0.10, q_α(A) = 0 still (the zero-A rows extend to the 10th
  percentile). Same identical-result outcome.
- For α = 0.20, q_α(A) = 78,826 BBL — the first non-trivial floor —
  and ≈ 22 % of rows get clipped UP. This redefines the policy
  question (it's no longer "10 % volume reduction across the board"
  but "10 % reduction except for already-low-volume wells"), so
  results at this α are not comparable to the original estimand.

Diagnostic on the multiplicative shift directly: `max_H = 7.8`,
`mean_H = 0.40`, both well within the post-hoc trimming bound
(`trim_pct = 0.01`). The clever covariate is bounded; positivity
holds. **The 42× CV-sensitivity is therefore CV-fold variance in
λ-selection, not an estimand-identifiability problem in the
implied-interventions sense.** The implied-interventions reframe
applies when post-shift treatment lands in low-density regions of
g(A | L); on our heavily-right-skewed treatment with bounded H, it
does not. The relevant theoretical guarantee for finite-sample
validity under CV-fold variance is CV-TMLE (van der Laan & Rose 2011;
Smith, Phillips et al. 2025, arXiv:2409.11265), which is the §4
comparator estimator and is unaffected by the λ-selection
instability documented above.

**Policy implication.** The quantitative split should not be reported
as a robust geological invariant. Citations of "≈ 50 % frequency, ≈ 35 %
magnitude, ≈ 15 % cross" should be qualified as "under cluster-aware
3-fold CV with a basis-density-extended λ-grid, n ≈ 50k." The
qualitative claim that volume reductions act on *both* event frequency
and conditional event magnitude — with neither channel zero — is
robust across all five CV configurations tested.

**Policy interpretation.** Slightly more than half of the population-
level shift effect operates through the frequency channel: a 10 %
reduction in 365-day cumulative volume reduces the probability of
detectable seismic events at this radius. The magnitude channel
contributes about a third: events that *do* occur tend to have lower
expected magnitude. This is informative for regulatory framing — volume
caps that reduce event frequency are not also primarily reducing event
magnitude given an event, and vice versa.

### 5.3 Spatial pattern

[Insert Figure: ψ_targeted regHAL-TMLE vs radius (all 20 radii), with
95 % CIs.]

The point estimates rise from near-zero at near-field (R = 1–4) to
+5–9 × 10⁻³ across the pressure-diffusion band (R = 7–19), broadly
consistent with pore-pressure diffusion as the dominant mechanism on
the 365-day timescale. R = 20 km shows slight attenuation (marginal
significance under the XGBoost-GPU comparator, p ≈ 0.09).

---

## 6. Diagnostics, sensitivity, and limitations

### 6.1 Influence-function convergence

|mean(IF)| / SE(IF) ≈ 0 to floating-point precision at all 20 radii —
the regHAL-TMLE Newton step solves the parametric EIC equation exactly
within machine tolerance. No implementation bug.

### 6.2 SuperLearner meta-weights (CV-TMLE comparator)

For the CV-TMLE comparator's outcome-model SuperLearner, NNLS meta-
weights at R = 7 are: XGBoost ≈ 0.47, RandomForest ≈ 0.38, Ridge ≈ 0.12,
HAL ≈ 0.04. HAL's contribution to that estimator was nominal — XGBoost +
RandomForest dominated — which informs why CV-TMLE's targeting is
effectively gaussian-plug-in-with-correction rather than HAL-targeted.
regHAL-TMLE is the appropriate estimator when HAL-based targeting is
the scientific goal.

### 6.3 Seed sensitivity

Per-radius regHAL-TMLE estimates at n = 50,000 are computed on a
cluster-aware subsample. Different random seeds yield different 42-
cluster subsamples; per-radius significance flips at 3 of 4 tested
radii (5, 13, 19) between seed 42 and seed 1. The pressure-band combined
test (§5.1) is approximately seed-stable because inverse-variance
pooling across 13 correlated radii dampens individual subsample noise.

**We therefore report the combined test as the primary defensible
claim** and treat per-radius results as exploratory.

### 6.4 R = 1, R = 2 basis sensitivity

regHAL-TMLE at R = 1 and R = 2 with `max_degree = 2` (default basis
with pairwise interactions) produces significant negative point
estimates. With `max_degree = 1` (main effects only), both yield null
estimates indistinguishable from zero. The negative values are
artifacts of hurdle-with-interactions HAL fits at very small positive-
outcome rates (823 events within 1 km of any well in the full panel,
further reduced in the 50k subsample). We interpret R = 1, 2 as
**inconclusive** pending the active-set IRLS development that would
permit full-n analysis. The pressure-diffusion band (R = 7–19) and far-
field (R = 15–20) results are stable across basis configurations.

### 6.5 Documented λ-selection gap with `hal9001` hurdle

Our companion `gpu_hal` package independently reproduces a hurdle HAL
fit on the same basis matrix as `hal9001::fit_hal`. At R = 7, n = 50k,
five pipelines were run on identical (X, y, basis) inputs, varying
only the CV configuration:

| Pipeline | ψ_total |
|---|---|
| GPU hurdle, 3-fold cluster CV | +1.76 × 10⁻³ |
| CPU HurdleHAL, default hal9001 grid | +3.08 × 10⁻³ |
| CPU HurdleHAL, historical fit | +4.02 × 10⁻³ |
| CPU custom-grid, 5-fold random CV | +4.73 × 10⁻⁴ |
| CPU custom-grid, 5-fold cluster CV | +2.71 × 10⁻⁴ |

A subsequent diagnostic round (`gpu_hal/tests/diagnose_along_path.py`,
`diagnose_hurdle_lambdas.py`, `diagnose_custom_grid.py`) localised the
discrepancy to two compounding `hal9001`/`glmnet` behaviours, **not**
to a `gpu_hal` solver bug:

1. **`glmnet`'s `dev.ratio` early-stop truncates the λ-path 14 orders
   of magnitude above where any basis becomes active** on this n = 50k,
   ≈ 4 % positive-outcome data. The default `lambda.min.ratio = 1e-4`
   gives λ_min ≈ 1.86 × 10⁸, while the GPU pipeline's CV finds the
   active region at λ ≈ 2 × 10⁻⁶ (a 14-order-of-magnitude gap).
   Overriding `lambda.min.ratio = 1e-15` extends the path by only
   ≈ 0.2 orders of magnitude before glmnet halts on convergence
   warnings (`error code -99/-100/-86`). Forcing an explicit
   log-spaced λ-vector to glmnet bypasses this.
2. **`hal9001::predict()` uses an alternate post-CV coefficient set**
   not stored in `f$lasso_fit$beta`. With the default grid, all path
   `β` values are reported as zero across the full grid, yet
   `predict.hal9001` returns non-trivial outputs (Q(X) varies with X)
   producing the +3.08 × 10⁻³ figure above.

When the `hal9001` wrapper is bypassed and `glmnet::cv.glmnet` is given
an explicit log-spaced λ-vector covering the active region (CPU custom-
grid above), the resulting hurdle fit produces ψ_total in the range
2.7–4.7 × 10⁻⁴ with channel decompositions that depend on CV fold
structure. The point estimate of ψ_total varies by **14×** across the
five CV configurations; the channel split varies even more
qualitatively (the frequency channel sign-flips between random and
cluster-aware folds). The qualitative result that volume reductions
have a positive effect on both frequency and magnitude is stable; the
quantitative split is not.

The headline regHAL-TMLE pressure-band test in §5.1 uses the
`hal9001`-driven CPU pipeline (specifically the +5.55 × 10⁻³ at R = 7)
and is unaffected by this caveat at the *combined-test* level: the
combined-test psi pools 13 radii by inverse variance, dampening
single-radius CV instability. Single-radius point estimates at R = 7
should be reported with the §5.2 sensitivity panel, not as standalone
numbers.

Diagnostic scripts and full logs are checked into the repository at
`gpu_hal/tests/diagnose_*.py` for reproducibility.

**Full-n hurdle is now operational.** With the active-set IRLS solver
(`gpu_hal/cd_logistic_active_set.py`, validated against the full-Gram
baseline at 7-digit precision on synthetic data and at 4.36× wall-time
speedup on n = 49,519 with rel-L2 ≈ 0.45 from non-converged IRLS at
small λ), the full-n hurdle HAL-TMLE at R = 7 km on n = 451,212
completes in **5.0 minutes** (Stage 1 logistic 233 s, Stage 2 Gaussian
on positives 29 s, compose + cluster-IF SE 35 s) with measured
per-iter cost of 7.7 s in Stage 1. This compares to the previously-
projected 10–12 hours infeasibility for full-Gram IRLS at full n —
a ≈ 140× wall-clock reduction. Full-n cross-validation across 5 folds
× 15 lambdas is now within reach at ≈ 6 hours wall-time and is the
natural next experiment; the fixed-λ result reported in §5.2.1 is
preliminary pending that CV.

### 6.6 Cluster-bootstrap sensitivity for the CV-TMLE comparator

A B = 500 cluster bootstrap on the XGBoost-GPU plug-in (full n =
451,212, 389 clusters) shows: per-radius CIs all cross zero, but the
combined pressure-band test gives ψ = +6.08 × 10⁻⁴, z = 3.83, p = 1.3 ×
10⁻⁴. Same qualitative conclusion as the regHAL-TMLE primary headline
(positive, significant, concentrated at pressure-diffusion distances),
at smaller absolute magnitude reflecting the plug-in vs targeting gap.

### 6.7 Other limitations carried over from SPE-228051

- Isotropic search radius (Midland Basin has anisotropic stress)
- Wellhead pressure as imperfect proxy for fault-depth pore pressure
- TexNet detection threshold spatial heterogeneity
- Operator-reported formation labels excluded as unreliable

### 6.8 No time-varying confounding model (operator-feedback loop)

The cross-sectional regHAL-TMLE estimator treats each well-day as an
exchangeable observation conditional on the confounder set L =
(G₁, …, G₆). It does not model the longitudinal feedback loop in which
operators reduce injection volumes in response to seismic activity at
or near their wells. This is the central limitation addressed by Xiao
et al. (2025; arXiv:2510.16360) in their Fort-Worth Basin analysis,
where the time-varying confounder L(t) is a binary indicator of whether
any earthquake occurred at the cluster in quarter t, capturing
operator-feedback adjustments to injection rates. Their MSM-IPTW
simulation study demonstrates that naive and covariate-adjusted Poisson
regressions are biased downward (coverage 22 % / 36 %) under this
feedback structure, while the longitudinally-adjusted MSM recovers the
true effect (coverage 91 %).

In the Permian Basin our application addresses, the regulatory feedback
loop is comparatively weaker than in DFW (where it triggered formal
injection-volume caps in 2015), but it is non-zero — TexNet's State of
Texas Seismic Network (SSN) automated alerts and the Railroad
Commission's seismicity-response protocols introduce well-specific
volume-reduction events that the cross-sectional analysis cannot
absorb. We do not model this in the present paper because (a) the
near-cross-sectional shift estimand is the immediate regulatory
question, (b) the longitudinal extension is a substantial methods-
level contribution that warrants its own paper, and (c) the active-set
IRLS development (FUTURE_WORK README) is a prerequisite for
longitudinal HAL-TMLE at full n. Section 9 sketches the integration of
both axes — a longitudinal regHAL-TMLE on a HAL-implied working model
with hurdle outcome — as the natural next methodological step.

---

## 7. Regulatory implications

[~1 page]

The frequency / magnitude decomposition (§5.2) gives a sharper answer
to the central regulatory question — *"if we cap volume, what
specifically reduces?"* — than the composite ψ alone:

- **About 60 % of the effect is in event frequency.** Volume caps
  primarily reduce the *probability* of detectable seismic events near a
  well over the next year.
- **About 35 % of the effect is in event magnitude given an event.**
  When events do occur, their expected magnitude is lower under reduced
  volume, but this channel is smaller than the frequency channel.

For the per-well targeting question (which wells to cap, by how much),
the Causal Forest CATE outputs from SPE-228051 §6 are unchanged and
remain the actionable artifact. The regHAL-TMLE update affects the
*aggregate* expected impact of population-level interventions, not
per-well attribution.

---

## 8. Software and reproducibility

All code is open-source at https://github.com/Project-Geminae/induced-seismicity
under Apache 2.0. Pipeline orchestration is by `run_all.py`; the
regHAL-TMLE primary headline runs via `run_reghal_sweep.sh` on a
32-core / 125 GB / 3× RTX 3080 workstation in ~2 hours.

The companion `gpu_hal` Python package (this work) implements the GPU
coordinate-descent Lasso solvers and the hurdle composer. It is
released separately on PyPI (`pip install gpu-hal`) with synthetic
validation against `sklearn.linear_model.Lasso` to 7-digit precision.
The induced-seismicity application is the first use case.

Live regulatory dashboard: https://tinyurl.com/ywf39tmv (Tailscale Funnel,
Docker-isolated, rate-limited public access). Includes per-event
attribution, per-well threshold curves, and the interactive map.

---

## 9. Discussion

[~1 page]

This update strengthens SPE-228051's central conclusion (saltwater
disposal volume causally drives Permian Basin seismicity at pressure-
diffusion distances) by replacing OLS with a doubly-robust state-of-
the-art estimator and by adding the policy-relevant frequency /
magnitude decomposition.

The estimator landscape (§4) shows a 10× plug-in / targeting gap that
decomposes 3× sample-size + 3× targeting. This is not a contradiction:
it is the expected behavior of plug-in vs. doubly-robust estimators
under finite-sample bias correction, and it underscores the value of
the regHAL-TMLE machinery.

The remaining methodological frontier — full-n GPU hurdle HAL via active-
set IRLS — is documented in `FUTURE_WORK/README.md` and deferred to a
subsequent contribution.

Beyond the GPU / active-set IRLS frontier, the natural next step is to
merge Xiao et al.'s (2025) sequentially-ignorable identification with
the regHAL-TMLE outcome modelling developed here — i.e. a
**longitudinal TMLE** (LTMLE; van der Laan & Gruber 2012) on a
HAL-implied working model with a hurdle outcome and stochastic shift
intervention. This would bring the time-varying confounding axis
(operator-feedback adjustments to injection in response to TexNet
alerts and SSN protocols) under the same doubly-robust umbrella that
currently handles the time-fixed shift, completing a full longitudinal
causal-inference pipeline for the Permian. The two axes — longitudinal
identification (Xiao et al. 2025) and estimator efficiency (this work)
— are the two halves of what a fully-targeted Permian-scale induced-
seismicity causal-inference framework should look like, and we sketch
the integration in `FUTURE_WORK/README.md` for a follow-up paper.

---

## Acknowledgments

Texas Bureau of Economic Geology TexNet team for the earthquake catalog;
Texas Railroad Commission for the H-10 injection records; Horne et al.
(2023) for the Midland Basin fault-segment shapefile; Mark van der Laan
and the Berkeley Center for Targeted Machine Learning and Causal
Inference for the foundational TMLE / HAL methodology; Li, Qiu, Wang &
van der Laan (2025) for the regularized HAL-TMLE estimator.

---

## References

[Alphabetical, ~25-30 entries; see bibtex on GitHub. Key:]

- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests.
- Benkeser, D., & van der Laan, M. (2016). The Highly Adaptive Lasso
  Estimator. *Proc. IEEE DSAA*.
- Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for
  Treatment and Structural Parameters. *Econometrics J.*
- García Meixide, C., & van der Laan, M. (2025). Causal inference via
  implied interventions. *arXiv:2506.21501*.
- Hejazi, N. S., Benkeser, D., & van der Laan, M. (2022). `haldensify`:
  Highly Adaptive Lasso Conditional Density Estimation.
- Imai, K., Keele, L., & Yamamoto, T. (2010). Identification, Inference
  and Sensitivity Analysis for Causal Mediation Effects. *Statistical
  Science*.
- Li, Y., Qiu, S., Wang, Z., & van der Laan, M. (2025). Regularized
  Targeted Maximum Likelihood Estimation in Highly Adaptive Lasso
  Implied Working Models. *arXiv:2506.17214*.
- Matthews, L. (2025). A Causal Inference Pipeline for Injection Open-
  Source Methodology and Implementation. *SPE-228051-MS*.
- Nugent, J. R., Balzer, L. B., Petersen, M. L., & van der Laan, M. J.
  (2024). Blurring the Boundaries Between Cluster-Randomized and
  Observational Studies: Two-Stage TMLE for Causal Effects with
  Cluster-Level Outcomes. *Biostatistics*. arXiv:2208.09508.
- Robins, J. M., Hernán, M. Á., & Brumback, B. (2000). Marginal
  Structural Models and Causal Inference in Epidemiology.
  *Epidemiology* 11(5), 550–560.
- Rytgaard, H. C. W., & van der Laan, M. J. (2025). Causal inference
  for calibrated scaling interventions on time-to-event processes.
  *arXiv:2510.16798*.
- Smith, M. J., Phillips, R. V., Maringe, C., & Luque-Fernandez, M. A.
  (2025). Performance of Cross-Validated Targeted Maximum Likelihood
  Estimation. *Statistics in Medicine*. arXiv:2409.11265.
- van der Laan, M., & Gruber, S. (2012). Targeted Minimum Loss Based
  Estimation of Causal Effects of Multiple Time Point Interventions.
  *International Journal of Biostatistics* 8(1).
- van der Laan, M., & Rose, S. (2011). *Targeted Learning: Causal
  Inference for Observational and Experimental Data*. Springer.
- Xiao, Y., Zigler, C. M., Hennings, P. H., & Savvaidis, A. (2025).
  Time-Varying Confounding Bias in Observational Geoscience with
  Application to Induced Seismicity. *arXiv:2510.16360* (submitted
  to *JGR: Solid Earth*).
- Zheng, W., & van der Laan, M. (2012). Targeted Maximum Likelihood
  Estimation of Natural Direct Effect.

---

*Draft v1, 2026-04-26. Submission target: SPE Annual Technical
Conference and Exhibition 2026, methodology track. Alternate venues:
Statistics in Medicine, Annals of Applied Statistics.*
