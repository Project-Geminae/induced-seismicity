# Causal Inference for Injection-Induced Seismicity

**A doubly-robust TMLE + Causal Forest framework for attributing Permian Basin earthquakes to specific saltwater disposal wells.**

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://tinyurl.com/ywf39tmv)
[![SPE-228051](https://img.shields.io/badge/SPE-228051-blue)](https://doi.org/10.2118/228051-MS)

> **Paper:** Matthews, Hennings & Lund Snee (2025). *A Causal Inference Framework for Assessing the Relationship Between Saltwater Disposal and Induced Seismicity in the Permian Basin.* SPE-228051-MS.

---

## What This Does

This pipeline takes publicly available earthquake and injection data from the Permian Basin and answers three questions a regulator or operator would ask:

1. **Population level:** "If all wells reduce volume by 10%, how much does earthquake risk drop?" → **TMLE shift intervention**
2. **Per-well level:** "How much did *this specific well* contribute to *this specific earthquake?*" → **Causal Forest CATE**
3. **Threshold level:** "At what volume does this well's contribution cross the regulatory limit?" → **Per-well dose-response curve**

### Live Dashboard

**https://tinyurl.com/ywf39tmv**

Interactive map of 7,424 TexNet earthquakes + 1,056 RRC SWD wells. Click any event → per-well CATE waterfall, dose-response curve, injection timeline, threshold curve. PDF report export. Mobile-friendly.

---

## Causal DAG

```
        G₁   G₂   G₃   G₄
        (fault dist, fault count, depth, days active)
         ↓     ↓    ↓    ↓     ← controlled for
         W  →  P  →  S
      (volume) (pressure) (seismicity)
```

- **W → P → S**: Injection volume builds pore pressure, which triggers fault slip
- **G → W, G → S**: Geology confounds both injection location and fault activity
- By controlling for G₁–G₄, the remaining W→P→S link is **causal, not correlational**

---

## Key Results (TMLE v2, April 2026)

### Shift Intervention (10% volume reduction)

| Radii significant | Mean ψ (ML) | Mean CI width | Mean max H |
|---|---|---|---|
| **17 / 20** | +8.8e-3 | 6.7e-3 | **5.3** |

*Previous TMLE (v1) had 0/20 significant radii due to positivity violations (max H = 335). Seven methodological fixes resolved this.*

### Per-Well Attribution (M4.8 event, texnet2025edml, 10 km)

| Wells | Σ CATE | Top contributor | Significant wells |
|---|---|---|---|
| 44 | +0.51 ML | API 31741240 (+0.10 ML) | 6 (CI excludes zero) |

### TMLE v2 Methodological Improvements

| Fix | Impact |
|---|---|
| NNLS metalearner (was RidgeCV) | Enforces convex combination of base learners |
| Data-adaptive density bins (was 20 fixed) | ~140 bins via Freedman-Diaconis rule |
| Data-adaptive positivity floor | 1% of 2.5th percentile (was fixed 0.5%) |
| H truncation at 99th percentile | Max H: 335 → 5.3 (targeting step stabilized) |
| Bessel-corrected cluster SE | Honest SEs with finite-cluster correction |
| 5-fold CV (was 3-fold) | More honest cross-fitted Q estimates |
| RandomForest added to SuperLearner | 4 diverse base learners (was 3) |

### Sensitivity Analyses

- **Formation vs depth-class proxy**: Total effect ±6–169% across radii; indirect (pressure) pathway robust (±7–18%). Direct effect is sensitive — use population-level results for policy. ([`formation_sensitivity.csv`](formation_sensitivity.csv))
- **Injection rate as confounder**: Rejected due to multicollinearity (corr=0.89 with treatment). Rate is a deterministic function of cumulative volume. ([`rate_definition_check.py`](rate_definition_check.py))
- **W×M interaction**: Not significant (p=0.91 at 7km). Mediation decomposition is defensible. ([`model_improvements.py`](model_improvements.py))
- **Positivity diagnostics**: Dose-response at 10⁸ BBL is pure extrapolation (0 observations). 10⁷ BBL is P99. ([`positivity_diagnostics.csv`](positivity_diagnostics.csv))

---

## Data

All data comes from public sources:

| Source | Records | Date |
|---|---|---|
| [TexNet](https://texnet.beg.utexas.edu) earthquake catalog | 7,424 quality-filtered events | Apr 2026 |
| [RRC](https://www.rrc.texas.gov/) H-10 injection reports | 1,056 SWD wells, 903,887 well-day cells | Apr 2026 |
| Horne et al. (2023) fault maps | Midland Basin basement fault segments | Static |

### Automated Ingestion

```bash
# Daily: fetch new earthquakes from IRIS FDSN + check Google Drive for updated injection data
python auto_ingest.py --events-only --check-drive

# Full rebuild (after new injection data):
python auto_ingest.py --full --parallel 16
```

---

## Pipeline

```bash
# Full pipeline (steps 0-15, ~30 min with --parallel-radii 16):
python run_all.py --parallel-radii 16

# Or run individual steps:
python run_all.py --only 0 1 2 3 4    # ingest + panel + spatial join + geoscience
python run_all.py --only 9 10 11       # TMLE shift + dose-response + mediation
```

| Step | Script | Output | Time |
|---|---|---|---|
| 0 | `swd_data_import.py` | `swd_data_filtered.csv` | 5s |
| 1 | `seismic_data_import.py` | `texnet_events_filtered.csv` | 2s |
| 2 | `build_well_day_panel.py` | `well_day_panel.csv` (320 MB) | 15s |
| 3 | `spatiotemporal_join.py` | `panel_with_outcomes_*.csv` × 20 | 2 min |
| 4 | `add_geoscience_to_panel.py` | `panel_with_faults_*.csv` × 20 | 2 min |
| 5-8 | `dowhy_*.py` | OLS mediation results | 5 min |
| 9 | `tmle_shift_analysis.py` | `tmle_shift_365d_*.csv` | 1.5 hr |
| 10 | `tmle_dose_response.py` | `tmle_dose_response_365d_*.csv` | 1.5 hr |
| 11 | `tmle_mediation_analysis.py` | `tmle_mediation_365d_*.csv` | 30 min |

### Causal Forest (per-well attribution)

```bash
# Train at all 20 radii (~27 min with 4 workers):
python build_causal_forest.py --radii 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --workers 4
```

---

## Dashboard

### Local Development

```bash
pip install -r requirements.txt
uvicorn dashboard.server:app --host 0.0.0.0 --port 8765
```

### Docker Deployment (production)

```bash
cd dashboard/
docker build -t seis-dashboard .
docker run -d --name seis-dashboard \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  --memory 8g --cpus 8 \
  -p 127.0.0.1:8766:8765 \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  seis-dashboard
```

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Landing page (primer + methodology overview) |
| `GET /dashboard` | Interactive map + attribution terminal |
| `GET /faq` | FAQ for regulators and operators |
| `GET /methodology` | Technical methodology |
| `GET /api/health` | Data freshness + model dates |
| `GET /api/events` | Earthquake catalog (map data) |
| `GET /api/event/{id}/wells` | Wells within radius with panel features |
| `GET /api/event/{id}/attribution` | Per-well CATE from Causal Forest |
| `GET /api/event/{id}/report` | Downloadable PDF report |
| `GET /api/wells/{api}/threshold` | Per-well volume threshold curve |
| `GET /api/wells/{api}/timeseries` | Daily injection + pressure history |
| `GET /api/tmle/summary` | Population TMLE context at radius |

---

## Installation

```bash
git clone https://github.com/Project-Geminae/induced-seismicity.git
cd induced-seismicity
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Setup

1. Download TexNet event + injection CSVs from the Google Drive links in `auto_ingest.py`
2. Place in `~/Downloads/`
3. Run: `python auto_ingest.py --full --parallel 16`

---

## Project Structure

```
induced-seismicity/
├── README.md                      # This file
├── CHANGES.md                     # Detailed changelog (old vs new pipeline)
├── requirements.txt               # Python dependencies
├── run_all.py                     # Pipeline orchestrator
│
├── # ── Data Ingestion ──
├── swd_data_import.py             # Filter RRC SWD injection data
├── seismic_data_import.py         # Filter TexNet earthquake catalog
├── auto_ingest.py                 # Automated daily FDSN + Drive polling
│
├── # ── Panel Construction ──
├── build_well_day_panel.py        # (well, day) panel with rolling features + BHP
├── spatiotemporal_join.py         # Link earthquakes to wells by radius
├── add_geoscience_to_panel.py     # Fault distance + segment count
├── column_maps.py                 # Single source of truth for column names
│
├── # ── Causal Analysis (OLS) ──
├── causal_core.py                 # OLS mediation + cluster bootstrap
├── dowhy_simple_all.py            # Well-day OLS (no bootstrap)
├── dowhy_ci.py                    # Well-day OLS + bootstrap CIs
├── dowhy_simple_all_aggregate.py  # Event-level OLS
├── dowhy_ci_aggregated.py         # Event-level OLS + bootstrap CIs
│
├── # ── Causal Analysis (TMLE) ──
├── tmle_core.py                   # TMLE primitives (SuperLearner, density, IF)
├── tmle_shift_analysis.py         # Stochastic shift intervention
├── tmle_dose_response.py          # Dose-response curve E[Y_a]
├── tmle_mediation_analysis.py     # NDE/NIE mediation decomposition
├── tmle_run_parallel.py           # Parallel TMLE driver (ProcessPoolExecutor)
│
├── # ── Per-Well Attribution ──
├── build_causal_forest.py         # CausalForestDML (EconML) per radius
├── build_attribution_q.py         # Legacy TMLE Q models (superseded by CF)
│
├── # ── Sensitivity / Diagnostics ──
├── positivity_diagnostics.py      # TMLE positivity & H-statistic report
├── formation_sensitivity.py       # Depth-proxy vs formation robustness
├── model_improvements.py          # 4-improvement impact assessment
├── rate_definition_check.py       # Injection rate multicollinearity check
├── compare_trim_results.py        # Trim vs no-trim TMLE comparison
├── evalue_sensitivity.py          # E-value bounds for unmeasured confounding
│
├── # ── Visualization ──
├── tmle_visualizations.py         # TMLE result plots
├── causal_poe_curves.py           # Probability-of-exceedance curves
├── induced_seismicity_scaling_plots.py
├── killer_visualizations.py
│
├── # ── Dashboard ──
├── dashboard/
│   ├── __init__.py
│   ├── server.py                  # FastAPI backend
│   ├── Dockerfile                 # Production container
│   ├── entrypoint.py              # Container startup
│   └── templates/
│       ├── index.html             # Bloomberg-terminal interactive map
│       ├── landing.html           # Landing page with primer
│       ├── faq.html               # Regulatory FAQ
│       ├── methodology.html       # Technical methodology
│       └── analytics.html         # Traffic analytics (key-gated)
│
├── # ── Reference Data ──
├── Horne_et_al._2023_MB_BSMT_FSP_V1.shp  # Fault segments shapefile
└── tmle_r_crossvalidation.R       # R tlverse cross-validation script
```

---

## Citation

```bibtex
@inproceedings{matthews2025causal,
  title={A Causal Inference Framework for Assessing the Relationship Between
         Saltwater Disposal and Induced Seismicity in the Permian Basin},
  author={Matthews, Lewis and Hennings, Peter and Lund Snee, Jens-Erik},
  booktitle={SPE Annual Technical Conference and Exhibition},
  year={2025},
  publisher={Society of Petroleum Engineers},
  doi={10.2118/228051-MS}
}
```

## License

Apache 2.0
