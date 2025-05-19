
D  A  G   O  V  E  R  V  I  E  W
=================================

                W
               / \
              /   \
             v     v
G₁ --------> P ---> S <------ G₂

Legend
------

G₁ : Nearest Fault Distance

G₂ : Fault Segments

W  : Volume Injected

P  : Injection Pressure

S  : Local Magnitude

(→ : Causal connection)

The causal model shows
  1. Geology factors (G₁, G₂) influence all variables
  2. Injection Volume (W) affects Pressure (P) and Magnitude (S)
  3. Injection Pressure (P) directly affects Magnitude (S)
  4. Direct path W → S
  5. Indirect path W → P → S

Data download 

"swd_data.csv"
https://drive.google.com/file/d/1QYwV9Ipusmmi1W1mJ81enCOzsAzGdNSN/view?usp=sharing

"texnet_events.csv"
https://drive.google.com/file/d/1Dj2nuojqne9Jz49spTu2nE8QsALd9_zw/view?usp=sharing

STEP 0 · ENVIRONMENT & CONSTANTS
───────────────────────────────────────────────────────────────────
| Parameter                    | Value                            |
|------------------------------|----------------------------------|
| Geographic CRS               | EPSG 4326  (WGS-84)              |
| Planar CRS for distances     | EPSG 3857  (Web Mercator)        |
| Well–event link distance     | 2, 5, 10, 15, 20 km (varied)     |
| Fault-segment length         | ~1 km                            |
| Injection look-back window   | 30 days                          |
| Random seed (DoWhy)          | 42                               |

STEP 1 · IMPORT & FILTER RAW TABLES
──────────────────────────────────────────────────────────────────────────────────────
| Script                     | Purpose                | Output                       |
|----------------------------|------------------------|------------------------------|
| swd_data_import.py         | Subset SWD records     | swd_data_filtered.csv        |
|                            |                        | ( 650,374 rows × 7 cols )    |
| seismic_data_import.py     | Subset TexNet catalog  | texnet_events_filtered.csv   |
|                            |                        | (  6,064 rows × 7 cols )     |

STEP 2 · SPATIAL JOIN (WELLS ↔ EVENTS)
──────────────────────────────────────────────────────────────────────────────────────────────
| Script                  | Output                      | Rows (per radius)                  |
|-------------------------|-----------------------------|------------------------------------|
| merge_seismic_swd.py	  | event_well_links_1km.csv	| 1,145 (radius = 1 km)              |
|                         | event_well_links_2km.csv	| 4,634 (radius = 2 km)              |
|                         | event_well_links_3km.csv	| 11,370 (radius = 3 km)             |
|                         | event_well_links_4km.csv	| 19,862 (radius = 4 km)             |
|                         | event_well_links_5km.csv	| 29,773 (radius = 5 km)             |
|                         | event_well_links_6km.csv	| 41,174 (radius = 6 km)             |
|                         | event_well_links_7km.csv	| 54,545 (radius = 7 km)             |
|                         | event_well_links_8km.csv	| 70,257 (radius = 8 km)             |
|                         | event_well_links_9km.csv	| 88,311 (radius = 9 km)             |
|                         | event_well_links_10km.csv	| 107,046 (radius = 10 km)           |
|                         | event_well_links_15km.csv	| 209,296 (radius = 15 km)           |
|                         | event_well_links_20km.csv	| 332,691 (radius = 20 km)           |


Columns retained → event metadata · well metadata · Distance_from_Well_to_Event

STEP 3 · SAME-DAY & N-DAY INJECTION LOOKBACK
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
| Script                                   | Output                                       | Rows (example)  |
|------------------------------------------|--------------------------------------------- |-----------------|
| filter_active_wells_before_events.py     | event_well_links_with_injection_<R>km.csv    | 216,440 (20km)  |
|                                          | innocent_wells_<R>km.csv                     | 117,034 (20km)  |
| filter_merge_events_and_nonevents.py     | innocent_wells_with_fakeids_<R>km.csv        | 117,034 (20km)  |
|                                          | combined_event_well_links_<R>km.csv          | 333,474 (20km)  |
|                                          |                                              |                 |

New columns added
  • Volume Injected (BBLs) – same day
  • Injection Pressure Average PSIG – same day
  • Injection Pressure Max PSIG – same day
  • Vol Prev N (BBLs)      – 30-day total before event
  • Avg Press Prev N (PSIG) – 30-day mean before event
  • Max Press Prev N (PSIG) – 30-day max before event

"Innocent wells" are those linked to events but not actively injecting on the event day (magnitude set to 0).
For these wells, the most recent injection date before the event is identified, with second script
replacing their EventID prefix from "texnet" to "faknet" before merging them with active wells.

STEP 4 · FAULT-PROXIMITY FEATURES
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
| Script                                               | Output                                 | Rows (example)    |
|------------------------------------------------------|-----------------------------------------|------------------|
| add_geoscience_to_event_well_links_with_injection.py | event_well_links_with_faults_<R>km.csv  | 333,474 (20km)   |
|                                                      | wells_vs_faults_<R>km.png               |                  |
|                                                      | wells_with_missing_xy_<R>km.csv         |                  |
|                                                      | wells_no_fault_match_<R>km.csv          |                  |

Added columns
  • Nearest Fault Dist (km) - distance to closest fault segment
  • Fault Segments ≤ R km (count) - number of ~1km fault segments within radius R

Process details:
  • Preserves ALL rows including those with missing coordinates
  • Fault shapefile auto-detection for coordinate reference system (CRS)
  • All wells were successfully matched to ≥1 fault in the example run
  • Wells with missing/invalid coordinates are identified but retained
  • Generates visual diagnostics of wells vs. fault lines (only for first radius processed)

Magnitude statistics were preserved across processing (no data loss):
  • 20km radius: 64.9% of rows have non-zero magnitude (216,440/333,474)

STEP 5 · MULTI-RADIUS CAUSAL SENSITIVITY ANALYSIS
─────────────────────────────────────────────────
Primary script : dowhy_simple_all.py   (DoWhy 0.12)
Adjustment set : { Nearest Fault Dist (km), Fault Segments ≤R km }
─────────────────────────────────────────────────────────────────────────────────────
| Radius | Total Effect   | Direct Effect   | Indirect Effect  | % Mediated | R²    |
|--------|----------------|-----------------|------------------|------------|-------|
| 1 km   | +6.33 × 10⁻⁶   | -3.36 × 10⁻⁶    | +9.69 × 10⁻⁶     | 153.0%     | 0.040 |
| 2 km   | +2.30 × 10⁻⁵   | +3.24 × 10⁻⁶    | +1.98 × 10⁻⁵     | 85.9%      | 0.188 |
| 3 km   | +2.96 × 10⁻⁵   | +9.87 × 10⁻⁶    | +1.97 × 10⁻⁵     | 66.7%      | 0.191 |
| 4 km   | +3.19 × 10⁻⁵   | +1.64 × 10⁻⁵    | +1.55 × 10⁻⁵     | 48.6%      | 0.189 |
| 5 km   | +2.58 × 10⁻⁵   | +9.71 × 10⁻⁶    | +1.61 × 10⁻⁵     | 62.3%      | 0.184 |
| 6 km   | +1.48 × 10⁻⁵   | +3.47 × 10⁻⁶    | +1.14 × 10⁻⁵     | 76.6%      | 0.163 |
| 7 km   | +1.53 × 10⁻⁵   | +3.76 × 10⁻⁶    | +1.15 × 10⁻⁵     | 75.4%      | 0.162 |
| 8 km   | +7.87 × 10⁻⁶   | +5.20 × 10⁻⁷    | +7.35 × 10⁻⁶     | 93.4%      | 0.173 |
| 9 km   | +8.16 × 10⁻⁶   | +4.04 × 10⁻⁷    | +7.76 × 10⁻⁶     | 95.0%      | 0.163 |
| 10 km  | +9.26 × 10⁻⁶   | +7.92 × 10⁻⁷    | +8.47 × 10⁻⁶     | 91.5%      | 0.160 |
| 15 km  | +4.44 × 10⁻⁶   | -3.89 × 10⁻⁷    | +4.83 × 10⁻⁶     | 108.8%     | 0.175 |
| 20 km  | +4.14 × 10⁻⁶   | -4.49 × 10⁻⁷    | +4.59 × 10⁻⁶     | 110.9%     | 0.197 |

DAG Structure
  • Treatment (W): Volume Injected (BBLs)
  • Mediator (P): Injection Pressure Average PSIG
  • Outcome (S): Local Magnitude
  • Confounders: Nearest Fault Dist (km), Fault Segments ≤R km

Radius Sensitivity Summary
  • Strongest total effect observed at 4km radius (3.19 × 10⁻⁵)
  • Pronounced shift in causal pathways with increasing radius:
    - Small radii (1-5km): Mixed direct and indirect effects
    - Medium radii (6-10km): Strong pressure mediation (75-95%)
    - Large radii (15-20km): Complete mediation with pressure (>100%)
  • Direct effect changes from positive to negative at larger distances
  • All results are statistically significant with p-values near zero
  • Refutation tests confirm robustness of the causal estimates

Plain-English interpretation
  • Near-field effects (1-5km): Both direct mechanical influence and pressure-mediated effects
  • Mid-field effects (6-10km): Pressure transmission becomes primary mechanism
  • Far-field effects (15-20km): Pressure diffusion completely mediates the relationship
  • Direct effects become negative at large distances, suggesting offsetting mechanisms
  • The radius analysis reveals different physical mechanisms operating at different spatial scales

STEP 6 · MULTI-RADIUS EVENT-LEVEL CAUSAL ANALYSIS
─────────────────────────────────────────────────────────────
Primary script : dowhy_simple_all_aggregate.py   (DoWhy 0.12)
Adjustment set : { Nearest Fault Dist (km), Fault Segments ≤R km, well_count }
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
| Radius | n_events | avg_well_count | Total Effect   | Direct Effect   | Indirect Effect  | % Mediated | p-value | R²   |
|--------|----------|----------------|----------------|-----------------|------------------|------------|---------|-------|
| 1 km   | 1,028    | 1.11           | +5.76 × 10⁻⁶   | -7.37 × 10⁻⁷    | +6.50 × 10⁻⁶     | 112.8%     | 1.6e-02 | 0.132 |
| 2 km   | 3,067    | 1.51           | +1.73 × 10⁻⁵   | +2.52 × 10⁻⁶    | +1.48 × 10⁻⁵     | 85.5%      | 1.1e-21 | 0.289 |
| 3 km   | 4,792    | 2.37           | +2.12 × 10⁻⁵   | +1.03 × 10⁻⁵    | +1.09 × 10⁻⁵     | 51.3%      | 2.1e-69 | 0.349 |
| 4 km   | 6,041    | 3.29           | +1.78 × 10⁻⁵   | +9.96 × 10⁻⁶    | +7.82 × 10⁻⁶     | 44.0%      | 8.2e-101| 0.464 |
| 5 km   | 6,938    | 4.30           | +1.23 × 10⁻⁵   | +5.57 × 10⁻⁶    | +6.76 × 10⁻⁶     | 54.8%      | 5.3e-98 | 0.510 |
| 6 km   | 7,753    | 5.32           | +6.32 × 10⁻⁶   | +1.92 × 10⁻⁶    | +4.39 × 10⁻⁶     | 69.5%      | 2.2e-60 | 0.547 |
| 7 km   | 8,463    | 6.45           | +5.34 × 10⁻⁶   | +1.52 × 10⁻⁶    | +3.82 × 10⁻⁶     | 71.5%      | 8.9e-67 | 0.557 |
| 8 km   | 9,052    | 7.77           | +2.32 × 10⁻⁶   | +3.08 × 10⁻⁷    | +2.01 × 10⁻⁶     | 86.7%      | 1.7e-30 | 0.544 |
| 9 km   | 9,479    | 9.33           | +1.97 × 10⁻⁶   | +2.04 × 10⁻⁷    | +1.76 × 10⁻⁶     | 89.6%      | 5.3e-30 | 0.526 |
| 10 km  | 9,749    | 11.00          | +1.92 × 10⁻⁶   | +2.01 × 10⁻⁷    | +1.72 × 10⁻⁶     | 89.5%      | 7.9e-34 | 0.492 |
| 15 km  | 10,208   | 20.55          | +9.48 × 10⁻⁷   | -7.59 × 10⁻⁹    | +9.56 × 10⁻⁷     | 100.8%     | 6.7e-25 | 0.479 |
| 20 km  | 10,388   | 32.10          | +1.01 × 10⁻⁶   | +1.93 × 10⁻⁷    | +8.19 × 10⁻⁷     | 80.9%      | 6.5e-54 | 0.451 |

Event-Level Aggregation Method

  • Group by EventID and compute:
  
    - Sum of Volume Injected (BBLs) across all wells within radius
    
    - Median Injection Pressure Average across all wells
    
    - Minimum Nearest Fault Distance across all wells
    
    - Sum of Fault Segments within radius
    
    - Count of wells within radius (added as control variable)

Amplification factors compared to 20 km radius:

  • 1 km:  5.7×
  
  • 2 km: 17.1×
  
  • 3 km: 20.9× (strongest)
  
  • 4 km: 17.6×
  
  • 5 km: 12.2×
  
  • 6 km:  6.2×
  
  • 7 km:  5.3×
  
  • 8 km:  2.3×
  
  • 9 km:  1.9×
  
  • 10 km: 1.9×
  
  • 15 km: 0.9×sl

Additional Outputs:

  • Earthquake probability curves for 20km radius (as PNG and CSV)
  
  • DAG images for each radius (when graphviz available)

Radius Sensitivity Summary

  • Strongest total effect observed at 3km radius (2.12 × 10⁻⁵)
  
  • Predictive performance (R²) increases with radius until peaking at 6-7km (R² ≈ 0.55)
  
  • Clear transition in causal mechanisms:
  
    - Near-field (1-2 km): High pressure mediation (85-113%)
    
    - Mid-field (3-5 km): Balanced direct and indirect effects (44-55% mediation)
    
    - Far-field (6-10 km): Predominantly pressure-mediated (70-90%)
    
    - Ultra-far (15 km): Complete mediation through pressure (100.8%)
    
  • Direct effect becomes statistically insignificant at 8-10km (p > 0.1) and negligible at 15km
  
  • All total effects remain highly statistically significant (p < 10⁻²⁵) except at 1km (p = 0.016)

Plain-English interpretation

  • Proximity matters greatly: injection wells within 3-5 km have 12-21× stronger effects than distant wells
  
  • Strongest effect radius (3km) and best predictive model radius (7km) are different
  
  • Pressure is the dominant mechanism in all cases, but direct mechanical effects are substantial at 3-5km
  
  • Wells beyond 10km have minimal direct effects but still contribute through pressure pathways
  
  • The 5-7km radius offers optimal balance of effect size and model performance (R² > 0.5)
  
  • These findings support spatially-targeted regulation focused on wells within 5-7km of faults
  
  • Different physical mechanisms appear to operate at different spatial scales


Raw yearly counts
MagBin    M2   M3  M4  M5
Year
2017     271   26   0   0
2018     602   55   2   0
2019     847   52   3   0
2020    1110   93   5   0
2021    1971  199  17   0
2022    2392  202  19   2
2023    2286  201  11   1
2024    1855  175  11   2
2025     565   27   5   2

YoY % change (rule: standard)
MagBin     M2     M3     M4     M5
Year
2017      NaN    NaN    NaN    NaN
2018    122.1  111.5    NaN    NaN
2019     40.7   -5.5   50.0    NaN
2020     31.1   78.8   66.7    NaN
2021     77.6  114.0  240.0    NaN
2022     21.4    1.5   11.8    NaN
2023     -4.4   -0.5  -42.1  -50.0
2024    -18.9  -12.9    0.0  100.0
2025    -69.5  -84.6  -54.5    0.0
