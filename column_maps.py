"""
column_maps.py
──────────────
Single source of truth for column names used by every causal-analysis script.

The old pipeline used substring auto-detection (`COL_FRAGS = {"W": ["volume",
"bbl"], ...}`) which silently picked whichever matching column appeared first
in the CSV — and so depended on column ordering. That's how the 30-day
pressure lookback ended up being computed but never used (the lookback
columns were named `Avg Press Prev N (PSIG)`, with "Press" not "Pressure",
so the substring filter never matched them).

This module replaces all of that with explicit, named column lists. Every
analysis script imports from here. If a column name in the panel changes,
this is the only file you need to update.
"""

# Lookback windows offered by build_well_day_panel.py
LOOKBACK_WINDOWS = [30, 90, 180, 365]


def cum_volume_col(window_days: int) -> str:
    """Cumulative injection volume over the past `window_days` days (BBL)."""
    return f"cum_vol_{window_days}d_BBL"


def vw_avg_psig_col(window_days: int) -> str:
    """Volume-weighted mean wellhead average pressure over the window (PSIG)."""
    return f"vw_avg_psig_{window_days}d"


def vw_max_psig_col(window_days: int) -> str:
    """Volume-weighted mean of wellhead max pressure over the window (PSIG)."""
    return f"vw_max_psig_{window_days}d"


def bhp_vw_avg_col(window_days: int) -> str:
    """Depth-corrected BHP from volume-weighted average WHP (psi)."""
    return f"bhp_vw_avg_{window_days}d"


def bhp_vw_max_col(window_days: int) -> str:
    """Depth-corrected BHP from volume-weighted max WHP (psi)."""
    return f"bhp_vw_max_{window_days}d"


def fault_segment_col(radius_km: int) -> str:
    """Per-radius fault segment count column (set by add_geoscience_to_panel.py)."""
    return f"Fault Segments <= {radius_km} km"


# Static columns (constant across all radii / windows)
COL_NEAREST_FAULT_KM = "Nearest Fault Dist (km)"
COL_PERF_DEPTH_FT    = "perf_depth_ft"
COL_FORMATION        = "formation"
COL_DAYS_ACTIVE      = "days_active"
COL_API              = "API Number"
COL_DATE             = "Date of Injection"
COL_LAT              = "Surface Latitude"
COL_LON              = "Surface Longitude"

# Outcome columns (set by spatiotemporal_join.py)
COL_OUTCOME_MAX_ML      = "outcome_max_ML"
COL_OUTCOME_EVENT_COUNT = "outcome_event_count"

# Same-day raw observations (kept for diagnostics, not used as treatment/mediator)
COL_SAME_DAY_VOL = "Volume Injected (BBLs)"
COL_SAME_DAY_AVG = "Injection Pressure Average PSIG"
COL_SAME_DAY_MAX = "Injection Pressure Max PSIG"


def confounder_columns(radius_km: int) -> list[str]:
    """Confounders G1..G4 used in the causal DAG.

    G1 = Nearest fault distance (km)
    G2 = Fault segment count within radius
    G3 = Injection interval midpoint depth (ft)
    G4 = Days since well first injected
    G5 = Total cum_vol_365d from all SWD wells within 7 km (spatial interference)
         Addresses the SUTVA violation: nearby wells share pressure fields.
         Including neighbor volume allows estimation of "effect of well i's
         injection, holding nearby wells' total injection constant."

    Note 1: formation was previously a confounder (one-hot from operator-reported
    `Current Injection Formations`). Dropped because RRC labels are unreliable.
    Replaced with a depth-class proxy (shallow/mid/deep) computed in
    causal_core.build_design_matrix(). Sensitivity analysis (formation_sensitivity.csv)
    shows the indirect (pressure-mediated) effect is robust to this substitution,
    but the direct effect is sensitive.

    Note 2: avg_rate_365d (avg daily injection rate) was tested as a confounder
    but rejected (rate_definition_check.py). Cumulative volume = rate × duration,
    so any rate definition is mechanically near-collinear with the treatment
    (correlation 0.84-0.89, VIF 3.7-5.5). Adding it inflated CATE estimates 12×
    with proportionally wider CIs, leaving zero wells statistically significant.
    The current model interprets effects as policy-relevant cumulative-volume
    effects averaged over the rate distribution.
    """
    return [
        COL_NEAREST_FAULT_KM,
        fault_segment_col(radius_km),
        COL_PERF_DEPTH_FT,
        COL_DAYS_ACTIVE,
        "neighbor_cum_vol_7km",
    ]


def treatment_column(window_days: int) -> str:
    """The treatment W = cumulative injection volume over the lookback window."""
    return cum_volume_col(window_days)


def mediator_column(window_days: int, use_bhp: bool = True) -> str:
    """The mediator P. By default uses depth-corrected BHP, not raw WHP.

    The old pipeline used same-day WHP — physically unmotivated because it
    ignores the column-weight contribution to bottom-hole pressure and the
    pore-pressure-diffusion timescale. The new default is BHP estimated from
    the volume-weighted average WHP over the same lookback window as W.
    """
    return bhp_vw_avg_col(window_days) if use_bhp else vw_avg_psig_col(window_days)


def outcome_column() -> str:
    return COL_OUTCOME_MAX_ML
