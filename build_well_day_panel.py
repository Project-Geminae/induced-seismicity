#!/usr/bin/env python3
"""
build_well_day_panel.py
───────────────────────
Construct the (well, day) panel that downstream causal analysis consumes.

For every Midland-Basin SWD well, build a dense daily calendar from its first
to its last reported injection record. For each (well, day) cell compute:

  • Same-day injection volume (BBL) and average/max wellhead pressure (PSIG)
  • Cumulative injection volume over rolling lookback windows of
    30 / 90 / 180 / 365 days
  • Volume-weighted mean wellhead pressure over the same windows
    (Σ P_i V_i  /  Σ V_i)
  • Depth-corrected bottom-hole-pressure estimate:
        BHP_est = WHP + 0.45 psi/ft * z_perf_midpoint_ft
    where z_perf_midpoint = mean(Completed Injection Interval Top/Bottom),
    falling back to Tubing Depth or 0.95 * Well Total Depth.
  • Days-active (days since the well's first record)
  • Primary formation (categorical; first entry of `Current Injection
    Formations`, "UNKNOWN" when missing)

Inputs
------
  swd_data_filtered.csv  ← produced by swd_data_import.py

Outputs
-------
  well_day_panel.csv     (~689k rows × ~30 cols)

Notes
-----
This script REPLACES the temporal half of the old pipeline that lived in
filter_active_wells_before_events.py. The spatial half is now in
spatiotemporal_join.py. The "innocent wells with magnitude=0" trick from the
old pipeline is gone — non-event days are introduced naturally by the dense
calendar, and the outcome is set to 0 only when no qualifying earthquake
occurred within the radius on that day.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────── Constants ──────────────────────────────────
INFILE  = Path("swd_data_filtered.csv")
OUTFILE = Path("well_day_panel.csv")

LOOKBACK_DAYS = [30, 90, 180, 365]   # rolling-window lengths

BRINE_GRADIENT_PSI_PER_FT = 0.45     # ~10.5 ppg SWD brine

# ──────────────────── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────── Helpers ────────────────────────────────────
def primary_formation(s: object) -> str:
    """First entry of a pipe-delimited formation string; UNKNOWN if blank.

    Whitespace is normalized so that 'CLEAR FORK' / 'CLEARFORK' / 'Clear  Fork'
    all collapse to a single label.
    """
    if not isinstance(s, str) or not s.strip():
        return "UNKNOWN"
    raw = s.split("|", 1)[0].strip().upper()
    return " ".join(raw.replace("-", " ").split())  # collapse internal whitespace


def estimate_perf_depth_ft(row: pd.Series) -> float:
    """Best-effort injection-interval midpoint in feet."""
    top = row["Completed Injection Interval Top"]
    bot = row["Completed Injection Interval Bottom"]
    if pd.notna(top) and pd.notna(bot):
        return float((top + bot) / 2)
    if pd.notna(row["Injection Top Interval"]) and pd.notna(row["Injection Bottom Interval"]):
        return float((row["Injection Top Interval"] + row["Injection Bottom Interval"]) / 2)
    if pd.notna(row["Tubing Depth (ft.)"]):
        return float(row["Tubing Depth (ft.)"])
    if pd.notna(row["Well Total Depth ft"]):
        return float(row["Well Total Depth ft"]) * 0.95
    return float("nan")


def build_well_metadata(swd: pd.DataFrame) -> pd.DataFrame:
    """One row per API: lat/lon, depth, formation. Constant across all days."""
    log.info("Aggregating per-well metadata (lat/lon, depth, formation)…")
    meta = (
        swd
        .groupby("API Number", as_index=False)
        .agg({
            "Surface Latitude":               "first",
            "Surface Longitude":              "first",
            "Completed Injection Interval Top":    "median",
            "Completed Injection Interval Bottom": "median",
            "Injection Top Interval":              "median",
            "Injection Bottom Interval":           "median",
            "Tubing Depth (ft.)":                  "median",
            "Well Total Depth ft":                 "median",
            "Current Injection Formations":        "first",
            "Permit Number":                       "first",
        })
    )
    meta["perf_depth_ft"] = meta.apply(estimate_perf_depth_ft, axis=1)
    meta["formation"] = meta["Current Injection Formations"].apply(primary_formation)

    n_unknown_depth = meta["perf_depth_ft"].isna().sum()
    if n_unknown_depth:
        log.warning("⚠️   %d well(s) have no usable depth — filling with global median",
                    n_unknown_depth)
        meta["perf_depth_ft"] = meta["perf_depth_ft"].fillna(meta["perf_depth_ft"].median())

    keep = [
        "API Number", "Surface Latitude", "Surface Longitude",
        "perf_depth_ft", "formation", "Permit Number",
    ]
    return meta[keep]


def collapse_duplicate_well_days(swd: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple records for the same (API, day): sum volume, mean pressures."""
    n_before = len(swd)
    collapsed = (
        swd
        .groupby(["API Number", "Date of Injection"], as_index=False)
        .agg({
            "Volume Injected (BBLs)":          "sum",
            "Injection Pressure Average PSIG": "mean",
            "Injection Pressure Max PSIG":     "mean",
        })
    )
    n_dups = n_before - len(collapsed)
    if n_dups:
        log.info("Collapsed %d duplicate (API, day) rows", n_dups)
    return collapsed


def make_dense_calendar(reports: pd.DataFrame) -> pd.DataFrame:
    """For each well: dense daily index from its first to last reported day.

    Missing days get Volume=0 and pressures=NaN.
    """
    log.info("Densifying calendar for each well…")
    pieces = []
    for api, g in reports.groupby("API Number", sort=False):
        idx = pd.date_range(g["Date of Injection"].min(),
                            g["Date of Injection"].max(),
                            freq="D")
        g = (g.set_index("Date of Injection")
              .reindex(idx)
              .rename_axis("Date of Injection")
              .reset_index())
        g["API Number"] = api
        g["Volume Injected (BBLs)"] = g["Volume Injected (BBLs)"].fillna(0.0)
        # leave pressures as NaN on no-injection days
        pieces.append(g)
    panel = pd.concat(pieces, ignore_index=True)
    log.info("Dense panel rows: %d (vs %d reported)", len(panel), len(reports))
    return panel


def add_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Cumulative volume and volume-weighted mean pressure over each lookback."""
    log.info("Computing rolling cumulative volume and volume-weighted pressure…")
    panel = panel.sort_values(["API Number", "Date of Injection"]).reset_index(drop=True)

    # P_avg * V and P_max * V for the volume-weighted means.
    # Treat missing pressure as 0 contribution (multiplied by V which may also be 0).
    p_avg = panel["Injection Pressure Average PSIG"].fillna(0.0)
    p_max = panel["Injection Pressure Max PSIG"].fillna(0.0)
    v     = panel["Volume Injected (BBLs)"]

    panel["_pv_avg"] = p_avg * v
    panel["_pv_max"] = p_max * v

    grp = panel.groupby("API Number", sort=False)

    for w in LOOKBACK_DAYS:
        win = f"{w}D"
        # Rolling sums must be indexed by date for time-aware windowing.
        sums = (
            panel.set_index("Date of Injection")
                 .groupby("API Number", sort=False)[["Volume Injected (BBLs)", "_pv_avg", "_pv_max"]]
                 .rolling(win, closed="left")  # exclude same-day to avoid look-ahead
                 .sum()
                 .reset_index(level=0, drop=True)
                 .reset_index(drop=True)
        )
        cum_v = sums["Volume Injected (BBLs)"]
        cum_p_avg = sums["_pv_avg"]
        cum_p_max = sums["_pv_max"]

        panel[f"cum_vol_{w}d_BBL"] = cum_v.values
        with np.errstate(divide="ignore", invalid="ignore"):
            panel[f"vw_avg_psig_{w}d"] = np.where(cum_v > 0, cum_p_avg / cum_v, np.nan)
            panel[f"vw_max_psig_{w}d"] = np.where(cum_v > 0, cum_p_max / cum_v, np.nan)

    panel = panel.drop(columns=["_pv_avg", "_pv_max"])
    return panel


def add_bhp_features(panel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Depth-corrected BHP estimate: BHP = WHP + 0.45 psi/ft * perf_depth_ft.

    Computed for both same-day pressures and the volume-weighted lookback means.
    """
    log.info("Computing depth-corrected BHP estimates…")
    panel = panel.merge(meta[["API Number", "perf_depth_ft"]], on="API Number", how="left")

    column_weight = BRINE_GRADIENT_PSI_PER_FT * panel["perf_depth_ft"]

    # Same-day BHP (from raw same-day WHP, NOT lookback-weighted)
    panel["bhp_avg_psi"] = panel["Injection Pressure Average PSIG"] + column_weight
    panel["bhp_max_psi"] = panel["Injection Pressure Max PSIG"]     + column_weight

    # Lookback BHP from each volume-weighted WHP window
    for w in LOOKBACK_DAYS:
        panel[f"bhp_vw_avg_{w}d"] = panel[f"vw_avg_psig_{w}d"] + column_weight
        panel[f"bhp_vw_max_{w}d"] = panel[f"vw_max_psig_{w}d"] + column_weight

    return panel


def add_well_age(panel: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing days_active per well…")
    first_seen = panel.groupby("API Number")["Date of Injection"].transform("min")
    panel["days_active"] = (panel["Date of Injection"] - first_seen).dt.days
    return panel


# ──────────────────── Main ───────────────────────────────────────
def main() -> None:
    if not INFILE.exists():
        sys.exit(f"❌  {INFILE} not found — run swd_data_import.py first")

    log.info("📄  Loading %s …", INFILE)
    swd = pd.read_csv(INFILE, low_memory=False)
    swd["Date of Injection"] = pd.to_datetime(swd["Date of Injection"]).dt.normalize()
    log.info("Loaded %d rows × %d cols", *swd.shape)

    # Drop bad volume rows (negatives + NaN) — measured 12 of these
    n0 = len(swd)
    swd = swd[(swd["Volume Injected (BBLs)"] >= 0) & swd["Volume Injected (BBLs)"].notna()]
    log.info("Dropped %d rows with negative or NaN volume", n0 - len(swd))

    # Per-well metadata (lat/lon, depth, formation) — constant across days
    meta = build_well_metadata(swd)
    log.info("Per-well metadata: %d wells", len(meta))

    # Collapse same-day duplicates
    reports = collapse_duplicate_well_days(
        swd[["API Number", "Date of Injection",
             "Volume Injected (BBLs)",
             "Injection Pressure Average PSIG",
             "Injection Pressure Max PSIG"]]
    )

    # Build dense daily calendar per well
    panel = make_dense_calendar(reports)

    # Rolling features
    panel = add_rolling_features(panel)

    # BHP features (depend on per-well perf_depth)
    panel = add_bhp_features(panel, meta)

    # Days active
    panel = add_well_age(panel)

    # Attach static metadata
    panel = panel.merge(
        meta[["API Number", "Surface Latitude", "Surface Longitude",
              "perf_depth_ft", "formation"]],
        on="API Number", how="left", suffixes=("", "_meta"),
    )
    # If perf_depth was already on panel from add_bhp_features, drop the duplicate
    if "perf_depth_ft_meta" in panel.columns:
        panel = panel.drop(columns=["perf_depth_ft_meta"])

    # Final column order
    static_cols = [
        "API Number", "Date of Injection", "days_active",
        "Surface Latitude", "Surface Longitude", "perf_depth_ft", "formation",
    ]
    same_day_cols = [
        "Volume Injected (BBLs)",
        "Injection Pressure Average PSIG",
        "Injection Pressure Max PSIG",
        "bhp_avg_psi", "bhp_max_psi",
    ]
    rolling_cols = []
    for w in LOOKBACK_DAYS:
        rolling_cols += [
            f"cum_vol_{w}d_BBL",
            f"vw_avg_psig_{w}d", f"vw_max_psig_{w}d",
            f"bhp_vw_avg_{w}d",  f"bhp_vw_max_{w}d",
        ]
    panel = panel[static_cols + same_day_cols + rolling_cols]

    log.info("Writing %s (%d rows × %d cols)…", OUTFILE, *panel.shape)
    panel.to_csv(OUTFILE, index=False)
    log.info("✅  Done")

    # Print a quick sanity summary
    print("\n📊  Panel sanity summary:")
    print(f"   Wells:                  {panel['API Number'].nunique():,}")
    print(f"   Date range:             {panel['Date of Injection'].min().date()} → {panel['Date of Injection'].max().date()}")
    print(f"   Total well-days:        {len(panel):,}")
    print(f"   Active well-days (V>0): {(panel['Volume Injected (BBLs)']>0).sum():,}")
    print(f"   Idle well-days   (V=0): {(panel['Volume Injected (BBLs)']==0).sum():,}")
    print(f"   Mean perf depth (ft):   {panel['perf_depth_ft'].mean():.0f}")
    print(f"   Top formations:")
    for f, n in panel.groupby("API Number")["formation"].first().value_counts().head(8).items():
        print(f"     {f:<25s} {n:>4d} wells")


if __name__ == "__main__":
    main()
