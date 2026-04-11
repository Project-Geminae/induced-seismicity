#!/usr/bin/env python3
"""
spatiotemporal_join.py
──────────────────────
Join the (well, day) panel to the filtered earthquake catalog.

For every radius R ∈ {1..20} km and every (well, day) cell in the panel,
the outcome is:

  outcome_max_ML = max Local Magnitude across all qualified earthquakes that
                   occurred on that calendar day within R km of the well's
                   surface location, OR 0 if no such event occurred.

Days with no nearby earthquake are LEGITIMATE zero-outcome controls — the
well actually existed and operated, no earthquake actually happened. This
replaces the old pipeline's synthetic-zero "innocent wells" trick that
silently overwrote real earthquake magnitudes with 0.

Inputs
------
  well_day_panel.csv          ← from build_well_day_panel.py
  texnet_events_filtered.csv  ← from seismic_data_import.py

Outputs
-------
  panel_with_outcomes_<R>km.csv  for R ∈ {1..20}

Each output is the panel with two extra columns:
  outcome_max_ML        : float, 0.0 on no-event days
  outcome_event_count   : int,   number of qualifying events that day

This script REPLACES merge_seismic_swd.py + filter_active_wells_before_events.py
+ filter_merge_events_and_nonevents.py from the old pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ──────────────────── Constants ──────────────────────────────────
PANEL_FILE  = Path("well_day_panel.csv")
EVENTS_FILE = Path("texnet_events_filtered.csv")
OUT_FMT     = "panel_with_outcomes_{R}km.csv"

RADII_KM = list(range(1, 21))  # 1..20 inclusive

EARTH_RADIUS_KM = 6371.0088

# ──────────────────── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────── Core ───────────────────────────────────────
def load_inputs():
    if not PANEL_FILE.exists():
        sys.exit(f"❌  {PANEL_FILE} not found — run build_well_day_panel.py first")
    if not EVENTS_FILE.exists():
        sys.exit(f"❌  {EVENTS_FILE} not found — run seismic_data_import.py first")

    log.info("📄  Loading panel: %s", PANEL_FILE)
    panel = pd.read_csv(PANEL_FILE, low_memory=False)
    panel["Date of Injection"] = pd.to_datetime(panel["Date of Injection"]).dt.normalize()
    log.info("Panel: %d well-days × %d cols", *panel.shape)

    log.info("📄  Loading events: %s", EVENTS_FILE)
    events = pd.read_csv(EVENTS_FILE, low_memory=False)
    events["Origin Date"] = pd.to_datetime(events["Origin Date"]).dt.normalize()
    log.info("Events: %d", len(events))

    return panel, events


def build_well_tree(panel: pd.DataFrame):
    """One row per well — used for spatial lookup of nearby wells per event."""
    wells = (
        panel[["API Number", "Surface Latitude", "Surface Longitude"]]
        .drop_duplicates(subset="API Number")
        .reset_index(drop=True)
    )
    coords_rad = np.radians(wells[["Surface Latitude", "Surface Longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    return wells, tree


def join_one_radius(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    wells: pd.DataFrame,
    tree: BallTree,
    radius_km: float,
) -> pd.DataFrame:
    """For one radius, build outcome (max ML, event count) per (well, day)."""
    log.info("🔗  Radius %d km — querying %d events against %d wells…",
             int(radius_km), len(events), len(wells))

    radius_rad = radius_km / EARTH_RADIUS_KM

    # For each event, find all wells within radius
    ev_coords = np.radians(events[["Latitude (WGS84)", "Longitude (WGS84)"]].values)
    nearby_well_idx = tree.query_radius(ev_coords, r=radius_rad)

    # Build a long table of (API, event_date, magnitude) — one row per
    # (event, nearby well) pair. We'll then groupby and pivot back.
    expanded = []
    for ev_pos, well_indices in enumerate(nearby_well_idx):
        if len(well_indices) == 0:
            continue
        ev_row = events.iloc[ev_pos]
        for w_idx in well_indices:
            expanded.append((
                wells.iloc[w_idx]["API Number"],
                ev_row["Origin Date"],
                ev_row["Local Magnitude"],
            ))

    if not expanded:
        log.warning("No (event, well) pairs at %d km — outcomes all zero", int(radius_km))
        outcomes = pd.DataFrame(columns=["API Number", "Date of Injection",
                                         "outcome_max_ML", "outcome_event_count"])
    else:
        ev_well = pd.DataFrame(expanded,
                               columns=["API Number", "Date of Injection", "Local Magnitude"])
        outcomes = (
            ev_well
            .groupby(["API Number", "Date of Injection"], as_index=False)
            .agg(outcome_max_ML=("Local Magnitude", "max"),
                 outcome_event_count=("Local Magnitude", "size"))
        )
    log.info("    %d (well, day) cells with ≥1 nearby event", len(outcomes))

    # Left-join onto panel; missing → 0
    out = panel.merge(outcomes, on=["API Number", "Date of Injection"], how="left")
    out["outcome_max_ML"] = out["outcome_max_ML"].fillna(0.0)
    out["outcome_event_count"] = out["outcome_event_count"].fillna(0).astype(int)

    n_event_cells = (out["outcome_max_ML"] > 0).sum()
    log.info("    Panel cells with outcome>0: %d (%.2f%%)",
             n_event_cells, 100 * n_event_cells / len(out))

    return out


def main() -> None:
    panel, events = load_inputs()
    wells, tree = build_well_tree(panel)
    log.info("Built BallTree on %d unique well locations", len(wells))

    for R in RADII_KM:
        out = join_one_radius(panel, events, wells, tree, R)
        outfile = OUT_FMT.format(R=R)
        out.to_csv(outfile, index=False)
        log.info("💾  Wrote %s (%d rows × %d cols)", outfile, *out.shape)

    log.info("✅  All radii complete")


if __name__ == "__main__":
    main()
