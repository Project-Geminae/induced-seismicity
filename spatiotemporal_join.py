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

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ──────────────────── Constants ──────────────────────────────────
PANEL_FILE  = Path("well_day_panel.csv")
EVENTS_FILE = Path("texnet_events_filtered.csv")
OUT_FMT     = "panel_with_outcomes_{R}km.csv"
LINKS_FMT   = "event_well_links_{R}km.csv"
LINKS_PARQUET = Path("event_well_links.parquet")
EVENT_INDEX_JSON = Path("event_index.json")

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For one radius, build outcome (max ML, event count) per (well, day) AND
    the per-(event, well) link table.

    Returns
    -------
    out : pd.DataFrame
        The panel left-joined with `outcome_max_ML` and `outcome_event_count`.
    links : pd.DataFrame
        One row per (event, nearby well) pair, with haversine distance, both
        coordinates, magnitude, and the radius_km that this row was joined
        at. Used by the dashboard to look up "wells near event X".
    """
    log.info("🔗  Radius %d km — querying %d events against %d wells…",
             int(radius_km), len(events), len(wells))

    radius_rad = radius_km / EARTH_RADIUS_KM

    # For each event, find all wells within radius — request distances too
    ev_coords = np.radians(events[["Latitude (WGS84)", "Longitude (WGS84)"]].values)
    nearby_well_idx, nearby_distances = tree.query_radius(
        ev_coords, r=radius_rad, return_distance=True,
    )

    # Build the link table AND the long (API, date, ML) table for the
    # downstream groupby aggregation.
    link_rows = []
    expanded = []
    for ev_pos, (well_indices, dists_rad) in enumerate(zip(nearby_well_idx, nearby_distances)):
        if len(well_indices) == 0:
            continue
        ev_row = events.iloc[ev_pos]
        for w_idx, d_rad in zip(well_indices, dists_rad):
            wrow = wells.iloc[w_idx]
            api = wrow["API Number"]
            distance_km = float(d_rad * EARTH_RADIUS_KM)
            link_rows.append({
                "EventID":         ev_row["EventID"],
                "API Number":      api,
                "event_date":      ev_row["Origin Date"],
                "event_lat":       float(ev_row["Latitude (WGS84)"]),
                "event_lon":       float(ev_row["Longitude (WGS84)"]),
                "well_lat":        float(wrow["Surface Latitude"]),
                "well_lon":        float(wrow["Surface Longitude"]),
                "local_magnitude": float(ev_row["Local Magnitude"]),
                "distance_km":     distance_km,
                "radius_km":       int(radius_km),
            })
            expanded.append((api, ev_row["Origin Date"], ev_row["Local Magnitude"]))

    links = pd.DataFrame(link_rows)

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
    log.info("    %d (well, day) cells with ≥1 nearby event, %d (event,well) link rows",
             len(outcomes), len(links))

    # Left-join onto panel; missing → 0
    out = panel.merge(outcomes, on=["API Number", "Date of Injection"], how="left")
    out["outcome_max_ML"] = out["outcome_max_ML"].fillna(0.0)
    out["outcome_event_count"] = out["outcome_event_count"].fillna(0).astype(int)

    n_event_cells = (out["outcome_max_ML"] > 0).sum()
    log.info("    Panel cells with outcome>0: %d (%.2f%%)",
             n_event_cells, 100 * n_event_cells / len(out))

    return out, links


def write_event_index(events: pd.DataFrame, all_links: pd.DataFrame) -> None:
    """Write event_index.json — one entry per event with summary metadata
    plus per-radius nearby-well counts. The dashboard map uses this to
    render markers without loading the full link table on page load.
    """
    log.info("Writing %s …", EVENT_INDEX_JSON)
    # Pre-compute nearby well counts per (event, radius)
    counts = (
        all_links
        .groupby(["EventID", "radius_km"])
        .size()
        .unstack(fill_value=0)
    )
    counts.columns = counts.columns.astype(str)

    index: dict[str, dict] = {}
    for _, ev in events.iterrows():
        eid = ev["EventID"]
        wc = counts.loc[eid].to_dict() if eid in counts.index else {}
        # Ensure all radii are present even if zero
        wc_full = {str(r): int(wc.get(str(r), 0)) for r in RADII_KM}
        index[eid] = {
            "lat":          float(ev["Latitude (WGS84)"]),
            "lon":          float(ev["Longitude (WGS84)"]),
            "date":         pd.Timestamp(ev["Origin Date"]).strftime("%Y-%m-%d"),
            "ml":           float(ev["Local Magnitude"]),
            "depth_km_msl": float(ev["Depth of Hypocenter (Km.  Rel to MSL)"])
                if pd.notna(ev["Depth of Hypocenter (Km.  Rel to MSL)"]) else None,
            "rms":          float(ev["RMS"]) if pd.notna(ev["RMS"]) else None,
            "phase_count":  int(ev["UsedPhaseCount"]) if pd.notna(ev["UsedPhaseCount"]) else None,
            "well_counts":  wc_full,
        }

    EVENT_INDEX_JSON.write_text(json.dumps(index, indent=None, separators=(",", ":")))
    log.info("✅  Wrote %s (%d events)", EVENT_INDEX_JSON, len(index))


def _process_one_radius(args: dict) -> dict:
    """Worker function for parallel-radius processing.

    Each worker re-loads the panel and events (BallTree isn't picklable)
    and processes a single radius. The ~3 sec data-load overhead is amortised
    across the radii since all workers load in parallel.
    """
    R           = args["R"]
    links_only  = args["links_only"]

    panel, events = load_inputs()
    wells, tree = build_well_tree(panel)
    out, links = join_one_radius(panel, events, wells, tree, R)

    if not links_only:
        outfile = OUT_FMT.format(R=R)
        out.to_csv(outfile, index=False)
        log.info("💾  Wrote %s (%d rows × %d cols)", outfile, *out.shape)

    links_file = LINKS_FMT.format(R=R)
    links.to_csv(links_file, index=False)
    log.info("💾  Wrote %s (%d rows)", links_file, len(links))

    return {"R": R, "links_file": links_file, "n_links": len(links)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--links-only",
        action="store_true",
        help="Skip the (well, day) panel rewrite. Only generate the new "
             "event_well_links_<R>km.csv files + event_well_links.parquet "
             "+ event_index.json. Useful for re-running just the dashboard "
             "data without paying the cost of the panel rewrite.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for per-radius processing. On minitim use "
             "--workers 16. Default 1 (sequential, for Mac).",
    )
    args = parser.parse_args()

    if args.workers <= 1:
        # ──── Sequential path (original behavior, for Mac) ────
        panel, events = load_inputs()
        wells, tree = build_well_tree(panel)
        log.info("Built BallTree on %d unique well locations", len(wells))

        all_links: list[pd.DataFrame] = []

        for R in RADII_KM:
            out, links = join_one_radius(panel, events, wells, tree, R)

            if not args.links_only:
                outfile = OUT_FMT.format(R=R)
                out.to_csv(outfile, index=False)
                log.info("💾  Wrote %s (%d rows × %d cols)", outfile, *out.shape)

            links_file = LINKS_FMT.format(R=R)
            links.to_csv(links_file, index=False)
            log.info("💾  Wrote %s (%d rows)", links_file, len(links))
            all_links.append(links)

    else:
        # ──── Parallel path (for minitim) ────
        log.info("🚀  Parallel mode: %d workers for %d radii", args.workers, len(RADII_KM))
        jobs = [{"R": R, "links_only": args.links_only} for R in RADII_KM]

        all_links = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one_radius, j): j for j in jobs}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    log.info("✓ R=%dkm complete (%d link rows)", res["R"], res["n_links"])
                    # Re-read the per-radius link CSV for consolidation
                    ldf = pd.read_csv(res["links_file"])
                    all_links.append(ldf)
                except Exception as e:
                    job = futures[fut]
                    log.error("✗ R=%dkm failed: %s", job["R"], e)

    # ──── Consolidate links + event index (both paths) ────
    # Re-load events for the event_index.json writer
    if not EVENTS_FILE.exists():
        sys.exit(f"❌  {EVENTS_FILE} not found")
    events = pd.read_csv(EVENTS_FILE, low_memory=False)
    events["Origin Date"] = pd.to_datetime(events["Origin Date"]).dt.normalize()

    if all_links:
        consolidated = pd.concat(all_links, ignore_index=True)
        log.info("Writing consolidated %s (%d rows)…", LINKS_PARQUET, len(consolidated))
        try:
            consolidated.to_parquet(LINKS_PARQUET, index=False, compression="snappy")
            log.info("✅  Wrote %s", LINKS_PARQUET)
        except Exception as e:
            fallback = LINKS_PARQUET.with_suffix(".csv")
            log.warning("Parquet write failed (%s); falling back to %s", e, fallback)
            consolidated.to_csv(fallback, index=False)

        write_event_index(events, consolidated)

    log.info("✅  All radii complete")


if __name__ == "__main__":
    main()
