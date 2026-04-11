#!/usr/bin/env python3
"""
add_geoscience_to_panel.py
──────────────────────────
Compute well-level fault-proximity features ONCE and join them onto every
panel_with_outcomes_<R>km.csv file.

For every unique well in the panel (687 wells in the current dataset):

  • Nearest Fault Dist (km)              ← constant, computed once
  • Fault Segments ≤R km                 ← depends on R, computed per radius
                                          (counts ~1-km fault segments inside
                                           a circular buffer around each well)

Inputs
------
  well_day_panel.csv                    ← from build_well_day_panel.py
  panel_with_outcomes_<R>km.csv         ← from spatiotemporal_join.py
  Horne_et_al._2023_MB_BSMT_FSP_V1.shp  ← Midland-Basin basement faults

Outputs
-------
  panel_with_faults_<R>km.csv            for R ∈ {1..20}
  well_fault_features.csv                (per-well static features, one-time)
  wells_vs_faults.png                    (sanity-check map, written once)

This script REPLACES add_geoscience_to_event_well_links_with_injection.py from
the old pipeline. The key efficiency win is that fault features are static at
the well level — the old script recomputed them for every (well, event) pair,
which scaled with the link table size; this version scales with the well count.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box
from shapely.ops import substring
from tqdm import tqdm

# ──────────────────── Constants ──────────────────────────────────
PANEL_FILE = Path("well_day_panel.csv")
FAULT_SHP  = Path("Horne_et_al._2023_MB_BSMT_FSP_V1.shp")

WELL_FAULT_OUT = Path("well_fault_features.csv")
PANEL_OUT_FMT  = "panel_with_faults_{R}km.csv"
PANEL_IN_FMT   = "panel_with_outcomes_{R}km.csv"
SANITY_PNG     = Path("wells_vs_faults.png")

RADII_KM = list(range(1, 21))

SEG_LEN_M  = 1000.0          # ~1 km segments
PROJ_CRS   = "EPSG:3857"     # planar metres for buffering / distance

# Candidate CRSs to try if the shapefile's declared CRS doesn't overlap wells
CANDIDATE_CRS = {
    "NAD83 / UTM 13N": "EPSG:26913",
    "NAD83 / UTM 14N": "EPSG:26914",
    "Web-Mercator":    "EPSG:3857",
}

# ──────────────────── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


@contextmanager
def phase(msg: str):
    log.info("⏳  %s …", msg)
    t0 = time.perf_counter()
    yield
    log.info("✅  %s — %.1fs", msg, time.perf_counter() - t0)


# ──────────────────── Geometry helpers ───────────────────────────
def split_ls(ls: LineString, seg_m: float) -> List[LineString]:
    """Split a LineString into ≈seg_m-metre chunks."""
    if ls is None or ls.is_empty:
        return []
    length = ls.length
    if length == 0 or math.isnan(length):
        return [ls] if length > 0 else []
    if length <= seg_m:
        return [ls]
    return [substring(ls, s, min(s + seg_m, length))
            for s in range(0, int(length), int(seg_m))]


def explode_faults(gdf: gpd.GeoDataFrame, seg_m: float) -> gpd.GeoDataFrame:
    pieces: list[LineString] = []
    for geom in tqdm(gdf.geometry, desc="splitting faults", unit="line"):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            pieces.extend(split_ls(geom, seg_m))
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                pieces.extend(split_ls(part, seg_m))
    return gpd.GeoDataFrame(geometry=pieces, crs=gdf.crs)


def fix_invalid(gdf: gpd.GeoDataFrame, tag: str) -> gpd.GeoDataFrame:
    bad = ~gdf.geometry.is_valid
    if bad.any():
        log.warning("Repairing %d invalid %s geometries via buffer(0)",
                    bad.sum(), tag)
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)
    return gdf.dropna(subset=["geometry"]).loc[lambda d: ~d.geometry.is_empty]


def overlap(a: gpd.GeoSeries, b: gpd.GeoSeries) -> bool:
    return box(*a.total_bounds).intersects(box(*b.total_bounds))


def reproject_faults(faults: gpd.GeoDataFrame,
                     wells_wgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Try declared CRS first; if that doesn't overlap wells, sweep candidates."""
    if faults.crs is not None:
        try:
            if overlap(wells_wgs.geometry, faults.to_crs("EPSG:4326").geometry):
                return faults
        except Exception:
            pass
    log.warning("Faults don't overlap wells in declared CRS — sweeping candidates")
    for label, epsg in CANDIDATE_CRS.items():
        try:
            cand = faults.set_crs(epsg, allow_override=True)
            if overlap(wells_wgs.geometry, cand.to_crs("EPSG:4326").geometry):
                log.info("🌎  Using %s (%s) for faults", label, epsg)
                return cand
        except Exception:
            continue
    log.error("❌  No overlapping CRS found — assuming EPSG:4326 (matches may be 0)")
    return faults.set_crs("EPSG:4326", allow_override=True)


# ──────────────────── Core ───────────────────────────────────────
def load_unique_wells() -> gpd.GeoDataFrame:
    if not PANEL_FILE.exists():
        sys.exit(f"❌  {PANEL_FILE} not found — run build_well_day_panel.py first")
    panel = pd.read_csv(PANEL_FILE,
                        usecols=["API Number", "Surface Latitude", "Surface Longitude"],
                        low_memory=False)
    wells = panel.drop_duplicates(subset="API Number").reset_index(drop=True)
    log.info("Unique wells in panel: %d", len(wells))
    return gpd.GeoDataFrame(
        wells,
        geometry=gpd.points_from_xy(wells["Surface Longitude"], wells["Surface Latitude"]),
        crs="EPSG:4326",
    )


def load_faults(wells_wgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not FAULT_SHP.exists():
        sys.exit(f"❌  {FAULT_SHP} not found")
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    with phase("Loading fault shapefile"):
        raw = gpd.read_file(FAULT_SHP)
    raw = fix_invalid(raw, "faults")
    return reproject_faults(raw, wells_wgs)


def compute_well_fault_features(wells_wgs: gpd.GeoDataFrame,
                                segs_m: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-well: nearest fault distance + fault segments inside each radius buffer."""
    wells_m = wells_wgs.to_crs(PROJ_CRS)

    with phase("Nearest-fault search"):
        nearest = gpd.sjoin_nearest(
            wells_m, segs_m[["geometry"]], how="left", distance_col="nearest_m"
        ).groupby(level=0)["nearest_m"].min().reindex(wells_m.index)

    out = pd.DataFrame({
        "API Number": wells_wgs["API Number"].values,
        "Nearest Fault Dist (km)": nearest.values / 1000.0,
    })

    for R in RADII_KM:
        with phase(f"Counting segments inside {R}-km buffers"):
            buffers = wells_m.geometry.buffer(R * 1000.0)
            seg_counts = (
                gpd.GeoDataFrame(geometry=buffers, crs=PROJ_CRS)
                .sjoin(segs_m, predicate="intersects")
                .groupby(level=0).size()
                .reindex(wells_m.index)
                .fillna(0).astype(int)
            )
        out[f"Fault Segments <= {R} km"] = seg_counts.values

    return out


def join_to_panels(well_features: pd.DataFrame) -> None:
    """For each panel_with_outcomes_<R>km.csv, attach the relevant fault columns."""
    for R in RADII_KM:
        infile  = Path(PANEL_IN_FMT.format(R=R))
        outfile = Path(PANEL_OUT_FMT.format(R=R))
        if not infile.exists():
            log.warning("⚠️   %s not found, skipping", infile)
            continue
        log.info("Joining fault features into %s", infile.name)
        panel = pd.read_csv(infile, low_memory=False)
        cols  = ["API Number", "Nearest Fault Dist (km)", f"Fault Segments <= {R} km"]
        out   = panel.merge(well_features[cols], on="API Number", how="left")
        out.to_csv(outfile, index=False)
        log.info("    → %s (%d rows × %d cols)", outfile.name, *out.shape)


def write_sanity_map(wells: gpd.GeoDataFrame, faults: gpd.GeoDataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    faults.to_crs("EPSG:4326").plot(ax=ax, linewidth=0.6, color="red",
                                    alpha=0.7, label="Fault polylines")
    wells.plot(ax=ax, markersize=4, color="blue", alpha=0.6, label="Wells")
    ax.set_title("Midland Basin SWD wells (blue) and basement faults (red)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SANITY_PNG, dpi=180)
    plt.close()
    log.info("Wrote sanity map → %s", SANITY_PNG)


# ──────────────────── Main ───────────────────────────────────────
def main() -> None:
    wells = load_unique_wells()
    faults = load_faults(wells)
    log.info("Faults loaded: %d polylines", len(faults))

    faults_m = faults.to_crs(PROJ_CRS)
    with phase(f"Splitting faults into {SEG_LEN_M:.0f}-m segments"):
        segs_m = explode_faults(faults_m, SEG_LEN_M)
    log.info("Total fault segments: %d", len(segs_m))

    well_features = compute_well_fault_features(wells, segs_m)
    well_features.to_csv(WELL_FAULT_OUT, index=False)
    log.info("Wrote per-well fault features → %s", WELL_FAULT_OUT)

    write_sanity_map(wells, faults)

    join_to_panels(well_features)
    log.info("✅  All panels augmented with fault features")


if __name__ == "__main__":
    main()
