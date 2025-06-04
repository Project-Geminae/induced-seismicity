#!/usr/bin/env python3
# add_geoscience_to_event_well_links_with_injection.py
"""
Batch augment multiple combined_event_well_links_<R>km.csv files with
fault-proximity metrics for *every* row in each file.

Outputs (for each radius R)
-------
event_well_links_with_faults_<R>km.csv   – ALL rows from input preserved
wells_with_missing_xy_<R>km.csv          – rows with bad/missing lat-lon (for reference only)
wells_no_fault_match_<R>km.csv           – rows that still found no fault
wells_vs_faults_<R>km.png                – full map of all wells + fault lines (only for first radius)
fault_bounds_check.png                   – bbox plot if things don't overlap (only once)

Key May-2025 fixes
──────────────────
- Processes multiple input files in a single run
- Preserves ALL rows from input files, even those with missing coordinates
- Robust CRS auto-detection for the fault shapefile
- Keeps NaN-length fault lines whole
- Collapses duplicate rows from `sjoin_nearest` (ties)
- Timed banners / tqdm so the console never looks frozen
"""
# ────────────────────────────────────────────────────────────────
from pathlib import Path
import os, sys, math, time, logging, re
from contextlib import contextmanager
from typing import List

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, box
from shapely.ops import substring
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt

    PLOT = True  # basic bbox plot + full map
except ImportError:
    PLOT = False

# ─── USER CONFIG ───────────────────────────────────────────────
# List of input files to process
LINKAGE_FILES = [
    "combined_event_well_links_1km.csv",
    "combined_event_well_links_2km.csv",
    "combined_event_well_links_3km.csv",
    "combined_event_well_links_4km.csv",
    "combined_event_well_links_5km.csv",
    "combined_event_well_links_6km.csv",
    "combined_event_well_links_7km.csv",
    "combined_event_well_links_8km.csv",
    "combined_event_well_links_9km.csv",
    "combined_event_well_links_10km.csv",
    "combined_event_well_links_11km.csv",
    "combined_event_well_links_12km.csv",
    "combined_event_well_links_13km.csv",
    "combined_event_well_links_14km.csv",
    "combined_event_well_links_15km.csv",
    "combined_event_well_links_16km.csv",
    "combined_event_well_links_17km.csv",
    "combined_event_well_links_18km.csv",
    "combined_event_well_links_19km.csv",
    "combined_event_well_links_20km.csv"
]

FAULT_SHP = "Horne_et_al._2023_MB_BSMT_FSP_V1.shp"

# Output file templates - {R} will be replaced with the radius value
OUT_CSV_TEMPLATE = "event_well_links_with_faults_{R}km.csv"
BAD_XY_CSV_TEMPLATE = "wells_with_missing_xy_{R}km.csv"  # Changed from "dropped" to "with"
NO_MATCH_CSV_TEMPLATE = "wells_no_fault_match_{R}km.csv"
FULL_MAP_PNG_TEMPLATE = "wells_vs_faults_{R}km.png"

# Only created once if needed
BOUNDS_PLOT = "fault_bounds_check.png"

PLOT_FULL_MAP = True  # flip to False to skip
SEG_LEN_KM = 1.0
PROJ_CRS = "EPSG:3857"  # planar for metres
LAT, LON = "Surface Latitude", "Surface Longitude"

LOG_LEVEL = "INFO"  # "DEBUG" for more chatter
# ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─── tiny helper for timed phases ──────────────────────────────
@contextmanager
def phase(msg: str):
    log.info("⏳  %s …", msg)
    t0 = time.perf_counter()
    yield
    log.info("✅  %s – done in %.1fs", msg, time.perf_counter() - t0)


# ─── geometry helpers ─────────────────────────────────────────
def split_ls(ls: LineString, seg_m: float) -> List[LineString]:
    """Split a LineString into ≈seg_m-metre chunks (keeps NaN-length whole)."""
    if ls is None or ls.is_empty:
        return []
    length = ls.length
    if length == 0:
        return []
    if math.isnan(length) or length <= seg_m:
        return [ls]
    return [substring(ls, s, min(s + seg_m, length))
            for s in range(0, int(length), int(seg_m))]


def explode_faults(gdf: gpd.GeoDataFrame, seg_m: float) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of ≈1-km fault segments with a progress bar."""
    pieces: list[LineString] = []
    for geom in tqdm(gdf.geometry, desc="faults split", unit="fault"):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            pieces.extend(split_ls(geom, seg_m))
        elif geom.geom_type == "MultiLineString":
            for part in geom:
                pieces.extend(split_ls(part, seg_m))
    return gpd.GeoDataFrame(geometry=pieces, crs=gdf.crs)


def fix_invalid(gdf: gpd.GeoDataFrame, tag: str) -> gpd.GeoDataFrame:
    """Repair invalid geometries with buffer(0) then drop empties."""
    bad = ~gdf.geometry.is_valid
    if bad.any():
        log.warning("Repairing %s invalid %s geometries via buffer(0)",
                    bad.sum(), tag)
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)
    return gdf.dropna(subset=["geometry"]).loc[lambda d: ~d.geometry.is_empty]


# ─── CRS auto-detection for faults ─────────────────────────────
CANDIDATE_CRS = {
    "NAD83 / UTM 13N": "EPSG:26913",
    "NAD83 / UTM 14N": "EPSG:26914",
    "Web-Mercator": "EPSG:3857",
}


def overlap(wells_wgs: gpd.GeoSeries, faults_any: gpd.GeoSeries) -> bool:
    """Return True if the two bounding boxes intersect."""
    return box(*wells_wgs.total_bounds).intersects(box(*faults_any.total_bounds))


def reproject_faults(faults: gpd.GeoDataFrame,
                     wells_wgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return faults in *some* CRS that overlaps the wells (or EPSG:4326)."""
    if faults.crs is not None:
        try:
            if overlap(wells_wgs, faults.to_crs("EPSG:4326").geometry):
                return faults
        except Exception:
            pass
    log.warning("Faults don't overlap wells in declared CRS – trying candidates …")
    for label, epsg in CANDIDATE_CRS.items():
        try:
            cand = faults.set_crs(epsg, allow_override=True)
            if overlap(wells_wgs, cand.to_crs("EPSG:4326").geometry):
                log.info("🌎  Using %s (%s) for faults (bbox now overlaps)", label, epsg)
                return cand
        except Exception:
            continue
    log.error("❌  No overlapping CRS found – assuming EPSG:4326 (matches may be zero)")
    return faults.set_crs("EPSG:4326", allow_override=True)


# ─── Extract radius from filename ────────────────────────────────
def get_radius_from_filename(filename: str) -> int:
    """Extract the radius value from a filename using regex."""
    match = re.search(r'(\d+)km', filename)
    if match:
        return int(match.group(1))
    return 10  # Default fallback value


# ─── Process one file ────────────────────────────────────────────
def process_file(linkage_file: str,
                 faults: gpd.GeoDataFrame = None,
                 segs: gpd.GeoDataFrame = None,
                 first_file: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Process a single linkage file and return faults and segments for reuse."""

    # Extract radius from filename
    radius_km = get_radius_from_filename(linkage_file)

    # Create output filenames based on templates and radius
    out_csv = OUT_CSV_TEMPLATE.format(R=radius_km)
    bad_xy_csv = BAD_XY_CSV_TEMPLATE.format(R=radius_km)
    no_match_csv = NO_MATCH_CSV_TEMPLATE.format(R=radius_km)
    full_map_png = FULL_MAP_PNG_TEMPLATE.format(R=radius_km)

    log.info("\n%s\nPROCESSING %s (radius: %d km)", "═" * 78, linkage_file, radius_km)
    t_start = time.perf_counter()

    # 1 ▸ linkage table -----------------------------------------------
    with phase(f"Loading linkage CSV ({linkage_file})"):
        if not Path(linkage_file).exists():
            log.error(f"❌  {linkage_file} not found, skipping")
            return faults, segs
        df = pd.read_csv(linkage_file, low_memory=False)
    log.info("Rows in linkage CSV : %s", f"{len(df):,}")

    # Identify rows with valid coordinates (but don't filter them out)
    good_coords = (
            df[[LAT, LON]].notna().all(axis=1)
            & df[LAT].between(-90, 90) & df[LON].between(-180, 180)
    )

    # Save rows with invalid/missing coordinates for reference
    if (~good_coords).any():
        df.loc[~good_coords].to_csv(bad_xy_csv, index=False)
        log.warning("%s rows have bad/missing coordinates (logged to %s but keeping all rows)",
                    (~good_coords).sum(), bad_xy_csv)

    # Log magnitude stats if available
    if "Local Magnitude" in df.columns:
        mag_col = "Local Magnitude"
        all_mag_mean = df[mag_col].mean()
        all_mag_nonzero = (df[mag_col] > 0).sum()
        good_mag_mean = df.loc[good_coords, mag_col].mean()
        good_mag_nonzero = (df.loc[good_coords, mag_col] > 0).sum()
        bad_mag_mean = df.loc[~good_coords, mag_col].mean() if (~good_coords).any() else 0
        bad_mag_nonzero = (df.loc[~good_coords, mag_col] > 0).sum() if (~good_coords).any() else 0

        log.info("Magnitude stats for ALL rows: mean=%.3f, non-zero=%d/%d (%.1f%%)",
                 all_mag_mean, all_mag_nonzero, len(df),
                 100 * all_mag_nonzero / len(df) if len(df) > 0 else 0)
        log.info("Magnitude stats for rows with GOOD coordinates: mean=%.3f, non-zero=%d/%d (%.1f%%)",
                 good_mag_mean, good_mag_nonzero, good_coords.sum(),
                 100 * good_mag_nonzero / good_coords.sum() if good_coords.sum() > 0 else 0)
        if (~good_coords).any():
            log.info("Magnitude stats for rows with BAD coordinates: mean=%.3f, non-zero=%d/%d (%.1f%%)",
                     bad_mag_mean, bad_mag_nonzero, (~good_coords).sum(),
                     100 * bad_mag_nonzero / (~good_coords).sum() if (~good_coords).sum() > 0 else 0)

    # Create a subset with only valid coordinates for geospatial operations
    df_good = df.loc[good_coords].copy()

    # Continue with geospatial processing using only the good coordinates
    wells = gpd.GeoDataFrame(df_good,
                             geometry=gpd.points_from_xy(df_good[LON], df_good[LAT]),
                             crs="EPSG:4326")
    wells = fix_invalid(wells, "wells")

    # 2 ▸ faults -- only load if not already loaded ------------------------------
    if faults is None:
        with phase("Loading fault shapefile"):
            if not Path(FAULT_SHP).exists():
                sys.exit(f"❌  {FAULT_SHP} not found")
            os.environ["SHAPE_RESTORE_SHX"] = "YES"
            faults_raw = gpd.read_file(FAULT_SHP)

        faults_raw = fix_invalid(faults_raw, "faults")
        faults = reproject_faults(faults_raw, wells)
        log.info("Fault polylines after CRS fix : %s", f"{len(faults):,}")

    # 3 ▸ reproject & explode -- only explode if not already done --------------
    wells_m = wells.to_crs(PROJ_CRS)
    faults_m = faults.to_crs(PROJ_CRS)

    if segs is None:
        with phase("Splitting faults into 1-km segments"):
            segs = explode_faults(faults_m, SEG_LEN_KM * 1_000)
        log.info("Total 1-km segments : %s", f"{len(segs):,}")

    # 4 ▸ nearest-fault distance -------------------------------------
    with phase("Nearest-fault search"):
        nearest_raw = gpd.sjoin_nearest(
            wells_m, segs[["geometry"]],
            how="left", distance_col="nearest_m"
        )[["nearest_m"]]

    nearest = (
        nearest_raw.groupby(level=0)["nearest_m"].min()  # shortest per well
        .to_frame().reindex(wells_m.index)
    )
    nearest["Nearest Fault Dist (km)"] = nearest["nearest_m"] / 1_000

    # 5 ▸ segment count inside buffer --------------------------------
    with phase(f"Counting segments ≤{radius_km} km"):
        buffers = wells_m.geometry.buffer(radius_km * 1_000)
        seg_counts = (
            gpd.GeoDataFrame(geometry=buffers, crs=PROJ_CRS)
            .sjoin(segs, predicate="intersects")
            .groupby(level=0).size()
            .reindex(wells_m.index).fillna(0).astype(int)
        )
    nearest[f"Fault Segments ≤{radius_km} km"] = seg_counts.values

    # 6 ▸ unresolved wells? -----------------------------------------
    nomatch = nearest["nearest_m"].isna()
    if nomatch.any():
        wells.loc[nomatch].drop(columns="geometry").to_csv(no_match_csv, index=False)
        log.warning("%s rows still found **no** fault match → %s",
                    nomatch.sum(), no_match_csv)

        if PLOT and first_file:  # bounding-box diagnostic only if needed and first file
            wbox = box(*wells.total_bounds)
            fbox = box(*faults.to_crs("EPSG:4326").total_bounds)
            fig, ax = plt.subplots(figsize=(6, 5))
            gpd.GeoSeries([wbox], crs="EPSG:4326").plot(
                ax=ax, facecolor="none", edgecolor="blue", label="Wells bbox")
            gpd.GeoSeries([fbox], crs="EPSG:4326").plot(
                ax=ax, facecolor="none", edgecolor="red", label="Faults bbox")
            ax.set_title("Bounding boxes – blue=wells  red=faults")
            ax.legend()
            plt.savefig(BOUNDS_PLOT, dpi=180)
            plt.close()
            log.info("Wrote bounding-box plot → %s", BOUNDS_PLOT)
    else:
        log.info("All wells matched to ≥1 fault ✔️")

    # 7 ▸ merge & write ---------------------------------------------
    # Initialize fault metrics columns in the full dataframe
    df["Nearest Fault Dist (km)"] = pd.NA
    df[f"Fault Segments ≤{radius_km} km"] = pd.NA

    # Update only the rows with good coordinates
    # Create a mapping from original index to good subset index
    idx_map = {old: new for new, old in enumerate(df_good.index)}

    # Update metrics only for rows with good coordinates
    for idx in df_good.index:
        if idx in idx_map:
            new_idx = idx_map[idx]
            if new_idx < len(nearest):
                df.loc[idx, "Nearest Fault Dist (km)"] = nearest.iloc[new_idx]["Nearest Fault Dist (km)"]
                df.loc[idx, f"Fault Segments ≤{radius_km} km"] = seg_counts.iloc[new_idx] if new_idx < len(
                    seg_counts) else pd.NA

    df.to_csv(out_csv, index=False)
    log.info("Saved augmented CSV with ALL rows → %s", out_csv)

    # 8 ▸ FULL MAP PNG - only for the first file -------------------------
    if PLOT and PLOT_FULL_MAP and first_file:
        with phase(f"Writing full wells_vs_faults map for {radius_km}km"):
            fig, ax = plt.subplots(figsize=(7, 7))
            faults.to_crs("EPSG:4326").plot(ax=ax, linewidth=0.6,
                                            edgecolor="red", alpha=0.7,
                                            label="Fault polylines")
            wells.plot(ax=ax, markersize=4, color="blue", alpha=0.6,
                       label="Wells")
            ax.set_title(f"All wells (blue) & fault polylines (red) - {radius_km}km")
            ax.legend()
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.tight_layout()
            plt.savefig(full_map_png, dpi=200)
            plt.close()
        log.info("Wrote full map → %s", full_map_png)

    log.info("🎉  Finished processing %s – runtime %.1fs",
             linkage_file, time.perf_counter() - t_start)

    # Return the faults and segments for reuse with next file
    return faults, segs


# ─── MAIN ──────────────────────────────────────────────────────
def main() -> None:
    log.info("\n%s\nBATCH FAULT-PROXIMITY AUGMENTATION starting …", "═" * 78)
    log.info("Processing %d input files", len(LINKAGE_FILES))

    # Record overall start time
    t_overall_start = time.perf_counter()

    # Process each file, reusing faults and segments for efficiency
    faults = None
    segs = None

    for i, linkage_file in enumerate(LINKAGE_FILES):
        log.info("\nProcessing file %d of %d: %s", i + 1, len(LINKAGE_FILES), linkage_file)
        faults, segs = process_file(
            linkage_file,
            faults=faults,
            segs=segs,
            first_file=(i == 0)
        )

    total_runtime = time.perf_counter() - t_overall_start
    log.info("\n%s\n🎉 ALL FILES PROCESSED SUCCESSFULLY – Total runtime: %.1fs",
             "═" * 78, total_runtime)


# ─── entry-point ───────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.error("Interrupted by user")