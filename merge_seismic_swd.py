#!/usr/bin/env python3
# merge_seismic_swd_multi.py ── link SWD wells to TexNet events for many radii
"""
Goal
────
For every salt-water-disposal (SWD) well, find seismic events that occurred
within several great-circle distance cut-offs and keep the well’s surface
coordinates.  Writes one CSV per radius:
    event_well_links_2km.csv
    event_well_links_5km.csv
    …
"""

# ── Standard-library imports ─────────────────────────────────────────────
import os, sys, logging
from typing import List

# ── Third-party imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm

# ────────────────────── USER-TUNABLE CONSTANTS ──────────────────────────
WELL_FILE   = "swd_data_filtered.csv"
EVENT_FILE  = "texnet_events_filtered.csv"
RADII_KM    = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0]      # NEW – list of cut-offs
CHUNK_SIZE  = 50_000
OUT_BASENAME = "event_well_links_{:.0f}km.csv"  # NEW – template for filenames
# ─────────────────────────────────────────────────────────────────────────

WELL_COLS = [
    "API Number", "Date of Injection",
    "Surface Longitude", "Surface Latitude",
]
EVENT_COLS = [
    "EventID", "Origin Date", "Local Magnitude",
    "Latitude (WGS84)", "Longitude (WGS84)",
    "Depth of Hypocenter (Km.  Rel to MSL)",
    "Depth of Hypocenter (Km. Rel to Ground Surface)",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ───────────────────────── Helper functions ─────────────────────────────
def haversine_km_vec(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2.astype(float)), np.radians(lon2.astype(float))
    dphi, dlambda = lat2 - lat1, lon2 - lon1
    a = (np.sin(dphi / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))

def load_csv(path: str, cols: List[str]) -> pd.DataFrame:
    if not os.path.isfile(path):
        log.error("❌  File not found: %s", path)
        sys.exit(1)
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    log.info("📄  Loaded %-28s → %7d rows × %d cols", path, len(df), len(df.columns))
    return df

# ──────────────────────────── main workflow ─────────────────────────────
def main() -> None:
    wells_raw = load_csv(WELL_FILE, WELL_COLS).dropna(subset=["API Number"])
    wells = wells_raw.drop_duplicates(subset="API Number", keep="first")
    log.info("🔧  Deduplicated wells: %d → %d unique APIs", len(wells_raw), len(wells))

    events = load_csv(EVENT_FILE, EVENT_COLS).dropna(
        subset=["Latitude (WGS84)", "Longitude (WGS84)"]
    )

    # Prepare one list per radius to gather link rows
    links_by_radius = {r: [] for r in RADII_KM}           # NEW

    log.info("🔗  Linking wells to events for radii: %s km", RADII_KM)
    for _, ev in tqdm(events.iterrows(), total=len(events), unit="event"):
        try:
            ev_lat = float(ev["Latitude (WGS84)"])
            ev_lon = float(ev["Longitude (WGS84)"])
        except ValueError:
            log.warning("⚠️   Bad lat/lon for EventID '%s' — skipped", ev["EventID"])
            continue

        for start in range(0, len(wells), CHUNK_SIZE):
            chunk = wells.iloc[start:start + CHUNK_SIZE]
            dists_km = haversine_km_vec(
                ev_lat, ev_lon,
                chunk["Surface Latitude"].values,
                chunk["Surface Longitude"].values,
            )

            # For each radius, capture the subset that lies inside
            for rad in RADII_KM:                           # NEW
                mask = dists_km <= rad
                if not mask.any():
                    continue
                nearby = chunk.loc[mask].copy()
                # event-level columns
                nearby["EventID"]   = ev["EventID"]
                nearby["Origin Date"] = ev["Origin Date"]
                nearby["Local Magnitude"] = ev["Local Magnitude"]
                nearby["Latitude (WGS84)"]  = ev_lat
                nearby["Longitude (WGS84)"] = ev_lon
                nearby["Depth of Hypocenter (Km.  Rel to MSL)"] = \
                    ev["Depth of Hypocenter (Km.  Rel to MSL)"]
                nearby["Depth of Hypocenter (Km. Rel to Ground Surface)"] = \
                    ev["Depth of Hypocenter (Km. Rel to Ground Surface)"]
                nearby["Distance from Well to Event"] = dists_km[mask]
                links_by_radius[rad].append(nearby[
                    [
                        "EventID", "Origin Date", "Local Magnitude",
                        "Latitude (WGS84)", "Longitude (WGS84)",
                        "Depth of Hypocenter (Km.  Rel to MSL)",
                        "Depth of Hypocenter (Km. Rel to Ground Surface)",
                        "API Number",
                        "Surface Latitude", "Surface Longitude",
                        "Distance from Well to Event",
                    ]
                ])

    # ── Write one CSV per radius ────────────────────────────────────────
    for rad in RADII_KM:
        frames = links_by_radius[rad]
        if not frames:
            log.warning("🚫  No (well, event) pairs found within %.1f km.", rad)
            continue
        linked_df = pd.concat(frames, ignore_index=True)
        outfile = OUT_BASENAME.format(rad)
        linked_df.to_csv(outfile, index=False)
        log.info("💾  %.1f km CSV written → %s  (%d rows)", rad, outfile, len(linked_df))

    log.info("✅  All radii complete.")

# ── Boilerplate guard ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()