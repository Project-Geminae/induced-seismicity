#!/usr/bin/env python3
"""
Add daily-injection data (same day + N-day look-back window) to the
event–well linkage tables, while keeping each well's surface coordinates.
Also creates separate lists of "Innocent" wells - those linked to events
but not actively injecting on the event day, with magnitude set to 0.

Processes multiple linkage files in one run.

Outputs for every input file foo.csv:
- foo → event_well_links_with_injection_<suffix>.csv - Wells with active injection
- foo → innocent_wells_<suffix>.csv - Wells without same-day injection (magnitude = 0)
"""

import os
import sys
import re
import logging
from datetime import timedelta
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# ──────────────────── User-tunable parameters ─────────────────────
LOOKBACK_DAYS = 30  # Number of days BEFORE the event to aggregate over

LINKS_FILES = [
    "event_well_links_1km.csv",
    "event_well_links_2km.csv",
    "event_well_links_3km.csv",
    "event_well_links_4km.csv",
    "event_well_links_5km.csv",
    "event_well_links_6km.csv",
    "event_well_links_7km.csv",
    "event_well_links_8km.csv",
    "event_well_links_9km.csv",
    "event_well_links_10km.csv",
    "event_well_links_15km.csv",
    "event_well_links_20km.csv"
]

SWD_FILE = "swd_data_filtered.csv"  # Input: per-well daily injection records
# ───────────────────────────────────────────────────────────────────

KEEP_SWD_COLS = [  # Columns to load from the SWD file
    "API Number",
    "Surface Longitude",
    "Surface Latitude",
    "Date of Injection",
    "Volume Injected (BBLs)",
    "Injection Pressure Average PSIG",
    "Injection Pressure Max PSIG",
]

# ────────────────────────── Logging ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ───────────────────── CSV loader helper ──────────────────────────
def load_csv(path: str, date_col: str, usecols=None) -> pd.DataFrame:
    """Read a CSV, convert *date_col* to Python date, report shape."""
    if not os.path.isfile(path):
        sys.exit(f"❌  File not found: {path}")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.columns = df.columns.str.strip()  # header hygiene
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    log.info(f"📄  Loaded {path:35s} → {df.shape[0]:7,} rows × {df.shape[1]} cols")
    return df


# ───────────────────── Core enrichment logic ───────────────────────
def enrich_links(
        links: pd.DataFrame,
        swd_groups: Dict[int, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (links_with_injection, innocent_links) for one linkage file."""

    # 1. Same-day inner join (API + calendar date) ------------------
    merged = links.merge(
        swd,
        how="inner",
        left_on=["API Number", "Origin Date"],
        right_on=["API Number", "Date of Injection"],
        indicator=True,
    )

    # FIX: Handle coordinate columns that were suffixed during merge
    # This must happen immediately after the merge
    # Check which suffixed columns exist and map them back to the original names
    coord_cols = {}
    if "Surface Latitude_x" in merged.columns:
        coord_cols["Surface Latitude_x"] = "Surface Latitude"
    if "Surface Longitude_x" in merged.columns:
        coord_cols["Surface Longitude_x"] = "Surface Longitude"

    # Rename the _x columns and drop the _y columns
    merged = merged.rename(columns=coord_cols)
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_y")])

    # 2. "Innocent" wells (linked but no same-day injection) --------
    links["link_id"] = links["EventID"] + "_" + links["API Number"].astype(str)
    merged["link_id"] = merged["EventID"] + "_" + merged["API Number"].astype(str)
    innocent = links[~links["link_id"].isin(set(merged["link_id"]))].copy()

    links.drop("link_id", axis=1, inplace=True)
    merged.drop("link_id", axis=1, inplace=True)

    log.info(f"🔗  Same-day injections matched: {len(merged):,}")
    log.info(f"🕊️   Innocent wells found        : {len(innocent):,}")

    # 3. Allocate look-back placeholders ----------------------------
    for df in (merged, innocent):
        df["Vol Prev N (BBLs)"] = 0.0
        df["Avg Press Prev N (PSIG)"] = pd.NA
        df["Max Press Prev N (PSIG)"] = pd.NA

    innocent["Local Magnitude"] = 0.0

    # Add Date of Injection and other fields for innocent wells
    innocent["Date of Injection"] = pd.NaT  # Initialize with NaT
    innocent["Volume Injected (BBLs)"] = 0.0
    innocent["Injection Pressure Average PSIG"] = 0.0
    innocent["Injection Pressure Max PSIG"] = 0.0

    # Find most recent injection date for innocent wells
    log.info(f"🔍  Finding most recent injection date for innocent wells...")
    for idx, row in tqdm(innocent.iterrows(), total=len(innocent), unit="row", leave=False):
        api = row["API Number"]
        ev_date = row["Origin Date"]

        g = swd_groups.get(api)
        if g is None:
            continue

        # Find the most recent injection date before the event date (if any)
        recent_injection = g[g["Date of Injection"] < ev_date]
        if not recent_injection.empty:
            most_recent = recent_injection.iloc[-1]  # Last row (most recent)
            innocent.at[idx, "Date of Injection"] = most_recent["Date of Injection"]
            # The following three fields are for that most recent injection date
            innocent.at[idx, "Volume Injected (BBLs)"] = most_recent["Volume Injected (BBLs)"]
            innocent.at[idx, "Injection Pressure Average PSIG"] = most_recent["Injection Pressure Average PSIG"]
            innocent.at[idx, "Injection Pressure Max PSIG"] = most_recent["Injection Pressure Max PSIG"]

    # 4. Compute look-back metrics ----------------------------------
    def fill_metrics(df: pd.DataFrame, label: str):
        log.info(f"📊  Computing {label} look-back metrics…")
        for idx, row in tqdm(df.iterrows(), total=len(df), unit="row", leave=False):
            api, ev_date = row["API Number"], row["Origin Date"]
            start = ev_date - timedelta(days=LOOKBACK_DAYS)

            g = swd_groups.get(api)
            if g is None:
                continue

            window = g[(g["Date of Injection"] >= start) &
                       (g["Date of Injection"] < ev_date)]
            if window.empty:
                continue

            df.at[idx, "Vol Prev N (BBLs)"] = window["Volume Injected (BBLs)"].sum()
            df.at[idx, "Avg Press Prev N (PSIG)"] = window["Injection Pressure Average PSIG"].mean()
            df.at[idx, "Max Press Prev N (PSIG)"] = window["Injection Pressure Max PSIG"].max()

    fill_metrics(merged, "matched")
    fill_metrics(innocent, "innocent")

    # 5. Final column order (adds NA columns if missing) ------------
    final_cols = [
        "EventID",
        "Origin Date",
        "Date of Injection",
        "Local Magnitude",
        "Latitude (WGS84)", "Longitude (WGS84)",
        "Depth of Hypocenter (Km.  Rel to MSL)",
        "Depth of Hypocenter (Km. Rel to Ground Surface)",
        "API Number",
        "Surface Latitude", "Surface Longitude",
        "Distance from Well to Event",
        "Volume Injected (BBLs)",
        "Injection Pressure Average PSIG",
        "Injection Pressure Max PSIG",
        "Vol Prev N (BBLs)",
        "Avg Press Prev N (PSIG)",
        "Max Press Prev N (PSIG)",
    ]

    for col in final_cols:  # guarantee schema
        if col not in merged.columns:
            merged[col] = pd.NA
        if col not in innocent.columns:
            innocent[col] = pd.NA

    return merged[final_cols], innocent[final_cols]

# ─────────────────────────── Main ───────────────────────────────────
if __name__ == "__main__":
    # A. Load SWD once & build quick lookup dict
    swd = load_csv(SWD_FILE, "Date of Injection", usecols=KEEP_SWD_COLS)
    swd_groups = {api: g.sort_values("Date of Injection")
                  for api, g in swd.groupby("API Number")}
    log.info(f"⚡  SWD table grouped by API ({len(swd_groups):,} APIs)")

    # B. Process each linkage file
    for links_path in LINKS_FILES:
        if not os.path.isfile(links_path):
            log.warning(f"🚫  Skipping missing file: {links_path}")
            continue

        log.info(f"\n────────── Processing {links_path} ──────────")
        links_df = load_csv(links_path, "Origin Date")

        enriched_df, innocent_df = enrich_links(links_df, swd_groups)

        # Derive radius suffix automatically
        base = os.path.splitext(links_path)[0]  # drop .csv
        out_matched = re.sub(r"event_well_links", "event_well_links_with_injection", base) + ".csv"
        out_innocent = re.sub(r"event_well_links", "innocent_wells", base) + ".csv"

        enriched_df.to_csv(out_matched, index=False)
        innocent_df.to_csv(out_innocent, index=False)

        log.info(f"✅  Saved {out_matched:34s} ({len(enriched_df):,} rows)")
        log.info(f"✅  Saved {out_innocent:34s} ({len(innocent_df):,} rows)")

    log.info("\n🎉  All linkage files processed.")