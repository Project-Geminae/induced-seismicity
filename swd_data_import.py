#!/usr/bin/env python3
# swd_data_import.py — filter statewide SWD data to wells inside the Midland Basin
#                     and show summary statistics before / after that spatial cut.

import os                                     # std-lib: path utilities
import sys                                    # std-lib: exit and platform info
from datetime import datetime                 # std-lib: for date demo (not used, kept for parity)

import pandas as pd                           # 3rd-party: fast table manipulation

# ──────────────────────────────── FILES ────────────────────────────────
FILENAME = "swd_data.csv"                     # raw input file (daily injection records, statewide)
OUTFILE  = "swd_data_filtered.csv"            # output limited to Midland Basin records

# Columns we must retain for later geospatial or causal work
KEEP_COLS = [
    "API Number",
    "Date of Injection",
    "Volume Injected (BBLs)",
    "Injection Pressure Average PSIG",
    "Injection Pressure Max PSIG",
    "Surface Longitude",
    "Surface Latitude",
]

# ──────────────────── SPATIAL BOUNDING BOX (WGS-84) ────────────────────
# Rough rectangle around the Midland Basin (used to discard far-away wells)
MIN_LAT, MAX_LAT = 30.6, 33.4                # south … north
MIN_LON, MAX_LON = -103.2, -100.2            # west  … east
# ----------------------------------------------------------------------

def quick_stats(df: pd.DataFrame, tag: str) -> None:
    """
    Emit a one-shot summary of key metrics for the supplied DataFrame.

    Parameters
    ----------
    df  : DataFrame with SWD injection rows
    tag : Short label for the console header (“BEFORE …” / “AFTER …”)
    """
    row_cnt   = len(df)                                    # total rows
    well_cnt  = df["API Number"].nunique()                 # distinct wells
    date_min  = df["Date of Injection"].min()              # earliest injection
    date_max  = df["Date of Injection"].max()              # latest injection
    vol_sum   = df["Volume Injected (BBLs)"].sum()         # total barrel count
    press_avg = df["Injection Pressure Average PSIG"].mean()  # mean pressure

    # Pretty print the summary block
    print(f"\n📈  {tag} — summary")
    print(f"   Rows .................... {row_cnt:,}")
    print(f"   Unique wells (API) ...... {well_cnt:,}")
    print(f"   Date range .............. {date_min} → {date_max}")
    print(f"   Total injected (BBL) .... {vol_sum:,.0f}")
    print(f"   Mean avg pressure (PSI).. {press_avg:,.1f}")

def main() -> None:
    """Pipeline entry-point."""
    # 1 ── Load CSV ───────────────────────────────────────────────
    if not os.path.isfile(FILENAME):                       # guard: file must exist
        print(f"❌  '{FILENAME}' not found in {os.getcwd()}")  # explicit path in error
        sys.exit(1)

    try:
        df = pd.read_csv(FILENAME, low_memory=False)       # load entire table
    except Exception as exc:
        print(f"❌  Could not read '{FILENAME}': {exc}")    # print root cause
        sys.exit(1)

    # 2 ── Validate column presence ──────────────────────────────
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:                                            # abort if any critical column is absent
        print(f"❌  Missing column(s): {missing}")
        sys.exit(1)

    # 3 ── Clean & subset ───────────────────────────────────────
    df = df[KEEP_COLS].copy()                              # keep only the columns we care about

    # Convert lat/lon strings to numerics; coerce bad rows to NaN (they’ll be dropped)
    lat_col, lon_col = "Surface Latitude", "Surface Longitude"
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # Drop rows with missing / zero coordinates (0,0 is an obvious bad marker)
    df = df.dropna(subset=[lat_col, lon_col])
    df = df[(df[lat_col] != 0) & (df[lon_col] != 0)]

    # 4 ── Show stats BEFORE spatial cut ─────────────────────────
    quick_stats(df, "BEFORE bounding-box filter")

    # 5 ── Apply Midland-Basin bounding box ─────────────────────
    lat_mask = df[lat_col].between(MIN_LAT, MAX_LAT, inclusive="both")
    lon_mask = df[lon_col].between(MIN_LON, MAX_LON, inclusive="both")
    df_filtered = df[lat_mask & lon_mask]                  # keep only rows inside the bbox

    # 6 ── Stats AFTER spatial cut ───────────────────────────────
    quick_stats(df_filtered, "AFTER  bounding-box filter")

    # 7 ── Quick preview + save to disk ──────────────────────────
    print("\n🔎  First 5 wells INSIDE the Midland Basin:\n")
    print(df_filtered.head())

    df_filtered.to_csv(OUTFILE, index=False)               # write filtered CSV
    print(f"\n💾  Filtered data saved → {OUTFILE} ({len(df_filtered):,} rows)")

# Bootstrapped main runner
if __name__ == "__main__":
    main()