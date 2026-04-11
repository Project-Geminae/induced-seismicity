#!/usr/bin/env python3
# seismic_data_import_and_plot.py
# ──────────────────────────────────────────────────────────────────────
# 1) Filters TexNet catalog to Midland Basin (ML ≥ 1.0)   → CSV
# 2) Plots a publication-quality map of those events      → PNG
# ---------------------------------------------------------------------

# ── Standard-library imports ──────────────────────────────────────────
import os, sys, datetime as dt
# ── Third-party imports ───────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── File paths / constants ────────────────────────────────────────────
FILENAME = "texnet_events.csv"
OUTFILE  = "texnet_events_filtered.csv"
PLOT_PNG = "midland_basin_events_map.png"

KEEP_COLS = [
    "EventID",
    "Evaluation Status",
    "Origin Date",
    "Local Magnitude",
    "Latitude (WGS84)",
    "Latitude Error (km)",
    "Longitude (WGS84)",
    "Longitude Error (km)",
    "Depth of Hypocenter (Km.  Rel to MSL)",
    "Depth of Hypocenter (Km. Rel to Ground Surface)",
    "Depth Uncertainty (Km. Corresponds to 1 st dev)",
    "RMS",
    "UsedPhaseCount",
    "UsedStationCount",
]

MIN_LAT, MAX_LAT = 30.6, 33.4
MIN_LON, MAX_LON = -103.2, -100.2

# Quality filtering thresholds (rationale documented in CHANGES.md)
MIN_LOCAL_MAG       = 1.0   # magnitude completeness threshold
MAX_DEPTH_UNCERT_KM = 2.0   # 1σ depth error
MAX_RMS             = 0.5   # location residual
MAX_HORIZ_ERR_KM    = 2.0   # max of lat/lon error
MIN_PHASE_COUNT     = 8     # minimum phases used in location
KEEP_EVAL_STATUS    = {"final"}  # exclude preliminary / Automatic
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1 ▸ Load raw CSV
    if not os.path.isfile(FILENAME):
        sys.exit(f"❌  '{FILENAME}' not found in {os.getcwd()}")

    try:
        df = pd.read_csv(FILENAME, low_memory=False)
    except Exception as exc:
        sys.exit(f"❌  Could not read '{FILENAME}': {exc}")

    # 2 ▸ Verify required columns
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        sys.exit(f"❌  Missing column(s): {missing}")

    # 3 ▸ Subset / cleanse
    df = df[KEEP_COLS].copy()
    lat, lon = "Latitude (WGS84)", "Longitude (WGS84)"

    numeric_cols = [
        lat, lon, "Local Magnitude",
        "Latitude Error (km)", "Longitude Error (km)",
        "Depth Uncertainty (Km. Corresponds to 1 st dev)",
        "RMS", "UsedPhaseCount", "UsedStationCount",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Origin Date"] = pd.to_datetime(df["Origin Date"], errors="coerce")

    df = df.dropna(subset=[lat, lon, "Local Magnitude"])
    df = df[(df[lat] != 0) & (df[lon] != 0)]

    # 4 ▸ Spatial filter
    df = df[
        df[lat].between(MIN_LAT, MAX_LAT) &
        df[lon].between(MIN_LON, MAX_LON)
    ]
    print(f"\n📊  After bbox + non-null lat/lon/mag: {len(df):,} rows")

    # 5 ▸ Quality filters (report each step)
    def report_filter(df_in: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
        kept = mask.sum()
        dropped = len(df_in) - kept
        print(f"   {label:<40s}  drop {dropped:>5,} → keep {kept:>5,}")
        return df_in[mask]

    print("\n🔍  Quality filters:")
    df_f = report_filter(df, df["Local Magnitude"] >= MIN_LOCAL_MAG,
                         f"Local Magnitude ≥ {MIN_LOCAL_MAG}")
    df_f = report_filter(df_f, df_f["Evaluation Status"].isin(KEEP_EVAL_STATUS),
                         f"Evaluation Status ∈ {sorted(KEEP_EVAL_STATUS)}")
    df_f = report_filter(df_f, df_f["RMS"].fillna(99) <= MAX_RMS,
                         f"RMS ≤ {MAX_RMS}")
    df_f = report_filter(df_f,
                         df_f["Depth Uncertainty (Km. Corresponds to 1 st dev)"].fillna(99) <= MAX_DEPTH_UNCERT_KM,
                         f"Depth Uncertainty ≤ {MAX_DEPTH_UNCERT_KM} km")
    horiz_err = df_f[["Latitude Error (km)", "Longitude Error (km)"]].max(axis=1).fillna(99)
    df_f = report_filter(df_f, horiz_err <= MAX_HORIZ_ERR_KM,
                         f"max(lat,lon) err ≤ {MAX_HORIZ_ERR_KM} km")
    df_f = report_filter(df_f, df_f["UsedPhaseCount"].fillna(0) >= MIN_PHASE_COUNT,
                         f"UsedPhaseCount ≥ {MIN_PHASE_COUNT}")

    print(f"\n✅  Final catalog: {len(df_f):,} of {len(df):,} bbox events ({len(df_f)/len(df):.1%} retained)")

    # 6 ▸ Save CSV
    df_f.to_csv(OUTFILE, index=False)
    print(f"💾  Saved → {OUTFILE}")

    # 7 ▸ World-class Matplotlib map
    if df_f.empty:
        print("⚠️  No events to plot.")
        return

    # ── styling --------------------------------------------------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150,
                            facecolor="#0e1117")
    fig.patch.set_facecolor("#0e1117")

    # colour by magnitude, size by magnitude² for pop
    mags   = df_f["Local Magnitude"]
    norm   = Normalize(mags.min(), mags.max())
    cmap   = plt.cm.turbo
    sizes  = ((mags - mags.min()) + 1.0)**2.3 * 6  # tweak to taste

    sc = ax.scatter(
        df_f[lon], df_f[lat],
        c=mags, cmap=cmap, norm=norm,
        s=sizes, alpha=0.85, edgecolor="black", linewidth=0.3
    )

    # bounding box
    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.set_aspect("equal", adjustable="box")

    # labels & grid
    ax.set_xlabel("Longitude (°E)", labelpad=8, color="white")
    ax.set_ylabel("Latitude (°N)",  labelpad=8, color="white")
    ax.grid(ls="--", lw=0.4, color="#333940", alpha=0.6)

    # title & subtitle
    t0, t1 = df_f["Origin Date"].min(), df_f["Origin Date"].max()
    year_span = f"{t0.year}‒{t1.year}" if t0.year != t1.year else str(t0.year)
    ax.set_title(
        f"Midland Basin Earthquakes (ML ≥ 1.0) • {year_span}",
        fontweight="bold", fontsize=17, pad=14
    )
    ax.set_title(f"{len(df_f):,} events", fontsize=12, loc="right", color="#bbbbbb")

    # colour-bar
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, pad=0.01, aspect=35, shrink=0.82
    )
    cbar.set_label("Local Magnitude", labelpad=6)

    # save
    plt.tight_layout()
    fig.savefig(PLOT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Event map saved → {PLOT_PNG}")

if __name__ == "__main__":
    main()