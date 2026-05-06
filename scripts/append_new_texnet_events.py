"""Merge: old filtered events (≤ April 10 with quality data) + ONLY events
strictly newer than April 10 from the May 5 raw → updated filtered CSV
through May 5.
"""
import pandas as pd
from pathlib import Path

OLD = Path("/opt/induced-seismicity/texnet_events_filtered.csv.backup")
NEW = Path("/opt/induced-seismicity/texnet_events.csv")
OUT = Path("/opt/induced-seismicity/texnet_events_filtered.csv")

old = pd.read_csv(OLD, low_memory=False)
new_raw = pd.read_csv(NEW, low_memory=False)
print(f"old: {len(old)} rows, dates {old['Origin Date'].min()} -> {old['Origin Date'].max()}")
print(f"new raw: {len(new_raw)} rows")

# Normalize Origin Date strings
new_raw["Origin Date"] = new_raw["Origin Date"].astype(str).str.split("T").str[0]
old_max = old["Origin Date"].max()
print(f"old max date = {old_max}; keeping new events with Origin Date > {old_max}")

# Permian/Midland Basin bbox
LAT_MIN, LAT_MAX = 30.6, 33.4
LON_MIN, LON_MAX = -103.2, -100.2

# Filter new events: strictly newer than old_max + minimal quality (ML≥1, final, bbox)
mask = (
    (new_raw["Origin Date"] > old_max)
    & (new_raw["Local Magnitude"].fillna(0) >= 1.0)
    & (new_raw["Evaluation Status"].astype(str) == "final")
    & (new_raw["Latitude (WGS84)"].between(LAT_MIN, LAT_MAX))
    & (new_raw["Longitude (WGS84)"].between(LON_MIN, LON_MAX))
    & new_raw["Latitude (WGS84)"].notna()
    & new_raw["Longitude (WGS84)"].notna()
)
new_kept = new_raw[mask].copy()
print(f"new events post-{old_max} after minimal filter: {len(new_kept)}")
if len(new_kept):
    print(f"  date range: {new_kept['Origin Date'].min()} -> {new_kept['Origin Date'].max()}")

# Reindex to old's columns and concat
new_kept = new_kept.reindex(columns=list(old.columns))
combined = pd.concat([old, new_kept], ignore_index=True)
combined = combined.sort_values("Origin Date").reset_index(drop=True)
combined.to_csv(OUT, index=False)
print(f"merged: {len(combined)} rows, dates {combined['Origin Date'].min()} -> {combined['Origin Date'].max()}")
print(f"wrote {OUT}")
