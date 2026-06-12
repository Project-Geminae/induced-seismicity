"""Build well_registry.json: API Number -> operator/lease metadata + current
365-day volume position. Shipped into the dashboard container to power
well-first navigation (/api/wells search + portfolio)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/opt/induced-seismicity")

# Operator/lease metadata from the raw SWD file (one row per well suffices)
print("reading swd_data.csv (metadata cols only)...")
meta = pd.read_csv(
    ROOT / "swd_data.csv",
    usecols=["API Number", "Operator Name", "Lease Name", "UIC Number"],
    dtype={"API Number": str},
    low_memory=False,
).drop_duplicates(subset="API Number", keep="last")
print(f"  {len(meta)} unique wells with metadata")

# Position + latest 365d volume from the panel
print("reading well_day_panel.parquet...")
panel = pd.read_parquet(
    ROOT / "well_day_panel.parquet",
    columns=["API Number", "Date of Injection", "Surface Latitude",
             "Surface Longitude", "perf_depth_ft", "days_active",
             "cum_vol_365d_BBL", "avg_rate_365d"],
)
panel["API Number"] = panel["API Number"].astype(str)
latest = (panel.sort_values("Date of Injection")
                .groupby("API Number")
                .tail(1)
                .set_index("API Number"))
print(f"  {len(latest)} wells in panel")

# Percentile of current 365d volume across wells (threshold-position proxy
# rendered alongside the per-well threshold curve in the UI)
latest["vol_pctile"] = latest["cum_vol_365d_BBL"].rank(pct=True).round(4)

meta = meta.set_index("API Number")
out = {}
for api, row in latest.iterrows():
    m = meta.loc[api] if api in meta.index else None
    out[api] = {
        "operator": (str(m["Operator Name"]) if m is not None and pd.notna(m["Operator Name"]) else None),
        "lease": (str(m["Lease Name"]) if m is not None and pd.notna(m["Lease Name"]) else None),
        "uic": (str(m["UIC Number"]) if m is not None and pd.notna(m["UIC Number"]) else None),
        "lat": round(float(row["Surface Latitude"]), 5),
        "lon": round(float(row["Surface Longitude"]), 5),
        "perf_depth_ft": (round(float(row["perf_depth_ft"]), 0) if pd.notna(row["perf_depth_ft"]) else None),
        "days_active": int(row["days_active"]) if pd.notna(row["days_active"]) else None,
        "last_panel_date": str(row["Date of Injection"].date()),
        "cum_vol_365d_BBL": round(float(row["cum_vol_365d_BBL"]), 0),
        "avg_rate_365d": (round(float(row["avg_rate_365d"]), 1) if pd.notna(row.get("avg_rate_365d")) else None),
        "vol_pctile": float(row["vol_pctile"]),
    }

dst = ROOT / "well_registry.json"
dst.write_text(json.dumps(out))
print(f"wrote {dst} ({len(out)} wells, {dst.stat().st_size/1024:.0f} KB)")
ops = {v['operator'] for v in out.values() if v['operator']}
print(f"unique operators: {len(ops)}")
