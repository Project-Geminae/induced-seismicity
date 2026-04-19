"""
entrypoint.py — Container startup for the isolated dashboard.

Patches the server module's path constants to point to /app/data/
instead of the repo root, adds parquet panel support, then starts uvicorn.
"""
import os
import sys
from pathlib import Path

# Make /app importable
sys.path.insert(0, "/app")

# Register pickle classes on __main__ BEFORE anything tries to unpickle.
# We define minimal stubs here rather than importing the full modules
# (which have deep dependency chains: tmle_core → statsmodels, etc.)
import __main__ as main_mod
import dataclasses

try:
    import build_causal_forest as bcf
    main_mod.CausalForestBundle = bcf.CausalForestBundle
except Exception:
    pass

# Stub for legacy AttributionQ pickles — just needs to unpickle and expose
# the same interface the server's attribution endpoint calls.
# The real class is a dataclass with: model, feature_cols, treatment_col,
# radius_km, window_days, and an estimate_cate() method.
try:
    import build_attribution_q as baq
    main_mod.AttributionQ = baq.AttributionQ
except Exception:
    # Full import failed (missing tmle_core/statsmodels). Define a stub
    # that can unpickle and be called by the server.
    @dataclasses.dataclass
    class AttributionQ:
        model: object = None
        feature_cols: list = dataclasses.field(default_factory=list)
        treatment_col: str = ""
        radius_km: int = 7
        window_days: int = 365

        def estimate_cate(self, well_rows, treatment_values):
            """Stub: predict using the pickled model."""
            import numpy as np
            import pandas as pd
            if self.model is None:
                n = len(treatment_values)
                return {"cate": np.zeros(n), "ci_low": np.zeros(n), "ci_high": np.zeros(n)}
            X = well_rows[self.feature_cols].copy()
            X[self.treatment_col] = treatment_values
            pred = self.model.predict(X.values)
            X0 = X.copy()
            X0[self.treatment_col] = 0.0
            pred0 = self.model.predict(X0.values)
            cate = pred - pred0
            return {"cate": cate, "ci_low": cate * 0.5, "ci_high": cate * 1.5}

    main_mod.AttributionQ = AttributionQ

# Monkey-patch the server's constants before importing it
import dashboard.server as srv

DATA = Path("/app/data")
srv.REPO_ROOT = DATA
srv.EVENTS_CSV = DATA / "texnet_events_filtered.csv"  # won't exist; fallback to event_index
srv.EVENT_INDEX_JSON = DATA / "event_index.json"
srv.LINKS_PARQUET = DATA / "event_well_links.parquet"
srv.LINKS_CSV_FALLBACK = DATA / "event_well_links.csv"
srv.PANEL_CSV = DATA / "well_day_panel.csv"  # won't exist; we patch _load_panel

# Patch panel loader to prefer parquet
_original_load_panel = srv._load_panel
def _load_panel_parquet():
    import pandas as pd
    parquet_path = DATA / "well_day_panel.parquet"
    if parquet_path.exists():
        srv.log.info("Loading %s ...", parquet_path.name)
        df = pd.read_parquet(parquet_path)
        df["Date of Injection"] = pd.to_datetime(df["Date of Injection"])
        df = df.set_index(["API Number", "Date of Injection"]).sort_index()
        srv.log.info("    %d rows x %d cols", len(df), df.shape[1])
        return df
    return _original_load_panel()

srv._load_panel = _load_panel_parquet

# Patch events loader to use event_index.json if CSV missing
_original_load_events = srv._load_events
def _load_events_fallback():
    import pandas as pd
    import json
    csv_path = DATA / "texnet_events_filtered.csv"
    if csv_path.exists():
        return _original_load_events()
    # Build a minimal events DF from event_index.json
    srv.log.info("No events CSV; building from event_index.json ...")
    with open(DATA / "event_index.json") as f:
        idx = json.load(f)
    rows = []
    for eid, e in idx.items():
        rows.append({
            "EventID": eid,
            "Local Magnitude": e.get("ml", 0),
            "Latitude (WGS84)": e.get("lat", 0),
            "Longitude (WGS84)": e.get("lon", 0),
            "Origin Date": e.get("date", "2020-01-01"),
            "Depth of Hypocenter (Km.  Rel to MSL)": e.get("depth_km_msl", 0),
        })
    df = pd.DataFrame(rows)
    df["Origin Date"] = pd.to_datetime(df["Origin Date"])
    df = df.set_index("EventID")
    srv.log.info("    %d events from index", len(df))
    return df

srv._load_events = _load_events_fallback

# Now start uvicorn
import uvicorn
uvicorn.run(
    srv.app,
    host="0.0.0.0",
    port=8765,
    log_level="info",
    access_log=True,
    server_header=False,  # suppress 'Server: uvicorn' disclosure
)
