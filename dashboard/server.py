"""
dashboard/server.py
───────────────────
FastAPI backend for the Induced-Seismicity TMLE dashboard.

Loads at startup:
- texnet_events_filtered.csv         (5,233 events × 15 cols)
- event_index.json                   (per-event metadata + per-radius well counts)
- event_well_links.parquet           (2.4M rows: every (event, well) pair within ≤20 km)
- well_day_panel.csv                 (689k rows of injection / pressure / depth features)
- tmle_*_365d_*.csv                  (latest TMLE result files for the summary panel)

Endpoints:
- GET  /                                              → renders index.html
- GET  /api/events                                    → list of events for the map
- GET  /api/event/{event_id}                          → single-event detail
- GET  /api/event/{event_id}/wells?radius_km=7        → wells within R km, with panel features
- GET  /api/tmle/summary?radius_km=7                  → population-level TMLE numbers at this radius
- GET  /api/wells/{api_number}/timeseries             → daily volume + pressure for a well

Run locally:
    .venv/bin/uvicorn dashboard.server:app --host 0.0.0.0 --port 8765 --log-level info

Run on minitim (under tmux):
    tmux new-session -d -s dashboard \\
      "cd ~/induced-seismicity && \\
       .venv/bin/uvicorn dashboard.server:app --host 0.0.0.0 --port 8765 --log-level info \\
       &> dashboard.log"
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# ──────────────────── Constants ──────────────────────────────────
REPO_ROOT          = Path(__file__).resolve().parent.parent
TEMPLATES_DIR      = Path(__file__).resolve().parent / "templates"
INDEX_HTML         = TEMPLATES_DIR / "index.html"

EVENTS_CSV         = REPO_ROOT / "texnet_events_filtered.csv"
EVENT_INDEX_JSON   = REPO_ROOT / "event_index.json"
LINKS_PARQUET      = REPO_ROOT / "event_well_links.parquet"
LINKS_CSV_FALLBACK = REPO_ROOT / "event_well_links.csv"  # only if pyarrow missing
PANEL_CSV          = REPO_ROOT / "well_day_panel.csv"

RADII_KM = list(range(1, 21))


# ──────────────────── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dashboard")


# ──────────────────── Data loaders ───────────────────────────────
def latest(glob: str) -> Optional[Path]:
    matches = sorted(REPO_ROOT.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_links() -> pd.DataFrame:
    """Load the consolidated event-well link table."""
    if LINKS_PARQUET.exists():
        log.info("📄  Loading %s …", LINKS_PARQUET.name)
        df = pd.read_parquet(LINKS_PARQUET)
    elif LINKS_CSV_FALLBACK.exists():
        log.info("📄  Loading %s …", LINKS_CSV_FALLBACK.name)
        df = pd.read_csv(LINKS_CSV_FALLBACK, parse_dates=["event_date"])
    else:
        raise FileNotFoundError(
            f"Neither {LINKS_PARQUET} nor {LINKS_CSV_FALLBACK} exists. "
            "Run `python spatiotemporal_join.py --links-only` first."
        )
    df["event_date"] = pd.to_datetime(df["event_date"])
    log.info("    %d rows × %d cols", len(df), df.shape[1])
    return df


def _load_panel() -> pd.DataFrame:
    """Load the well-day panel; index for fast (API, date) lookup."""
    if not PANEL_CSV.exists():
        raise FileNotFoundError(f"{PANEL_CSV} not found — run build_well_day_panel.py first")
    log.info("📄  Loading %s …", PANEL_CSV.name)
    df = pd.read_csv(PANEL_CSV, low_memory=False)
    df["Date of Injection"] = pd.to_datetime(df["Date of Injection"])
    df = df.set_index(["API Number", "Date of Injection"]).sort_index()
    log.info("    %d rows × %d cols (indexed by (API Number, Date of Injection))",
             len(df), df.shape[1])
    return df


def _load_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"{EVENTS_CSV} not found — run seismic_data_import.py first")
    log.info("📄  Loading %s …", EVENTS_CSV.name)
    df = pd.read_csv(EVENTS_CSV, low_memory=False)
    df["Origin Date"] = pd.to_datetime(df["Origin Date"])
    df = df.set_index("EventID")
    log.info("    %d events", len(df))
    return df


def _load_event_index() -> dict:
    if not EVENT_INDEX_JSON.exists():
        raise FileNotFoundError(f"{EVENT_INDEX_JSON} not found — run spatiotemporal_join.py --links-only")
    log.info("📄  Loading %s …", EVENT_INDEX_JSON.name)
    return json.loads(EVENT_INDEX_JSON.read_text())


def _load_tmle_summary() -> dict[int, dict]:
    """Load the latest tmle_*_365d_*.csv files into a per-radius dict."""
    out: dict[int, dict] = {r: {} for r in RADII_KM}

    shift_path = latest("tmle_shift_365d_*.csv")
    if shift_path is not None:
        log.info("📄  Loading shift TMLE: %s", shift_path.name)
        for _, r in pd.read_csv(shift_path).iterrows():
            R = int(r["radius_km"])
            out[R].update({
                "shift_psi":      float(r["psi"]),
                "shift_ci_low":   float(r["ci_low"]),
                "shift_ci_high":  float(r["ci_high"]),
                "shift_pval":     float(r["pval"]),
            })

    dose_path = latest("tmle_dose_response_365d_*.csv")
    if dose_path is not None:
        log.info("📄  Loading dose TMLE: %s", dose_path.name)
        df = pd.read_csv(dose_path)
        for R in RADII_KM:
            sub = df[df["radius_km"] == R].sort_values("a_star")
            if not sub.empty:
                out[R]["dose_grid"] = [
                    {
                        "a":       float(row["a_star"]),
                        "psi":     float(row["psi"]),
                        "ci_low":  float(row["ci_low"]),
                        "ci_high": float(row["ci_high"]),
                    }
                    for _, row in sub.iterrows()
                ]
                # Convenience: dose @ 1e7 BBL
                row_1e7 = sub[np.isclose(sub["a_star"], 1e7)]
                if not row_1e7.empty:
                    out[R]["dose_at_1e7"]        = float(row_1e7["psi"].iloc[0])
                    out[R]["dose_at_1e7_ci_low"] = float(row_1e7["ci_low"].iloc[0])
                    out[R]["dose_at_1e7_ci_high"]= float(row_1e7["ci_high"].iloc[0])

    med_path = latest("tmle_mediation_365d_*.csv")
    if med_path is not None:
        log.info("📄  Loading mediation TMLE: %s", med_path.name)
        for _, r in pd.read_csv(med_path).iterrows():
            R = int(r["radius_km"])
            out[R].update({
                "TE":           float(r["TE"]),
                "TE_ci_low":    float(r["TE_ci_low"]),
                "TE_ci_high":   float(r["TE_ci_high"]),
                "NDE":          float(r["NDE"]),
                "NIE":          float(r["NIE"]),
                "pct_mediated": float(r["pct_mediated"]),
                "a_high":       float(r["a_high"]),
                "a_low":        float(r["a_low"]),
            })

    return out


# ──────────────────── App + state ────────────────────────────────
app = FastAPI(
    title="Induced Seismicity TMLE Dashboard",
    description="Click an event on the map → see contributing wells with TMLE-derived metrics.",
    version="0.1.0",
)


class State:
    """In-memory state. Loaded once at startup."""
    events:       pd.DataFrame  # indexed by EventID
    event_index:  dict          # from event_index.json
    links:        pd.DataFrame  # consolidated event-well links across all radii
    panel:        pd.DataFrame  # indexed by (API Number, Date of Injection)
    tmle_summary: dict[int, dict]
    loaded_at:    datetime


state = State()


@app.on_event("startup")
def _startup() -> None:
    log.info("=" * 60)
    log.info("Loading dashboard state …")
    state.events       = _load_events()
    state.event_index  = _load_event_index()
    state.links        = _load_links()
    state.panel        = _load_panel()
    state.tmle_summary = _load_tmle_summary()
    state.loaded_at    = datetime.now()
    log.info("✅  Dashboard ready (loaded at %s)", state.loaded_at.isoformat())
    log.info("=" * 60)


# ──────────────────── Schemas ────────────────────────────────────
# We return plain dicts for simplicity; FastAPI handles JSON encoding via
# the default jsonable_encoder. Numpy floats / ints get coerced to native
# Python types automatically.


# ──────────────────── Routes: HTML ───────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not INDEX_HTML.exists():
        return HTMLResponse(
            "<h1>dashboard/templates/index.html not found</h1>"
            "<p>The frontend hasn't been written yet.</p>",
            status_code=500,
        )
    return HTMLResponse(INDEX_HTML.read_text())


# ──────────────────── Routes: API ────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    return {
        "status":      "ok",
        "loaded_at":   state.loaded_at.isoformat(),
        "n_events":    len(state.events),
        "n_panel":     len(state.panel),
        "n_links":     len(state.links),
        "radii_km":    RADII_KM,
    }


@app.get("/api/events")
def list_events(
    since:   Optional[str]   = Query(None,  description="ISO date, YYYY-MM-DD"),
    until:   Optional[str]   = Query(None,  description="ISO date, YYYY-MM-DD"),
    min_ml:  Optional[float] = Query(None,  description="Minimum local magnitude"),
    max_ml:  Optional[float] = Query(None,  description="Maximum local magnitude"),
    limit:   int             = Query(10000, description="Max events to return"),
) -> JSONResponse:
    """List events for the map. Returns lat/lon/magnitude/date metadata only.

    The full per-event well counts are in event_index.json so the frontend
    can render different marker sizes based on the radius slider without
    additional fetches.
    """
    df = state.events
    if since:
        df = df[df["Origin Date"] >= pd.Timestamp(since)]
    if until:
        df = df[df["Origin Date"] <= pd.Timestamp(until)]
    if min_ml is not None:
        df = df[df["Local Magnitude"] >= min_ml]
    if max_ml is not None:
        df = df[df["Local Magnitude"] <= max_ml]

    df = df.head(limit)
    rows = []
    for eid, ev in df.iterrows():
        idx_entry = state.event_index.get(eid, {})
        rows.append({
            "id":           eid,
            "lat":          float(ev["Latitude (WGS84)"]),
            "lon":          float(ev["Longitude (WGS84)"]),
            "date":         ev["Origin Date"].strftime("%Y-%m-%d"),
            "ml":           float(ev["Local Magnitude"]),
            "depth_km":     (float(ev["Depth of Hypocenter (Km.  Rel to MSL)"])
                              if pd.notna(ev["Depth of Hypocenter (Km.  Rel to MSL)"]) else None),
            "well_counts":  idx_entry.get("well_counts", {}),
        })
    return JSONResponse({"count": len(rows), "events": rows})


@app.get("/api/event/{event_id}")
def get_event(event_id: str) -> dict:
    if event_id not in state.events.index:
        raise HTTPException(404, f"event {event_id} not found")
    ev = state.events.loc[event_id]
    return {
        "id":            event_id,
        "lat":           float(ev["Latitude (WGS84)"]),
        "lon":           float(ev["Longitude (WGS84)"]),
        "date":          ev["Origin Date"].strftime("%Y-%m-%d"),
        "ml":            float(ev["Local Magnitude"]),
        "depth_km_msl":  (float(ev["Depth of Hypocenter (Km.  Rel to MSL)"])
                           if pd.notna(ev["Depth of Hypocenter (Km.  Rel to MSL)"]) else None),
        "depth_unc_km":  (float(ev["Depth Uncertainty (Km. Corresponds to 1 st dev)"])
                           if pd.notna(ev["Depth Uncertainty (Km. Corresponds to 1 st dev)"]) else None),
        "rms":           float(ev["RMS"]) if pd.notna(ev["RMS"]) else None,
        "phase_count":   int(ev["UsedPhaseCount"]) if pd.notna(ev["UsedPhaseCount"]) else None,
        "station_count": int(ev["UsedStationCount"]) if pd.notna(ev["UsedStationCount"]) else None,
        "lat_err_km":    float(ev["Latitude Error (km)"]) if pd.notna(ev["Latitude Error (km)"]) else None,
        "lon_err_km":    float(ev["Longitude Error (km)"]) if pd.notna(ev["Longitude Error (km)"]) else None,
        "well_counts":   state.event_index.get(event_id, {}).get("well_counts", {}),
    }


# Panel feature columns we expose to the frontend (a curated subset)
_PANEL_FEATURES = [
    "Volume Injected (BBLs)",
    "Injection Pressure Average PSIG",
    "cum_vol_30d_BBL", "cum_vol_90d_BBL", "cum_vol_180d_BBL", "cum_vol_365d_BBL",
    "vw_avg_psig_30d", "vw_avg_psig_90d", "vw_avg_psig_180d", "vw_avg_psig_365d",
    "bhp_vw_avg_30d",  "bhp_vw_avg_90d",  "bhp_vw_avg_180d",  "bhp_vw_avg_365d",
    "perf_depth_ft", "formation", "days_active",
]


def _well_features_at(api: int, date: pd.Timestamp) -> dict:
    """Look up the panel features for a single (API, date) cell.

    Returns an empty dict if the well had no panel row on that date (which
    shouldn't normally happen since the panel is dense per well, but is
    possible at the edges of a well's lifecycle).
    """
    try:
        row = state.panel.loc[(api, date)]
    except KeyError:
        return {}
    if isinstance(row, pd.DataFrame):
        # Defensive: if the index returns multiple rows somehow, take the first
        row = row.iloc[0]
    out: dict = {}
    for col in _PANEL_FEATURES:
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v):
            out[col] = None
        elif isinstance(v, (np.integer, np.floating)):
            out[col] = float(v)
        else:
            out[col] = v
    return out


@app.get("/api/event/{event_id}/wells")
def event_wells(
    event_id:  str,
    radius_km: int = Query(7, ge=1, le=20),
) -> dict:
    """List wells within `radius_km` of the event, with their panel features
    on the event date. Sorted by distance ascending."""
    if event_id not in state.events.index:
        raise HTTPException(404, f"event {event_id} not found")
    ev = state.events.loc[event_id]
    ev_date = ev["Origin Date"]

    sub = state.links[
        (state.links["EventID"] == event_id)
        & (state.links["radius_km"] == radius_km)
    ].sort_values("distance_km")

    wells = []
    for _, link in sub.iterrows():
        api = int(link["API Number"])
        feats = _well_features_at(api, ev_date)
        wells.append({
            "api":         api,
            "lat":         float(link["well_lat"]),
            "lon":         float(link["well_lon"]),
            "distance_km": float(link["distance_km"]),
            "panel":       feats,
        })

    return {
        "event": {
            "id":   event_id,
            "lat":  float(ev["Latitude (WGS84)"]),
            "lon":  float(ev["Longitude (WGS84)"]),
            "date": ev_date.strftime("%Y-%m-%d"),
            "ml":   float(ev["Local Magnitude"]),
        },
        "radius_km": radius_km,
        "n_wells":   len(wells),
        "wells":     wells,
    }


@app.get("/api/tmle/summary")
def tmle_summary(radius_km: int = Query(7, ge=1, le=20)) -> dict:
    """Population-level TMLE numbers for the given radius."""
    return {
        "radius_km": radius_km,
        **state.tmle_summary.get(radius_km, {}),
    }


@app.get("/api/tmle/all")
def tmle_all() -> dict:
    """The full per-radius TMLE summary table — for the population context plot."""
    return state.tmle_summary


@app.get("/api/wells/{api_number}/timeseries")
def well_timeseries(
    api_number: int,
    around:     str = Query(..., description="Center date, YYYY-MM-DD"),
    days:       int = Query(180, ge=7, le=2000),
) -> dict:
    """Daily injection volume + pressure for a well, centered on a date.

    Used by the dashboard to show "what was this well doing in the months
    leading up to the event?"
    """
    center = pd.Timestamp(around)
    start  = center - pd.Timedelta(days=days)
    end    = center + pd.Timedelta(days=days // 4)  # a bit of post-event tail
    try:
        sub = state.panel.loc[api_number]
    except KeyError:
        raise HTTPException(404, f"API {api_number} not in panel")
    sub = sub.loc[start:end]
    return {
        "api":      api_number,
        "center":   center.strftime("%Y-%m-%d"),
        "n_days":   len(sub),
        "series": [
            {
                "date":     d.strftime("%Y-%m-%d"),
                "volume":   float(row.get("Volume Injected (BBLs)", 0.0) or 0.0),
                "psig_avg": float(row.get("Injection Pressure Average PSIG", 0.0) or 0.0),
                "cum_365d": float(row.get("cum_vol_365d_BBL", 0.0) or 0.0),
            }
            for d, row in sub.iterrows()
        ],
    }
