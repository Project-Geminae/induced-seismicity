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
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import time as _time

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False


# ──────────────────── Constants ──────────────────────────────────
REPO_ROOT          = Path(__file__).resolve().parent.parent
TEMPLATES_DIR      = Path(__file__).resolve().parent / "templates"
INDEX_HTML         = TEMPLATES_DIR / "index.html"
METHODOLOGY_HTML   = TEMPLATES_DIR / "methodology.html"

EVENTS_CSV         = REPO_ROOT / "texnet_events_filtered.csv"
EVENT_INDEX_JSON   = REPO_ROOT / "event_index.json"
LINKS_PARQUET      = REPO_ROOT / "event_well_links.parquet"
LINKS_CSV_FALLBACK = REPO_ROOT / "event_well_links.csv"  # only if pyarrow missing
PANEL_CSV          = REPO_ROOT / "well_day_panel.csv"
Q_PKL_FMT          = "q_attribution_{R}km.pkl"    # legacy TMLE pickles (unused)
CF_PKL_FMT         = "cf_cate_{R}km.pkl"          # Causal Forest DML pickles

RADII_KM = list(range(1, 21))

# Hydrostatic baseline (psi/ft) used to compute the counterfactual BHP for
# a "shut-off" well. Matches the value used in build_well_day_panel.py.
BRINE_GRADIENT_PSI_PER_FT = 0.45


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


def _load_causal_forests() -> dict[int, object]:
    """Load per-radius CausalForestDML bundles from cf_cate_<R>km.pkl.

    These are produced by build_causal_forest.py. Falls back to the old
    TMLE AttributionQ pickles if no CF pickles are found.
    """
    out: dict[int, object] = {}

    # Register build classes for pickle dispatch
    try:
        import build_causal_forest as bcf
        import __main__ as main_mod
        if not hasattr(main_mod, "CausalForestBundle"):
            main_mod.CausalForestBundle = bcf.CausalForestBundle
    except Exception as e:
        log.warning("Could not register CausalForestBundle: %s", e)

    # Try Causal Forest pickles first
    for R in RADII_KM:
        path = REPO_ROOT / CF_PKL_FMT.format(R=R)
        if not path.exists():
            continue
        try:
            with path.open("rb") as f:
                out[R] = pickle.load(f)
            log.info("📄  Loaded CausalForest: %s", path.name)
        except Exception as e:
            log.warning("Failed to load %s: %s", path.name, e)

    # Also load legacy TMLE AttributionQ pickles for radii without a Causal Forest
    if out:
        log.info("Loaded %d CausalForest models; checking legacy Q models for remaining radii", len(out))
    try:
        import build_attribution_q as baq
        import __main__ as main_mod
        if not hasattr(main_mod, "AttributionQ"):
            main_mod.AttributionQ = baq.AttributionQ
    except Exception:
        pass
    for R in RADII_KM:
        if R in out:
            continue  # already have a CausalForest for this radius
        path = REPO_ROOT / Q_PKL_FMT.format(R=R)
        if not path.exists():
            continue
        try:
            with path.open("rb") as f:
                out[R] = pickle.load(f)
            log.info("📄  Loaded legacy Q model: %s", path.name)
        except Exception as e:
            log.warning("Failed to load %s: %s", path.name, e)

    if not out:
        log.warning("⚠️   No CATE models found. Run build_causal_forest.py.")
    return out


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
app.add_middleware(GZipMiddleware, minimum_size=500)  # gzip responses > 500 bytes

# ──────────────────── Access logging middleware ──────────────────
ACCESS_LOG = Path(os.environ.get("ACCESS_LOG", "/tmp/access.log"))
_access_logger = logging.getLogger("access")
_access_handler = logging.FileHandler(ACCESS_LOG)
_access_handler.setFormatter(logging.Formatter("%(message)s"))
_access_logger.addHandler(_access_handler)
_access_logger.setLevel(logging.INFO)
# Also log to stdout so docker logs captures it
_access_logger.addHandler(logging.StreamHandler(sys.stdout))

_SKIP_LOG_PATHS = {"/analytics", "/api/analytics", "/api/beacon"}

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = _time.time()
        response = await call_next(request)
        elapsed = (_time.time() - start) * 1000  # ms

        path = request.url.path
        if path in _SKIP_LOG_PATHS:
            return response

        xff = request.headers.get("x-forwarded-for", "")
        ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "-")
        ua = request.headers.get("user-agent", "-")
        method = request.method
        qs = str(request.url.query)
        if qs:
            path = f"{path}?{qs}"
        status = response.status_code

        _access_logger.info(
            '%s  %s  %s  %d  %.0fms  "%s"',
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            ip, method + " " + path, status, elapsed, ua[:120]
        )
        return response

app.add_middleware(AccessLogMiddleware)

# ──────────────────── Rate limiting ──────────────────────────────
def _get_real_ip(request: Request) -> str:
    """Extract the real client IP, respecting X-Forwarded-For from Tailscale Funnel."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)

if _HAS_SLOWAPI:
    limiter = Limiter(key_func=_get_real_ip)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    log.info("Rate limiting enabled (slowapi, keyed by X-Forwarded-For)")
else:
    limiter = None
    log.info("slowapi not installed — rate limiting disabled")

def ratelimit(limit_str: str):
    """Decorator: apply rate limit if slowapi is available, no-op otherwise."""
    if limiter is not None:
        return limiter.limit(limit_str)
    def _noop(func):
        return func
    return _noop


class State:
    """In-memory state. Loaded once at startup."""
    events:        pd.DataFrame  # indexed by EventID
    event_index:   dict          # from event_index.json
    links:         pd.DataFrame  # consolidated event-well links across all radii
    panel:         pd.DataFrame  # indexed by (API Number, Date of Injection)
    tmle_summary:  dict[int, dict]
    attribution_q: dict[int, object]  # per-radius AttributionQ models
    loaded_at:     datetime


state = State()


@app.on_event("startup")
def _startup() -> None:
    # build_attribution_q.py needs to be importable so the unpickler can
    # resolve the AttributionQ class. Add the repo root to sys.path before
    # _load_attribution_qs() runs.
    sys.path.insert(0, str(REPO_ROOT))

    log.info("=" * 60)
    log.info("Loading dashboard state …")
    state.events        = _load_events()
    state.event_index   = _load_event_index()
    state.links         = _load_links()
    state.panel         = _load_panel()
    state.tmle_summary  = _load_tmle_summary()
    state.attribution_q = _load_causal_forests()
    state.loaded_at     = datetime.now()
    log.info("✅  Dashboard ready (loaded at %s)", state.loaded_at.isoformat())
    log.info("    %d attribution Q models available: %s",
             len(state.attribution_q), sorted(state.attribution_q.keys()))
    log.info("=" * 60)


# ──────────────────── Schemas ────────────────────────────────────
# We return plain dicts for simplicity; FastAPI handles JSON encoding via
# the default jsonable_encoder. Numpy floats / ints get coerced to native
# Python types automatically.


LANDING_HTML = TEMPLATES_DIR / "landing.html"
ANALYTICS_HTML = TEMPLATES_DIR / "analytics.html"

# ──────────────────── Routes: HTML ───────────────────────────────
@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    if not LANDING_HTML.exists():
        # Fallback to dashboard if no landing page
        return HTMLResponse(INDEX_HTML.read_text())
    return HTMLResponse(LANDING_HTML.read_text())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_app() -> HTMLResponse:
    if not INDEX_HTML.exists():
        return HTMLResponse("<h1>dashboard not found</h1>", status_code=500)
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/methodology", response_class=HTMLResponse)
def methodology() -> HTMLResponse:
    if not METHODOLOGY_HTML.exists():
        return HTMLResponse(
            "<h1>dashboard/templates/methodology.html not found</h1>",
            status_code=500,
        )
    return HTMLResponse(METHODOLOGY_HTML.read_text())


FAQ_HTML = TEMPLATES_DIR / "faq.html"

@app.get("/faq", response_class=HTMLResponse)
def faq() -> HTMLResponse:
    if not FAQ_HTML.exists():
        return HTMLResponse("<h1>FAQ not found</h1>", status_code=500)
    return HTMLResponse(FAQ_HTML.read_text())


ANALYTICS_KEY = os.environ.get("ANALYTICS_KEY", "geminae2026")

@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(key: str = Query("")) -> HTMLResponse:
    if key != ANALYTICS_KEY:
        raise HTTPException(404)
    if not ANALYTICS_HTML.exists():
        return HTMLResponse("<h1>Analytics not found</h1>", status_code=500)
    return HTMLResponse(ANALYTICS_HTML.read_text())


BEACON_LOG = Path(os.environ.get("BEACON_LOG", "/tmp/beacon.log"))
_beacon_logger = logging.getLogger("beacon")
_beacon_handler = logging.FileHandler(BEACON_LOG)
_beacon_handler.setFormatter(logging.Formatter("%(message)s"))
_beacon_logger.addHandler(_beacon_handler)
_beacon_logger.setLevel(logging.INFO)


@app.post("/api/beacon")
async def beacon(request: Request) -> dict:
    """Receive client-side interaction events."""
    try:
        body = await request.json()
        xff = request.headers.get("x-forwarded-for", "")
        ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "-")
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        event = body.get("event", "unknown")
        data = body.get("data", {})
        page = body.get("page", "-")
        sid = body.get("sid", "-")
        tz = body.get("tz", "-")
        screen = body.get("screen", "-")
        lang = body.get("lang", "-")
        ref = body.get("ref", "-")
        _beacon_logger.info(
            '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s',
            ts, ip, sid, event, json.dumps(data), page, tz, screen, lang, ref
        )
        return {"ok": True}
    except Exception:
        return {"ok": False}


@app.get("/api/analytics")
def analytics_data(key: str = Query("")) -> dict:
    """Return parsed access log lines for the analytics dashboard."""
    if key != ANALYTICS_KEY:
        raise HTTPException(404)
    if not ACCESS_LOG.exists():
        return {"lines": []}
    try:
        lines = ACCESS_LOG.read_text().strip().split("\n") if ACCESS_LOG.exists() else []
        beacons = BEACON_LOG.read_text().strip().split("\n") if BEACON_LOG.exists() else []
        return {"lines": lines[-10000:], "beacons": beacons[-10000:]}
    except Exception as e:
        return {"lines": [], "beacons": [], "error": str(e)}


# ──────────────────── Routes: API ────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    # Data freshness: latest event date and model dates
    try:
        dates = state.events["Origin Date"]
        latest_event = dates.max().strftime("%Y-%m-%d") if hasattr(dates.max(), 'strftime') else str(dates.max())[:10]
    except Exception:
        latest_event = "unknown"

    tmle_files = sorted(REPO_ROOT.glob("tmle_shift_365d_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    tmle_date = datetime.fromtimestamp(tmle_files[0].stat().st_mtime).strftime("%Y-%m-%d") if tmle_files else "never"

    cf_files = sorted(REPO_ROOT.glob("cf_cate_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    cf_date = datetime.fromtimestamp(cf_files[0].stat().st_mtime).strftime("%Y-%m-%d") if cf_files else "never"

    return {
        "status":           "ok",
        "loaded_at":        state.loaded_at.isoformat(),
        "n_events":         len(state.events),
        "n_panel":          len(state.panel),
        "n_links":          len(state.links),
        "n_cate_models":    len(state.attribution_q),
        "radii_km":         RADII_KM,
        "latest_event_date": latest_event,
        "tmle_run_date":    tmle_date,
        "causal_forest_date": cf_date,
    }


@app.get("/api/events")
@ratelimit("120/minute")
def list_events(
    request: Request,
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
@ratelimit("120/minute")
def event_wells(
    request: Request,
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


@app.get("/api/event/{event_id}/attribution")
@ratelimit("60/minute")
def event_attribution(
    request: Request,
    event_id:  str,
    radius_km: int = Query(7, ge=1, le=20),
) -> dict:
    """Per-well in-model attribution of expected seismic outcome.

    For each well within `radius_km` of the event, compute the model's
    estimate of this well's contribution to the predicted local seismic
    outcome (max ML within R km on the event date) by g-computation:

        contribution_i = Q(this well, actual injection)
                        − Q(this well, zero injection)

    The Q model used here is `q_attribution_<R>km.pkl`, fit at the well-day
    level on `panel_with_faults_<R>km.csv` by `build_attribution_q.py`. It
    is a different (per-well) Q than the one used by the TMLE shift /
    dose-response / mediation drivers (which fit at the cluster-day level).

    The "zero injection" counterfactual sets:
      - cum_vol_<window>d_BBL → 0 (the treatment)
      - bhp_vw_avg_<window>  → 0.45 psi/ft × perf_depth_ft (depth-only
                                 hydrostatic baseline; matches the panel's
                                 BHP construction with WHP=0)

    All other features (formation, depth, fault distance, segment count,
    days_active) are held at their actual values.

    Returns:
      - per-well: {api, distance_km, q_factual, q_counterfactual,
                   contribution_ml, share, pn}
        where `pn` is a simplified probability-of-necessity proxy:
            pn = 1 − Q_counterfactual / max(Q_factual, ε)   clamped to [0, 1]
        Higher `pn` means "more of the model's predicted outcome would go
        away if this well had been shut off." It is NOT Pearl's formal PN;
        it is the in-model fractional reduction.
      - aggregate: total_contribution, top_contributor, n_wells_with_pn_gt_50
      - disclaimer: a one-line caveat the frontend should display

    This endpoint requires `q_attribution_<R>km.pkl` on disk. If missing,
    returns HTTP 503 with a hint to run `build_attribution_q.py`.
    """
    if event_id not in state.events.index:
        raise HTTPException(404, f"event {event_id} not found")
    if radius_km not in state.attribution_q:
        raise HTTPException(503,
            f"q_attribution_{radius_km}km.pkl not loaded. "
            f"Run `python build_attribution_q.py --radii {radius_km}` and "
            f"restart the server.")

    aq = state.attribution_q[radius_km]
    ev = state.events.loc[event_id]
    ev_date = ev["Origin Date"]

    sub = state.links[
        (state.links["EventID"] == event_id)
        & (state.links["radius_km"] == radius_km)
    ].sort_values("distance_km")

    if sub.empty:
        return {
            "event":          {"id": event_id, "ml": float(ev["Local Magnitude"])},
            "radius_km":      radius_km,
            "window_days":    aq.window_days,
            "n_wells":        0,
            "wells":          [],
            "aggregate":      {"total_contribution_ml": 0.0,
                               "top_contributor": None,
                               "n_wells_with_pn_gt_50": 0},
            "disclaimer":     ATTRIBUTION_DISCLAIMER,
        }

    # Build a feature dataframe for ALL wells in one shot, then call Q.predict
    # twice (factual + counterfactual) and zip the results.
    W_col = f"cum_vol_{aq.window_days}d_BBL"
    P_col = f"bhp_vw_avg_{aq.window_days}d"

    def _safe(v, fallback=0.0):
        """Coerce to float, replacing None/NaN with `fallback`."""
        if v is None:
            return fallback
        try:
            f = float(v)
        except (TypeError, ValueError):
            return fallback
        return fallback if (f != f) else f  # NaN check

    feature_rows = []
    api_list = []
    for _, link in sub.iterrows():
        api = int(link["API Number"])
        api_list.append(api)
        feats = _well_features_at(api, ev_date)
        perf_depth = _safe(feats.get("perf_depth_ft"), fallback=7000.0)
        feature_rows.append({
            W_col:                    _safe(feats.get(W_col)),
            P_col:                    _safe(feats.get(P_col)),
            "perf_depth_ft":          perf_depth,
            "days_active":            _safe(feats.get("days_active")),
            "Nearest Fault Dist (km)": _safe(_well_static(api, "Nearest Fault Dist (km)")),
            f"Fault Segments <= {radius_km} km": _safe(_well_static(api, f"Fault Segments <= {radius_km} km")),
            "formation":              feats.get("formation") or "UNKNOWN",
            "_distance_km":           float(link["distance_km"]),
            "_well_lat":              float(link["well_lat"]),
            "_well_lon":              float(link["well_lon"]),
        })

    fdf = pd.DataFrame(feature_rows)

    # Counterfactual: zero out treatment AND set BHP to depth-only baseline
    fdf_cf = fdf.copy()
    fdf_cf[W_col] = 0.0
    fdf_cf[P_col] = BRINE_GRADIENT_PSI_PER_FT * fdf_cf["perf_depth_ft"]

    # ── Causal Forest DML CATE estimation ──
    # If aq is a CausalForestBundle (from build_causal_forest.py), use the
    # forest's .estimate_cate() which gives honest per-well CIs from sample
    # splitting. Falls back to plain g-computation for legacy TMLE pickles.
    is_cf = hasattr(aq, 'estimate_cate')

    if is_cf:
        treatment_vals = fdf[W_col].to_numpy()
        result = aq.estimate_cate(fdf, treatment_vals)
        cate_arr  = result["cate"]
        ci_lo_arr = result["ci_low"]
        ci_hi_arr = result["ci_high"]
    else:
        # Legacy fallback: plain g-computation (no CIs)
        q_factual        = np.asarray(aq.predict(fdf), dtype=float)
        q_counterfactual = np.asarray(aq.predict(fdf_cf), dtype=float)
        cate_arr  = q_factual - q_counterfactual
        ci_lo_arr = np.full(len(fdf), float("nan"))
        ci_hi_arr = np.full(len(fdf), float("nan"))

    total_contribution = float(cate_arr.sum())
    pos_total = float(np.maximum(cate_arr, 0).sum())

    wells_out = []
    for i, row in fdf.iterrows():
        cate = float(cate_arr[i])
        ci_lo = float(ci_lo_arr[i])
        ci_hi = float(ci_hi_arr[i])
        share = (max(cate, 0.0) / pos_total) if pos_total > 0 else 0.0
        wells_out.append({
            "api":               int(api_list[i]),
            "distance_km":       float(row["_distance_km"]),
            "lat":               float(row["_well_lat"]),
            "lon":               float(row["_well_lon"]),
            "cate_ml":           cate,
            "cate_ci_low":       ci_lo if not np.isnan(ci_lo) else None,
            "cate_ci_high":      ci_hi if not np.isnan(ci_hi) else None,
            "share":             share,
            "formation":         row["formation"],
            "perf_depth_ft":     float(row["perf_depth_ft"])
                                  if pd.notna(row["perf_depth_ft"]) else None,
            "cum_vol_window":    float(row[W_col]),
        })

    # Sort by CATE descending (regulator-relevant ranking)
    wells_out.sort(key=lambda w: w["cate_ml"], reverse=True)

    top_contributor = wells_out[0] if wells_out else None

    return {
        "event": {
            "id":   event_id,
            "lat":  float(ev["Latitude (WGS84)"]),
            "lon":  float(ev["Longitude (WGS84)"]),
            "date": ev_date.strftime("%Y-%m-%d"),
            "ml":   float(ev["Local Magnitude"]),
        },
        "radius_km":   radius_km,
        "window_days": aq.window_days,
        "n_wells":     len(wells_out),
        "wells":       wells_out,
        "aggregate": {
            "total_cate_ml":     total_contribution,
            "positive_total_ml": pos_total,
            "top_contributor":   (top_contributor and {
                "api":       top_contributor["api"],
                "cate_ml":   top_contributor["cate_ml"],
                "share":     top_contributor["share"],
            }),
        },
        "estimand": "Causal Forest DML (Athey, Tibshirani & Wager 2019)" if is_cf else "g-computation (legacy)",
        "model_meta": {
            "n_train":      aq.n_train,
            "n_pos":        aq.n_pos,
            "method":       "CausalForestDML" if is_cf else "HurdleSuperLearner",
            "feature_cols": aq.confounder_cols if is_cf else getattr(aq, "feature_cols", []),
        },
        "disclaimer":  ATTRIBUTION_DISCLAIMER,
    }


ATTRIBUTION_DISCLAIMER = (
    "Causal Forest DML (Athey, Tibshirani & Wager 2019; Chernozhukov et al. "
    "2018): per-well CATE estimated via honest Causal Forest with Double "
    "Machine Learning. Cross-fitted XGBoost nuisance models for E[Y|L] and "
    "E[A|L]; the forest splits on treatment-effect heterogeneity with sample "
    "splitting for valid CIs. Doubly robust: misspecify the outcome or "
    "treatment model (not both) and the CATE is still consistent. "
    "Identified under: no unmeasured confounding, positivity, consistency. "
    "See /methodology for full discussion."
)


def _well_static(api: int, col: str) -> float:
    """Look up a constant per-well feature (fault distance, fault count) from
    any panel row for that API. Falls back to None if missing."""
    try:
        sub = state.panel.loc[api]
    except KeyError:
        return float("nan")
    if isinstance(sub, pd.Series):
        return float(sub.get(col, float("nan")))
    if col not in sub.columns:
        return float("nan")
    val = sub[col].dropna()
    return float(val.iloc[0]) if not val.empty else float("nan")


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
@ratelimit("120/minute")
def well_timeseries(
    request: Request,
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


def _safe_float(v, fallback=0.0):
    """Coerce to float, replacing None/NaN with fallback."""
    if v is None:
        return fallback
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fallback
    return fallback if (f != f) else f


@app.get("/api/wells/{api_number}/threshold")
@ratelimit("60/minute")
def well_threshold(
    request: Request,
    api_number: int,
    radius_km: int   = Query(7, ge=1, le=20),
    event_date: str  = Query(..., description="Event date YYYY-MM-DD (for panel lookup)"),
    n_points:   int  = Query(25, ge=5, le=100),
    max_vol:    float = Query(0, description="Max volume for grid (0 = auto 2× current)"),
    cate_threshold: float = Query(0.03, description="Regulatory CATE threshold (ML)"),
) -> dict:
    """Well-specific dose-response curve: CATE vs cumulative volume at this
    well's covariate profile, evaluated by the Causal Forest at a grid of
    treatment levels.

    The regulator uses this to find: "at what cumulative volume does this
    well's CATE become statistically significant (CI lower bound > 0) or
    exceed a regulatory threshold?"

    Returns the curve + the intersection point (max allowable volume).
    """
    if radius_km not in state.attribution_q:
        raise HTTPException(503, f"No CausalForest model for {radius_km}km")

    aq = state.attribution_q[radius_km]
    is_cf = hasattr(aq, 'estimate_cate')
    if not is_cf:
        raise HTTPException(503, "Threshold curve requires CausalForest model")

    ev_date = pd.Timestamp(event_date)
    feats = _well_features_at(api_number, ev_date)
    if not feats:
        raise HTTPException(404, f"No panel data for API {api_number} on {event_date}")

    W_col = f"cum_vol_{aq.window_days}d_BBL"
    current_vol = _safe_float(feats.get(W_col), 0.0)
    depth_ft    = _safe_float(feats.get("perf_depth_ft"), 7000.0)

    if max_vol <= 0:
        max_vol = max(current_vol * 2.5, 1_000_000)

    # Build the confounder row (same for all grid points — only treatment changes)
    well_row = pd.DataFrame([{
        "perf_depth_ft":               depth_ft,
        "days_active":                 _safe_float(feats.get("days_active")),
        "Nearest Fault Dist (km)":     _safe_float(_well_static(api_number, "Nearest Fault Dist (km)")),
        f"Fault Segments <= {radius_km} km": _safe_float(_well_static(api_number, f"Fault Segments <= {radius_km} km")),
        "formation":                   feats.get("formation", "UNKNOWN"),
    }])

    # Volume grid: 0 to max_vol, linearly spaced
    vol_grid = np.linspace(0, max_vol, n_points)

    # Evaluate CATE at each grid point by replicating the well row
    well_rows = pd.concat([well_row] * n_points, ignore_index=True)
    result = aq.estimate_cate(well_rows, vol_grid)

    curve = []
    threshold_vol = None
    significance_vol = None

    for i in range(n_points):
        v = float(vol_grid[i])
        c = float(result["cate"][i])
        lo = float(result["ci_low"][i])
        hi = float(result["ci_high"][i])
        curve.append({"vol": v, "cate": c, "ci_low": lo, "ci_high": hi})

        # Find where CATE exceeds the regulatory threshold
        if threshold_vol is None and c >= cate_threshold:
            threshold_vol = v

        # Find where CI lower bound > 0 (statistical significance)
        if significance_vol is None and lo > 0:
            significance_vol = v

    return {
        "api":              api_number,
        "radius_km":        radius_km,
        "event_date":       event_date,
        "current_vol":      current_vol,
        "current_cate":     float(result["cate"][np.argmin(np.abs(vol_grid - current_vol))]) if current_vol > 0 else 0.0,
        "depth_ft":         depth_ft,
        "cate_threshold":   cate_threshold,
        "threshold_vol":    threshold_vol,
        "significance_vol": significance_vol,
        "max_vol":          max_vol,
        "curve":            curve,
    }


@app.get("/api/event/{event_id}/report")
@ratelimit("10/minute")
def event_report(
    request: Request,
    event_id:  str,
    radius_km: int = Query(7, ge=1, le=20),
) -> Response:
    """Generate a regulatory-ready PDF report for an event with charts."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    if event_id not in state.events.index:
        raise HTTPException(404, f"event {event_id} not found")

    ev = state.events.loc[event_id]
    ev_date = ev["Origin Date"]
    ev_ml = _safe_float(ev.get("Local Magnitude", 0))
    ev_lat = _safe_float(ev.get("Latitude (WGS84)", 0))
    ev_lon = _safe_float(ev.get("Longitude (WGS84)", 0))
    ev_depth = _safe_float(ev.get("Depth of Hypocenter (Km.  Rel to MSL)", 0))

    # Attribution
    has_cate = radius_km in state.attribution_q
    wells_data = []
    total_cate = 0.0
    positive_total = 0.0
    if has_cate:
        try:
            attr_response = event_attribution(request, event_id, radius_km)
            wells_data = attr_response.get("wells", [])
            agg = attr_response.get("aggregate", {})
            total_cate = _safe_float(agg.get("total_cate_ml", 0))
            positive_total = _safe_float(agg.get("positive_total_ml", 0))
        except Exception:
            pass

    tmle = state.tmle_summary.get(radius_km, {})

    # Data vintage
    try:
        latest_event = state.events["Origin Date"].max()
        latest_str = latest_event.strftime("%Y-%m-%d") if hasattr(latest_event, "strftime") else str(latest_event)[:10]
    except Exception:
        latest_str = "unknown"
    tmle_files = sorted(REPO_ROOT.glob("tmle_shift_365d_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    tmle_date = datetime.fromtimestamp(tmle_files[0].stat().st_mtime).strftime("%Y-%m-%d") if tmle_files else "unknown"
    cf_files = sorted(REPO_ROOT.glob("cf_cate_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    cf_date = datetime.fromtimestamp(cf_files[0].stat().st_mtime).strftime("%Y-%m-%d") if cf_files else "unknown"

    # ── Build PDF ──
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.5*inch, rightMargin=0.5*inch,
        topMargin=0.5*inch, bottomMargin=0.5*inch,
    )
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    AMBER = colors.HexColor("#C47000")
    DARK = colors.HexColor("#222222")
    GRAY = colors.HexColor("#888888")
    RED = colors.HexColor("#CC2222")
    GREEN = colors.HexColor("#008A52")

    title_style = ParagraphStyle(
        "title", parent=styles["Heading1"], fontSize=16, textColor=DARK,
        spaceAfter=4, alignment=TA_LEFT,
    )
    subtitle_style = ParagraphStyle(
        "subtitle", parent=styles["Normal"], fontSize=9, textColor=GRAY,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "section", parent=styles["Heading2"], fontSize=10, textColor=AMBER,
        spaceAfter=4, spaceBefore=8, fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "body", parent=styles["Normal"], fontSize=9, textColor=DARK,
        spaceAfter=4,
    )
    small_style = ParagraphStyle(
        "small", parent=styles["Normal"], fontSize=7, textColor=GRAY,
    )

    # Header
    story.append(Paragraph(
        f"<font color='#222222'>Induced Seismicity Causal Attribution Report</font>",
        title_style
    ))
    story.append(Paragraph(
        f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} &bull; "
        f"<a href='https://tinyurl.com/ywf39tmv' color='#C47000'>https://tinyurl.com/ywf39tmv</a>",
        subtitle_style
    ))

    # Event details table
    story.append(Paragraph("EVENT DETAILS", section_style))
    event_table = Table([
        ["Event ID:", event_id, "Magnitude:", f"M{ev_ml:.1f}"],
        ["Date:", str(ev_date)[:10], "Depth:", f"{ev_depth:.2f} km MSL"],
        ["Location:", f"{ev_lat:.4f}\u00b0N, {ev_lon:.4f}\u00b0W", "Search Radius:", f"{radius_km} km"],
        ["Wells Found:", f"{len(wells_data)}", "Total CATE:", f"{total_cate:+.4f} ML"],
    ], colWidths=[1.2*inch, 2.2*inch, 1.2*inch, 2.2*inch])
    event_table.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 9),
        ("TEXTCOLOR", (0,0), (0,-1), AMBER),
        ("TEXTCOLOR", (2,0), (2,-1), AMBER),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("TOPPADDING", (0,0), (-1,-1), 2),
    ]))
    story.append(event_table)
    story.append(Spacer(1, 12))

    # ── CATE Waterfall Chart ──
    if wells_data:
        wells_sorted = sorted(wells_data, key=lambda w: -_safe_float(w.get("cate_ml", 0)))[:15]
        labels = [f"#{i+1} \u00b7 {str(w.get('api',''))[-4:]} \u00b7 {_safe_float(w.get('distance_km',0)):.1f}km"
                  for i, w in enumerate(wells_sorted)]
        cates = [_safe_float(w.get("cate_ml", 0)) for w in wells_sorted]
        ci_los = [_safe_float(w.get("cate_ci_low", 0)) for w in wells_sorted]
        ci_his = [_safe_float(w.get("cate_ci_high", 0)) for w in wells_sorted]

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        y_pos = range(len(labels))
        bar_colors = []
        for c, lo in zip(cates, ci_los):
            if c < 0:
                bar_colors.append("#2266CC")
            elif lo > 0:
                bar_colors.append("#CC2222")
            else:
                bar_colors.append("#C47000")

        errors = [[c - lo for c, lo in zip(cates, ci_los)], [hi - c for c, hi in zip(cates, ci_his)]]
        ax.barh(y_pos, cates, color=bar_colors, height=0.7, alpha=0.85,
                xerr=errors, ecolor="#888888", capsize=2, error_kw={"elinewidth":0.8})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="#C47000", linewidth=0.8)
        ax.set_xlabel("CATE (ML contribution)", fontsize=8)
        ax.set_title(f"Per-Well Causal Attribution (Top {len(wells_sorted)} of {len(wells_data)})",
                     fontsize=10, color="#C47000", loc="left")
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        img_buf.seek(0)
        story.append(Paragraph("PER-WELL CATE (CAUSAL FOREST DML)", section_style))
        story.append(Image(img_buf, width=7.2*inch, height=3.6*inch))
        story.append(Spacer(1, 6))

        # Significant wells callout
        sig_wells = [w for w in wells_data if _safe_float(w.get("cate_ci_low", 0)) > 0]
        if sig_wells:
            sig_text = f"<b>Statistically significant contributors (95% CI excludes zero): {len(sig_wells)}</b><br/>"
            for w in sig_wells[:5]:
                sig_text += (
                    f"&nbsp;&nbsp;API {w.get('api','')} &bull; CATE = "
                    f"{_safe_float(w.get('cate_ml',0)):+.4f} ML &bull; "
                    f"CI [{_safe_float(w.get('cate_ci_low',0)):+.4f}, "
                    f"{_safe_float(w.get('cate_ci_high',0)):+.4f}]<br/>"
                )
            story.append(Paragraph(sig_text, body_style))
        else:
            story.append(Paragraph(
                "<b><font color='#CC2222'>No wells with statistically significant individual contribution.</font></b><br/>"
                "All 95% CIs cross zero \u2014 this is distributed causation. "
                "Area-wide volume reduction (TMLE shift) is the appropriate regulatory tool.",
                body_style
            ))

    story.append(Spacer(1, 8))

    # ── Per-well table ──
    if wells_data:
        story.append(Paragraph("PER-WELL ATTRIBUTION TABLE", section_style))
        header = ["#", "API", "Dist km", "CATE (ML)", "95% CI", "Depth ft", "Vol 365d", "Sig?"]
        table_rows = [header]
        for i, w in enumerate(wells_data[:25]):
            cate = _safe_float(w.get("cate_ml", 0))
            ci_lo = _safe_float(w.get("cate_ci_low", 0))
            ci_hi = _safe_float(w.get("cate_ci_high", 0))
            dist = _safe_float(w.get("distance_km", 0))
            depth = _safe_float(w.get("perf_depth_ft", 0))
            vol = _safe_float(w.get("cum_vol_window", 0))
            sig = "YES" if ci_lo > 0 else "--"
            table_rows.append([
                str(i+1), str(w.get("api","")), f"{dist:.2f}",
                f"{cate:+.4f}",
                f"[{ci_lo:+.3f}, {ci_hi:+.3f}]",
                f"{depth:,.0f}" if depth else "-",
                f"{vol:,.0f}" if vol else "-",
                sig,
            ])
        tbl = Table(table_rows, colWidths=[0.3*inch, 0.7*inch, 0.6*inch, 0.8*inch, 1.3*inch, 0.7*inch, 0.9*inch, 0.4*inch])
        tbl_style = [
            ("FONT", (0,0), (-1,-1), "Helvetica", 7),
            ("BACKGROUND", (0,0), (-1,0), AMBER),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#dddddd")),
            ("ALIGN", (2,0), (-1,-1), "RIGHT"),
        ]
        # Highlight significant rows
        for i, row in enumerate(table_rows[1:], start=1):
            if row[-1] == "YES":
                tbl_style.append(("TEXTCOLOR", (-1,i), (-1,i), GREEN))
                tbl_style.append(("FONTNAME", (-1,i), (-1,i), "Helvetica-Bold"))
        tbl.setStyle(TableStyle(tbl_style))
        story.append(tbl)

    story.append(Spacer(1, 10))

    # ── Population TMLE Context ──
    if tmle:
        story.append(Paragraph(f"POPULATION TMLE CONTEXT @ {radius_km} KM", section_style))
        tmle_table = Table([
            ["Total Effect (P90 vs P10):", f"{tmle.get('TE', '--'):.4e}" if isinstance(tmle.get('TE'), (int,float)) else str(tmle.get('TE','--')),
             "% Mediated (via WHP):", f"{tmle.get('pct_mediated', '--'):.1f}%" if isinstance(tmle.get('pct_mediated'), (int,float)) else str(tmle.get('pct_mediated','--'))],
            ["Natural Direct Effect (NDE):", f"{tmle.get('NDE', '--'):.4e}" if isinstance(tmle.get('NDE'), (int,float)) else str(tmle.get('NDE','--')),
             "E[Y | A=1E7 BBL]:", f"{tmle.get('dose_1e7', '--'):.4e}" if isinstance(tmle.get('dose_1e7'), (int,float)) else str(tmle.get('dose_1e7','--'))],
            ["Natural Indirect Effect (NIE):", f"{tmle.get('NIE', '--'):.4e}" if isinstance(tmle.get('NIE'), (int,float)) else str(tmle.get('NIE','--')),
             "Shift (10% reduction):", f"{tmle.get('shift_10pct', '--'):.4e}" if isinstance(tmle.get('shift_10pct'), (int,float)) else str(tmle.get('shift_10pct','--'))],
        ], colWidths=[1.9*inch, 1.5*inch, 1.7*inch, 1.5*inch])
        tmle_table.setStyle(TableStyle([
            ("FONT", (0,0), (-1,-1), "Helvetica", 8),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0,0), (0,-1), AMBER),
            ("TEXTCOLOR", (2,0), (2,-1), AMBER),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ("TOPPADDING", (0,0), (-1,-1), 2),
        ]))
        story.append(tmle_table)

    # ── Footer ──
    story.append(Spacer(1, 12))
    story.append(Paragraph("DATA VINTAGE & METHODOLOGY", section_style))
    vintage_text = (
        f"Event catalog through: <b>{latest_str}</b> &bull; "
        f"TMLE models: <b>{tmle_date}</b> &bull; "
        f"Causal Forest models: <b>{cf_date}</b> &bull; "
        f"CATE radii: <b>{len(state.attribution_q)}</b> (1-20 km) &bull; "
        f"Panel observations: <b>{len(state.panel):,}</b><br/>"
        f"<br/>"
        "<b>Method:</b> Causal Forest DML (Athey, Tibshirani &amp; Wager 2019; Chernozhukov et al. 2018). "
        "Honest sample splitting, cross-fitted XGBoost nuisance models. Doubly robust: consistent if either "
        "outcome or treatment model is correctly specified.<br/>"
        "<b>Population TMLE:</b> van der Laan &amp; Rose 2011. SuperLearner ensemble. Validated against R tlverse (&lt;0.1%).<br/>"
        "<b>Data sources:</b> TexNet Earthquake Catalog &bull; RRC H-10 Injection Records<br/>"
        "<b>Interpretation:</b> A 95% CI excluding zero indicates a statistically significant individual contribution. "
        "A CI crossing zero does not exonerate a well \u2014 it indicates the data is insufficient to distinguish its "
        "effect from noise. Distributed causation (all CIs crossing zero) supports area-wide rather than targeted action."
    )
    story.append(Paragraph(vintage_text, small_style))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="seis_report_{event_id}_{radius_km}km.pdf"',
        },
    )
