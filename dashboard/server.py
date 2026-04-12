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
import pickle
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
METHODOLOGY_HTML   = TEMPLATES_DIR / "methodology.html"

EVENTS_CSV         = REPO_ROOT / "texnet_events_filtered.csv"
EVENT_INDEX_JSON   = REPO_ROOT / "event_index.json"
LINKS_PARQUET      = REPO_ROOT / "event_well_links.parquet"
LINKS_CSV_FALLBACK = REPO_ROOT / "event_well_links.csv"  # only if pyarrow missing
PANEL_CSV          = REPO_ROOT / "well_day_panel.csv"
Q_PKL_FMT          = "q_attribution_{R}km.pkl"

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


def _load_attribution_qs() -> dict[int, object]:
    """Load the per-radius attribution Q models, if any are on disk.

    These are produced by build_attribution_q.py and are optional —
    the dashboard works without them, but the per-event attribution
    endpoint will return an error if the requested radius's Q is missing.

    Pickle dispatch: the AttributionQ class is defined in
    build_attribution_q.py, which on the build side runs as `__main__`. To
    let pickle find the class regardless of how the build script was
    invoked, we register the class under both `__main__` AND
    `build_attribution_q` namespaces before unpickling.
    """
    out: dict[int, object] = {}

    # Make `build_attribution_q.AttributionQ` AND `__main__.AttributionQ`
    # both resolve to the same class object
    try:
        import build_attribution_q as baq
        import __main__ as main_mod
        if not hasattr(main_mod, "AttributionQ"):
            main_mod.AttributionQ = baq.AttributionQ
    except Exception as e:
        log.warning("Could not register AttributionQ for pickle: %s", e)

    for R in RADII_KM:
        path = REPO_ROOT / Q_PKL_FMT.format(R=R)
        if not path.exists():
            continue
        try:
            with path.open("rb") as f:
                out[R] = pickle.load(f)
            log.info("📄  Loaded Q model: %s", path.name)
        except Exception as e:
            log.warning("Failed to load %s: %s", path.name, e)
    if not out:
        log.warning("⚠️   No q_attribution_*.pkl files found — "
                    "attribution endpoint will be unavailable. "
                    "Run `python build_attribution_q.py` to fit them.")
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
    state.attribution_q = _load_attribution_qs()
    state.loaded_at     = datetime.now()
    log.info("✅  Dashboard ready (loaded at %s)", state.loaded_at.isoformat())
    log.info("    %d attribution Q models available: %s",
             len(state.attribution_q), sorted(state.attribution_q.keys()))
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


@app.get("/methodology", response_class=HTMLResponse)
def methodology() -> HTMLResponse:
    if not METHODOLOGY_HTML.exists():
        return HTMLResponse(
            "<h1>dashboard/templates/methodology.html not found</h1>",
            status_code=500,
        )
    return HTMLResponse(METHODOLOGY_HTML.read_text())


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


@app.get("/api/event/{event_id}/attribution")
def event_attribution(
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

    # ── TMLE-targeted predictions (not plain g-computation) ──
    # Q̂*(a, l) = Q̂(a, l) + ε · H(a, l)
    # Uses the pre-computed ε from the offline targeting step.
    # hasattr check for backwards compatibility with old pickles that
    # don't have the g model.
    has_tmle = hasattr(aq, 'g') and aq.g is not None and hasattr(aq, 'epsilon')

    if has_tmle:
        q_factual        = np.asarray(aq.predict_targeted(fdf), dtype=float)
        q_counterfactual = np.asarray(aq.predict_targeted(fdf_cf), dtype=float)
        cate_se          = aq.cate_se()  # population-level SE as approximation
    else:
        # Fallback to plain g-computation (no targeting)
        q_factual        = np.asarray(aq.predict(fdf), dtype=float)
        q_counterfactual = np.asarray(aq.predict(fdf_cf), dtype=float)
        cate_se          = float("nan")

    contributions = q_factual - q_counterfactual
    total_contribution = float(contributions.sum())

    # Share normalization: only over POSITIVE contributions
    pos_total = float(np.maximum(contributions, 0).sum())

    z95 = 1.959963984540054
    wells_out = []
    for i, row in fdf.iterrows():
        qf = float(q_factual[i])
        qc = float(q_counterfactual[i])
        cate = qf - qc
        share = (max(cate, 0.0) / pos_total) if pos_total > 0 else 0.0
        wells_out.append({
            "api":               int(api_list[i]),
            "distance_km":       float(row["_distance_km"]),
            "lat":               float(row["_well_lat"]),
            "lon":               float(row["_well_lon"]),
            "q_targeted":        qf,
            "q_cf_targeted":     qc,
            "cate_ml":           cate,
            "cate_ci_low":       cate - z95 * cate_se if not np.isnan(cate_se) else None,
            "cate_ci_high":      cate + z95 * cate_se if not np.isnan(cate_se) else None,
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
            "cate_se":           cate_se if not np.isnan(cate_se) else None,
            "top_contributor":   (top_contributor and {
                "api":       top_contributor["api"],
                "cate_ml":   top_contributor["cate_ml"],
                "share":     top_contributor["share"],
            }),
        },
        "estimand": "CATE-TMLE" if has_tmle else "g-computation (no targeting)",
        "model_meta": {
            "n_train":     aq.n_train,
            "n_pos":       aq.n_pos,
            "epsilon":     float(aq.epsilon) if has_tmle else None,
            "if_var":      float(aq.if_var) if has_tmle else None,
            "feature_cols": aq.feature_cols,
        },
        "disclaimer":  ATTRIBUTION_DISCLAIMER,
    }


ATTRIBUTION_DISCLAIMER = (
    "CATE-TMLE: Conditional Average Treatment Effect estimated via Targeted "
    "Maximum Likelihood. CATE(l) = Q̂*(actual, l) − Q̂*(shut-off, l) where "
    "Q̂* is the TMLE-targeted outcome regression (hurdle Super Learner + "
    "epsilon-fluctuation via the clever covariate from the conditional density "
    "g(A|L)). The 95% CI uses the influence-function-based SE from the "
    "population-level targeting step. Identified under: no unmeasured "
    "confounding, positivity, consistency. See /methodology for full discussion."
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
