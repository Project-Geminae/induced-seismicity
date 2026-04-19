#!/usr/bin/env python3
"""
auto_ingest.py — Automated daily data ingestion for the SEIS pipeline.

EARTHQUAKE DATA (daily, automated):
  Fetches from IRIS FDSN web service — the same data as TexNet, federated
  through IRIS. Free, no auth, real-time. Updates event_index.json and
  texnet_events_filtered.csv.

INJECTION DATA (manual, quarterly):
  RRC H-10 injection data is NOT available via API. It must be manually
  downloaded from TexNet's Google Drive link and placed in ~/Downloads/.
  This script detects new injection files and triggers a full pipeline re-run.

Usage:
  # Daily cron (earthquakes only — fast, ~2 min):
  python auto_ingest.py --events-only

  # Full pipeline (after manual injection data update — slow, ~60 min):
  python auto_ingest.py --full

  # Check data freshness only:
  python auto_ingest.py --status

  # Force re-run everything:
  python auto_ingest.py --full --force
"""

import argparse
import csv
import io
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import urllib.request

# ── Config ──
REPO = Path(__file__).resolve().parent
DATA_DIR = REPO
DOWNLOADS = Path.home() / "Downloads"

FDSN_URL = "https://service.iris.edu/fdsnws/event/1/query"
MIDLAND_BBOX = dict(minlat=30.6, maxlat=33.4, minlon=-103.2, maxlon=-100.2)
MIN_MAG = 1.0
START_DATE = "2017-01-01"

# Quality filters (matching seismic_data_import.py)
MAX_RMS = 0.5
MAX_DEPTH_UNC = 2.0
MAX_HORIZ_ERR = 2.0
MIN_PHASE_COUNT = 8

# Google Drive file IDs (TexNet public data)
GDRIVE_FILES = {
    "events": {
        "id": "1vaky_Tk-brHWi_Mxd8e090BR6JgAdz1E",
        "filename": "texnet_events_gdrive.csv",
        "symlink": "texnet_events.csv",
    },
    "injection": {
        "id": "1O2a61OQI9iMkuLjLC39GHHqYesjYhqg3",
        "filename": "texnet_injection_gdrive.csv",
        "symlink": "swd_data.csv",
    },
}
GDRIVE_TIMESTAMPS_FILE = REPO / ".gdrive_timestamps.json"

LOG_FILE = REPO / "auto_ingest.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger("auto_ingest")


def _load_gdrive_timestamps() -> dict:
    if GDRIVE_TIMESTAMPS_FILE.exists():
        with open(GDRIVE_TIMESTAMPS_FILE) as f:
            return json.load(f)
    return {}


def _save_gdrive_timestamps(ts: dict):
    with open(GDRIVE_TIMESTAMPS_FILE, "w") as f:
        json.dump(ts, f, indent=2)


def check_gdrive_update(file_key: str) -> dict:
    """Check if a Google Drive file has been updated since our last download.

    Returns: {"updated": bool, "remote_modified": str, "local_modified": str, "size": int}
    """
    info = GDRIVE_FILES[file_key]
    url = f"https://drive.usercontent.google.com/download?id={info['id']}&export=download&confirm=t"

    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "SEIS-AutoIngest/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            remote_modified = resp.headers.get("Last-Modified", "")
            content_length = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        log.warning("Failed to check Drive file %s: %s", file_key, e)
        return {"updated": False, "error": str(e)}

    saved = _load_gdrive_timestamps()
    local_modified = saved.get(file_key, "")

    is_updated = remote_modified != local_modified and remote_modified != ""

    return {
        "updated": is_updated,
        "remote_modified": remote_modified,
        "local_modified": local_modified,
        "size_mb": round(content_length / 1e6, 1),
    }


def download_gdrive_file(file_key: str) -> Path:
    """Download a Google Drive file and update the timestamp record."""
    info = GDRIVE_FILES[file_key]
    url = f"https://drive.usercontent.google.com/download?id={info['id']}&export=download&confirm=t"
    dest = DOWNLOADS / info["filename"]

    log.info("Downloading %s (%s) ...", file_key, url[:80])
    req = urllib.request.Request(url, headers={"User-Agent": "SEIS-AutoIngest/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        remote_modified = resp.headers.get("Last-Modified", "")
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    size_mb = dest.stat().st_size / 1e6
    log.info("Downloaded %s: %.1f MB → %s", file_key, size_mb, dest)

    # Update symlink
    symlink = DATA_DIR / info["symlink"]
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(dest)
    log.info("Symlink: %s → %s", info["symlink"], dest.name)

    # Save timestamp
    saved = _load_gdrive_timestamps()
    saved[file_key] = remote_modified
    _save_gdrive_timestamps(saved)

    return dest


def check_all_gdrive() -> dict:
    """Check both Drive files for updates. Returns summary."""
    results = {}
    for key in GDRIVE_FILES:
        results[key] = check_gdrive_update(key)
        if results[key].get("updated"):
            log.info("Google Drive %s has been UPDATED (remote: %s, local: %s)",
                     key, results[key]["remote_modified"], results[key]["local_modified"])
        else:
            log.info("Google Drive %s unchanged", key)
    return results


def fetch_fdsn_events(start_date: str, end_date: str) -> list[dict]:
    """Fetch earthquake events from IRIS FDSN web service."""
    params = {
        **MIDLAND_BBOX,
        "minmagnitude": MIN_MAG,
        "starttime": start_date,
        "endtime": end_date,
        "format": "text",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{FDSN_URL}?{query}"
    log.info("Fetching FDSN: %s", url)

    req = urllib.request.Request(url, headers={"User-Agent": "SEIS-AutoIngest/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8")

    lines = text.strip().split("\n")
    if not lines or lines[0].startswith("#"):
        header = lines[0] if lines else ""
        data_lines = lines[1:] if lines else []
    else:
        data_lines = lines

    events = []
    for line in data_lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 13:
            continue
        try:
            events.append({
                "EventID": parts[8].strip() if parts[8].strip().startswith("tx") else parts[0].strip(),
                "Origin Date": parts[1].strip()[:10],
                "Origin Time": parts[1].strip(),
                "Latitude (WGS84)": float(parts[2]),
                "Longitude (WGS84)": float(parts[3]),
                "Depth of Hypocenter (Km.  Rel to MSL)": float(parts[4]) if parts[4].strip() else 0,
                "Local Magnitude": float(parts[10]) if parts[10].strip() else 0,
                "MagType": parts[9].strip(),
                "Author": parts[5].strip(),
                "Catalog": parts[6].strip(),
                "Contributor": parts[7].strip(),
                "ContributorID": parts[8].strip(),
                "EventLocationName": parts[12].strip() if len(parts) > 12 else "",
            })
        except (ValueError, IndexError) as e:
            log.warning("Skipping malformed FDSN line: %s", e)
            continue

    log.info("Fetched %d events from FDSN", len(events))
    return events


def apply_quality_filters(events: list[dict]) -> list[dict]:
    """Apply the same quality filters as seismic_data_import.py.

    Note: FDSN text format doesn't include RMS, phase count, etc.
    We filter on magnitude and spatial bounds only. The full quality
    filters are applied by seismic_data_import.py on the merged CSV.
    """
    filtered = [
        e for e in events
        if e["Local Magnitude"] >= MIN_MAG
        and MIDLAND_BBOX["minlat"] <= e["Latitude (WGS84)"] <= MIDLAND_BBOX["maxlat"]
        and MIDLAND_BBOX["minlon"] <= e["Longitude (WGS84)"] <= MIDLAND_BBOX["maxlon"]
    ]
    log.info("After basic filters: %d events (from %d)", len(filtered), len(events))
    return filtered


def update_event_index(events: list[dict]) -> int:
    """Merge new events into event_index.json. Returns count of new events added."""
    index_path = DATA_DIR / "event_index.json"
    if index_path.exists():
        with open(index_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    new_count = 0
    for e in events:
        eid = e["EventID"]
        if eid not in existing:
            existing[eid] = {
                "lat": e["Latitude (WGS84)"],
                "lon": e["Longitude (WGS84)"],
                "ml": e["Local Magnitude"],
                "date": e["Origin Date"],
                "depth_km_msl": e["Depth of Hypocenter (Km.  Rel to MSL)"],
            }
            new_count += 1

    # Backup and write
    if new_count > 0:
        backup = index_path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
        if index_path.exists():
            shutil.copy2(index_path, backup)
            log.info("Backed up event_index.json → %s", backup.name)

        with open(index_path, "w") as f:
            json.dump(existing, f)
        log.info("Updated event_index.json: %d new events (total: %d)", new_count, len(existing))
    else:
        log.info("No new events to add (all %d already in index)", len(events))

    return new_count


def write_events_csv(events: list[dict]) -> Path:
    """Write events to the TexNet-compatible CSV format."""
    today = datetime.now().strftime("%d%b%Y").upper()
    csv_path = DOWNLOADS / f"texnet_event_data_{today}.csv"

    # Build the CSV with TexNet column names
    fieldnames = [
        "EventID", "Evaluation Status", "Origin Date", "Origin Time",
        "Local Magnitude", "Preferred Magnitude", "Latitude (WGS84)",
        "Latitude Error (km)", "Longitude (WGS84)", "Longitude Error (km)",
        "Depth of Hypocenter (Km.  Rel to MSL)",
        "Depth of Hypocenter (Km. Rel to Ground Surface)",
        "Depth Uncertainty (Km. Corresponds to 1 st dev)",
        "RMS", "UsedPhaseCount", "UsedStationCount",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            row = {
                "EventID": e["EventID"],
                "Evaluation Status": "final",
                "Origin Date": e.get("Origin Time", e["Origin Date"]),
                "Origin Time": e.get("Origin Time", ""),
                "Local Magnitude": e["Local Magnitude"],
                "Preferred Magnitude": e["Local Magnitude"],
                "Latitude (WGS84)": e["Latitude (WGS84)"],
                "Latitude Error (km)": "",
                "Longitude (WGS84)": e["Longitude (WGS84)"],
                "Longitude Error (km)": "",
                "Depth of Hypocenter (Km.  Rel to MSL)": e["Depth of Hypocenter (Km.  Rel to MSL)"],
                "Depth of Hypocenter (Km. Rel to Ground Surface)": "",
                "Depth Uncertainty (Km. Corresponds to 1 st dev)": "",
                "RMS": "",
                "UsedPhaseCount": "",
                "UsedStationCount": "",
            }
            writer.writerow(row)

    log.info("Wrote %d events to %s", len(events), csv_path)

    # Update symlink
    symlink = DATA_DIR / "texnet_events.csv"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(csv_path)
    log.info("Symlink: texnet_events.csv → %s", csv_path.name)

    return csv_path


def check_injection_data() -> dict:
    """Check for the latest injection data file in Downloads."""
    pattern = "texnet_injection_data_"
    candidates = sorted(DOWNLOADS.glob(f"{pattern}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    symlink = DATA_DIR / "swd_data.csv"

    current_target = None
    if symlink.is_symlink():
        current_target = symlink.resolve()

    if not candidates:
        return {"status": "missing", "file": None, "current": str(current_target)}

    latest = candidates[0]
    is_new = current_target != latest.resolve() if current_target else True

    return {
        "status": "new" if is_new else "current",
        "file": str(latest),
        "current": str(current_target),
        "age_days": (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).days,
    }


def update_injection_symlink(injection_path: Path):
    """Update the swd_data.csv symlink to point to the new injection file."""
    symlink = DATA_DIR / "swd_data.csv"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(injection_path)
    log.info("Symlink: swd_data.csv → %s", injection_path.name)


def run_pipeline(steps: str = "0,1,2,3,4", parallel: int = 1):
    """Run the data processing pipeline."""
    cmd = [
        sys.executable, str(REPO / "run_all.py"),
        "--only", *steps.split(","),
        "--parallel-radii", str(parallel),
        "--continue-on-error",
    ]
    log.info("Running pipeline: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        log.error("Pipeline failed (exit %d):\n%s", result.returncode, result.stderr[-2000:])
        return False
    log.info("Pipeline completed successfully")
    return True


def rebuild_docker(host: str = "minitim@100.65.23.59"):
    """Rebuild and restart the Docker container on minitim."""
    log.info("Rebuilding Docker container on minitim...")
    # This would need SSH access — placeholder for now
    log.warning("Docker rebuild requires manual SSH. Run on minitim:")
    log.warning("  cd ~/dashboard-docker && docker build -t seis-dashboard . && "
                "docker stop seis-dashboard && docker rm seis-dashboard && "
                "docker run -d --name seis-dashboard ...")


def get_status() -> dict:
    """Return current data freshness status."""
    # Event index
    ei_path = DATA_DIR / "event_index.json"
    if ei_path.exists():
        with open(ei_path) as f:
            events = json.load(f)
        dates = [e.get("date", "") for e in events.values() if e.get("date")]
        latest_event = max(dates) if dates else "?"
        event_count = len(events)
    else:
        latest_event = "missing"
        event_count = 0

    # Injection data
    inj = check_injection_data()

    # TMLE results
    tmle_files = sorted(DATA_DIR.glob("tmle_shift_365d_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    tmle_date = datetime.fromtimestamp(tmle_files[0].stat().st_mtime).strftime("%Y-%m-%d") if tmle_files else "never"

    # Causal Forest
    cf_files = sorted(DATA_DIR.glob("cf_cate_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    cf_date = datetime.fromtimestamp(cf_files[0].stat().st_mtime).strftime("%Y-%m-%d") if cf_files else "never"

    return {
        "events": {"count": event_count, "latest_date": latest_event},
        "injection": inj,
        "tmle_last_run": tmle_date,
        "causal_forest_last_run": cf_date,
        "pipeline_log": str(REPO / "pipeline_run.log"),
    }


def main():
    parser = argparse.ArgumentParser(description="SEIS automated data ingestion")
    parser.add_argument("--events-only", action="store_true",
                        help="Fetch new earthquake data only (fast, ~2 min)")
    parser.add_argument("--full", action="store_true",
                        help="Full pipeline re-run (slow, ~60 min on minitim)")
    parser.add_argument("--status", action="store_true",
                        help="Print data freshness status and exit")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if data hasn't changed")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Parallel workers for pipeline (default: 1 on Mac, use 16 on minitim)")
    parser.add_argument("--check-drive", action="store_true",
                        help="Check Google Drive for updated TexNet files and download if new")
    args = parser.parse_args()

    if args.status:
        status = get_status()
        # Also check Drive freshness
        log.info("Checking Google Drive for updates...")
        drive_status = check_all_gdrive()
        status["gdrive"] = drive_status
        print(json.dumps(status, indent=2))
        return

    log.info("="*60)
    log.info("SEIS Auto-Ingest starting")
    log.info("="*60)

    # ── Check Google Drive for updated files ──
    drive_updated = False
    if args.check_drive or args.full:
        log.info("Checking Google Drive for TexNet data updates...")
        drive_status = check_all_gdrive()

        for key, info in drive_status.items():
            if info.get("updated"):
                log.info("⬇ Downloading updated %s from Google Drive (%s MB)...",
                         key, info.get("size_mb", "?"))
                download_gdrive_file(key)
                drive_updated = True

        if not drive_updated:
            log.info("No Drive updates found.")

    # ── Fetch earthquake data from FDSN ──
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    events = fetch_fdsn_events(START_DATE, end_date)
    events = apply_quality_filters(events)
    new_count = update_event_index(events)
    csv_path = write_events_csv(events)

    if args.events_only and not args.check_drive:
        if new_count > 0:
            log.info("✅ %d new events added. Run with --full to update the pipeline.", new_count)
        else:
            log.info("✅ No new events. Data is current.")
        status = get_status()
        log.info("Status: %s", json.dumps(status, indent=2))
        return

    if args.check_drive and not args.full:
        if drive_updated:
            log.info("✅ Drive data updated. Run with --full to re-run the pipeline.")
        else:
            log.info("✅ No updates. Data is current.")
        if new_count > 0:
            log.info("   Also: %d new FDSN events added.", new_count)
        return

    if args.full:
        # Check injection data — prefer Drive download, fall back to local
        inj = check_injection_data()
        if inj["status"] == "new":
            log.info("New injection data found: %s", inj["file"])
            update_injection_symlink(Path(inj["file"]))
        elif inj["status"] == "missing" and not drive_updated:
            log.error("No injection data found. Run with --check-drive to fetch from Google Drive.")
            sys.exit(1)
        else:
            if not args.force and new_count == 0:
                log.info("No new data. Use --force to re-run anyway.")
                return
            log.info("Using existing injection data: %s (age: %d days)", inj["file"], inj["age_days"])

        # Run pipeline steps 0-4 (ingest + panel + spatiotemporal + geoscience)
        log.info("Running pipeline steps 0-4...")
        if not run_pipeline("0 1 2 3 4", args.parallel):
            log.error("Pipeline failed at steps 0-4")
            sys.exit(1)

        # Run TMLE (steps 9-11)
        log.info("Running TMLE analysis (steps 9-11)...")
        if not run_pipeline("9 10 11", args.parallel):
            log.warning("TMLE steps had errors (non-fatal)")

        log.info("✅ Pipeline complete. Rebuild Docker container to deploy.")
        rebuild_docker()

    status = get_status()
    log.info("Final status: %s", json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
