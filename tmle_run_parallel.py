#!/usr/bin/env python3
"""
tmle_run_parallel.py
────────────────────
Parallel-radii driver for the TMLE sweep on minitim. Uses
ProcessPoolExecutor to fan out 20 radii across however many workers are
configured (default = nproc / 3, so all three drivers can run concurrently
without oversubscribing).

Each worker is a fresh Python process, which sidesteps the GIL completely
for the SuperLearner stack and gives each radius its own ~3-4 GB RSS
without competing for shared memory.

Usage on minitim:

    # Single driver, parallel across radii (default workers = nproc/3 = 10)
    .venv/bin/python tmle_run_parallel.py shift     --window 365 --shift 0.10
    .venv/bin/python tmle_run_parallel.py dose      --window 365 \\
        --grid 1e4 1e5 1e6 1e7 1e8
    .venv/bin/python tmle_run_parallel.py mediation --window 365

    # Override worker count if you want to be aggressive
    TMLE_WORKERS=24 .venv/bin/python tmle_run_parallel.py shift

Outputs the same CSV format as the single-process drivers; the per-radius
rows are concatenated and written incrementally so a partial run still has
useful output if you Ctrl-C out.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import causal_core as cc
import tmle_core as tmle

PANEL_FMT = "panel_with_faults_{R}km.csv"
RADII     = list(range(1, 21))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────── Per-radius worker functions ────────────────
# Each of these is a top-level function (picklable) that can be sent to a
# ProcessPoolExecutor worker.

def _worker_shift(args: dict) -> dict | None:
    R = args["R"]
    window = args["window"]
    shift_pct = args["shift_pct"]
    path = Path(PANEL_FMT.format(R=R))
    if not path.exists():
        return None
    panel = cc.load_panel(str(path), radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=window)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=window)
    data = data.copy()
    data["_cluster"] = cluster.values
    t0 = time.time()
    trim_pct = float(args.get("trim_pct", 0.01))
    use_cv = bool(args.get("cv_tmle", False))
    if use_cv:
        result = tmle.cv_tmle_shift(
            df=data, A_col=W, L_cols=confs, Y_col=S, cluster_col="_cluster",
            shift_pct=shift_pct,
        )
    else:
        result = tmle.tmle_shift(
            df=data, A_col=W, L_cols=confs, Y_col=S, cluster_col="_cluster",
            shift_pct=shift_pct, trim_pct=trim_pct,
        )
    elapsed = time.time() - t0
    return {
        "radius_km":     R,
        "window_days":   window,
        "shift_pct":     shift_pct,
        "treatment_col": W,
        "n":             result.n,
        "n_clusters":    result.n_clusters,
        "psi":           result.psi,
        "se_iid":        result.se_iid,
        "se_cluster":    result.se_cluster,
        "ci_low":        result.ci_low,
        "ci_high":       result.ci_high,
        "pval":          result.pval,
        "epsilon":       result.epsilon,
        "psi_under_shift": result.notes.get("psi_under_shift"),
        "psi_no_shift":    result.notes.get("psi_no_shift"),
        "mean_H":        result.notes.get("mean_H"),
        "max_H":         result.notes.get("max_H"),
        "elapsed_sec":   elapsed,
    }


def _worker_dose(args: dict) -> pd.DataFrame | None:
    R = args["R"]
    window = args["window"]
    grid = np.array(args["grid"], dtype=float)
    path = Path(PANEL_FMT.format(R=R))
    if not path.exists():
        return None
    panel = cc.load_panel(str(path), radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=window)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=window)
    data = data.copy()
    data["_cluster"] = cluster.values
    trim_pct = float(args.get("trim_pct", 0.01))
    df = tmle.tmle_dose_response(
        df=data, A_col=W, L_cols=confs, Y_col=S, cluster_col="_cluster",
        a_grid=grid, trim_pct=trim_pct,
    )
    df.insert(0, "radius_km", R)
    df.insert(1, "window_days", window)
    return df


def _worker_mediation(args: dict) -> dict | None:
    R = args["R"]
    window = args["window"]
    high_pctl = args["high_pctl"]
    low_pctl  = args["low_pctl"]
    n_iter_boot = args["n_iter_boot"]
    path = Path(PANEL_FMT.format(R=R))
    if not path.exists():
        return None
    panel = cc.load_panel(str(path), radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=window)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=window)
    data = data.copy()
    data["_cluster"] = cluster.values
    a_high = float(np.quantile(data[W], high_pctl))
    a_low  = float(np.quantile(data[W], low_pctl))
    t0 = time.time()
    res = tmle.tmle_mediation(
        df=data, A_col=W, M_col=P, L_cols=confs, Y_col=S, cluster_col="_cluster",
        a_high=a_high, a_low=a_low, n_iter_boot=n_iter_boot,
    )
    elapsed = time.time() - t0
    return {
        "radius_km":     R,
        "window_days":   window,
        "n":             res["n"],
        "n_clusters":    res["n_clusters"],
        "a_high":        a_high,
        "a_low":         a_low,
        "TE":            res["TE"],
        "TE_ci_low":     res["TE_ci"][0],
        "TE_ci_high":    res["TE_ci"][1],
        "NDE":           res["NDE"],
        "NDE_ci_low":    res["NDE_ci"][0],
        "NDE_ci_high":   res["NDE_ci"][1],
        "NIE":           res["NIE"],
        "NIE_ci_low":    res["NIE_ci"][0],
        "NIE_ci_high":   res["NIE_ci"][1],
        "pct_mediated":  res["pct_mediated"],
        "n_iter_boot":   res["n_iter_boot"],
        "elapsed_sec":   elapsed,
    }


# ──────────────────── Driver dispatch ────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("driver", choices=["shift", "dose", "mediation"],
                   help="Which TMLE driver to run")
    p.add_argument("--window", type=int, default=365,
                   help="Lookback window in days. Default 365.")
    p.add_argument("--shift", type=float, default=0.10,
                   help="(shift driver) Shift fraction. Default 0.10.")
    p.add_argument("--grid", type=float, nargs="+",
                   default=[1e4, 1e5, 1e6, 1e7, 1e8],
                   help="(dose driver) Cumulative-volume grid points (BBL).")
    p.add_argument("--high-pctl", type=float, default=0.90,
                   help="(mediation driver) High contrast quantile.")
    p.add_argument("--low-pctl", type=float, default=0.10,
                   help="(mediation driver) Low contrast quantile.")
    p.add_argument("--n-iter-boot", type=int, default=100,
                   help="(mediation driver) Bootstrap iterations. Default 100.")
    p.add_argument("--trim-pct", type=float, default=0.01,
                   help="(shift, dose) Truncate H at (1-trim_pct) percentile. "
                        "Default 0.01. Ignored if --cv-tmle is set.")
    p.add_argument("--cv-tmle", action="store_true",
                   help="Use Cross-Validated TMLE (no H-truncation needed). "
                        "~5× slower but principled positivity handling.")
    p.add_argument("--radii", type=int, nargs="+", default=RADII,
                   help="Radii (km) to analyze.")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel worker processes. "
                        "Default = nproc/3 (or env TMLE_WORKERS).")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path.")
    return p.parse_args()


def default_workers() -> int:
    env = os.environ.get("TMLE_WORKERS")
    if env:
        return int(env)
    nproc = os.cpu_count() or 4
    return max(1, nproc // 3)


def run_parallel(driver: str, jobs: list[dict], n_workers: int,
                 outfile: Path) -> list:
    """Dispatch jobs to a process pool, write results incrementally."""
    log.info("Spawning %d worker processes for %d %s jobs",
             n_workers, len(jobs), driver)
    func = {
        "shift":     _worker_shift,
        "dose":      _worker_dose,
        "mediation": _worker_mediation,
    }[driver]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_job = {pool.submit(func, j): j for j in jobs}
        for fut in as_completed(future_to_job):
            job = future_to_job[fut]
            try:
                res = fut.result()
            except Exception as e:
                log.error("Job R=%dkm failed: %s", job["R"], e)
                continue
            if res is None:
                continue
            if isinstance(res, pd.DataFrame):
                results.append(res)
                pd.concat(results, ignore_index=True).to_csv(outfile, index=False)
            else:
                results.append(res)
                pd.DataFrame(results).sort_values("radius_km").to_csv(outfile, index=False)
            log.info("✓  R=%dkm done (cumulative %d/%d)",
                     job["R"], len(results), len(jobs))
    log.info("All workers complete in %.1fs (%.1f min)",
             time.time() - t0, (time.time() - t0) / 60)
    return results


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.window}d_{timestamp}"
    default_out = {
        "shift":     f"tmle_shift_{suffix}.csv",
        "dose":      f"tmle_dose_response_{suffix}.csv",
        "mediation": f"tmle_mediation_{suffix}.csv",
    }[args.driver]
    outfile = Path(args.output or default_out)
    n_workers = args.workers or default_workers()

    log.info("PARALLEL TMLE — %s driver", args.driver.upper())
    log.info("  window: %d days  radii: %s", args.window, args.radii)
    log.info("  workers: %d  output: %s", n_workers, outfile)
    log.info("  TMLE_N_FOLDS=%d  TMLE_BIG_LIBRARY=%d  TMLE_XGB_N=%d  TMLE_GBM_N=%d",
             tmle.N_FOLDS_CROSSFIT, int(tmle.BIG_LIBRARY),
             tmle.XGB_N_ESTIMATORS, tmle.GBM_N_ESTIMATORS)

    if args.driver == "shift":
        jobs = [{"R": R, "window": args.window, "shift_pct": args.shift,
                 "trim_pct": args.trim_pct, "cv_tmle": args.cv_tmle}
                for R in args.radii]
    elif args.driver == "dose":
        jobs = [{"R": R, "window": args.window, "grid": args.grid,
                 "trim_pct": args.trim_pct}
                for R in args.radii]
    else:  # mediation
        jobs = [{"R": R, "window": args.window,
                 "high_pctl": args.high_pctl, "low_pctl": args.low_pctl,
                 "n_iter_boot": args.n_iter_boot}
                for R in args.radii]

    run_parallel(args.driver, jobs, n_workers, outfile)
    log.info("✅  Wrote %s", outfile)


if __name__ == "__main__":
    main()
