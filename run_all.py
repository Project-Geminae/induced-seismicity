#!/usr/bin/env python3
"""
run_all.py
──────────
Orchestrate the full induced-seismicity causal pipeline end-to-end.

Differences from the old run_all.py
-----------------------------------
1. **No interactive prompts**. The old version blocked on `input("Continue?
   (y/n): ")` whenever a step failed, which made nohup / cron / CI runs hang
   forever. The new version exits non-zero on the first failure unless you
   pass `--continue-on-error`.
2. **New step list** matching the cleaned pipeline:
       0. swd_data_import.py             (extended schema with depth/formation)
       1. seismic_data_import.py         (with quality filtering)
       2. build_well_day_panel.py        (NEW — replaces filter_active_wells)
       3. spatiotemporal_join.py         (NEW — replaces merge_seismic_swd)
       4. add_geoscience_to_panel.py     (renamed, panel-aware)
       5. dowhy_simple_all.py            (well-day, no bootstrap)
       6. dowhy_simple_all_aggregate.py  (event-level, no bootstrap)
       7. dowhy_ci.py                    (well-day with cluster-bootstrap CI)
       8. dowhy_ci_aggregated.py         (event-level with cluster-bootstrap CI)
       9. causal_poe_curves.py           (PoE figures from the new panel)
      10. killer_visualizations.py       (CSV-driven dashboards)
      11. induced_seismicity_scaling_plots.py (CSV-driven scaling plots)
      12. measure_balrog.py              (catalog-driven magnitude histogram)
3. **Each step is timed** and the elapsed times are dumped at the end so you
   can see where the runtime is going without grepping the log.

Usage
-----
    python run_all.py                       # all steps, exit on first failure
    python run_all.py --continue-on-error   # tolerate failures, run everything
    python run_all.py --skip 9 10 11        # skip step indices 9, 10, 11
    python run_all.py --only 0 1 2 3 4      # ingest + panel only
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ──────────────────── Pipeline definition ────────────────────────
STEPS: list[tuple[str, str]] = [
    ("swd_data_import.py",                "Filter SWD records to Midland Basin (extended schema)"),
    ("seismic_data_import.py",            "Filter TexNet catalog with quality cuts"),
    ("build_well_day_panel.py",           "Build (well, day) panel + rolling features + BHP"),
    ("spatiotemporal_join.py",            "Spatiotemporal join: events → panel for 1..20 km"),
    ("add_geoscience_to_panel.py",        "Per-well fault distance + per-radius segment counts"),
    ("dowhy_simple_all.py",               "Well-day causal analysis (no bootstrap)"),
    ("dowhy_simple_all_aggregate.py",     "Event-level (cluster-day) causal analysis (no bootstrap)"),
    ("dowhy_ci.py",                       "Well-day analysis with cluster-bootstrap CIs + refutations"),
    ("dowhy_ci_aggregated.py",            "Event-level analysis with cluster-bootstrap CIs + refutations"),
    ("tmle_shift_analysis.py",            "TMLE: stochastic shift intervention (+10% cumulative volume)"),
    ("tmle_dose_response.py",             "TMLE: causal dose-response curve E[Y_a]"),
    ("tmle_mediation_analysis.py",        "TMLE: NDE/NIE mediation decomposition (p90 vs p10 contrast)"),
    ("causal_poe_curves.py",              "PoE curves and per-radius figures"),
    ("killer_visualizations.py",          "Individual vs aggregate comparison dashboards"),
    ("induced_seismicity_scaling_plots.py", "Effect-vs-distance scaling plots"),
    ("measure_balrog.py",                 "Animated magnitude histogram"),
]


# ──────────────────── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("pipeline_run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("run_all")


# Scripts that support --workers N for per-radius parallelism
_PARALLELIZABLE = {
    "spatiotemporal_join.py",
    "add_geoscience_to_panel.py",
    "build_attribution_q.py",
}


def run_one(script: str, description: str, continue_on_error: bool,
            parallel_radii: int = 1) -> tuple[bool, float]:
    """Run a single pipeline step. Returns (success, elapsed_seconds)."""
    if not Path(script).exists():
        log.error("✗  %s — script not found", script)
        if not continue_on_error:
            sys.exit(2)
        return False, 0.0

    log.info("=" * 70)
    log.info("▶  %s", script)
    log.info("   %s", description)
    log.info("=" * 70)

    cmd = [sys.executable, "-u", script]
    if parallel_radii > 1 and script in _PARALLELIZABLE:
        cmd.extend(["--workers", str(parallel_radii)])
        log.info("   (parallel: --workers %d)", parallel_radii)

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log.error("✗  %s — exit %d after %.1fs", script, proc.returncode, elapsed)
        if not continue_on_error:
            log.error("Aborting (use --continue-on-error to keep going)")
            sys.exit(proc.returncode)
        return False, elapsed

    log.info("✓  %s — done in %.1fs", script, elapsed)
    return True, elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the induced-seismicity causal pipeline end-to-end."
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Don't abort on a failed step; report and move on",
    )
    p.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=[],
        help="Step indices to skip (see --list)",
    )
    p.add_argument(
        "--only",
        type=int,
        nargs="*",
        default=None,
        help="Only run these step indices (see --list)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the step list and exit",
    )
    p.add_argument(
        "--parallel-radii", type=int, default=1,
        help="Parallel workers for per-radius steps (spatiotemporal_join, "
             "add_geoscience_to_panel, build_attribution_q). On minitim "
             "use --parallel-radii 16. Default 1 (Mac).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("Pipeline steps:")
        for i, (script, desc) in enumerate(STEPS):
            print(f"  {i:>2}. {script:<38s}  {desc}")
        sys.exit(0)

    log.info(
        "INDUCED-SEISMICITY PIPELINE — %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    selected = list(range(len(STEPS)))
    if args.only is not None:
        selected = [i for i in args.only if 0 <= i < len(STEPS)]
    selected = [i for i in selected if i not in set(args.skip)]

    log.info("Running steps: %s", selected)

    pipeline_t0 = time.time()
    timings: list[tuple[int, str, bool, float]] = []
    for i in selected:
        script, description = STEPS[i]
        ok, elapsed = run_one(script, description, args.continue_on_error,
                              parallel_radii=args.parallel_radii)
        timings.append((i, script, ok, elapsed))

    pipeline_elapsed = time.time() - pipeline_t0

    # ── Final timing summary ────────────────────────────────────
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE — total %.1fs (%.1f min)",
             pipeline_elapsed, pipeline_elapsed / 60)
    log.info("=" * 70)
    log.info("%-3s %-40s %-6s %s", "#", "Script", "Status", "Elapsed")
    for i, script, ok, elapsed in timings:
        status = "OK" if ok else "FAIL"
        log.info("%-3d %-40s %-6s %.1fs", i, script, status, elapsed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.error("Interrupted by user")
        sys.exit(1)
