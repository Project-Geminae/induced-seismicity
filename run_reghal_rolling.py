#!/usr/bin/env python3
"""Rolling-window regHAL-TMLE driver to test for time-trend attenuation
in the basin-scale Estimator A. Runs the same regHAL-TMLE pipeline as
run_reghal_tmle.py at a fixed radius, but partitions the panel into
rolling windows of `--window-days` width stepping by `--step-days`.

For each window:
  - Filter panel rows to that window's date range
  - Aggregate to event level
  - Subsample to --max-n clusters
  - Fit HAL + targeted Δ-method
  - Record (window_start, window_end, n_events, n_clusters, psi, SE, p)

If RRC volume controls (implemented 2024) really attenuated basin-scale
coupling, the rolling estimate should drop noticeably between
windows centered on 2023 and 2026. If the signal is stable, the
weakening from April-vintage +7.65e-3 to May-vintage +1.03e-3 is
finite-sample noise from a different basis selection, not a real
attenuation.

Usage:
    .venv/bin/python run_reghal_rolling.py --radius 7 \\
        --window-days 540 --step-days 90 --max-n 40000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import causal_core as cc
import reghal_tmle as rht


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=int, default=7)
    p.add_argument("--window-days", type=int, default=540,
                   help="Width of each rolling window (default ~18 months).")
    p.add_argument("--step-days", type=int, default=90,
                   help="Step between successive window starts.")
    p.add_argument("--start", type=str, default="2020-01-01",
                   help="Earliest window-end date to compute.")
    p.add_argument("--max-n", type=int, default=40000,
                   help="Cluster-aware subsample size per window.")
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-degree", type=int, default=2)
    p.add_argument("--num-knots", type=str, default="25,10")
    p.add_argument("--ridge-eta", type=float, default=1e-4)
    p.add_argument("--shift-pct", type=float, default=0.10)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    R = args.radius
    num_knots = tuple(int(x) for x in args.num_knots.split(","))

    print(f"=== regHAL-TMLE rolling-window driver at R={R} km ===")
    print(f"  window={args.window_days}d  step={args.step_days}d  "
          f"start={args.start}  max_n={args.max_n}")

    panel = cc.load_panel(f"panel_with_faults_{R}km.csv", radius_km=R)
    panel["Date of Injection"] = pd.to_datetime(panel["Date of Injection"])
    print(f"  full panel: {len(panel):,} rows, "
          f"{panel['Date of Injection'].min().date()} -> "
          f"{panel['Date of Injection'].max().date()}")

    panel_min = panel["Date of Injection"].min()
    panel_max = panel["Date of Injection"].max()
    window_d = pd.Timedelta(days=args.window_days)
    step_d = pd.Timedelta(days=args.step_days)

    # Window-end dates: start at max(start, panel_min + window) and step.
    earliest_end = max(pd.Timestamp(args.start),
                       panel_min + window_d)
    ends = []
    e = earliest_end
    while e <= panel_max:
        ends.append(e)
        e = e + step_d

    print(f"  → {len(ends)} windows from {ends[0].date()} to {ends[-1].date()}")

    rows = []
    out = Path(args.out or f"reghal_rolling_R{R}km.csv")

    for i, end in enumerate(ends):
        start = end - window_d
        win = panel[(panel["Date of Injection"] >= start) &
                     (panel["Date of Injection"] <= end)]
        if len(win) < 1000:
            print(f"  [{i+1}/{len(ends)}] {start.date()} -> {end.date()}: "
                  f"only {len(win)} rows, skipping")
            continue

        try:
            agg = cc.aggregate_panel_to_event_level(
                win, R, window_days=365)
            data, W, P, S, confs, cluster = cc.build_design_matrix(
                agg, R, window_days=365)
            data = data.copy()
            data["_cluster"] = cluster.values
        except Exception as ex:
            print(f"  [{i+1}/{len(ends)}] {start.date()} -> {end.date()}: "
                  f"aggregation failed: {ex}")
            continue

        n_full = len(data)
        n_pos = int(data[S].gt(0).sum())
        if n_pos < 20:
            print(f"  [{i+1}/{len(ends)}] {start.date()} -> {end.date()}: "
                  f"only {n_pos} positive events, skipping")
            continue

        # Subsample clusters for HAL tractability.
        if args.max_n and len(data) > args.max_n:
            rng = np.random.default_rng(args.seed)
            clusters_all = data["_cluster"].values
            uniq = np.unique(clusters_all)
            rng.shuffle(uniq)
            kept, n_so_far = [], 0
            for c in uniq:
                nc = int((clusters_all == c).sum())
                if n_so_far + nc > args.max_n and n_so_far > 0:
                    break
                kept.append(c)
                n_so_far += nc
            mask = np.isin(clusters_all, kept)
            data = data.loc[mask].reset_index(drop=True)

        t0 = time.time()
        try:
            result = rht.reghal_tmle_shift(
                df=data, A_col=W, L_cols=confs, Y_col=S,
                cluster_col="_cluster",
                shift_pct=args.shift_pct,
                ridge_eta=args.ridge_eta,
                max_iter=args.max_iter,
                hal_kwargs={
                    "max_degree":        args.max_degree,
                    "num_knots":         num_knots,
                    "smoothness_orders": 1,
                },
                verbose=False,
            )
        except Exception as ex:
            print(f"  [{i+1}/{len(ends)}] {start.date()} -> {end.date()}: "
                  f"fit failed: {ex}")
            continue
        elapsed = time.time() - t0

        z = result.psi_targeted / result.se_if if result.se_if else np.nan
        rows.append({
            "window_start":  start.date(),
            "window_end":    end.date(),
            "n_full":        n_full,
            "n_positives":   n_pos,
            "n_subsample":   result.n,
            "n_clusters":    result.n_clusters,
            "n_basis":       result.n_basis,
            "psi_plugin":    result.psi_plugin,
            "psi_targeted":  result.psi_targeted,
            "se_cluster":    result.se_if,
            "z":             float(z),
            "pval":          result.pval,
            "converged":     result.converged,
            "elapsed_sec":   elapsed,
        })

        # Incremental write so partial runs are useful.
        pd.DataFrame(rows).to_csv(out, index=False)

        print(f"  [{i+1}/{len(ends)}] {start.date()} -> {end.date()}  "
              f"n={result.n}  n_pos={n_pos}  "
              f"ψ={result.psi_targeted:+.3e}  z={z:+.2f}  "
              f"p={result.pval:.3e}  ({elapsed:.0f}s)")

    print(f"\nWrote {out} with {len(rows)} windows")


if __name__ == "__main__":
    main()
