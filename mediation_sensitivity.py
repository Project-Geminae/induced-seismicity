#!/usr/bin/env python3
"""
mediation_sensitivity.py
────────────────────────
Imai et al. (2010) sensitivity analysis for mediational TMLE estimates.

Assesses robustness of NDE/NIE decomposition to unmeasured confounding of the
mediator-outcome relationship. The key sensitivity parameter is rho, the
correlation between the error terms in the mediator model and the outcome model.
Under sequential ignorability (our identifying assumption), rho = 0. If rho != 0,
the NDE and NIE are biased.

The adjustment formula (simplified from Imai, Keele & Yamamoto 2010, JRSS-B):

    NDE_adj(rho) = NDE - rho * sigma_M * sigma_Y
    NIE_adj(rho) = NIE + rho * sigma_M * sigma_Y

where sigma_M and sigma_Y are the residual standard deviations from the mediator
and outcome models respectively. This is a first-order sensitivity analysis:
it asks "how much unmeasured mediator-outcome confounding would be needed to
nullify the direct/indirect effect?"

Key outputs:
  - rho_NDE_flip: the rho at which NDE changes sign
  - rho_NIE_nonsig: the rho at which NIE becomes non-significant (CI crosses 0)
  - Full table of adjusted NDE/NIE at each rho value

References:
  Imai, K., Keele, L., & Yamamoto, T. (2010). Identification, Inference and
  Sensitivity Analysis for Causal Mediation Effects. Statistical Science,
  25(1), 51-71.

Usage:
    python mediation_sensitivity.py --radius 7
    python mediation_sensitivity.py --radius 7 --ci-method influence
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import causal_core as cc
import tmle_core as tmle


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PANEL_FMT       = "panel_with_faults_{R}km.csv"
DEFAULT_WINDOW  = 365
DEFAULT_HIGH_PCTL = 0.90
DEFAULT_LOW_PCTL  = 0.10


def compute_residual_sds(
    df: pd.DataFrame,
    A_col: str,
    M_col: str,
    L_cols: list[str],
    Y_col: str,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute residual standard deviations for mediator and outcome models.

    Fits XGBoost models for E[M | A, L] and E[Y | A, M, L], then returns
    the residual SDs. These are the sigma_M and sigma_Y in the Imai formula.
    """
    import xgboost as xgb

    A = df[A_col].to_numpy(dtype=float)
    M = df[M_col].to_numpy(dtype=float)
    L = df[L_cols].to_numpy(dtype=float)
    Y = df[Y_col].to_numpy(dtype=float)

    # Mediator model: E[M | A, L]
    M_model = xgb.XGBRegressor(
        n_estimators=tmle.XGB_N_ESTIMATORS, max_depth=4, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=seed,
    )
    M_model.fit(np.column_stack([A, L]), M)
    M_hat = M_model.predict(np.column_stack([A, L]))
    sigma_M = float(np.std(M - M_hat, ddof=1))

    # Outcome model: E[Y | A, M, L]
    AML = np.column_stack([A, M, L])
    Q = tmle.HurdleSuperLearner(random_state=seed)
    Q.fit(AML, Y)
    Y_hat = Q.predict(AML)
    sigma_Y = float(np.std(Y - Y_hat, ddof=1))

    return sigma_M, sigma_Y


def sensitivity_analysis(
    NDE: float,
    NIE: float,
    TE: float,
    NDE_ci: tuple[float, float],
    NIE_ci: tuple[float, float],
    sigma_M: float,
    sigma_Y: float,
    rho_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run Imai et al. (2010) sensitivity analysis over a grid of rho values.

    Parameters
    ----------
    NDE, NIE, TE : point estimates from TMLE mediation
    NDE_ci, NIE_ci : 95% CI tuples (low, high)
    sigma_M, sigma_Y : residual SDs from mediator/outcome models
    rho_grid : correlation values to sweep (default: 0 to 0.5 in 0.05 steps)

    Returns
    -------
    DataFrame with columns: rho, NDE_adj, NIE_adj, TE (constant),
    NDE_adj_ci_low, NDE_adj_ci_high, NIE_adj_ci_low, NIE_adj_ci_high,
    NDE_sign_flipped, NIE_nonsig
    """
    if rho_grid is None:
        rho_grid = np.arange(0.0, 0.505, 0.05)

    bias_unit = sigma_M * sigma_Y
    NDE_se = (NDE_ci[1] - NDE_ci[0]) / (2 * 1.96)  # back out SE from CI
    NIE_se = (NIE_ci[1] - NIE_ci[0]) / (2 * 1.96)

    rows = []
    for rho in rho_grid:
        bias = rho * bias_unit
        nde_adj = NDE - bias
        nie_adj = NIE + bias

        # Adjusted CIs (shift by same bias; SE unchanged)
        nde_adj_ci_low  = nde_adj - 1.96 * NDE_se
        nde_adj_ci_high = nde_adj + 1.96 * NDE_se
        nie_adj_ci_low  = nie_adj - 1.96 * NIE_se
        nie_adj_ci_high = nie_adj + 1.96 * NIE_se

        # Check: has NDE flipped sign relative to rho=0?
        nde_sign_flipped = (np.sign(nde_adj) != np.sign(NDE)) if abs(NDE) > 1e-15 else False
        # Check: is NIE non-significant (CI contains 0)?
        nie_nonsig = (nie_adj_ci_low <= 0 <= nie_adj_ci_high)

        rows.append({
            "rho":              float(rho),
            "bias":             float(bias),
            "NDE_adj":          float(nde_adj),
            "NIE_adj":          float(nie_adj),
            "TE":               float(TE),
            "NDE_adj_ci_low":   float(nde_adj_ci_low),
            "NDE_adj_ci_high":  float(nde_adj_ci_high),
            "NIE_adj_ci_low":   float(nie_adj_ci_low),
            "NIE_adj_ci_high":  float(nie_adj_ci_high),
            "NDE_sign_flipped": bool(nde_sign_flipped),
            "NIE_nonsig":       bool(nie_nonsig),
        })

    return pd.DataFrame(rows)


def find_critical_rho(
    NDE: float,
    NIE: float,
    NDE_ci: tuple[float, float],
    NIE_ci: tuple[float, float],
    sigma_M: float,
    sigma_Y: float,
) -> dict:
    """Find the critical rho values analytically.

    rho_NDE_flip: rho at which NDE_adj = 0 => rho = NDE / (sigma_M * sigma_Y)
    rho_NIE_nonsig: rho at which NIE_adj CI crosses 0.
    """
    bias_unit = sigma_M * sigma_Y
    NIE_se = (NIE_ci[1] - NIE_ci[0]) / (2 * 1.96)

    # NDE flips sign when: NDE - rho * bias_unit = 0
    rho_NDE_flip = NDE / bias_unit if bias_unit > 1e-15 else float("inf")

    # NIE becomes non-significant when: |NIE + rho * bias_unit| <= 1.96 * SE
    # If NIE > 0: becomes nonsig when NIE + rho * bias_unit < 1.96 * SE
    #   (rho pushes NIE further positive, so it can only become nonsig if NIE
    #    was already borderline, OR if we allow negative rho)
    # If NIE < 0: NIE_adj = NIE + rho * bias_unit, crosses 0 at rho = -NIE / bias_unit
    #   Becomes nonsig when lower CI crosses 0, i.e., NIE_adj - 1.96*SE = 0
    #   => rho = (1.96*SE - NIE) / bias_unit
    # General: NIE_adj is nonsig when 0 is in [NIE_adj - 1.96*SE, NIE_adj + 1.96*SE]
    # The first rho (positive) at which this happens:
    if NIE > 0:
        # NIE_adj increases with rho (positive bias), so it won't become nonsig
        # via positive rho. Report inf (or the negative-rho value).
        rho_NIE_nonsig = float("inf")
    elif NIE < 0:
        # NIE_adj = NIE + rho * bias_unit. It crosses zero at rho = -NIE/bias_unit.
        # Becomes nonsig (CI includes 0) when NIE_adj + 1.96*SE >= 0
        # => NIE + rho * bias_unit + 1.96*SE >= 0
        # => rho >= (-NIE - 1.96*SE) / bias_unit
        rho_val = (-NIE - 1.96 * NIE_se) / bias_unit if bias_unit > 1e-15 else float("inf")
        rho_NIE_nonsig = max(0.0, rho_val)
    else:
        rho_NIE_nonsig = 0.0  # already zero

    return {
        "rho_NDE_flip":   float(rho_NDE_flip),
        "rho_NIE_nonsig": float(rho_NIE_nonsig),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radius", type=int, default=7,
                   help="Radius in km. Default: 7.")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"Lookback window in days. Default {DEFAULT_WINDOW}.")
    p.add_argument("--high-pctl", type=float, default=DEFAULT_HIGH_PCTL,
                   help="High contrast quantile for cumulative volume.")
    p.add_argument("--low-pctl", type=float, default=DEFAULT_LOW_PCTL,
                   help="Low contrast quantile for cumulative volume.")
    p.add_argument("--ci-method", type=str, default="influence",
                   choices=["bootstrap", "influence"],
                   help="CI method for mediation TMLE. Default: influence (fast).")
    p.add_argument("--n-boot", type=int, default=30,
                   help="Bootstrap iterations (only used if ci-method=bootstrap).")
    p.add_argument("--output", type=str, default="mediation_sensitivity.csv",
                   help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    R = args.radius

    path = Path(PANEL_FMT.format(R=R))
    if not path.exists():
        log.error("Panel file %s not found. Run build_well_day_panel.py first.", path)
        sys.exit(1)

    log.info("MEDIATION SENSITIVITY ANALYSIS (Imai et al. 2010)")
    log.info("  radius: %dkm  window: %dd  ci_method: %s", R, args.window, args.ci_method)

    # Load and prepare data
    log.info("Loading panel %s ...", path)
    panel = cc.load_panel(str(path), radius_km=R)
    agg = cc.aggregate_panel_to_event_level(panel, R, window_days=args.window)
    data, W, P, S, confs, cluster = cc.build_design_matrix(agg, R, window_days=args.window)
    data = data.copy()
    data["_cluster"] = cluster.values

    a_high = float(np.quantile(data[W], args.high_pctl))
    a_low  = float(np.quantile(data[W], args.low_pctl))
    log.info("Contrast: a_high=%.2e (p%d), a_low=%.2e (p%d)",
             a_high, int(args.high_pctl * 100), a_low, int(args.low_pctl * 100))

    # Step 1: Fit mediation TMLE to get NDE, NIE, TE + CIs
    log.info("Fitting mediation TMLE (ci_method=%s) ...", args.ci_method)
    t0 = time.time()
    med_result = tmle.tmle_mediation(
        df=data, A_col=W, M_col=P, L_cols=confs, Y_col=S,
        cluster_col="_cluster", a_high=a_high, a_low=a_low,
        ci_method=args.ci_method, n_iter_boot=args.n_boot,
    )
    elapsed_tmle = time.time() - t0
    log.info("  TE=%+.3e  NDE=%+.3e  NIE=%+.3e  %%med=%.1f%%  (%.0fs)",
             med_result["TE"], med_result["NDE"], med_result["NIE"],
             med_result["pct_mediated"], elapsed_tmle)

    # Step 2: Compute residual SDs for the sensitivity formula
    log.info("Computing residual SDs (sigma_M, sigma_Y) ...")
    sigma_M, sigma_Y = compute_residual_sds(data, W, P, confs, S)
    log.info("  sigma_M=%.4f  sigma_Y=%.4f  sigma_M*sigma_Y=%.4e",
             sigma_M, sigma_Y, sigma_M * sigma_Y)

    # Step 3: Run sensitivity analysis over rho grid
    log.info("Running sensitivity sweep rho = 0.00 to 0.50 ...")
    sens_df = sensitivity_analysis(
        NDE=med_result["NDE"],
        NIE=med_result["NIE"],
        TE=med_result["TE"],
        NDE_ci=med_result["NDE_ci"],
        NIE_ci=med_result["NIE_ci"],
        sigma_M=sigma_M,
        sigma_Y=sigma_Y,
    )

    # Step 4: Find critical rho values
    critical = find_critical_rho(
        NDE=med_result["NDE"],
        NIE=med_result["NIE"],
        NDE_ci=med_result["NDE_ci"],
        NIE_ci=med_result["NIE_ci"],
        sigma_M=sigma_M,
        sigma_Y=sigma_Y,
    )

    # Add metadata columns
    sens_df["radius_km"] = R
    sens_df["window_days"] = args.window
    sens_df["sigma_M"] = sigma_M
    sens_df["sigma_Y"] = sigma_Y
    sens_df["rho_NDE_flip"] = critical["rho_NDE_flip"]
    sens_df["rho_NIE_nonsig"] = critical["rho_NIE_nonsig"]

    # Save
    outpath = Path(args.output)
    sens_df.to_csv(outpath, index=False)
    log.info("Wrote %s (%d rows)", outpath, len(sens_df))

    # Print summary
    print()
    print("=" * 80)
    print("MEDIATION SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"  Radius: {R} km   Window: {args.window} days")
    print(f"  TE  = {med_result['TE']:+.4e}")
    print(f"  NDE = {med_result['NDE']:+.4e}  CI: [{med_result['NDE_ci'][0]:+.4e}, {med_result['NDE_ci'][1]:+.4e}]")
    print(f"  NIE = {med_result['NIE']:+.4e}  CI: [{med_result['NIE_ci'][0]:+.4e}, {med_result['NIE_ci'][1]:+.4e}]")
    print(f"  %mediated = {med_result['pct_mediated']:.1f}%")
    print()
    print(f"  sigma_M = {sigma_M:.4f}   sigma_Y = {sigma_Y:.4f}")
    print(f"  Bias unit (sigma_M * sigma_Y) = {sigma_M * sigma_Y:.4e}")
    print()
    print(f"  rho at which NDE flips sign:        {critical['rho_NDE_flip']:.4f}")
    print(f"  rho at which NIE becomes non-signif: {critical['rho_NIE_nonsig']:.4f}")
    print()
    print("Sensitivity table:")
    print("-" * 80)
    print(f"{'rho':>6} {'NDE_adj':>12} {'NIE_adj':>12} {'NDE_flipped':>12} {'NIE_nonsig':>11}")
    print("-" * 80)
    for _, row in sens_df.iterrows():
        print(f"{row['rho']:>6.2f} {row['NDE_adj']:>+12.4e} {row['NIE_adj']:>+12.4e} "
              f"{'YES' if row['NDE_sign_flipped'] else 'no':>12} "
              f"{'YES' if row['NIE_nonsig'] else 'no':>11}")
    print("=" * 80)


if __name__ == "__main__":
    main()
