#!/usr/bin/env python3  # Tell the shell to execute this file with the system's python3
"""
Batch Flexible-Header Causal Analysis **with Mediation**
────────────────────────────────────────────────────────
* Processes multiple event_well_links_with_faults_<R>km.csv files
* Prints a mapping-table **and** a missing-value summary for each file
* Retains only rows containing both treatment (W) **and** outcome (S)
* Median-fills any remaining NA values in covariates (edit to change)
* Carries out a formal mediation analysis for the W → P → S pathway
* Uses "timed banners" so the console never looks frozen
* Saves results for each radius to a summary CSV file
"""

# 0 ▸ IMPORTS ──────────────────────────────────────────────────
import re, sys, time, logging, warnings, os  # Std-lib utilities
from contextlib import contextmanager  # For the timed-banner helper
from pathlib import Path  # For file path handling

import numpy as np  # Fast numerical arrays
import pandas as pd  # Tabular data handling
import networkx as nx  # DAG representation
from tqdm import tqdm  # Nicely formatted progress bars

from dowhy import CausalModel  # Causal-inference engine
import statsmodels.api as sm  # Classical stats (for p-values)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Silence noisy warnings to keep the console readable
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("dowhy").setLevel(logging.ERROR)

# List of input CSV files to process (one per radius)
CSV_FILES = [
    "event_well_links_with_faults_1km.csv",
    "event_well_links_with_faults_2km.csv",
    "event_well_links_with_faults_3km.csv",
    "event_well_links_with_faults_4km.csv",
    "event_well_links_with_faults_5km.csv",
    "event_well_links_with_faults_6km.csv",
    "event_well_links_with_faults_7km.csv",
    "event_well_links_with_faults_8km.csv",
    "event_well_links_with_faults_9km.csv",
    "event_well_links_with_faults_10km.csv",
    "event_well_links_with_faults_11km.csv",
    "event_well_links_with_faults_12km.csv",
    "event_well_links_with_faults_13km.csv",
    "event_well_links_with_faults_14km.csv",
    "event_well_links_with_faults_15km.csv",
    "event_well_links_with_faults_16km.csv",
    "event_well_links_with_faults_17km.csv",
    "event_well_links_with_faults_18km.csv",
    "event_well_links_with_faults_19km.csv",
    "event_well_links_with_faults_20km.csv"
]

# Summary results file
RESULTS_CSV = "causal_analysis_results_by_radius.csv"

# 1 ▸ COLUMN-NAME FRAGMENTS  (edit if your headers differ) ─────
COL_FRAGS = {
    "W": ["volume", "bbl"],  # Treatment  – injection volume
    "P": ["pressure", "psig"],  # Mediator   – average injection pressure
    "S": ["local", "mag"],  # Outcome    – earthquake magnitude
    "G1": ["nearest", "fault", "dist"],  # Confounder 1
    "G2": ["fault", "segment"],  # Confounder 2
}


# helper – timed banner ---------------------------------------
@contextmanager
def step(title: str):
    """Context-manager that times a code block and prints an animated banner."""
    print(f"\n⏳  {title} …", end="", flush=True)  # Start banner
    t0 = time.time()  # Start timer
    yield  # Run the wrapped code
    print(f" done in {time.time() - t0:.1f}s")  # End banner with elapsed time


def safe(col: str) -> str:
    """Convert a column name to a strict Python identifier."""
    return re.sub(r"\W", "_", col)


def find_column(frags, columns):
    """Return the first column that contains **all** substrings in *frags*."""
    for c in columns:
        if all(f.lower() in c.lower() for f in frags):
            return c
    return None


def extract_radius(filename: str) -> int:
    """Extract the radius value (in km) from the filename."""
    match = re.search(r'(\d+)km', filename)
    if match:
        return int(match.group(1))
    return 0  # Default fallback


def process_file(csv_file: str) -> dict:
    """Process a single CSV file and return the results as a dictionary."""
    radius = extract_radius(csv_file)
    print(f"\n{'=' * 80}\nPROCESSING: {csv_file} (radius: {radius} km)\n{'=' * 80}")

    # 2 ▸ LOAD CSV & AUTO-MAP  ────────────────────────────────────
    with step(f"Loading CSV ({csv_file})"):
        if not Path(csv_file).exists():
            print(f"❌ File not found: {csv_file} - skipping")
            return None
        df_raw = pd.read_csv(csv_file, low_memory=False)  # Read entire file

    print("\n🔎  Auto-detecting required columns:")
    found = {}  # Map role → column name
    for k, frags in COL_FRAGS.items():  # Loop treatment/mediator/outcome/controls
        col = find_column(frags, df_raw.columns)  # Try to auto-find by substrings
        print(f"   {k:>2}  ←  {col if col else '❌ not found'}")  # Show result
        if col:
            found[k] = col  # Store successful matches
    missing = [k for k in COL_FRAGS if k not in found]  # Which roles are still missing?
    if missing:  # Abort if any crucial roles unidentified
        print(f"\n❌  Column(s) not found: {', '.join(missing)} — skipping file.")
        return None

    # 3 ▸ MISSING-VALUE SUMMARY & CLEANING  -----------------------
    print("\n🕵️  Missing-value summary:")
    na = df_raw[found.values()].isna().sum().sort_values(ascending=False)  # Count NA per col
    for c, n in na.items():  # Pretty print summary
        print(f"   {c:<35} : {n:,} NA ({n / len(df_raw):.1%})")

    # Rename relevant columns to safe identifiers (W, P, S, G1, G2)
    rename_map = {found[k]: safe(k) for k in found}
    df = df_raw[list(found.values())].rename(columns=rename_map)

    # Keep only rows with non-NA treatment **and** outcome
    essential = [safe("W"), safe("S")]
    rows_before = len(df)
    df = df.dropna(subset=essential)
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    print(f"Dropped {rows_dropped:,} rows missing treatment or outcome ({rows_dropped / rows_before:.1%})")

    # Median-fill missing values in covariates
    covariates = [safe(c) for c in ["P", "G1", "G2"] if safe(c) in df.columns]
    df[covariates] = df[covariates].fillna(df[covariates].median(numeric_only=True))

    # Abort if nothing left
    if df.empty:
        print("\n❌  No rows retain both treatment and outcome — skipping file.")
        return None
    W, P, S, G1, G2 = map(safe, ["W", "P", "S", "G1", "G2"])  # Shorthand vars
    print("✅  Rows after cleaning:", len(df))

    # 4 ▸ BUILD DAG ------------------------------------------------
    G = nx.DiGraph(
        [
            (G1, W), (G1, P), (G1, S),  # Confounder paths from G1
            (G2, W), (G2, P), (G2, S),  # Confounder paths from G2
            (W, P),  # Treatment → Mediator
            (P, S),  # Mediator  → Outcome
            (W, S),  # Direct Treatment → Outcome
        ]
    )

    # 5 ▸ BUILD MODEL & ESTIMATE EFFECT  --------------------------
    with step("Building DoWhy model"):
        model = CausalModel(df, treatment=W, outcome=S, graph=G)  # Initialise
    estimand = model.identify_effect()  # Identify causal estimand

    # Extract which variables must be controlled (back-door adjustment set)
    backdoor_vars = sorted({v for lst in estimand.backdoor_variables.values() for v in lst})
    print("📝  Back-door controls:", ", ".join(backdoor_vars) if backdoor_vars else "None")

    # Estimate **total** causal effect of W on S using linear regression
    with step("Estimating total causal effect"):
        est = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )

    # Compute two-sided p-value for W coefficient via statsmodels
    X_sm = sm.add_constant(df[[W] + backdoor_vars])  # Add intercept
    pval = sm.OLS(df[S], X_sm).fit().pvalues[W]  # Fit model & pull p-value

    print("\n📈  TOTAL causal effect (ΔMagnitude per extra BBL)")
    print(f"    DoWhy point estimate : {est.value:+.4g}")
    print(f"    Statsmodels p-value  : {pval:.2e}")

    # 5.5 ▸ MEDIATION ANALYSIS (W → P → S) -------------------------
    print("\n🔄  Running mediation analysis...")

    # Step 1: path a (W → P)
    model_a = sm.OLS(df[P], sm.add_constant(df[[W] + backdoor_vars])).fit()
    a_coef, a_pval = model_a.params[W], model_a.pvalues[W]

    # Step 2: path b (P → S | W)
    model_b = sm.OLS(df[S], sm.add_constant(df[[P, W] + backdoor_vars])).fit()
    b_coef, b_pval = model_b.params[P], model_b.pvalues[P]

    # Step 3: path c (total W → S)
    model_c = sm.OLS(df[S], sm.add_constant(df[[W] + backdoor_vars])).fit()
    c_coef, c_pval = model_c.params[W], model_c.pvalues[W]

    # Step 4: path c′ (direct W → S without mediator)
    c_prime_coef, c_prime_pval = model_b.params[W], model_b.pvalues[W]

    # Indirect (mediated) effect = a * b
    indirect_effect = a_coef * b_coef
    proportion_mediated = (indirect_effect / c_coef) * 100 if c_coef != 0 else 0

    # Pretty-print mediation table
    print(f"Path a (W → P)  : {a_coef:+.4g} (p={a_pval:.3g})")
    print(f"Path b (P → S)  : {b_coef:+.4g} (p={b_pval:.3g})")
    print(f"Total effect    : {c_coef:+.4g} (p={c_pval:.3g})")
    print(f"Direct effect   : {c_prime_coef:+.4g} (p={c_prime_pval:.3g})")
    print(f"Indirect effect : {indirect_effect:+.4g}")
    print(f"Proportion mediated: {proportion_mediated:.1f}%")

    # 6 ▸ QUICK REFUTATIONS ---------------------------------------
    with step("Running refutation checks"):
        placebo = model.refute_estimate(  # Shuffle treatment to break causality
            estimand, est, method_name="placebo_treatment_refuter",
            placebo_type="permute")
        subset = model.refute_estimate(  # Re-estimate on a random 80 % subset
            estimand, est, method_name="data_subset_refuter",
            subset_fraction=0.8)

    print("\n🔍  Refutation checks (effect should stay similar)")
    print(f"    Placebo : {placebo.new_effect:+.4g}")
    print(f"    Subset  : {subset.new_effect:+.4g}")

    # 7 ▸ PREDICTIVE SANITY CHECK  --------------------------------
    with step("Training predictive sanity-check model"):
        X = df[[W, P, G1, G2]].copy()  # Feature matrix for a quick prediction test
        X[W] = np.log1p(X[W])  # Mild log-transform of volumes
        y = df[S]  # Target = magnitude
        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True)
        pred = LinearRegression().fit(x_tr, y_tr).predict(x_te)

    rmse = np.sqrt(mean_squared_error(y_te, pred))  # Regression RMSE
    r2 = r2_score(y_te, pred)  # R-squared
    print("\n⚙️   Prediction metrics (non-causal sanity check)")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R²   : {r2:.4f}")

    # 8 ▸ OPTIONAL DAG PNG  ---------------------------------------
    dag_filename = f"dag_radius_{radius}km.png"
    try:
        from graphviz import Source  # Try to render a DAG image
        import networkx.drawing.nx_pydot as nxpd
        Source(nxpd.to_pydot(G)).render(f"dag_radius_{radius}km", format="png", quiet=True)
        print(f"\n🖼️   DAG image saved as {dag_filename}")
    except Exception:
        print("\n🖼️   graphviz or pydot not installed → DAG image skipped")

    # 9 ▸ SUMMARY OF FINDINGS -------------------------------------
    print(f"\n📋  SUMMARY OF MEDIATION ANALYSIS (RADIUS: {radius} KM)")
    print("─────────────────────────────────────────────────────")
    print(f"Total effect (W → S)            : {c_coef:+.4g}")
    print(f"Direct effect (W → S)           : {c_prime_coef:+.4g}")
    print(f"Indirect effect (W → P → S)     : {indirect_effect:+.4g}")
    print(f"Percentage mediated by pressure : {proportion_mediated:.1f}%")
    print("─────────────────────────────────────────────────────")

    # Return results as a dictionary for later saving to CSV
    return {
        "radius": radius,
        "n_rows": len(df),
        "total_effect": c_coef,
        "total_effect_pvalue": c_pval,
        "direct_effect": c_prime_coef,
        "direct_effect_pvalue": c_prime_pval,
        "indirect_effect": indirect_effect,
        "proportion_mediated": proportion_mediated,
        "path_a_coef": a_coef,
        "path_a_pvalue": a_pval,
        "path_b_coef": b_coef,
        "path_b_pvalue": b_pval,
        "rmse": rmse,
        "r_squared": r2,
        "refutation_placebo": placebo.new_effect,
        "refutation_subset": subset.new_effect,
    }


# Main execution
if __name__ == "__main__":
    # Record overall start time
    start_time = time.time()
    print(f"\n{'=' * 80}\nBATCH CAUSAL ANALYSIS WITH MEDIATION\n{'=' * 80}")

    # Process each file and collect results
    results = []
    for i, csv_file in enumerate(CSV_FILES):
        print(f"\nProcessing file {i + 1} of {len(CSV_FILES)}: {csv_file}")
        try:
            result = process_file(csv_file)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {str(e)}")

    # If we have any results, save them to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("radius")
        results_df.to_csv(RESULTS_CSV, index=False)
        print(f"\n✅ Results saved to {RESULTS_CSV}")

        # Print comparative summary
        print(f"\n{'=' * 80}\nCOMPARATIVE SUMMARY ACROSS RADII\n{'=' * 80}")
        print("\nEffect Estimates by Radius:")
        summary_df = results_df[["radius", "n_rows", "total_effect", "direct_effect",
                                 "indirect_effect", "proportion_mediated"]]
        print(summary_df.to_string(index=False,
                                   float_format=lambda x: f"{x:+.5f}" if isinstance(x, (int, float)) and not isinstance(
                                       x, bool) else str(x)))

        # Find radius with strongest effect
        strongest_idx = results_df["total_effect"].abs().idxmax()
        strongest = results_df.loc[strongest_idx]

        # Find radius with most mediation
        most_mediated_idx = results_df["proportion_mediated"].idxmax()
        most_mediated = results_df.loc[most_mediated_idx]

        print(f"\nStrongest total effect found at {strongest['radius']}km radius: {strongest['total_effect']:.5f}")
        print(
            f"Most mediation by pressure found at {most_mediated['radius']}km radius: {most_mediated['proportion_mediated']:.1f}%")
    else:
        print("\n❌ No valid results obtained from any file.")

    # Print total runtime
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}\nAnalysis complete! Total runtime: {total_time:.1f} seconds\n{'=' * 80}")