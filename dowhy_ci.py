import numpy as np
import pandas as pd
import warnings
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dowhy import CausalModel
import dowhy.causal_estimators.linear_regression_estimator
import networkx as nx
import os
import re
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Current working directory:", os.getcwd())

# Navigate to the parent directory if we're in .git-rewrite
if os.getcwd().endswith('.git-rewrite'):
    parent_dir = Path(os.getcwd()).parent
    os.chdir(parent_dir)
    print(f"Changed to parent directory: {os.getcwd()}")

# Also try going up one more level if needed
if not any(f.endswith('.csv') for f in os.listdir('.') if os.path.isfile(f)):
    if Path('..').exists():
        os.chdir('..')
        print(f"Changed to: {os.getcwd()}")

print("\nLooking for data files...")


def find_csv_files():
    csv_files = []

    # Look in current directory
    for file in os.listdir('.'):
        if file.endswith('.csv') and os.path.isfile(file):
            csv_files.append(file)

    # Look in common subdirectories
    for subdir in ['data', 'Data', 'csv', 'CSV']:
        if os.path.exists(subdir) and os.path.isdir(subdir):
            try:
                for file in os.listdir(subdir):
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(subdir, file))
            except PermissionError:
                continue

    return csv_files


def extract_radius(filename):
    """Extract radius from filename"""
    match = re.search(r'(\d+)km', filename)
    return int(match.group(1)) if match else 0


def find_column(frags, columns):
    """Return the first column that contains all substrings in frags."""
    for c in columns:
        if all(f.lower() in c.lower() for f in frags):
            return c
    return None


def safe(col: str) -> str:
    """Convert a column name to a strict Python identifier."""
    return re.sub(r"\W", "_", col)


def calculate_vif(X):
    """Calculate VIF for each variable in design matrix"""
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data


def calculate_mediation_effects_dowhy(data):
    """Calculate mediation effects using DoWhy framework with Baron & Kenny approach"""

    # Build the causal DAG
    G = nx.DiGraph([
        ('G1', 'W'), ('G1', 'P'), ('G1', 'S'),  # Confounder paths from G1
        ('G2', 'W'), ('G2', 'P'), ('G2', 'S'),  # Confounder paths from G2
        ('W', 'P'),  # Treatment → Mediator
        ('P', 'S'),  # Mediator  → Outcome
        ('W', 'S'),  # Direct Treatment → Outcome
    ])

    # Step 1: Path a (W → P) using DoWhy
    try:
        model_a_dowhy = CausalModel(data, treatment='W', outcome='P', graph=G)
        estimand_a = model_a_dowhy.identify_effect()
        estimate_a = model_a_dowhy.estimate_effect(
            estimand_a,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )
        a_coef_dowhy = estimate_a.value

        # Get p-value from statsmodels for path a
        X_a = sm.add_constant(data[['W', 'G1', 'G2']])
        model_a_sm = sm.OLS(data['P'], X_a).fit()
        a_pval = model_a_sm.pvalues['W']

    except Exception as e:
        print(f"Warning: DoWhy path a estimation failed: {e}")
        # Fallback to statsmodels
        model_a_sm = sm.OLS(data['P'], sm.add_constant(data[['W', 'G1', 'G2']])).fit()
        a_coef_dowhy = model_a_sm.params['W']
        a_pval = model_a_sm.pvalues['W']

    # Step 2: Path b (P → S | W) using DoWhy
    try:
        model_b_dowhy = CausalModel(data, treatment='P', outcome='S', graph=G)
        estimand_b = model_b_dowhy.identify_effect()
        estimate_b = model_b_dowhy.estimate_effect(
            estimand_b,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )
        b_coef_dowhy = estimate_b.value

        # Get p-value from statsmodels for path b
        X_b = sm.add_constant(data[['P', 'W', 'G1', 'G2']])
        model_b_sm = sm.OLS(data['S'], X_b).fit()
        b_pval = model_b_sm.pvalues['P']

    except Exception as e:
        print(f"Warning: DoWhy path b estimation failed: {e}")
        # Fallback to statsmodels
        model_b_sm = sm.OLS(data['S'], sm.add_constant(data[['P', 'W', 'G1', 'G2']])).fit()
        b_coef_dowhy = model_b_sm.params['P']
        b_pval = model_b_sm.pvalues['P']

    # Step 3: Total effect (W → S) using DoWhy
    try:
        model_c_dowhy = CausalModel(data, treatment='W', outcome='S', graph=G)
        estimand_c = model_c_dowhy.identify_effect()
        estimate_c = model_c_dowhy.estimate_effect(
            estimand_c,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )
        c_coef_dowhy = estimate_c.value

        # Get p-value and R² from statsmodels for total effect
        X_c = sm.add_constant(data[['W', 'G1', 'G2']])
        model_c_sm = sm.OLS(data['S'], X_c).fit()
        c_pval = model_c_sm.pvalues['W']
        model_c_r2 = model_c_sm.rsquared

    except Exception as e:
        print(f"Warning: DoWhy total effect estimation failed: {e}")
        # Fallback to statsmodels
        model_c_sm = sm.OLS(data['S'], sm.add_constant(data[['W', 'G1', 'G2']])).fit()
        c_coef_dowhy = model_c_sm.params['W']
        c_pval = model_c_sm.pvalues['W']
        model_c_r2 = model_c_sm.rsquared

    # Step 4: Direct effect (W → S | P) - use statsmodels for consistency
    model_direct = sm.OLS(data['S'], sm.add_constant(data[['P', 'W', 'G1', 'G2']])).fit()
    c_prime_coef = model_direct.params['W']
    c_prime_pval = model_direct.pvalues['W']

    return {
        'total_effect': c_coef_dowhy,
        'total_pval': c_pval,
        'direct_effect': c_prime_coef,
        'direct_pval': c_prime_pval,
        'indirect_effect': a_coef_dowhy * b_coef_dowhy,
        'path_a_coef': a_coef_dowhy,
        'path_a_pval': a_pval,
        'path_b_coef': b_coef_dowhy,
        'path_b_pval': b_pval,
        'model_c_r2': model_c_r2,
        'dowhy_estimates': {
            'path_a': estimate_a if 'estimate_a' in locals() else None,
            'path_b': estimate_b if 'estimate_b' in locals() else None,
            'total': estimate_c if 'estimate_c' in locals() else None
        }
    }


def bootstrap_mediation_effects_dowhy(data, n_bootstrap=100):
    """Calculate bootstrap confidence intervals for DoWhy mediation effects"""
    total_effects = []
    direct_effects = []
    indirect_effects = []

    for i in range(n_bootstrap):
        boot_data = resample(data, random_state=i, replace=True)

        try:
            boot_effects = calculate_mediation_effects_dowhy(boot_data)
            total_effects.append(boot_effects['total_effect'])
            direct_effects.append(boot_effects['direct_effect'])
            indirect_effects.append(boot_effects['indirect_effect'])
        except Exception as e:
            print(f"Bootstrap iteration {i} failed: {e}")
            continue

    if len(total_effects) < n_bootstrap * 0.8:
        print(f"Warning: Only {len(total_effects)}/{n_bootstrap} bootstrap iterations succeeded")

    total_ci = np.percentile(total_effects, [2.5, 97.5]) if total_effects else [np.nan, np.nan]
    direct_ci = np.percentile(direct_effects, [2.5, 97.5]) if direct_effects else [np.nan, np.nan]
    indirect_ci = np.percentile(indirect_effects, [2.5, 97.5]) if indirect_effects else [np.nan, np.nan]

    return {
        'total_ci': total_ci,
        'direct_ci': direct_ci,
        'indirect_ci': indirect_ci,
        'bootstrap_success_rate': len(total_effects) / n_bootstrap
    }


def run_dowhy_refutations(data):
    """Run DoWhy refutation tests"""
    try:
        # Build the causal DAG
        G = nx.DiGraph([
            ('G1', 'W'), ('G1', 'P'), ('G1', 'S'),
            ('G2', 'W'), ('G2', 'P'), ('G2', 'S'),
            ('W', 'P'), ('P', 'S'), ('W', 'S'),
        ])

        # Total effect model for refutations
        model = CausalModel(data, treatment='W', outcome='S', graph=G)
        estimand = model.identify_effect()
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )

        # Placebo treatment refutation
        placebo_refute = model.refute_estimate(
            estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )

        # Data subset refutation
        subset_refute = model.refute_estimate(
            estimand, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8
        )

        return {
            'placebo_effect': placebo_refute.new_effect,
            'subset_effect': subset_refute.new_effect,
            'original_effect': estimate.value
        }

    except Exception as e:
        print(f"Warning: DoWhy refutations failed: {e}")
        return {
            'placebo_effect': np.nan,
            'subset_effect': np.nan,
            'original_effect': np.nan
        }


def calculate_predictive_r2(data):
    """Calculate predictive R² with log transformation (like DoWhy implementation)"""
    try:
        X = data[['W', 'P', 'G1', 'G2']].copy()
        X['W'] = np.log1p(X['W'])  # Log transform volumes
        y = data['S']

        # Train-test split
        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True
        )

        # Fit and predict
        pred = LinearRegression().fit(x_tr, y_tr).predict(x_te)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        r2 = r2_score(y_te, pred)

        return {'predictive_r2': r2, 'rmse': rmse}

    except Exception as e:
        print(f"Warning: Predictive R² calculation failed: {e}")
        return {'predictive_r2': np.nan, 'rmse': np.nan}


def process_file(csv_file):
    """Process a single file and return results"""
    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {csv_file}")
    print(f"{'=' * 60}")

    radius = extract_radius(csv_file)

    try:
        # Load the data
        df_raw = pd.read_csv(csv_file, low_memory=False)
        print(f"✅ Loaded {len(df_raw):,} rows")

        # Auto-detect columns
        COL_FRAGS = {
            "W": ["volume", "bbl"],  # Treatment – injection volume
            "P": ["pressure", "psig"],  # Mediator – average injection pressure
            "S": ["local", "mag"],  # Outcome – earthquake magnitude
            "G1": ["nearest", "fault", "dist"],  # Confounder 1
            "G2": ["fault", "segment"],  # Confounder 2
        }

        found = {}
        for k, frags in COL_FRAGS.items():
            col = find_column(frags, df_raw.columns)
            if col:
                found[k] = col

        # Check if all required columns were found
        missing = [k for k in COL_FRAGS.keys() if k not in found]
        if missing:
            print(f"❌ Missing columns: {missing}")
            return None

        # Rename columns to safe identifiers
        rename_map = {found[k]: safe(k) for k in found}
        data = df_raw[list(found.values())].rename(columns=rename_map)

        # Clean the data
        essential = [safe("W"), safe("S")]
        rows_before = len(data)
        data = data.dropna(subset=essential)
        rows_after = len(data)

        # Median-fill missing values in covariates
        covariates = [safe(c) for c in ["P", "G1", "G2"] if safe(c) in data.columns]
        data[covariates] = data[covariates].fillna(data[covariates].median(numeric_only=True))

        # Rename columns to standard names
        data = data.rename(columns={
            safe("W"): "W",
            safe("P"): "P",
            safe("S"): "S",
            safe("G1"): "G1",
            safe("G2"): "G2"
        })

        print(f"✅ Clean data: {len(data):,} rows (dropped {rows_before - rows_after:,})")

        # Calculate DoWhy mediation effects
        print("🔄 Running DoWhy mediation analysis...")
        effects = calculate_mediation_effects_dowhy(data)

        # Bootstrap confidence intervals with DoWhy
        print("🔄 Running bootstrap confidence intervals...")
        ci_results = bootstrap_mediation_effects_dowhy(data, n_bootstrap=50)

        # Run DoWhy refutations
        print("🔄 Running DoWhy refutation tests...")
        refutations = run_dowhy_refutations(data)

        # Calculate predictive R²
        print("🔄 Calculating predictive metrics...")
        pred_metrics = calculate_predictive_r2(data)

        # Calculate VIFs
        X_a = sm.add_constant(data[['W', 'G1', 'G2']])
        vif_a = calculate_vif(X_a)
        vif_a_avg = vif_a['VIF'].mean()

        X_b = sm.add_constant(data[['W', 'P', 'G1', 'G2']])
        vif_b = calculate_vif(X_b)
        vif_b_avg = vif_b['VIF'].mean()

        # Calculate proportion mediated
        prop_mediated = (effects['indirect_effect'] / effects['total_effect']) * 100 if effects[
                                                                                            'total_effect'] != 0 else 0

        print(f"✅ DoWhy analysis complete")
        print(f"   Total effect: {effects['total_effect']:+.3e} (p={effects['total_pval']:.3e})")
        print(f"   Direct effect: {effects['direct_effect']:+.3e} (p={effects['direct_pval']:.3e})")
        print(f"   Indirect effect: {effects['indirect_effect']:+.3e}")
        print(f"   Proportion mediated: {prop_mediated:.1f}%")
        print(f"   Bootstrap success rate: {ci_results['bootstrap_success_rate']:.1%}")
        print(f"   Predictive R²: {pred_metrics['predictive_r2']:.3f}")

        return {
            'radius': radius,
            'filename': csv_file,
            'n_rows': len(data),
            'total_effect': effects['total_effect'],
            'total_pval': effects['total_pval'],
            'total_ci_low': ci_results['total_ci'][0],
            'total_ci_high': ci_results['total_ci'][1],
            'direct_effect': effects['direct_effect'],
            'direct_pval': effects['direct_pval'],
            'direct_ci_low': ci_results['direct_ci'][0],
            'direct_ci_high': ci_results['direct_ci'][1],
            'indirect_effect': effects['indirect_effect'],
            'indirect_ci_low': ci_results['indirect_ci'][0],
            'indirect_ci_high': ci_results['indirect_ci'][1],
            'prop_mediated': prop_mediated,
            'path_a_coef': effects['path_a_coef'],
            'path_a_pval': effects['path_a_pval'],
            'path_b_coef': effects['path_b_coef'],
            'path_b_pval': effects['path_b_pval'],
            'causal_r2': effects['model_c_r2'],
            'predictive_r2': pred_metrics['predictive_r2'],
            'rmse': pred_metrics['rmse'],
            'vif_a_avg': vif_a_avg,
            'vif_b_avg': vif_b_avg,
            'bootstrap_success_rate': ci_results['bootstrap_success_rate'],
            'placebo_effect': refutations['placebo_effect'],
            'subset_effect': refutations['subset_effect'],
            'dowhy_original_effect': refutations['original_effect']
        }

    except Exception as e:
        print(f"❌ Error processing {csv_file}: {str(e)}")
        return None


# Find event files
csv_files = find_csv_files()
event_files = [f for f in csv_files if 'event_well_links_with_faults' in f]

if not event_files:
    print("❌ No event files found!")
    exit(1)

# Sort event files by radius
event_files.sort(key=lambda x: extract_radius(x))

print(f"\n🎯 Found {len(event_files)} event files to process:")
for i, file in enumerate(event_files):
    radius = extract_radius(file)
    size = os.path.getsize(file) / (1024 * 1024)
    print(f"  {i + 1:2d}. {radius:2d}km - {file} ({size:.1f} MB)")

# Process all files
print(f"\n{'=' * 80}")
print("BATCH DOWHY MEDIATION ANALYSIS WITH BOOTSTRAP CI")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}")

results = []
for i, csv_file in enumerate(event_files):
    print(f"\n[{i + 1}/{len(event_files)}] Processing {csv_file}...")

    result = process_file(csv_file)
    if result:
        results.append(result)

# Create summary table
if results:
    results_df = pd.DataFrame(results).sort_values('radius')

    print(f"\n{'=' * 140}")
    print("📊 COMPREHENSIVE DOWHY MEDIATION ANALYSIS SUMMARY")
    print(f"{'=' * 140}")

    # Main results table
    print("\n🎯 CAUSAL EFFECTS BY RADIUS:")
    print("─" * 140)
    print(
        f"{'Radius':>6} {'N':>7} {'Total Effect':>13} {'95% CI':>20} {'Direct Effect':>14} {'95% CI':>20} {'Indirect':>11} {'% Med':>6} {'Causal R²':>9} {'Pred R²':>8}")
    print("─" * 140)

    for _, row in results_df.iterrows():
        print(f"{row['radius']:>4}km {row['n_rows']:>7,} "
              f"{row['total_effect']:>+10.3e} "
              f"[{row['total_ci_low']:>+7.2e},{row['total_ci_high']:>+7.2e}] "
              f"{row['direct_effect']:>+11.3e} "
              f"[{row['direct_ci_low']:>+7.2e},{row['direct_ci_high']:>+7.2e}] "
              f"{row['indirect_effect']:>+8.2e} "
              f"{row['prop_mediated']:>5.1f}% "
              f"{row['causal_r2']:>8.3f} "
              f"{row['predictive_r2']:>7.3f}")

    print("─" * 140)

    # Statistical significance and quality metrics table
    print("\n📈 STATISTICAL SIGNIFICANCE & QUALITY METRICS:")
    print("─" * 120)
    print(
        f"{'Radius':>6} {'Total p-val':>12} {'Direct p-val':>13} {'Path a p-val':>13} {'Path b p-val':>13} {'Bootstrap':>10} {'VIF':>6}")
    print("─" * 120)

    for _, row in results_df.iterrows():
        total_sig = "***" if row['total_pval'] < 0.001 else "**" if row['total_pval'] < 0.01 else "*" if row[
                                                                                                             'total_pval'] < 0.05 else ""
        direct_sig = "***" if row['direct_pval'] < 0.001 else "**" if row['direct_pval'] < 0.01 else "*" if row[
                                                                                                                'direct_pval'] < 0.05 else ""
        path_a_sig = "***" if row['path_a_pval'] < 0.001 else "**" if row['path_a_pval'] < 0.01 else "*" if row[
                                                                                                                'path_a_pval'] < 0.05 else ""
        path_b_sig = "***" if row['path_b_pval'] < 0.001 else "**" if row['path_b_pval'] < 0.01 else "*" if row[
                                                                                                                'path_b_pval'] < 0.05 else ""

        print(f"{row['radius']:>4}km "
              f"{row['total_pval']:>8.2e}{total_sig:<3} "
              f"{row['direct_pval']:>8.2e}{direct_sig:<3} "
              f"{row['path_a_pval']:>8.2e}{path_a_sig:<3} "
              f"{row['path_b_pval']:>8.2e}{path_b_sig:<3} "
              f"{row['bootstrap_success_rate']:>8.1%} "
              f"{row['vif_b_avg']:>5.2f}")

    print("─" * 120)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

    # DoWhy refutation results
    print("\n🔍 DOWHY REFUTATION TESTS:")
    print("─" * 80)
    print(f"{'Radius':>6} {'Original':>12} {'Placebo':>12} {'Subset':>12} {'Status':>15}")
    print("─" * 80)

    for _, row in results_df.iterrows():
        placebo_ratio = abs(row['placebo_effect'] / row['total_effect']) if row['total_effect'] != 0 else np.inf
        subset_ratio = abs((row['subset_effect'] - row['total_effect']) / row['total_effect']) if row[
                                                                                                      'total_effect'] != 0 else np.inf

        # Status assessment
        status = "✅ PASS" if placebo_ratio < 0.1 and subset_ratio < 0.2 else "⚠️ CAUTION" if placebo_ratio < 0.3 else "❌ FAIL"

        print(f"{row['radius']:>4}km "
              f"{row['total_effect']:>+9.2e} "
              f"{row['placebo_effect']:>+9.2e} "
              f"{row['subset_effect']:>+9.2e} "
              f"{status:>15}")

    print("─" * 80)

    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    print("─" * 60)

    strongest_total = results_df.loc[results_df['total_effect'].abs().idxmax()]
    highest_mediation = results_df.loc[results_df['prop_mediated'].idxmax()]
    best_causal_r2 = results_df.loc[results_df['causal_r2'].idxmax()]
    best_pred_r2 = results_df.loc[results_df['predictive_r2'].idxmax()]

    print(f"• Strongest total effect:    {strongest_total['radius']}km ({strongest_total['total_effect']:+.3e})")
    print(f"• Highest mediation:         {highest_mediation['radius']}km ({highest_mediation['prop_mediated']:.1f}%)")
    print(f"• Best causal model fit:     {best_causal_r2['radius']}km (R²={best_causal_r2['causal_r2']:.3f})")
    print(f"• Best predictive fit:       {best_pred_r2['radius']}km (R²={best_pred_r2['predictive_r2']:.3f})")
    print(f"• Sample size range:         {results_df['n_rows'].min():,} - {results_df['n_rows'].max():,} observations")

    # Mediation patterns
    near_field = results_df[results_df['radius'] <= 5]
    far_field = results_df[results_df['radius'] >= 10]

    print(f"• Near-field mediation:      {near_field['prop_mediated'].mean():.1f}% (≤5km)")
    print(f"• Far-field mediation:       {far_field['prop_mediated'].mean():.1f}% (≥10km)")

    # Quality metrics
    avg_bootstrap_success = results_df['bootstrap_success_rate'].mean()
    avg_vif = results_df['vif_b_avg'].mean()

    print(f"• Average bootstrap success: {avg_bootstrap_success:.1%}")
    print(f"• Average VIF:               {avg_vif:.2f}")

    print("─" * 60)

    # Save results to CSV
    output_file = f"dowhy_mediation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

else:
    print("\n❌ No valid results obtained from any files.")

print(f"\n{'=' * 80}")
print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}")