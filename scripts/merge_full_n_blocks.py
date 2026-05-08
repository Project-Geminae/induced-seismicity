"""Merge full_n_cv (R=7 row) + full_n_combined_test (pooled + per-radius)
into cf_targeted_7km.json, preserving the four cf_targeted blocks
(cluster_robust, cluster_bootstrap, tmle_targeted, hurdle).

Usage (from /opt/induced-seismicity):
    python3 merge_full_n_blocks_alphanet.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/opt/induced-seismicity")
JSON_PATH = ROOT / "cf_targeted_7km.json"
COMBINED = ROOT / "hurdle_full_n_combined_test.csv"
PER_RADIUS = ROOT / "hurdle_full_n_combined_test.per_radius.csv"
R7 = ROOT / "hurdle_full_n_cv_R7km.csv"

# Load existing JSON (keep cf_targeted blocks intact).
d = json.loads(JSON_PATH.read_text())
print("Before merge - keys:", list(d.keys()))

# ----- full_n_cv (R=7 single-radius row) -----
r7 = pd.read_csv(R7).iloc[0].to_dict()
full_n_cv = {
    "n": int(r7["n"]),
    "n_clusters": int(r7["n_clusters"]),
    "n_positives": int(r7["n_positives"]),
    "lambda_pos_cv": float(r7["lambda_pos_cv"]),
    "lambda_mag_cv": float(r7["lambda_mag_cv"]),
    "n_active_pos": int(r7["n_active_pos"]),
    "n_active_mag": int(r7["n_active_mag"]),
    "shift_pct": float(r7["shift_pct"]),
    "psi_total": float(r7["psi_total"]),
    "psi_freq": float(r7["psi_freq"]),
    "psi_mag": float(r7["psi_mag"]),
    "psi_cross": float(r7["psi_cross"]),
    "se_iid_mean": float(r7["se_iid_mean"]),
    "se_cluster_mean": float(r7["se_cluster_mean"]),
    "design_effect": float(r7["design_effect"]),
    "ci_low": float(r7["ci_low"]),
    "ci_high": float(r7["ci_high"]),
    "z": float(r7["z"]),
    "pval": float(r7["pval"]),
    "total_wall_min": float(r7["total_wall_min"]),
    "estimator": str(r7["estimator"]),
}

# ----- full_n_combined_test (pooled + per_radius) -----
pooled = pd.read_csv(COMBINED).iloc[0].to_dict()
per_radius_df = pd.read_csv(PER_RADIUS)
per_radius = []
for _, row in per_radius_df.iterrows():
    per_radius.append({
        "radius_km": int(row["radius_km"]),
        "psi_total": float(row["psi_total"]),
        "psi_freq": float(row["psi_freq"]),
        "psi_mag": float(row["psi_mag"]),
        "psi_cross": float(row["psi_cross"]),
        "se_cluster_mean": float(row["se_cluster_mean"]),
        "n": int(row["n"]),
        "n_clusters": int(row["n_clusters"]),
        "n_active_pos": int(row["n_active_pos"]),
        "n_active_mag": int(row["n_active_mag"]),
        "lambda_pos_cv": float(row["lambda_pos_cv"]),
        "lambda_mag_cv": float(row["lambda_mag_cv"]),
    })

full_n_combined_test = {
    "n_radii": int(pooled["n_radii"]),
    "radii": str(pooled["radii"]),
    "psi_pooled": float(pooled["psi_pooled"]),
    "se_pooled": float(pooled["se_pooled"]),
    "ci_low": float(pooled["ci_low"]),
    "ci_high": float(pooled["ci_high"]),
    "z": float(pooled["z"]),
    "pval": float(pooled["pval"]),
    "psi_freq_pooled": float(pooled["psi_freq_pooled"]),
    "psi_mag_pooled": float(pooled["psi_mag_pooled"]),
    "psi_cross_pooled": float(pooled["psi_cross_pooled"]),
    "per_radius": per_radius,
    "estimator": str(pooled["estimator"]),
}

d["full_n_cv"] = full_n_cv
d["full_n_combined_test"] = full_n_combined_test

print("After merge - keys:", list(d.keys()))
print(f"  full_n_cv (R=7): psi={full_n_cv['psi_total']:.4e}, "
      f"z={full_n_cv['z']:.3f}, p={full_n_cv['pval']:.4f}")
print(f"  full_n_combined_test: psi_pooled={full_n_combined_test['psi_pooled']:.4e}, "
      f"z={full_n_combined_test['z']:.3f}, p={full_n_combined_test['pval']:.4e}")
print(f"  per_radius entries: {len(full_n_combined_test['per_radius'])}")

JSON_PATH.write_text(json.dumps(d, indent=2))
print(f"Wrote {JSON_PATH}")
