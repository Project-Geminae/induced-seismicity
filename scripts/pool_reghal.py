"""Inverse-variance pool the regHAL-TMLE per-radius shift CSVs into
the basin-scale combined-test headline. Run from /opt/induced-seismicity
under the venv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path("/opt/induced-seismicity")

rows = []
for R in range(1, 21):
    p = ROOT / f"reghal_shift_{R}km.csv"
    if not p.exists():
        continue
    df = pd.read_csv(p).iloc[0]
    rows.append({
        "R": R,
        "n": int(df["n"]),
        "n_clusters": int(df["n_clusters"]),
        "psi_plugin": float(df["psi_plugin"]),
        "psi_targeted": float(df["psi_targeted"]),
        "se_cluster": float(df["se_cluster"]),
        "pval": float(df["pval"]),
        "converged": bool(df["converged"]),
    })

d = pd.DataFrame(rows)
print("=== Per-radius regHAL-TMLE (May 6 vintage) ===")
print(d.to_string(index=False))
print()


def pool(band_df: pd.DataFrame, label: str):
    w = 1 / band_df["se_cluster"] ** 2
    psi_p = float((w * band_df["psi_targeted"]).sum() / w.sum())
    se_p = float(np.sqrt(1 / w.sum()))
    z = psi_p / se_p
    p = float(2 * (1 - norm.cdf(abs(z))))
    print(f"=== Pooled: {label} ({len(band_df)} radii) ===")
    print(f"  psi_pooled = {psi_p:+.4e}")
    print(f"  se_pooled  = {se_p:.4e}")
    print(f"  CI95       = [{psi_p - 1.96*se_p:+.4e}, {psi_p + 1.96*se_p:+.4e}]")
    print(f"  z          = {z:+.3f}")
    print(f"  p          = {p:.4e}")
    print()
    return {
        "n_radii": len(band_df),
        "radii": ",".join(str(int(r)) for r in band_df["R"].tolist()),
        "n": int(band_df["n"].iloc[0]),
        "n_clusters": int(band_df["n_clusters"].iloc[0]),
        "psi_pooled": psi_p,
        "se_pooled": se_p,
        "ci_low": psi_p - 1.96 * se_p,
        "ci_high": psi_p + 1.96 * se_p,
        "z": float(z),
        "pval": p,
    }


band_pb = pool(d[(d["R"] >= 7) & (d["R"] <= 19)], "pressure band 7-19")
band_nf = pool(d[(d["R"] >= 1) & (d["R"] <= 6)], "near-field 1-6")
band_full = pool(d[d["R"] >= 1], "full sweep 1-20")

out = {
    "estimator": "regHAL-TMLE Delta-method (Li/Qiu/Wang/van der Laan 2025)",
    "vintage": "2026-05-06",
    "pressure_band_km": [7, 19],
    "pressure_band": band_pb,
    "near_field_km": [1, 6],
    "near_field": band_nf,
    "full_sweep_1_20": band_full,
    "per_radius": d.to_dict(orient="records"),
}

OUT = ROOT / "reghal_combined_test_may06.json"
OUT.write_text(json.dumps(out, indent=2))
print(f"Wrote {OUT}")
