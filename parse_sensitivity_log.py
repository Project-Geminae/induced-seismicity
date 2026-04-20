#!/usr/bin/env python3
"""Parse mediation_sensitivity_sweep.log into a per-radius summary CSV."""
import re
from pathlib import Path
import pandas as pd

LOG = Path("mediation_sensitivity_sweep.log")
OUT = Path("mediation_sensitivity_summary.csv")

RADIUS_RE = re.compile(r"=== Radius (\d+) km ===")
TE_RE = re.compile(r"TE\s+=\s+([+-][\d.eE+-]+)")
NDE_RE = re.compile(r"NDE\s+=\s+([+-][\d.eE+-]+)\s+CI:\s+\[([+-][\d.eE+-]+),\s*([+-][\d.eE+-]+)\]")
NIE_RE = re.compile(r"NIE\s+=\s+([+-][\d.eE+-]+)\s+CI:\s+\[([+-][\d.eE+-]+),\s*([+-][\d.eE+-]+)\]")
PCT_RE = re.compile(r"%mediated\s+=\s+([+-]?[\d.]+)%")
SIGMA_RE = re.compile(r"sigma_M\s+=\s+([\d.]+)\s+sigma_Y\s+=\s+([\d.]+)")
BIAS_RE = re.compile(r"Bias unit .* =\s+([\d.eE+-]+)")
RHO_NDE_RE = re.compile(r"rho at which NDE flips sign:\s+([\d.eE+-]+)")
RHO_NIE_RE = re.compile(r"rho at which NIE becomes non-signif:\s+([\d.eE+-]+)")

rows = []
cur = None
text = LOG.read_text()
blocks = re.split(r"(=== Radius \d+ km ===)", text)
for i in range(1, len(blocks), 2):
    header = blocks[i]
    body = blocks[i + 1] if i + 1 < len(blocks) else ""
    R = int(RADIUS_RE.search(header).group(1))
    te = TE_RE.search(body)
    nde = NDE_RE.search(body)
    nie = NIE_RE.search(body)
    pct = PCT_RE.search(body)
    sigma = SIGMA_RE.search(body)
    bias = BIAS_RE.search(body)
    rho_nde = RHO_NDE_RE.search(body)
    rho_nie = RHO_NIE_RE.search(body)
    if not all([te, nde, nie]):
        continue
    rows.append({
        "radius_km": R,
        "TE": float(te.group(1)),
        "NDE": float(nde.group(1)),
        "NDE_ci_low": float(nde.group(2)),
        "NDE_ci_high": float(nde.group(3)),
        "NIE": float(nie.group(1)),
        "NIE_ci_low": float(nie.group(2)),
        "NIE_ci_high": float(nie.group(3)),
        "pct_mediated": float(pct.group(1)) if pct else float("nan"),
        "sigma_M": float(sigma.group(1)) if sigma else float("nan"),
        "sigma_Y": float(sigma.group(2)) if sigma else float("nan"),
        "bias_unit": float(bias.group(1)) if bias else float("nan"),
        "rho_NDE_flip": float(rho_nde.group(1)) if rho_nde else float("nan"),
        "rho_NIE_nonsig": float(rho_nie.group(1)) if rho_nie else float("nan"),
    })

df = pd.DataFrame(rows).sort_values("radius_km").reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} — {len(df)} radii")
print(df.to_string(index=False))
