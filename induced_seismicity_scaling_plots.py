#!/usr/bin/env python3
"""
====================================
Publication-quality plots of induced-seismic magnitude vs. (i) distance
and (ii) well-count multiplier, including 95 % bootstrap confidence bands.

Scenario
--------
• Same total injection volume.
• Well counts N = 1 (baseline), 2, …, 10.
• Direct mechanical component  ~ 1 / N.
• Indirect pressure-mediated component ≈ constant.

Data inputs
-----------
Point estimates **and** 95 % CIs for the single-well (1 ×) case come
directly from Step-7 bootstrap results you supplied.

Outputs  (written to ./plots/)
--------------------------------
total_effect_vs_distance.png
direct_effect_vs_distance.png
indirect_effect_vs_distance.png
total_effect_vs_multiplier.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# ───────────────────────────────────────────────────────────────────────────────
# 1.  BASELINE (1 × wells) POINT ESTIMATES  &  95 % CIs  (radius 1–20 km)
# ───────────────────────────────────────────────────────────────────────────────
radii_km = np.arange(1, 21)

#         total_base, direct_base, indirect_base (from Step-7 table)
total_base = np.array([
    6.472e-06, 2.337e-05, 3.031e-05, 3.137e-05, 2.456e-05,
    1.417e-05, 1.461e-05, 7.607e-06, 7.969e-06, 9.095e-06,
    8.176e-06, 6.199e-06, 4.725e-06, 4.727e-06, 4.464e-06,
    4.115e-06, 4.105e-06, 3.901e-06, 4.087e-06, 4.170e-06,
])
direct_base = np.array([
   -3.386e-06,  3.731e-06,  1.136e-05,  1.626e-05,  8.988e-06,
    2.929e-06,  3.173e-06,  3.877e-07,  3.420e-07,  7.399e-07,
    2.520e-07, -4.069e-07, -4.844e-07, -2.739e-07, -3.448e-07,
   -3.370e-07, -3.573e-07, -4.376e-07, -3.980e-07, -4.159e-07,
])
indirect_base = total_base - direct_base   # consistent with table

# 95 % CI bounds for TOTAL and DIRECT (lower, upper)  —  Step-7 bootstrap table
total_lo = np.array([
    4.12e-07, 1.56e-05, 2.21e-05, 2.59e-05, 1.92e-05,
    1.07e-05, 1.09e-05, 5.62e-06, 6.47e-06, 7.17e-06,
    6.30e-06, 4.86e-06, 3.79e-06, 3.89e-06, 3.61e-06,
    3.50e-06, 3.52e-06, 3.14e-06, 3.51e-06, 3.60e-06,
])
total_hi = np.array([
    4.06e-05, 4.38e-05, 4.12e-05, 3.68e-05, 3.11e-05,
    1.97e-05, 1.98e-05, 1.09e-05, 1.02e-05, 1.19e-05,
    9.78e-06, 7.23e-06, 6.05e-06, 5.57e-06, 5.30e-06,
    5.18e-06, 4.93e-06, 4.74e-06, 4.79e-06, 4.72e-06,
])

direct_lo = np.array([
   -5.65e-06, -7.00e-07,  5.51e-06,  9.51e-06,  4.29e-06,
    1.10e-06,  1.66e-06, -3.74e-07, -3.52e-07,  6.18e-08,
   -3.92e-07, -8.45e-07, -8.47e-07, -5.35e-07, -6.63e-07,
   -5.91e-07, -5.66e-07, -6.73e-07, -6.42e-07, -6.50e-07,
])
direct_hi = np.array([
    2.09e-05,  1.89e-05,  2.01e-05,  2.31e-05,  1.43e-05,
    6.45e-06,  5.62e-06,  1.89e-06,  1.29e-06,  1.77e-06,
    9.35e-07,  4.07e-08, -5.90e-08,  1.02e-07, -3.50e-08,
   -1.74e-08, -7.95e-08, -1.65e-07, -1.40e-07, -1.54e-07,
])

# Indirect CI = Total CI − Direct CI  (exact from bootstrap pairing)
indirect_lo = total_lo - direct_hi   # conservative (worst-case)
indirect_hi = total_hi - direct_lo   # conservative

# ───────────────────────────────────────────────────────────────────────────────
# 2.  WELL-COUNT MULTIPLIERS
# ───────────────────────────────────────────────────────────────────────────────
multipliers = np.arange(1, 11)  # 1 … 10

def _scale(arr, n):          # helper: divide by n but keep shape
    return arr / n

def direct_scaled(n: int):
    return _scale(direct_base, n), _scale(direct_lo, n), _scale(direct_hi, n)

def indirect_scaled(n: int):
    # volume controlled → unchanged
    return indirect_base, indirect_lo, indirect_hi

def total_scaled(n: int):
    d, d_lo, d_hi = direct_scaled(n)
    i, i_lo, i_hi = indirect_scaled(n)
    return (d + i,
            d_lo + i_lo,          # lower bound (conservative addition)
            d_hi + i_hi)          # upper bound

# ───────────────────────────────────────────────────────────────────────────────
# 3.  MATPLOTLIB STYLE  (white, clean)
# ───────────────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (9, 6),
    "savefig.dpi":    350,
    "axes.prop_cycle": cycler(color=plt.cm.tab10.colors),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.frameon": False,
})

Path("plots").mkdir(exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# 4.  TOTAL EFFECT vs DISTANCE  (+ CI ribbons for every multiplier)
# ───────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots()

for n in multipliers:
    mean, lo, hi = total_scaled(n)
    style = "--" if n == 1 else "-"
    lw    = 2.8 if n == 1 else 2.0
    label = "Baseline (1×)" if n == 1 else f"{n}× wells"

    ax1.plot(radii_km, mean, style, lw=lw, marker="o", ms=4, label=label)
    ax1.fill_between(radii_km, lo, hi, alpha=0.15 if n == 1 else 0.08)

ax1.set(
    xlabel="Distance from injection wells (km)",
    ylabel="Total causal effect on magnitude (ΔM)",
    title="Total induced-seismic effect vs. distance  •  same volume, varying wells",
)
ax1.legend(fontsize=9)
fig1.tight_layout()
fig1.savefig("plots/total_effect_vs_distance.png")

# ───────────────────────────────────────────────────────────────────────────────
# 5.  DIRECT EFFECT vs DISTANCE  (+ CIs)
# ───────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots()

for n in multipliers:
    mean, lo, hi = direct_scaled(n)
    style = "--" if n == 1 else "-"
    lw    = 2.8 if n == 1 else 2.0
    label = "1× baseline" if n == 1 else f"{n}×"

    ax2.plot(radii_km, mean, style, lw=lw, marker="s", ms=4, label=label)
    ax2.fill_between(radii_km, lo, hi, alpha=0.15 if n == 1 else 0.08)

ax2.set(
    xlabel="Distance from injection wells (km)",
    ylabel="Direct mechanical effect (ΔM)",
    title=r"Direct component $\propto\;1 / (\#\ \text{wells})$   "
          r"with 95 % CI ribbons",
)
ax2.legend(title="Multiplier", ncol=2, fontsize=9)
fig2.tight_layout()
fig2.savefig("plots/direct_effect_vs_distance.png")

# ───────────────────────────────────────────────────────────────────────────────
# 6.  INDIRECT EFFECT vs DISTANCE  (+ single CI ribbon)
# ───────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots()
ax3.plot(radii_km, indirect_base, color="tab:green", lw=3, marker="^", ms=5,
         label="Indirect mean")
ax3.fill_between(radii_km, indirect_lo, indirect_hi, color="tab:green", alpha=0.15)

ax3.set(
    xlabel="Distance from injection wells (km)",
    ylabel="Indirect pressure-mediated effect (ΔM)",
    title="Indirect pathway (volume-controlled) with 95 % CI",
)
ax3.legend()
fig3.tight_layout()
fig3.savefig("plots/indirect_effect_vs_distance.png")

# ───────────────────────────────────────────────────────────────────────────────
# 7.  TOTAL EFFECT vs MULTIPLIER  (+ CI error bars at select radii)
# ───────────────────────────────────────────────────────────────────────────────
select_radii = [1, 3, 5, 10, 20]  # km
fig4, ax4 = plt.subplots()

for r in select_radii:
    idx = r - 1
    means  = []
    lowers = []
    uppers = []
    for n in multipliers:
        mean, lo, hi = total_scaled(n)
        means.append(mean[idx])
        lowers.append(mean[idx] - lo[idx])   # distance from mean to lower
        uppers.append(hi[idx] - mean[idx])   # distance from mean to upper
    ax4.errorbar(multipliers, means, yerr=[lowers, uppers],
                 marker="o", ms=4, lw=1.5, capsize=3, label=f"{r} km")

ax4.set(
    xlabel="Well-count multiplier (×)",
    ylabel="Total ΔM  (with 95 % CI)",
    title="Total effect vs. well density   •   same volume",
)
ax4.set_xticks(multipliers)
ax4.legend(title="Distance", fontsize=9)
fig4.tight_layout()
fig4.savefig("plots/total_effect_vs_multiplier.png")

print("✅  Plots with confidence intervals saved in ./plots/")