#!/usr/bin/env python3
"""
magnitude_histogram_log.py
──────────────────────────
Animated MP4 of cumulative earthquake counts (M2–M6) on a *log* scale.

Output: magnitude_histogram.mp4   (1920×1080, 1 fps)
"""

# ── Imports ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ── 0 ▸ global style tweaks (dark theme + nicer fonts) --------------------
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#0e1117",
    "axes.edgecolor":   "#333940",
    "axes.labelcolor":  "white",
    "text.color":       "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "font.size":        16,
    "figure.dpi":       120,
})

# Custom colour palette: cyan → green → amber → red → violet
COLOURS = {
    "M2": "#00bcd4",   # bright cyan
    "M3": "#4caf50",   # fresh green
    "M4": "#ffc107",   # golden amber
    "M5": "#f44336",   # hot red
    "M6": "#9c27b0",   # royal violet   ← new
}

def main() -> None:
    # ── 1 ▸ raw data ------------------------------------------------------
    data = {
        "Year": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "M2":   [271,  602,  847, 1110, 1971, 2392, 2286, 1855,  565],
        "M3":   [26,    55,   52,   93,  199,  202,  201,  175,   27],
        "M4":   [0,      2,    3,    5,   17,   19,   11,   11,    5],
        "M5":   [0,      0,    0,    0,    0,    2,    1,    2,    2],
        "M6":   [0,      0,    0,    0,    0,    0,    0,    0,    0],
    }

    df        = pd.DataFrame(data).set_index("Year")
    cumul     = df.cumsum()

    plot_bins = ["M2", "M3", "M4", "M5", "M6"]
    eps       = 0.5                                 # avoids log(0)
    cumul_cln = cumul.clip(lower=eps)
    y_max     = cumul_cln[plot_bins].to_numpy().max() * 1.3

    # ── 2 ▸ animation setup ---------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12)

    # Static elements
    ax.set_yscale("log")
    ax.set_ylim(eps, y_max)
    ax.set_ylabel("Cumulative event count (log scale)", labelpad=10)
    ax.set_xlabel("Magnitude bin", labelpad=10)
    ax.set_title("Permian Basin M6?", fontsize=26, pad=20, weight="bold")

    # legend
    legend_handles = [
        plt.Line2D([0], [0], color=COLOURS[m], lw=8, label=m) for m in plot_bins
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", fontsize=14)

    # grid
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(1, 10) * 0.1))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#333940", alpha=0.8)

    # initial bars
    bars = ax.bar(
        plot_bins,
        cumul_cln.iloc[0][plot_bins],
        color=[COLOURS[m] for m in plot_bins],
        width=0.6,
    )

    anno = ax.text(
        0.98, 0.92, "", transform=ax.transAxes,
        ha="right", va="center", fontsize=18, color="white", weight="bold"
    )

    def update(frame_idx: int):
        year   = cumul_cln.index[frame_idx]
        counts = cumul_cln.loc[year, plot_bins]

        for bar, height in zip(bars, counts):
            bar.set_height(height)

        anno.set_text(f"Through {year}")
        return (*bars, anno)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(cumul_cln),
        interval=1200,
        blit=True,
        repeat=False,
    )

    # ── 3 ▸ save MP4 ------------------------------------------------------
    out_file = "magnitude_histogram.mp4"
    writer   = FFMpegWriter(fps=1, bitrate=6000)
    anim.save(out_file, writer=writer)
    print(f"✅  Saved {out_file}")


if __name__ == "__main__":
    main()