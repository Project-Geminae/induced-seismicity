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

def load_event_counts(events_csv: str = "texnet_events_filtered.csv") -> pd.DataFrame:
    """Compute the year × magnitude-bin count table directly from the catalog.

    The old version of this script hardcoded a 9 × 5 table from a prior pipeline
    run. This version derives the table from the filtered TexNet catalog so the
    figure stays in sync with whatever quality filters are currently in force.
    """
    events = pd.read_csv(events_csv, low_memory=False)
    events["Year"] = pd.to_datetime(events["Origin Date"]).dt.year
    bin_edges = [2, 3, 4, 5, 6, 7]   # right-open: [2,3), [3,4), [4,5), [5,6), [6,7)
    bin_labels = ["M2", "M3", "M4", "M5", "M6"]
    events["bin"] = pd.cut(events["Local Magnitude"],
                           bins=bin_edges, labels=bin_labels, right=False)
    table = (
        events.dropna(subset=["bin"])
              .pivot_table(index="Year", columns="bin", values="EventID",
                           aggfunc="count", fill_value=0)
              .reindex(columns=bin_labels, fill_value=0)
              .sort_index()
    )
    return table


def main() -> None:
    # ── 1 ▸ raw data ------------------------------------------------------
    df    = load_event_counts()
    cumul = df.cumsum()

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