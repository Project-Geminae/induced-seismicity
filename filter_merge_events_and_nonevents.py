#!/usr/bin/env python3
"""
Batch version
─────────────
For each radius R in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20} km:

1. Read innocent_wells_<R>km.csv
2. Replace the `EventID` prefix  texnet → faknet
3. Save the patched file as           innocent_wells_with_fakeids_<R>km.csv
4. Concatenate that table with
       event_well_links_with_injection_<R>km.csv
   and write                           combined_event_well_links_<R>km.csv

Usage:  python batch_concat_innocent_and_links.py
"""

from pathlib import Path
import pandas as pd

# ── file templates ─────────────────────────────────────────────────────────
RADII = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]  # km

INNOCENT_FMT   = "innocent_wells_{R}km.csv"
PATCHED_FMT    = "innocent_wells_with_fakeids_{R}km.csv"
EVENT_LINK_FMT = "event_well_links_with_injection_{R}km.csv"
OUTPUT_FMT     = "combined_event_well_links_{R}km.csv"


def patch_event_ids(src: Path, dst: Path) -> pd.DataFrame:
    """Replace EventID prefix (texnet → faknet) and save to *dst*."""
    df = pd.read_csv(src, dtype=str)
    if "EventID" not in df.columns:
        raise KeyError(f"'EventID' column not found in {src.name}")

    df["EventID"] = df["EventID"].str.replace(r"^texnet", "faknet", regex=True)
    df.to_csv(dst, index=False)
    print(f"✅  {dst.name:40}  ({len(df):,} rows)")
    return df


def concatenate(event_path: Path, fake_df: pd.DataFrame, out_path: Path) -> None:
    """Concatenate *fake_df* with event_well_links file and save."""
    event_df = pd.read_csv(event_path, dtype=str)
    combined = pd.concat([event_df, fake_df], ignore_index=True, sort=False)
    combined.to_csv(out_path, index=False)
    print(f"✅  {out_path.name:40}  ({len(combined):,} rows)")


def main() -> None:
    for R in RADII:
        # Resolve paths for this radius
        innocent    = Path(INNOCENT_FMT.format(R=R))
        patched     = Path(PATCHED_FMT.format(R=R))
        event_link  = Path(EVENT_LINK_FMT.format(R=R))
        output_file = Path(OUTPUT_FMT.format(R=R))

        print(f"\n─── Processing {R} km files ─────────────────────────────────────")
        fake_df = patch_event_ids(innocent, patched)
        concatenate(event_link, fake_df, output_file)


if __name__ == "__main__":
    main()