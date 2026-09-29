"""
Extracts the 11 physics variables from the collocated CMEMS/ERA5/GPM historical dataset
into a single, hourly, chronologically-sorted table with no gaps.
Handles small gaps via interpolation and asserts zero missing values.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SOURCES = [
    PROJECT_ROOT / "data" / "interim" / "collocated.parquet",
    PROJECT_ROOT.parent / "demand-forecast" / "data" / "interim" / "collocated.parquet",
    PROJECT_ROOT.parent / "CapstoneProject_ML" / "data" / "interim" / "collocated.parquet",
    PROJECT_ROOT.parent / "Capstone_Project_ML" / "data" / "interim" / "collocated.parquet",
]

PHYSICS_VARIABLES = [
    "hs",
    "tp",
    "swell_height",
    "wind_wave_height",
    "wind_speed",
    "wind_gust",
    "wind_dir",
    "slp",
    "current_u",
    "current_v",
    "rain_rate_mm_hr",
]

def extract_and_clean_dataset():
    source_path = None
    for p in CANDIDATE_SOURCES:
        if p.exists():
            source_path = p
            break

    if not source_path:
        raise FileNotFoundError(f"Could not find collocated source dataset in candidates: {CANDIDATE_SOURCES}")

    print(f"Reading source dataset from: {source_path}")
    df_raw = pd.read_parquet(source_path)
    print(f"Source shape: {df_raw.shape}, index: {df_raw.index.min()} to {df_raw.index.max()}")

    # Ensure index is datetime
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw.index = pd.to_datetime(df_raw.index)

    # Sort chronologically
    df_sorted = df_raw.sort_index()

    # Check for duplicate timestamps
    if df_sorted.index.duplicated().any():
        dups = df_sorted.index.duplicated().sum()
        print(f"Warning: Found {dups} duplicate timestamps. Dropping duplicates (keeping first)...")
        df_sorted = df_sorted[~df_sorted.index.duplicated(keep="first")]

    # Extract the 11 variables
    missing_cols = [c for c in PHYSICS_VARIABLES if c not in df_sorted.columns]
    if missing_cols:
        raise KeyError(f"Missing requested physics variables in dataset: {missing_cols}")

    df_11 = df_sorted[PHYSICS_VARIABLES].copy()

    # Construct the complete continuous hourly datetime grid
    min_ts = df_11.index.min()
    max_ts = df_11.index.max()
    full_hourly_index = pd.date_range(start=min_ts, end=max_ts, freq="1h", name="timestamp")
    print(f"Canonical hourly grid: {len(full_hourly_index)} hours from {min_ts} to {max_ts}")

    # Reindex onto full hourly grid to reveal any missing hourly timestamps
    df_reindexed = df_11.reindex(full_hourly_index)
    gaps_count = df_reindexed.isna().sum()
    print("Missing values before interpolation:")
    print(gaps_count.to_dict())

    # Detect gap sizes per column if any
    for col in PHYSICS_VARIABLES:
        null_mask = df_reindexed[col].isna()
        if null_mask.any():
            gap_blocks = (~null_mask).cumsum()[null_mask]
            gap_sizes = gap_blocks.value_counts()
            large_gaps = gap_sizes[gap_sizes > 3]
            if len(large_gaps) > 0:
                print(f"  [FLAG] {col} has {len(large_gaps)} gap(s) larger than 3 hours!")
            else:
                print(f"  [INFO] {col} has only small gaps (<= 3 hours).")

    # Interpolate small gaps
    # For angular direction (wind_dir), convert to sin/cos before interpolating
    if "wind_dir" in df_reindexed.columns and df_reindexed["wind_dir"].isna().any():
        rad = np.deg2rad(df_reindexed["wind_dir"])
        sin_dir = np.sin(rad).interpolate(method="linear", limit=3)
        cos_dir = np.cos(rad).interpolate(method="linear", limit=3)
        interp_dir = np.rad2deg(np.arctan2(sin_dir, cos_dir)) % 360
        df_reindexed["wind_dir"] = interp_dir

    # Linear interpolation for physical scalars up to 3 consecutive missing steps
    df_clean = df_reindexed.interpolate(method="linear", limit=3)

    # Edge fill (forward fill / backward fill for boundaries if any)
    df_clean = df_clean.bfill().ffill()

    # Final assertion: no NaNs and exact row count
    assert df_clean.isna().sum().sum() == 0, f"NaNs remain after cleaning: {df_clean.isna().sum().to_dict()}"
    assert len(df_clean) == len(full_hourly_index), f"Length mismatch: {len(df_clean)} vs {len(full_hourly_index)}"

    # Save outputs
    interim_dir = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full collocated dataset in data/interim/
    df_raw.to_parquet(interim_dir / "collocated.parquet")
    print(f"Saved full collocated table to {interim_dir / 'collocated.parquet'}")

    # 2. Save cleaned 11-variable table in data/processed/
    out_parquet = processed_dir / "historical_11_physics_hourly.parquet"
    out_csv = processed_dir / "historical_11_physics_hourly.csv"
    df_clean.to_parquet(out_parquet)
    df_clean.to_csv(out_csv)

    print(f"Successfully exported {len(df_clean)} hourly records with {df_clean.shape[1]} physics variables:")
    print(f"  Parquet: {out_parquet} ({out_parquet.stat().st_size:,} bytes)")
    print(f"  CSV:     {out_csv} ({out_csv.stat().st_size:,} bytes)")
    print("Summary statistics of 11 physics variables:")
    print(df_clean.describe().T[["mean", "std", "min", "50%", "max"]])

    return df_clean

if __name__ == "__main__":
    extract_and_clean_dataset()
