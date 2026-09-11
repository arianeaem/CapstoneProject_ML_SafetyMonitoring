"""
Runs the four per-source ingestion scripts, then collocates them into one
hourly, 15-variable Parquet file ready for feature engineering (Day 4).

Tide (tides.py) is intentionally excluded: it required NAMRIA gauge-calibrated
harmonic fitting, and the historical gauge CSV was not obtainable within the
capstone sprint window. This is a stated, documented scope reduction (same
category as the 3-year data window and capped Optuna trials) — channel-current
strength is still partially represented via current_speed/current_dir from
CMEMS GLORYS12V1, just without the tide-derived slack-water signal. Revisit
post-defense via a pre-fit global harmonic model (e.g. TPXO9 through pyTMD)
that doesn't require local gauge calibration, if time allows.

Usage: python run_all.py
"""

import pandas as pd

import cmems_waves
import cmems_currents
import era5_wind_pressure
import gpm_precip
from config import INTERIM_DIR, target_hourly_index

SOURCES = {
    "waves": f"{INTERIM_DIR}/cmems_waves.parquet",
    "currents": f"{INTERIM_DIR}/cmems_currents.parquet",
    "wind_pressure": f"{INTERIM_DIR}/era5_wind_pressure.parquet",
    "precip": f"{INTERIM_DIR}/gpm_precip.parquet",
}


def run_ingestion():
    cmems_waves.download(); cmems_waves.to_interim()
    cmems_currents.download(); cmems_currents.to_interim()
    era5_wind_pressure.download(); era5_wind_pressure.to_interim()
    files = gpm_precip.download(); gpm_precip.to_interim(files)


def collocate():
    frames = [pd.read_parquet(path) for path in SOURCES.values()]
    collocated = frames[0]
    for f in frames[1:]:
        collocated = collocated.join(f, how="outer")

    missing_pct = collocated.isna().mean().sort_values(ascending=False)
    print("missing % by column (investigate anything above ~2%):")
    print(missing_pct)

    expected_index = target_hourly_index()
    assert collocated.index.equals(expected_index), (
        f"collocated index doesn't match expected range — "
        f"got {collocated.index.min()} to {collocated.index.max()}, "
        f"{len(collocated)} rows vs expected {len(expected_index)}"
    )
    expected_cols = 15  # 4 waves + 4 currents + 6 wind_pressure + 1 precip
    assert collocated.shape[1] == expected_cols, (
        f"expected {expected_cols} columns, got {collocated.shape[1]} — "
        f"check SOURCES dict matches what's actually being merged"
    )

    collocated.to_parquet(f"{INTERIM_DIR}/collocated.parquet")
    print(f"wrote {len(collocated)} rows, {collocated.shape[1]} columns -> collocated.parquet")


if __name__ == "__main__":
    run_ingestion()
    collocate()