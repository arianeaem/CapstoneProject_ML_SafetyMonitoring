
"""
Source: ECMWF ERA5 Reanalysis, via the Copernicus Climate Data Store (CDS) API.
Pulls:  10u, 10v (wind vectors), i10fg (instantaneous 10m gust), msl (sea level pressure).
Auth:   requires a ~/.cdsapirc file with your CDS API key (from cds.climate.copernicus.eu).

IMPORTANT: CDS enforces a per-request "cost" limit. Requesting 3 years x 24 hours
in a single call gets rejected with "403 cost limits exceeded". This version
requests ONE MONTH at a time (36 separate requests for a 3-year window) and
merges the results afterward — each monthly request is small enough to clear
the limit reliably.
"""

import glob
import calendar
import numpy as np
import xarray as xr
import cdsapi

from config import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, START_DATE, END_DATE, RAW_DIR, INTERIM_DIR, target_hourly_index

OUT_RAW_DIR = f"{RAW_DIR}/era5_wind_pressure"
OUT_INTERIM = f"{INTERIM_DIR}/era5_wind_pressure.parquet"


def _year_months(start_date: str, end_date: str):
    start_year, start_month = int(start_date[:4]), int(start_date[5:7])
    end_year, end_month = int(end_date[:4]), int(end_date[5:7])
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def download():
    import os
    os.makedirs(OUT_RAW_DIR, exist_ok=True)
    c = cdsapi.Client()

    for year, month in _year_months(START_DATE, END_DATE):
        out_file = f"{OUT_RAW_DIR}/era5_{year}_{month:02d}.nc"
        if os.path.exists(out_file):
            print(f"skipping {year}-{month:02d}, already downloaded")
            continue

        days_in_month = calendar.monthrange(year, month)[1]
        print(f"requesting ERA5 {year}-{month:02d} ({days_in_month} days)...")

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "10m_u_component_of_wind", "10m_v_component_of_wind",
                    "instantaneous_10m_wind_gust", "mean_sea_level_pressure",
                ],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": [f"{d:02d}" for d in range(1, days_in_month + 1)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # [North, West, South, East]
                "format": "netcdf",
            },
            out_file,
        )
        print(f"  saved {out_file}")


def to_interim():
    files = sorted(glob.glob(f"{OUT_RAW_DIR}/era5_*.nc"))
    if not files:
        raise FileNotFoundError(f"no monthly ERA5 files found in {OUT_RAW_DIR} — run download() first")

    ds = xr.open_mfdataset(files, combine="by_coords")
    df = ds.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()

    # ECMWF's newer CDS backend renamed the time coordinate from "time" to
    # "valid_time" for ERA5 reanalysis data — handle both so this doesn't break
    # again on a different cdsapi/backend version.
    if "valid_time" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"valid_time": "time"})

    df = df.rename(columns={
        "u10": "wind_u", "v10": "wind_v", "i10fg": "wind_gust", "msl": "slp",
    })

    df["wind_speed"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)
    df["wind_dir"] = (270 - np.degrees(np.arctan2(df["wind_v"], df["wind_u"]))) % 360
    df["slp"] = df["slp"] / 100.0  # Pa -> hPa

    df = df.set_index("time")[["wind_u", "wind_v", "wind_speed", "wind_gust", "wind_dir", "slp"]]
    df = df.sort_index()

    full_idx = target_hourly_index()
    missing = full_idx.difference(df.index)
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} hours missing from ERA5 pull — check for a failed/short month")
    df = df.reindex(full_idx)

    df.to_parquet(OUT_INTERIM)
    print(f"wrote {len(df)} hourly rows -> {OUT_INTERIM}")


if __name__ == "__main__":
    download()
    to_interim() 

