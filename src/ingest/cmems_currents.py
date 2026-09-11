"""
Source: CMEMS GLORYS12V1 (GLOBAL_MULTIYEAR_PHY_001_030)
Pulls:  zonal (uo) and meridional (vo) surface current velocity.
Auth:   same `copernicusmarine login` session as cmems_waves.py.
"""

import numpy as np
import xarray as xr
import copernicusmarine as cm

from config import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, START_DATE, END_DATE, RAW_DIR, INTERIM_DIR, target_hourly_index

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"  # daily-mean reanalysis
OUT_RAW = f"{RAW_DIR}/cmems_currents.nc"
OUT_INTERIM = f"{INTERIM_DIR}/cmems_currents.parquet"


def download():
    cm.subset(
        dataset_id=DATASET_ID,
        variables=["uo", "vo"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        minimum_depth=0, maximum_depth=1,  # surface layer only
        start_datetime=f"{START_DATE}T00:00:00", end_datetime=f"{END_DATE}T23:59:59",
        output_filename=OUT_RAW,
    )


def to_interim():
    ds = xr.open_dataset(OUT_RAW)
    df = ds.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()
    df = df.rename(columns={"uo": "current_u", "vo": "current_v"})[["time", "current_u", "current_v"]]

    # Note: GLORYS12V1 here is daily-mean, coarser than the hourly target grid.
    # Forward-fill within each day, then let the tidal-rate feature (Day 4) carry the
    # sub-daily current variability instead — daily-mean current + hourly tide phase
    # is the standard approximation used for this dataset.
    df = df.set_index("time").resample("1h").ffill()
    df = df.reindex(target_hourly_index()).ffill()
    df["current_speed"] = np.sqrt(df["current_u"] ** 2 + df["current_v"] ** 2)
    df["current_dir"] = (np.degrees(np.arctan2(df["current_v"], df["current_u"]))) % 360
    df.to_parquet(OUT_INTERIM)
    print(f"wrote {len(df)} hourly rows -> {OUT_INTERIM}")


if __name__ == "__main__":
    download()
    to_interim()