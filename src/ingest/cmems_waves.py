"""
Source: CMEMS WAVERYS (GLOBAL_MULTIYEAR_WAV_001_032)
Pulls:  Hs, Tp, swell height, wind-wave height — one API call, one file.
Auth:   `copernicusmarine login` once, interactively, before running this script.
"""

import xarray as xr
import copernicusmarine as cm

from config import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, START_DATE, END_DATE, RAW_DIR, INTERIM_DIR, target_hourly_index

DATASET_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"  # multi-year reanalysis, 3-hourly
OUT_RAW = f"{RAW_DIR}/cmems_waves.nc"
OUT_INTERIM = f"{INTERIM_DIR}/cmems_waves.parquet"


def download():
    cm.subset(
        dataset_id=DATASET_ID,
        variables=["VHM0", "VTPK", "VHM0_SW1", "VHM0_WW"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=f"{START_DATE}T00:00:00", end_datetime=f"{END_DATE}T23:59:59",
        output_filename=OUT_RAW,
    )


def to_interim():
    ds = xr.open_dataset(OUT_RAW)
    df = ds.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()
    df = df.rename(columns={
        "VHM0": "hs", "VTPK": "tp", "VHM0_SW1": "swell_height", "VHM0_WW": "wind_wave_height",
    })[["time", "hs", "tp", "swell_height", "wind_wave_height"]]

    # WAVERYS is 3-hourly — upsample to hourly with linear interpolation, not forward-fill,
    # so the feature-engineering step downstream doesn't see stair-stepped values.
    df = df.set_index("time").resample("1h").interpolate(method="linear")

    # Clip / reindex to canonical target_hourly_index (26,304 rows ending at 2024-12-31 23:00:00)
    full_idx = target_hourly_index()
    df = df.reindex(full_idx).ffill()

    df.to_parquet(OUT_INTERIM)
    print(f"wrote {len(df)} hourly rows -> {OUT_INTERIM}")


if __name__ == "__main__":
    download()
    to_interim()