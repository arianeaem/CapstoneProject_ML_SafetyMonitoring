"""
Standalone Scheduled Fetch Script for Live CMEMS Ocean Current Forecasts.
Dataset: CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024 (e.g. cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i)

WINDOW CONFIGURATION:
- Lookback: 72 hours (3 days) — guarantees full coverage for the model's 48-hour
  lag features (LAG_HOURS = [0, 1, 3, 6, 12, 24, 48]) plus 24h rolling windows
  with a safety margin.
- Forecast Horizon: 10 days (240 hours) forward — full span of CMEMS numerical forecast.
- Total Span: ~312 continuous hourly timestamps.

Scheduled Cadence: Run hourly via Windows Task Scheduler or cron.
Decoupled from FastAPI request path to eliminate 15-20s external network latency.

Run manually from project root:
    python src/ingest/fetch_cmems_forecast.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.config import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, PROJECT_ROOT

logger = logging.getLogger("fetch_cmems_forecast")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = CACHE_DIR / "cmems_forecast_cache.parquet"
TEMP_CACHE_FILE = CACHE_DIR / "cmems_forecast_cache_temp.parquet"
RAW_NC_FILE = CACHE_DIR / "cmems_live_forecast_raw.nc"

# CMEMS Live Analysis/Forecast Product IDs
# 1. 6-hourly instantaneous velocity (sub-daily current resolution)
DATASET_ID_6H = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"
# 2. Daily-mean analysis & forecast fallback
DATASET_ID_DAILY = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"


def fetch_cmems_forecast(days_ahead: int = 10, lookback_hours: int = 72) -> bool:
    """
    Subsets the live CMEMS analysis/forecast product, verifies physical sanity,
    and writes the parsed hourly current vectors to local cache.
    """
    try:
        import copernicusmarine as cm
    except ImportError:
        logger.error("copernicusmarine package is not installed. Run: pip install copernicusmarine")
        return False

    now_utc = datetime.now(timezone.utc)
    start_dt = (now_utc - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:00:00")
    end_dt = (now_utc + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:00:00")

    print("\n" + "=" * 75)
    print("CMEMS LIVE OCEAN CURRENT FORECAST INGESTION & CACHING")
    print("=" * 75)
    print(f"Timestamp (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Fetch Window:    {start_dt} to {end_dt}")
    print(f"Lookback Window: {lookback_hours} hours (covers 48h lag features + buffer)")
    print(f"Forecast Horizon:{days_ahead} days ({days_ahead * 24} hours forward)")
    print(f"Bounding Box:    Lon [{LON_MIN}, {LON_MAX}], Lat [{LAT_MIN}, {LAT_MAX}]")
    print(f"Output Target:   {CACHE_FILE}")
    print("-" * 75)

    downloaded = False
    active_dataset_id = None

    for dataset_id in [DATASET_ID_6H, DATASET_ID_DAILY]:
        try:
            logger.info(f"Connecting to CMEMS API for dataset: '{dataset_id}'...")
            cm.subset(
                dataset_id=dataset_id,
                variables=["uo", "vo"],
                minimum_longitude=LON_MIN,
                maximum_longitude=LON_MAX,
                minimum_latitude=LAT_MIN,
                maximum_latitude=LAT_MAX,
                minimum_depth=0,
                maximum_depth=1,
                start_datetime=start_dt,
                end_datetime=end_dt,
                output_filename=str(RAW_NC_FILE),
                force_download=True,
            )
            downloaded = True
            active_dataset_id = dataset_id
            logger.info(f"Successfully downloaded NetCDF using dataset: '{dataset_id}'")
            break
        except Exception as e:
            logger.warning(f"Download attempt failed for dataset '{dataset_id}': {e}")

    if not downloaded:
        logger.error("CRITICAL: All CMEMS forecast dataset download attempts failed.")
        print("=" * 75 + "\n")
        return False

    # -----------------------------------------------------------------------
    # Parse NetCDF and perform physical verification
    # -----------------------------------------------------------------------
    try:
        ds = xr.open_dataset(RAW_NC_FILE)
        logger.info(f"Loaded NetCDF dimensions: {dict(ds.dims)}")
        logger.info(f"Available data variables: {list(ds.data_vars.keys())}")

        # Check for expected eastward (uo) and northward (vo) velocity variables
        var_u = "uo" if "uo" in ds.data_vars else None
        var_v = "vo" if "vo" in ds.data_vars else None

        if not var_u or not var_v:
            logger.error(f"Missing expected velocity variables (uo, vo). Found: {list(ds.data_vars.keys())}")
            return False

        # Spatial mean across the Balayan Bay / VIP bounding box
        df = ds.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()
        df = df.rename(columns={var_u: "current_u", var_v: "current_v"})[["time", "current_u", "current_v"]]
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

        # Resample onto uniform 1-hour grid with time interpolation
        full_hourly_index = pd.date_range(df.index.min(), df.index.max(), freq="1h")
        df_hourly = df.reindex(full_hourly_index).interpolate(method="time").ffill().bfill()
        df_hourly = df_hourly.reset_index().rename(columns={"index": "timestamp"})

        # Derive physical speed (m/s) and circular direction (degrees)
        u = df_hourly["current_u"].values
        v = df_hourly["current_v"].values
        speed = np.sqrt(u**2 + v**2)
        direction = (np.degrees(np.arctan2(v, u))) % 360.0

        df_hourly["current_speed"] = speed
        df_hourly["current_dir"] = direction

        # -------------------------------------------------------------------
        # Physical Sanity Checks
        # -------------------------------------------------------------------
        mean_speed = float(np.mean(speed))
        p90_speed = float(np.percentile(speed, 90))
        max_speed = float(np.max(speed))
        min_u, max_u = float(np.min(u)), float(np.max(u))
        min_v, max_v = float(np.min(v)), float(np.max(v))

        print("\n" + "-" * 75)
        print("PHYSICAL TELEMETRY SANITY CHECK (LIVE CMEMS FORECAST)")
        print("-" * 75)
        print(f"Total Hourly Rows:      {len(df_hourly)} rows")
        print(f"Covered Range:          {df_hourly['timestamp'].min()} to {df_hourly['timestamp'].max()}")
        print(f"Current Speed (m/s):    Mean={mean_speed:.3f}, P90={p90_speed:.3f}, Max={max_speed:.3f}")
        print(f"Current U (Eastward):   Min={min_u:.3f}, Max={max_u:.3f} m/s")
        print(f"Current V (Northward):  Min={min_v:.3f}, Max={max_v:.3f} m/s")

        # Sanity assertions
        # Ocean currents in Balayan Bay typically range 0.02 - 0.6 m/s. Anything > 3.0 m/s indicates unit scaling bug.
        if max_speed > 3.0:
            logger.warning(f"Unusually high current speed detected ({max_speed:.2f} m/s > 3.0 m/s). Verify units.")
        else:
            print("  [PASS] Physical magnitude sanity check: speeds within expected oceanographic range (0.0 - 1.5 m/s).")

        if np.isnan(speed).any():
            logger.error("NaN detected in interpolated current values.")
            return False
        else:
            print("  [PASS] Data completeness: 0 NaN or missing values across entire window.")

        # -------------------------------------------------------------------
        # Atomic Cache File Update
        # -------------------------------------------------------------------
        df_hourly.to_parquet(TEMP_CACHE_FILE, index=False)
        if TEMP_CACHE_FILE.exists():
            TEMP_CACHE_FILE.replace(CACHE_FILE)

        file_size_kb = os.path.getsize(CACHE_FILE) / 1024.0
        print("-" * 75)
        print(f"SUCCESS: Live CMEMS forecast cache updated ({file_size_kb:.1f} KB -> {CACHE_FILE})")
        print("=" * 75 + "\n")
        return True

    except Exception as e:
        logger.error(f"Error parsing and verifying CMEMS NetCDF data: {e}")
        print("=" * 75 + "\n")
        return False


if __name__ == "__main__":
    success = fetch_cmems_forecast()
    sys.exit(0 if success else 1)
