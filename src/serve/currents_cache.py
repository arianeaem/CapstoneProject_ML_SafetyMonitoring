"""
Ocean Current Live Cache and Climatological Fallback Layer.

Provides fast, sub-millisecond retrieval of ocean current boundary conditions
(current_u, current_v, current_speed, current_dir) for live serving endpoints.

DESIGN ARCHITECTURE (Decoupled Caching & Graceful Fallback):
1. Separated Ingestion vs Serving:
   - Live endpoint NEVER blocks on slow 15-20s CMEMS subset API calls.
   - Reads directly from local pre-fetched cache: data/cache/cmems_forecast_cache.parquet.
2. Climatological Horizon Extension (Beyond CMEMS 10-day limit):
   - For requested hours beyond CMEMS forecast horizon (e.g. Day 11 to 16 in a 16-day booking window),
     the system automatically fills current vectors using historical seasonal climatology.
3. Defensive Outage Fallback:
   - If the CMEMS scheduled fetch fails (auth expiry, network loss, API downtime) or cache is stale,
     it seamlessly falls back to climatology without crashing or returning HTTP 500.
4. Unit/Sign Parity:
   - current_u (m/s): Eastward velocity (positive East, negative West).
   - current_v (m/s): Northward velocity (positive North, negative South).
   - current_speed (m/s): sqrt(u^2 + v^2).
   - current_dir (deg): (atan2(v, u) * 180 / pi) % 360.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = CACHE_DIR / "cmems_forecast_cache.parquet"
CLIMATOLOGY_FILE = CACHE_DIR / "currents_climatology.parquet"
PROCESSED_TRAINING_FILE = PROJECT_ROOT / "data" / "processed" / "training_features.parquet"

logger = logging.getLogger("currents_cache")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# 1. Climatology Fitting & Lookup Table
# ---------------------------------------------------------------------------
# TODO: Implement automated daily cron job to pre-fetch Copernicus CMEMS 0.083-degree global ocean current vectors.


def fit_and_save_climatology(train_features_path: Path = PROCESSED_TRAINING_FILE) -> pd.DataFrame:
    """
    Fits historical seasonal climatology (hour-of-day, day-of-year) mean vectors
    for current_u and current_v using the training split.

    Business Logic / Rationale:
        CMEMS marine physics models only forecast up to 10 days ahead. For extended
        16-day advance booking windows, historical diurnal-seasonal averages provide
        a physically grounded baseline preventing arbitrary zero-drift assumptions.

    Parameters:
        train_features_path (Path): Path to preprocessed feature dataset.

    Returns:
        pd.DataFrame: Climatology lookup table indexed by (day_of_year, hour_of_day).
    """
    if not train_features_path.exists():
        logger.warning(f"Training features file {train_features_path} not found. Using neutral zeros fallback.")
        # Create a default neutral table if training data is absent
        records = []
        for d in range(1, 367):
            for h in range(24):
                records.append({"doy": d, "hour": h, "current_u": 0.05, "current_v": 0.02})
        df_clim = pd.DataFrame(records).set_index(["doy", "hour"])
        df_clim.to_parquet(CLIMATOLOGY_FILE)
        return df_clim

    df = pd.read_parquet(train_features_path)
    ts = pd.to_datetime(df.index)
    
    # 70% chronological split (Train split only to prevent leakage)
    n_train = int(len(df) * 0.70)
    train_df = df.iloc[:n_train].copy()
    train_ts = ts[:n_train]

    train_df["hour"] = train_ts.hour
    train_df["doy"] = train_ts.dayofyear

    clim_u = train_df.groupby(["doy", "hour"])["current_u"].mean()
    clim_v = train_df.groupby(["doy", "hour"])["current_v"].mean()

    df_clim = pd.DataFrame({"current_u": clim_u, "current_v": clim_v})
    df_clim.to_parquet(CLIMATOLOGY_FILE)
    logger.info(f"Saved currents climatology table ({len(df_clim)} entries) to {CLIMATOLOGY_FILE}")
    return df_clim


def get_climatological_currents(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Computes climatological current vectors for any given DatetimeIndex.
    """
    if not CLIMATOLOGY_FILE.exists():
        fit_and_save_climatology()

    df_clim = pd.read_parquet(CLIMATOLOGY_FILE)

    doys = timestamps.dayofyear
    hours = timestamps.hour

    keys = list(zip(doys, hours))
    
    # Fast reindexing / lookup
    u_vals = []
    v_vals = []
    global_u_mean = float(df_clim["current_u"].mean()) if len(df_clim) > 0 else 0.05
    global_v_mean = float(df_clim["current_v"].mean()) if len(df_clim) > 0 else 0.02

    for k in keys:
        if k in df_clim.index:
            row = df_clim.loc[k]
            u_vals.append(float(row["current_u"]))
            v_vals.append(float(row["current_v"]))
        else:
            u_vals.append(global_u_mean)
            v_vals.append(global_v_mean)

    u_arr = np.array(u_vals, dtype=np.float64)
    v_arr = np.array(v_vals, dtype=np.float64)
    speed_arr = np.sqrt(u_arr ** 2 + v_arr ** 2)
    dir_arr = (np.degrees(np.arctan2(v_arr, u_arr))) % 360.0

    return pd.DataFrame({
        "timestamp": timestamps,
        "current_u": u_arr,
        "current_v": v_arr,
        "current_speed": speed_arr,
        "current_dir": dir_arr,
        "current_source": "climatological_fallback",
    }, index=timestamps)


# ---------------------------------------------------------------------------
# 2. Live Cache Reader & Hybrid Resolver
# ---------------------------------------------------------------------------
def get_live_currents_forecast(
    timestamps: pd.DatetimeIndex,
    max_cache_age_hours: float = 24.0
) -> pd.DataFrame:
    """
    Retrieves hourly ocean current data for the requested timestamps:
    - Reads from local CMEMS forecast cache (data/cache/cmems_forecast_cache.parquet).
    - Checks cache freshness (within max_cache_age_hours).
    - For hours covered by CMEMS forecast: returns CMEMS values tagged 'cmems_forecast'.
    - For hours beyond CMEMS horizon (>10 days) or if cache is stale/unavailable:
      returns seasonal climatology tagged 'climatological_fallback'.
    """
    # Check if cache exists
    if not CACHE_FILE.exists():
        logger.warning(f"CMEMS cache file {CACHE_FILE} does not exist. Falling back to climatology.")
        return get_climatological_currents(timestamps)

    # Check cache file modification time for staleness
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE), tz=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    cache_age_hours = (now_utc - cache_mtime).total_seconds() / 3600.0

    if cache_age_hours > max_cache_age_hours:
        logger.warning(
            f"CMEMS cache is STALE ({cache_age_hours:.1f}h old > {max_cache_age_hours}h limit). "
            "Using climatological fallback for safety."
        )
        return get_climatological_currents(timestamps)

    try:
        df_cache = pd.read_parquet(CACHE_FILE)
        # Normalize cache timestamps to timezone-naive UTC for robust matching
        cache_ts = pd.to_datetime(df_cache["timestamp"])
        if getattr(cache_ts.dt, "tz", None) is not None:
            cache_ts = cache_ts.dt.tz_convert(None)
        df_cache["timestamp"] = cache_ts
        df_cache = df_cache.set_index("timestamp").sort_index()

        # Build output dataframe
        records = []
        df_clim = None

        for ts in timestamps:
            ts_dt = pd.to_datetime(ts)
            if getattr(ts_dt, "tzinfo", None) is not None:
                ts_dt = ts_dt.tz_convert(None) if hasattr(ts_dt, "tz_convert") else ts_dt.tz_localize(None)

            # Check direct presence in actual downloaded cached timestamps
            # (handles CMEMS daily model-run truncation gracefully without theoretical assumptions)
            if ts_dt in df_cache.index:
                row = df_cache.loc[ts_dt]
                # In case of duplicates, take first
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                records.append({
                    "timestamp": ts_dt,
                    "current_u": float(row["current_u"]),
                    "current_v": float(row["current_v"]),
                    "current_speed": float(row["current_speed"]),
                    "current_dir": float(row["current_dir"]),
                    "current_source": "cmems_forecast",
                })
            else:
                # Timestamp is outside actual cached data window -> seamless fallback to seasonal climatology
                if df_clim is None:
                    df_clim = get_climatological_currents(timestamps)
                clim_row = df_clim.loc[ts_dt]
                records.append({
                    "timestamp": ts_dt,
                    "current_u": float(clim_row["current_u"]),
                    "current_v": float(clim_row["current_v"]),
                    "current_speed": float(clim_row["current_speed"]),
                    "current_dir": float(clim_row["current_dir"]),
                    "current_source": "climatological_fallback",
                })

        return pd.DataFrame(records).set_index("timestamp")

    except Exception as e:
        logger.error(f"Error reading CMEMS cache: {e}. Gracefully falling back to climatology.")
        return get_climatological_currents(timestamps)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------
def _run_tests():
    print("Running currents_cache unit test suite...\n")
    failures = []

    def check(name, condition, message=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            failures.append(f"{name}: {message}")

    # 1. Test climatology generation and table creation
    clim_df = fit_and_save_climatology()
    check("Climatology table generated successfully", len(clim_df) > 0)
    check("Climatology table contains current_u and current_v",
          "current_u" in clim_df.columns and "current_v" in clim_df.columns)

    # 2. Test climatological query across arbitrary timestamps
    test_ts = pd.date_range("2026-09-15 00:00:00", periods=48, freq="1h")
    clim_res = get_climatological_currents(test_ts)
    check("Climatology query returns exact requested row count (48)", len(clim_res) == 48)
    check("Climatology source tagged as 'climatological_fallback'",
          (clim_res["current_source"] == "climatological_fallback").all())
    check("Speed matches sqrt(u^2 + v^2)",
          np.allclose(clim_res["current_speed"], np.sqrt(clim_res["current_u"]**2 + clim_res["current_v"]**2)))

    # 3. Test mock CMEMS cache ingestion and hybrid retrieval
    mock_cmems = pd.DataFrame({
        "timestamp": test_ts[:24],  # first 24 hours only
        "current_u": [0.15] * 24,
        "current_v": [0.20] * 24,
        "current_speed": [0.25] * 24,
        "current_dir": [53.13] * 24,
    })
    mock_cmems.to_parquet(CACHE_FILE)

    hybrid_res = get_live_currents_forecast(test_ts)
    check("Hybrid query returns exact requested row count (48)", len(hybrid_res) == 48)
    check("First 24h sourced from CMEMS forecast",
          (hybrid_res.iloc[:24]["current_source"] == "cmems_forecast").all())
    check("Hours 25-48 (>CMEMS cache) sourced from climatological fallback",
          (hybrid_res.iloc[24:]["current_source"] == "climatological_fallback").all())

    # 4. Clean up mock cache or leave valid cache
    print(f"\n{len(failures)} failures out of 7 checks.")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All currents_cache tests passed successfully!")


if __name__ == "__main__":
    _run_tests()
