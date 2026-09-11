"""
Source: NASA GPM IMERG Final Run V07B (GPM_3IMERGHH.07)
Access: Remote spatial subsetting via NASA OPeNDAP DAP2

Optimized Architecture:
- Concurrent ThreadPoolExecutor with request jitter to prevent burst 503 throttling.
- Exponential backoff retry with random jitter on 503/429/502/504 errors.
- Incremental batch checkpointing: saves to NetCDF every batch so progress is
  never lost if paused or interrupted.
- Full auto-resume: skips timestamps already present on disk.
- Scientific integrity: missing values preserved as NaN; no blind interpolation.
"""

from datetime import timedelta
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import xarray as xr

from config import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, START_DATE, END_DATE, RAW_DIR, INTERIM_DIR

RAW_SUBDIR = f"{RAW_DIR}/gpm_precip"
OUT_RAW = f"{RAW_SUBDIR}/gpm_precip_raw.nc"
OUT_INTERIM = f"{INTERIM_DIR}/gpm_precip.parquet"

LAT_IDX_MIN = int(np.floor((LAT_MIN + 89.95) / 0.1))
LAT_IDX_MAX = int(np.ceil((LAT_MAX + 89.95) / 0.1))
LON_IDX_MIN = int(np.floor((LON_MIN + 179.95) / 0.1))
LON_IDX_MAX = int(np.ceil((LON_MAX + 179.95) / 0.1))

LAT_COORDS = np.array([round(-89.95 + idx * 0.1, 2) for idx in range(LAT_IDX_MIN, LAT_IDX_MAX + 1)], dtype=np.float32)
LON_COORDS = np.array([round(-179.95 + idx * 0.1, 2) for idx in range(LON_IDX_MIN, LON_IDX_MAX + 1)], dtype=np.float32)


def create_session(pool_size: int = 20) -> requests.Session:
    """Configures a thread-safe requests session with sufficient connection pooling."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=max(pool_size * 2, 20),
        pool_maxsize=max(pool_size * 2, 20),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_opendap_url(dt: pd.Timestamp) -> str:
    year = dt.strftime("%Y")
    doy = dt.strftime("%j")
    d_str = dt.strftime("%Y%m%d")
    s_str = dt.strftime("%H%M%S")
    dt_end = dt + timedelta(minutes=29, seconds=59)
    e_str = dt_end.strftime("%H%M%S")
    minutes = dt.hour * 60 + dt.minute
    fn = f"3B-HHR.MS.MRG.3IMERG.{d_str}-S{s_str}-E{e_str}.{minutes:04d}.V07B.HDF5"

    base_url = f"https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/{year}/{doy}/{fn}"
    constraint = f"precipitation[0:1:0][{LON_IDX_MIN}:1:{LON_IDX_MAX}][{LAT_IDX_MIN}:1:{LAT_IDX_MAX}]"
    return f"{base_url}.ascii?{constraint}"


def fetch_slice_grid(dt: pd.Timestamp, session: requests.Session, max_retries: int = 5) -> np.ndarray:
    url = build_opendap_url(dt)
    for attempt in range(1, max_retries + 1):
        try:
            # Request jitter to avoid synchronous burst pressure on NASA GES DISC
            time.sleep(random.uniform(0.04, 0.15))
            response = session.get(url, timeout=25)
            if response.status_code in (429, 502, 503, 504):
                sleep_time = (1.5 ** attempt) + random.uniform(0.5, 2.0)
                time.sleep(sleep_time)
                continue

            response.raise_for_status()
            rows = []
            for line in response.text.splitlines():
                if line.startswith("precipitation["):
                    parts = line.split(",", 1)
                    if len(parts) > 1:
                        vals = [float(x.strip()) for x in parts[1].split(",") if x.strip()]
                        vals = [v if v >= 0.0 else np.nan for v in vals]
                        rows.append(vals)
            if len(rows) == len(LON_COORDS):
                return np.array(rows, dtype=np.float32)
        except Exception as e:
            if attempt == max_retries:
                print(f"  warning: failed to fetch {dt} after {max_retries} attempts: {e}")
            time.sleep((1.5 ** attempt) + random.uniform(0.5, 1.5))

    return np.full((len(LON_COORDS), len(LAT_COORDS)), np.nan, dtype=np.float32)


def load_existing_timestamps() -> tuple[set, xr.Dataset | None]:
    """Loads existing timestamps from disk into memory and closes file handles."""
    if not os.path.exists(OUT_RAW):
        return set(), None
    try:
        with xr.open_dataset(OUT_RAW) as ds:
            existing_ds = ds.load()
        existing_times = set(pd.to_datetime(existing_ds.time.values))
        return existing_times, existing_ds
    except Exception as e:
        print(f"Notice: Could not load existing NetCDF file ({e}), starting fresh.")
        return set(), None


def validate_dataset(ds: xr.Dataset):
    raw_arr = np.asarray(ds["precipitation"].values, dtype=np.float64).ravel()
    raw_vals: list[float] = [float(x) for x in raw_arr.tolist()]
    valid_vals = [x for x in raw_vals if not np.isnan(x)]
    total_points = len(raw_vals)
    nan_count = total_points - len(valid_vals)
    missing_pct = (nan_count / total_points) * 100.0 if total_points > 0 else 0.0
    val_min = min(valid_vals) if len(valid_vals) > 0 else float("nan")
    val_max = max(valid_vals) if len(valid_vals) > 0 else float("nan")
    val_mean = sum(valid_vals) / len(valid_vals) if len(valid_vals) > 0 else float("nan")

    print("\n" + "=" * 50)
    print("GPM IMERG V07B Dataset Validation:")
    print(f"  Time range:     {str(ds.time.values[0])[:19]} to {str(ds.time.values[-1])[:19]}")
    print(f"  Time steps:     {len(ds.time)} half-hourly timestamps")
    print(f"  Rain rate min:  {val_min:.3f} mm/hr")
    print(f"  Rain rate max:  {val_max:.3f} mm/hr")
    print(f"  Rain rate mean: {val_mean:.3f} mm/hr")
    print(f"  Missing (NaN):  {nan_count}/{total_points} cells ({missing_pct:.2f}%)")
    print("=" * 50 + "\n")


def download(start_date: str = START_DATE, end_date: str = END_DATE,
             test_days: int = 0, max_workers: int = 20, batch_size: int = 720) -> str:
    """
    Subsets GPM IMERG with concurrent workers, exponential retry backoff,
    and incremental checkpoint saving every batch_size timestamps.
    """
    os.makedirs(RAW_SUBDIR, exist_ok=True)

    if test_days > 0:
        target_start = f"{start_date} 00:00:00"
        target_end = (pd.Timestamp(start_date) + timedelta(days=test_days) - timedelta(minutes=30))
        print(f"--- TEST MODE: {test_days} day(s) from {start_date} ---")
    else:
        target_start = f"{start_date} 00:00:00"
        target_end = f"{end_date} 23:30:00"
        print(f"--- Full ingest: {start_date} to {end_date} ---")

    all_timestamps = pd.date_range(start=target_start, end=target_end, freq="30min")

    existing_times, combined = load_existing_timestamps()
    remaining = [t for t in all_timestamps if t not in existing_times]
    skipped = len(all_timestamps) - len(remaining)
    if skipped > 0:
        print(f"Resuming: {skipped} timestamps already on disk, fetching {len(remaining)} remaining.")

    if not remaining:
        print("Nothing to fetch — all timestamps already present.")
        if combined is not None:
            validate_dataset(combined)
        return OUT_RAW

    est_minutes = len(remaining) * 1.5 / max_workers / 60
    print(f"Fetching {len(remaining)} half-hourly granules with {max_workers} concurrent workers "
          f"(estimate: ~{est_minutes:.0f} min)...")

    session = create_session(pool_size=max_workers)

    # Process in batches for incremental saving to disk
    num_batches = int(np.ceil(len(remaining) / batch_size))
    total_pbar = tqdm(total=len(remaining), desc="OPeNDAP Subsetting", dynamic_ncols=True)

    for b in range(num_batches):
        batch_timestamps = remaining[b * batch_size : (b + 1) * batch_size]
        grids: list[np.ndarray] = [np.zeros((len(LON_COORDS), len(LAT_COORDS)), dtype=np.float32) for _ in batch_timestamps]

        def _fetch(i_dt):
            i, dt = i_dt
            return i, fetch_slice_grid(dt, session)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fetch, (i, dt)) for i, dt in enumerate(batch_timestamps)]
            for future in as_completed(futures):
                i, grid = future.result()
                grids[i] = grid
                total_pbar.update(1)

        batch_3d = np.array(grids, dtype=np.float32)
        batch_ds = xr.Dataset(
            data_vars={"precipitation": (["time", "lon", "lat"], batch_3d)},
            coords={"time": pd.DatetimeIndex(batch_timestamps), "lon": LON_COORDS, "lat": LAT_COORDS},
            attrs={
                "title": "NASA GPM IMERG Final Run V07B Remote Spatial Subset",
                "region": "Balayan Bay / Verde Island Passage",
                "units": "mm/hr",
            },
        )

        if combined is not None:
            combined = xr.concat([combined, batch_ds], dim="time").sortby("time")
        else:
            combined = batch_ds

        # Checkpoint save after every batch
        temp_out = f"{OUT_RAW}.tmp"
        combined.to_netcdf(temp_out)
        if os.path.exists(OUT_RAW):
            os.remove(OUT_RAW)
        os.rename(temp_out, OUT_RAW)

    total_pbar.close()
    if combined is not None:
        validate_dataset(combined)
        size_kb = os.path.getsize(OUT_RAW) / 1024.0
        print(f"Saved -> {OUT_RAW} ({size_kb:.1f} KB, {len(combined.time)} total timestamps)")
    return OUT_RAW


def to_interim(raw_source: str = OUT_RAW) -> str:
    os.makedirs(INTERIM_DIR, exist_ok=True)
    if not os.path.exists(raw_source):
        raise FileNotFoundError(f"Raw subset file not found: {raw_source}")

    ds = xr.open_dataset(raw_source)
    regional_30min = ds["precipitation"].mean(dim=["lat", "lon"], skipna=True)
    df_30min = regional_30min.to_dataframe(name="rain_rate_mm_hr")[["rain_rate_mm_hr"]]

    df_hourly = df_30min.resample("1h").mean()
    df_hourly.index.name = "timestamp"

    nan_hours = df_hourly["rain_rate_mm_hr"].isna().sum()
    total_hours = len(df_hourly)
    print(f"Hourly aggregation: {total_hours} rows ({nan_hours} NaN, {nan_hours/total_hours*100:.1f}%)")

    df_hourly.to_parquet(OUT_INTERIM)
    print(f"wrote {len(df_hourly)} hourly rows -> {OUT_INTERIM}")
    return OUT_INTERIM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA GPM IMERG Ingestion via Remote OPeNDAP Subsetting")
    parser.add_argument("--full", action="store_true", help="Run full 3-year historical ingestion.")
    parser.add_argument("--test-days", type=int, default=1,
                         help="Number of days to test with when --full is not passed (default: 1).")
    parser.add_argument("--workers", type=int, default=20,
                         help="Concurrent request threads (default: 20).")
    args = parser.parse_args()

    test_days_arg = 0 if args.full else args.test_days
    raw_path = download(test_days=test_days_arg, max_workers=args.workers)
    to_interim(raw_path)