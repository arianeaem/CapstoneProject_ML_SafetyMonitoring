"""
Constructs multi-horizon forward targets and stacks horizons into long format.

Horizons:
- 1h (dockside Go/No-Go), 6h, 12h (half-day dispatch), 24h (1-day),
  48h (2-day weekend), 72h (3-day), 96h (4-day), 144h (6-day weekly outlook).
- Stacks all 8 horizons into one long-format table with `horizon` as a feature,
  allowing a single model per variable to serve all horizons.
- Decomposes circular wind_dir into target_wind_dir_sin and target_wind_dir_cos.
- Sorts by index (timestamp) so temporal splits preserve exact chronological ordering
  across all horizons simultaneously.
"""

import numpy as np
import pandas as pd

HORIZONS = [1, 6, 12, 24, 48, 72, 96, 144]  # hours: 1h, 6h, 12h, 24h, 48h, 72h, 96h, 144h
TARGET_VARS = [
    "hs", "tp", "swell_height", "wind_wave_height",
    "wind_speed", "wind_gust", "wind_dir", "slp",
    "current_u", "current_v",
]


def build_stacked_dataset(df: pd.DataFrame, lagged_features: pd.DataFrame) -> pd.DataFrame:
    stacked = []
    for h in HORIZONS:
        block = lagged_features.copy()
        block["horizon"] = h
        for var in TARGET_VARS:
            block[f"target_{var}"] = df[var].shift(-h)  # value h hours AFTER this row

        # Sin/cos decomposition for circular wind direction at t+H
        if "wind_dir" in df.columns:
            rad = np.radians(df["wind_dir"].shift(-h))
            block["target_wind_dir_sin"] = np.sin(rad)
            block["target_wind_dir_cos"] = np.cos(rad)

        stacked.append(block)

    result = pd.concat(stacked, axis=0)
    # Drop rows where either the lag/rolling window or the horizon shift
    # produced NaN — these are the unavoidable edges of the dataset
    # (very start: not enough history; very end: not enough future).
    # .sort_index() ensures all horizons for time t stay chronologically ordered.
    result = result.dropna().sort_index()
    return result
