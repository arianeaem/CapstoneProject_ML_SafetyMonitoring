"""
Builds lagged and rolling historical features for multi-step-ahead forecasting.

Features:
- RAW_VARS at LAG_HOURS [0, 1, 3, 6, 12, 24, 48]
  (0 = current observed value at time t; 1..48 = historical lags)
- Trailing 24h rolling statistics (mean, std, min, max) computed on .shift(1)
  so that rolling stats cover t-1 back to t-24 without touching t itself.
"""

import pandas as pd

RAW_VARS = [
    "hs", "tp", "swell_height", "wind_wave_height", "current_u", "current_v",
    "wind_u", "wind_v", "wind_speed", "wind_gust", "slp", "rain_rate_mm_hr",
]

LAG_HOURS = [0, 1, 3, 6, 12, 24, 48]  # 0 = the current/most-recent observed value.
# Including lag=0 is deliberate, not an oversight to avoid: it's fully valid,
# non-leaking information (it's KNOWN at time t, same as any other lag) when
# forecasting t+H for any H >= 1. Omitting it would handicap the model
# relative to the persistence baseline it's required to beat in Step 4 —
# persistence IS essentially "just use lag=0," so the model needs access to
# that same information at minimum to have a fair chance of doing better.
ROLLING_WINDOW = 24


def build_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}

    for var in RAW_VARS:
        for lag in LAG_HOURS:
            cols[f"{var}_lag{lag}h"] = df[var].shift(lag)

        # .shift(1) BEFORE .rolling() is deliberate: a rolling window computed
        # directly on df[var] would include the CURRENT hour in its own "history"
        # — shifting first means the window covers t-1 back to t-ROLLING_WINDOW,
        # never touching t itself. This is the exact leakage mistake this whole
        # project has already caught and fixed once (the original random-split
        # bug); don't reintroduce a subtler version of it here.
        shifted = df[var].shift(1)
        cols[f"{var}_roll_mean{ROLLING_WINDOW}h"] = shifted.rolling(ROLLING_WINDOW).mean()
        cols[f"{var}_roll_std{ROLLING_WINDOW}h"] = shifted.rolling(ROLLING_WINDOW).std()
        cols[f"{var}_roll_min{ROLLING_WINDOW}h"] = shifted.rolling(ROLLING_WINDOW).min()
        cols[f"{var}_roll_max{ROLLING_WINDOW}h"] = shifted.rolling(ROLLING_WINDOW).max()

    return pd.DataFrame(cols, index=df.index)
