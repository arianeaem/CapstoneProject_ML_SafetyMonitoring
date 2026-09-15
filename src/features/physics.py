"""
Physics-Informed Feature Engineering module.
Reusable module called during both offline training and live inference.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def compute_marine_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-informed marine and atmospheric interaction features:
    - Wave steepness (Hs / wavelength proxy)
    - Swell ratio (swell energy fraction)
    - 3-hour barometric pressure tendency (delta_p_3h)
    - Wind-current / wind-wave directional alignment
    - Diurnal and seasonal cyclical temporal encodings (hour_sin/cos, doy_sin/cos)

    Note: Tide-dependent features (tidal_rate, slack_tide_proxy) are intentionally
    excluded per scope reduction.
    """
    df = df.copy()
    g = 9.80665

    # 1. Wave Steepness: H / L proxy ~ (2 * pi * Hs) / (g * Tp^2)
    df["wave_steepness"] = (2 * np.pi * df["hs"]) / (g * (df["tp"] ** 2) + 1e-6)

    # 2. Swell Ratio: Fraction of sea surface energy driven by long-period swell
    df["swell_ratio"] = df["swell_height"] / (df["hs"] + 1e-5)

    # 3. Barometric Pressure Tendency: 3-hour SLP change (ΔP3h) in hPa
    df["delta_p_3h"] = df["slp"] - df["slp"].shift(3)

    # 4. Directional Alignments: Angular difference on [0, 180] deg circle
    if "wave_dir" in df.columns:
        angle_diff_wave = np.abs(df["wind_dir"] - df["wave_dir"])
        df["wind_wave_alignment"] = np.minimum(angle_diff_wave, 360 - angle_diff_wave)

    if "current_dir" in df.columns:
        angle_diff_curr = np.abs(df["wind_dir"] - df["current_dir"])
        df["wind_current_alignment"] = np.minimum(angle_diff_curr, 360 - angle_diff_curr)

    # 5. Cyclical Temporal Encodings from DatetimeIndex
    if hasattr(df.index, "hour"):
        hours = df.index.hour
        doy = df.index.dayofyear
    elif "hour" in df.columns and "day_of_year" in df.columns:
        hours = df["hour"]
        doy = df["day_of_year"]
    else:
        time_col = pd.to_datetime(df.index)
        hours = time_col.hour
        doy = time_col.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df
