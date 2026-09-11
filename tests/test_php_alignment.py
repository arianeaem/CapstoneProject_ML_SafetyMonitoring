"""
Unit tests comparing the pandas/vectorized label generation functions in
src/labels/build_safety_labels.py against a literal, unoptimized ground-truth
reimplementation of PHP's WeatherForecastService.php methods.

Tests boundary conditions, inclusive vs exclusive edges, descending bin orders,
joint wind conditions, and wind direction sequential matching.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labels.build_safety_labels import (
    score_wave_height,
    score_wind_wave_height,
    score_wave_period,
    score_ocean_current,
    score_wind_speed,
    score_rain,
    score_sea_level_pressure,
    score_wind_direction,
    compute_subscores,
    compute_final_scores,
    score_to_tier,
    KMH_TO_MS,
    WEIGHTS,
    SYNERGY_HAZARD_FEATURES,
)


# ===========================================================================
# 1. Literal, unoptimized PHP ground-truth reference functions
# ===========================================================================

def php_score_wave_height(v: float) -> int:
    # PHP: <0.30 -> 0, <0.50 -> 1, <0.80 -> 2, <1.00 -> 3, else 4
    if v < 0.30:
        return 0
    elif v < 0.50:
        return 1
    elif v < 0.80:
        return 2
    elif v < 1.00:
        return 3
    else:
        return 4


def php_score_wind_wave_height(v: float) -> int:
    # PHP: <0.20 -> 0, <0.40 -> 1, <0.60 -> 2, <0.80 -> 3, else 4
    if v < 0.20:
        return 0
    elif v < 0.40:
        return 1
    elif v < 0.60:
        return 2
    elif v < 0.80:
        return 3
    else:
        return 4


def php_score_wave_period(v: float) -> int:
    # PHP: >7.0 -> 0, >5.0 -> 1, >3.0 -> 2, >2.0 -> 3, else 4
    if v > 7.0:
        return 0
    elif v > 5.0:
        return 1
    elif v > 3.0:
        return 2
    elif v > 2.0:
        return 3
    else:
        return 4


def php_score_ocean_current(v: float) -> int:
    # PHP: <0.10 -> 0, <0.30 -> 1, <0.50 -> 2, <0.80 -> 3, else 4
    if v < 0.10:
        return 0
    elif v < 0.30:
        return 1
    elif v < 0.50:
        return 2
    elif v < 0.80:
        return 3
    else:
        return 4


def php_score_wind_speed(sustained_kmh: float, gust_kmh: float) -> int:
    # PHP joint condition:
    # if sustained < 12.0 and gust < 18.0 -> 0
    # elseif sustained < 20.0 and gust < 28.0 -> 1
    # elseif sustained < 28.0 and gust < 38.0 -> 2
    # elseif sustained < 38.0 and gust < 48.0 -> 3
    # else -> 4
    if sustained_kmh < 12.0 and gust_kmh < 18.0:
        return 0
    elif sustained_kmh < 20.0 and gust_kmh < 28.0:
        return 1
    elif sustained_kmh < 28.0 and gust_kmh < 38.0:
        return 2
    elif sustained_kmh < 38.0 and gust_kmh < 48.0:
        return 3
    else:
        return 4


def php_score_rain(v: float) -> int:
    # PHP: <0.2 -> 0, <2.5 -> 1, <7.5 -> 2, <20.0 -> 3, else 4
    if v < 0.2:
        return 0
    elif v < 2.5:
        return 1
    elif v < 7.5:
        return 2
    elif v < 20.0:
        return 3
    else:
        return 4


def php_score_sea_level_pressure(v: float) -> int:
    # PHP: >=1012.0 -> 0, >=1008.0 -> 1, >=1004.0 -> 2, >=1000.0 -> 3, else 4
    if v >= 1012.0:
        return 0
    elif v >= 1008.0:
        return 1
    elif v >= 1004.0:
        return 2
    elif v >= 1000.0:
        return 3
    else:
        return 4


def php_score_wind_direction(deg: float) -> int:
    # PHP sequential if-elseif:
    # if (d >= 0 and d <= 90) or d > 315 -> 0
    # elseif d <= 135 -> 1
    # elseif d <= 180 or d > 270 -> 2
    # else -> 3
    d = deg % 360
    if ((d >= 0 and d <= 90) or d > 315):
        return 0
    elif d <= 135:
        return 1
    elif d <= 180 or d > 270:
        return 2
    else:
        return 3


# ===========================================================================
# 2. Test Execution
# ===========================================================================

def test_wave_height():
    test_vals = [0.0, 0.29, 0.30, 0.31, 0.49, 0.50, 0.51, 0.79, 0.80, 0.81, 0.99, 1.00, 1.01, 2.5]
    s = pd.Series(test_vals)
    pandas_res = score_wave_height(s).tolist()
    php_res = [php_score_wave_height(v) for v in test_vals]
    assert pandas_res == php_res, f"Wave height mismatch: {pandas_res} vs {php_res}"
    print("[PASS] score_wave_height matches PHP reference across all boundaries.")


def test_wind_wave_height():
    test_vals = [0.0, 0.19, 0.20, 0.21, 0.39, 0.40, 0.41, 0.59, 0.60, 0.61, 0.79, 0.80, 0.81, 2.0]
    s = pd.Series(test_vals)
    pandas_res = score_wind_wave_height(s).tolist()
    php_res = [php_score_wind_wave_height(v) for v in test_vals]
    assert pandas_res == php_res, f"Wind wave height mismatch: {pandas_res} vs {php_res}"
    print("[PASS] score_wind_wave_height matches PHP reference across all boundaries.")


def test_wave_period():
    # Boundary points: 7.0, 5.0, 3.0, 2.0 and edges
    test_vals = [0.5, 1.9, 2.0, 2.01, 2.1, 2.9, 3.0, 3.01, 3.1, 4.9, 5.0, 5.01, 5.1, 6.9, 7.0, 7.01, 7.1, 15.0]
    s = pd.Series(test_vals)
    pandas_res = score_wave_period(s).tolist()
    php_res = [php_score_wave_period(v) for v in test_vals]
    assert pandas_res == php_res, f"Wave period mismatch: {pandas_res} vs {php_res}"
    print("[PASS] score_wave_period matches PHP reference across all descending boundaries.")


def test_ocean_current():
    test_vals = [0.0, 0.09, 0.10, 0.11, 0.29, 0.30, 0.31, 0.49, 0.50, 0.51, 0.79, 0.80, 0.81, 1.5]
    s = pd.Series(test_vals)
    pandas_res = score_ocean_current(s).tolist()
    php_res = [php_score_ocean_current(v) for v in test_vals]
    assert pandas_res == php_res, f"Ocean current mismatch: {pandas_res} vs {php_res}"
    print("[PASS] score_ocean_current matches PHP reference across all boundaries.")


def test_wind_speed():
    # Test matched pairs, boundary values, and mismatched pairs (calm sustained with high gust)
    pairs = [
        (5.0, 10.0),    # Tier 0
        (11.9, 17.9),  # Tier 0 (just under)
        (12.0, 10.0),  # Tier 1 (sustained triggers tier 1)
        (5.0, 18.0),   # Tier 1 (gust triggers tier 1)
        (19.9, 27.9),  # Tier 1 (just under)
        (20.0, 20.0),  # Tier 2
        (10.0, 28.0),  # Tier 2 (gust trigger)
        (27.9, 37.9),  # Tier 2 (just under)
        (28.0, 20.0),  # Tier 3 (sustained trigger)
        (10.0, 38.0),  # Tier 3 (gust trigger)
        (37.9, 47.9),  # Tier 3 (just under)
        (38.0, 20.0),  # Tier 4 (sustained breach)
        (10.0, 48.0),  # Tier 4 (gust breach)
        (50.0, 70.0),  # Tier 4 (extreme)
    ]
    sustained_kmh = [p[0] for p in pairs]
    gust_kmh = [p[1] for p in pairs]

    s_ms = pd.Series([v * KMH_TO_MS for v in sustained_kmh])
    g_ms = pd.Series([v * KMH_TO_MS for v in gust_kmh])

    pandas_res = score_wind_speed(s_ms, g_ms).tolist()
    php_res = [php_score_wind_speed(s, g) for s, g in pairs]
    assert pandas_res == php_res, f"Wind speed mismatch: {pandas_res} vs {php_res}"
    print("[PASS] score_wind_speed joint evaluation matches PHP reference across all combinations.")


def test_rain():
    test_vals = [0.0, 0.19, 0.20, 0.21, 2.49, 2.50, 2.51, 7.49, 7.50, 7.51, 19.9, 20.0, 20.1, 50.0]
    s = pd.Series(test_vals)
    pandas_res = score_rain(s).tolist()
    php_res = [php_score_rain(v) for v in test_vals]
    assert pandas_res == php_res, f"Rain mismatch: {pandas_res} vs {php_res}"
    print("[PASS] score_rain matches PHP reference across all boundaries.")


def test_sea_level_pressure():
    # Boundary points: 1012, 1008, 1004, 1000 and edges
    test_vals = [990.0, 999.9, 1000.0, 1000.1, 1003.9, 1004.0, 1004.1, 1007.9, 1008.0, 1008.1, 1011.9, 1012.0, 1012.1, 1025.0]
    s = pd.Series(test_vals)
    pandas_res = score_sea_level_pressure(s).tolist()
    php_res = [php_score_sea_level_pressure(v) for v in test_vals]
    assert pandas_res == php_res, f"SLP mismatch: {pandas_res} vs {php_res}"
    print("[PASS] score_sea_level_pressure matches PHP reference across all descending boundaries.")


def test_wind_direction():
    # Test all 16 cardinal points and exact transition angles: 0, 45, 90, 91, 135, 136, 180, 181, 225, 270, 271, 315, 316, 359, 360, 405 (-45)
    test_vals = [0.0, 45.0, 90.0, 90.1, 110.0, 135.0, 135.1, 160.0, 180.0, 180.1, 225.0, 269.9, 270.0, 270.1, 300.0, 315.0, 315.1, 330.0, 359.9, 360.0, 720.0]
    s = pd.Series(test_vals)
    pandas_res = score_wind_direction(s).tolist()
    php_res = [php_score_wind_direction(v) for v in test_vals]
    assert pandas_res == php_res, f"Wind direction mismatch: {pandas_res} vs {php_res}"
    print("[PASS] score_wind_direction sequential if-chain matches PHP reference across all angles.")


def test_synergy_and_weights():
    # Ensure weights sum to 1.000
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.000"
    assert len(SYNERGY_HAZARD_FEATURES) == 5, "Synergy hazards must be exactly the 5-feature subset"
    print("[PASS] WEIGHTS sum to 1.000 and SYNERGY_HAZARD_FEATURES is 5-feature subset.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHP ALIGNMENT VERIFICATION TEST SUITE")
    print("=" * 70)
    test_wave_height()
    test_wind_wave_height()
    test_wave_period()
    test_ocean_current()
    test_wind_speed()
    test_rain()
    test_sea_level_pressure()
    test_wind_direction()
    test_synergy_and_weights()
    print("=" * 70)
    print("ALL PHP ALIGNMENT TESTS PASSED PERFECTLY!")
    print("=" * 70)
