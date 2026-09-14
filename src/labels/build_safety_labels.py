"""
Safety label generation, ALIGNED EXACTLY to PHP's WeatherForecastService.php —
weights, sub-scoring bands, and physical safety thresholds all matched to the live
PHP implementation, not independently re-derived. This is what makes the two
parallel paths (native PHP scorer using live Open-Meteo directly, and this
ML path using our own forecaster models) comparable in the way the panel
presentation needs them to be.

KEY METEOROLOGICAL & SYSTEM DESIGN ALIGNMENTS:

1. Wind safety threshold: Standardized on 38.0 km/h sustained (10.56 m/s)
   and 48.0 km/h gusts (13.33 m/s), matching updateAllForecasts() and
   this project's safety_thresholds.py.

2. 3-Hour Pressure Drop Omission: The 3-hour pressure drop limit (>= 2.0 hPa)
   is deliberately EXCLUDED from the window-level safety thresholds, matching PHP's
   operative evaluateWindowNatively(). In tropical maritime environments like
   the Philippines, the semi-diurnal atmospheric solar tide (S2 oscillation)
   routinely causes natural 2.0+ hPa barometric pressure swings over 3-4 hour
   windows twice daily (10:00->16:00 and 22:00->04:00) during calm, clear weather.
   Enforcing a 3-hour pressure drop threshold at hourly resolution acts as a
   massive false-positive generator. Extreme low pressure is correctly guarded
   by the absolute slp <= 998.0 hPa threshold and slp sub-scoring.

Run from the project root: python src/labels/build_safety_labels.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "training_features.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "safety_labels.parquet"

KMH_TO_MS = 1000 / 3600
TIER_NAMES = ["Very Safe", "Safe", "Moderate", "High Risk", "Critical Risk"]

# ---------------------------------------------------------------------------
# Critical Safety Thresholds — matched to PHP's operational window-level limits
# (evaluateWindowNatively), converted to this project's standard SI units
# (m/s, meters, hPa, mm/hr).
# ---------------------------------------------------------------------------
SAFETY_THRESHOLDS = {
    "wind_speed_ms": 38.0 * KMH_TO_MS,  # 38 km/h sustained
    "wind_gust_ms": 48.0 * KMH_TO_MS,   # 48 km/h gusts
    "wave_height_m": 1.80,              # 1.80 m significant wave height
    "swell_height_m": 1.80,             # 1.80 m swell height
    "current_ms": 0.80,                 # 0.80 m/s ocean current speed
    "rain_mm_hr": 25.0,                 # 25.0 mm/hr heavy rain rate
    "pressure_hpa": 998.0,              # <= 998.0 hPa cyclone / severe low
}

# Backward-compatibility alias
HARD_GATE = SAFETY_THRESHOLDS


def check_safety_thresholds(df: pd.DataFrame) -> pd.Series:
    """
    Evaluates raw physical marine readings against mandatory physical safety limits.

    Business Logic / Rationale:
        Physical threshold enforcement guarantees that regardless of ML classifier
        confidence, any severe single-variable oceanographic hazard (e.g., gale-force
        gusts >= 48 km/h or sea state >= 1.8m) instantly triggers a Critical Risk verdict.
        This provides a deterministic safety floor preventing false negatives.

    Parameters:
        df (pd.DataFrame): Time-indexed atmospheric and oceanographic readings.

    Returns:
        pd.Series (bool): Boolean mask where True indicates at least one physical limit is breached.
    """
    breach = pd.Series(False, index=df.index)
    breach |= df["wind_speed"] >= SAFETY_THRESHOLDS["wind_speed_ms"]
    breach |= df["wind_gust"] >= SAFETY_THRESHOLDS["wind_gust_ms"]
    breach |= df["hs"] >= SAFETY_THRESHOLDS["wave_height_m"]
    breach |= df["swell_height"] >= SAFETY_THRESHOLDS["swell_height_m"]
    breach |= df["current_speed"] >= SAFETY_THRESHOLDS["current_ms"]
    breach |= df["rain_rate_mm_hr"] >= SAFETY_THRESHOLDS["rain_mm_hr"]
    breach |= df["slp"] <= SAFETY_THRESHOLDS["pressure_hpa"]
    return breach


check_hard_gate = check_safety_thresholds


def individual_safety_threshold_breaches(df: pd.DataFrame) -> dict[str, int]:
    """
    Diagnostic helper reporting breach frequency for each physical threshold.

    Parameters:
        df (pd.DataFrame): Input dataset containing raw physical variables.

    Returns:
        dict[str, int]: Mapping of parameter threshold labels to total breach counts.
    """
    return {
        "wind_speed (>= 38 km/h)": int((df["wind_speed"] >= SAFETY_THRESHOLDS["wind_speed_ms"]).sum()),
        "wind_gust (>= 48 km/h)": int((df["wind_gust"] >= SAFETY_THRESHOLDS["wind_gust_ms"]).sum()),
        "wave_height (>= 1.8 m)": int((df["hs"] >= SAFETY_THRESHOLDS["wave_height_m"]).sum()),
        "swell_height (>= 1.8 m)": int((df["swell_height"] >= SAFETY_THRESHOLDS["swell_height_m"]).sum()),
        "current_speed (>= 0.8 m/s)": int((df["current_speed"] >= SAFETY_THRESHOLDS["current_ms"]).sum()),
        "rain_rate (>= 25 mm/hr)": int((df["rain_rate_mm_hr"] >= SAFETY_THRESHOLDS["rain_mm_hr"]).sum()),
        "pressure (<= 998 hPa)": int((df["slp"] <= SAFETY_THRESHOLDS["pressure_hpa"]).sum()),
    }


individual_hard_gate_breaches = individual_safety_threshold_breaches


# ---------------------------------------------------------------------------
# Sub-scoring bands (0=Optimal .. 4=Severe), matched EXACTLY to PHP's
# score*() functions — including PHP's own choice of strict vs inclusive
# comparison operators, which vary per function in the source.
# ---------------------------------------------------------------------------
def score_wave_height(v: pd.Series) -> pd.Series:
    """
    Scores significant wave height (Hs) into 5 risk severity tiers.
    
    Rationale:
        Freediving open-water training lines become dangerous above 0.80m due to
        surface chop disrupting breathing relaxation and diver visibility.
    """
    return pd.cut(v, bins=[-np.inf, 0.30, 0.50, 0.80, 1.00, np.inf],
                   labels=[0, 1, 2, 3, 4], right=False).astype(int)


def score_wind_wave_height(v: pd.Series) -> pd.Series:
    """
    Scores wind-driven sea chop (wind wave height) into 5 risk severity tiers.
    
    Rationale:
        Short-period wind chop causes abrupt boat pitching and diver disorientation
        near buoy platforms even when long-period swell is minimal.
    """
    return pd.cut(v, bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
                   labels=[0, 1, 2, 3, 4], right=False).astype(int)


def score_wave_period(v: pd.Series) -> pd.Series:
    """
    Scores dominant wave period (Tp) into 5 risk severity tiers (inverted scale).
    
    Rationale:
        Long wave periods (>7s) indicate organized, predictable swell energy. Short
        periods (<=3s) represent chaotic, steep wind chop that increases aspiration risk.
    """
    # PHP: v>7.0->0, v>5.0->1, v>3.0->2, v>2.0->3, else 4 (strict '>' throughout)
    return pd.cut(v, bins=[-np.inf, 2.0, 3.0, 5.0, 7.0, np.inf],
                   labels=[4, 3, 2, 1, 0], right=True).astype(int)


def score_ocean_current(v: pd.Series) -> pd.Series:
    """
    Scores ocean current speed into 5 risk severity tiers.
    
    Rationale:
        Currents above 0.50 m/s (~1 knot) cause freediving shot lines to drift at an
        angle, increasing ascent fatigue and risking diver separation from safety divers.
    """
    return pd.cut(v, bins=[-np.inf, 0.10, 0.30, 0.50, 0.80, np.inf],
                   labels=[0, 1, 2, 3, 4], right=False).astype(int)


def score_wind_speed(sustained_ms: pd.Series, gust_ms: pd.Series) -> pd.Series:
    """
    Jointly scores sustained wind speed and wind gusts.

    Rationale:
        Wind hazard in Anilao coastal waters is a compound effect of continuous wind
        driving currents and sudden gusts capsizing small banca boats. Both sustained
        and gust speeds must remain within a tier's bound to qualify for lower risk.
    """
    sustained_kmh = sustained_ms / KMH_TO_MS
    gust_kmh = gust_ms / KMH_TO_MS
    score = pd.Series(4, index=sustained_ms.index)
    score = score.where(~((sustained_kmh < 38.0) & (gust_kmh < 48.0)), 3)
    score = score.where(~((sustained_kmh < 28.0) & (gust_kmh < 38.0)), 2)
    score = score.where(~((sustained_kmh < 20.0) & (gust_kmh < 28.0)), 1)
    score = score.where(~((sustained_kmh < 12.0) & (gust_kmh < 18.0)), 0)
    return score.astype(int)


def score_rain(v: pd.Series) -> pd.Series:
    """
    Scores precipitation rate into 5 risk severity tiers.
    
    Rationale:
        Heavy rainfall (>7.5 mm/hr) drastically impairs spotter surface visibility,
        while extreme rain (>20 mm/hr) indicates convective thunderstorm squalls.
    """
    return pd.cut(v, bins=[-np.inf, 0.2, 2.5, 7.5, 20.0, np.inf],
                   labels=[0, 1, 2, 3, 4], right=False).astype(int)


def score_sea_level_pressure(v: pd.Series) -> pd.Series:
    """
    Scores mean sea level pressure (SLP) into 5 risk severity tiers.
    
    Rationale:
        Standard tropical atmospheric pressure is ~1012 hPa. Barometric depression
        below 1004 hPa signals monsoon troughs or approaching tropical cyclones.
    """
    return pd.cut(v, bins=[-np.inf, 1000.0, 1004.0, 1008.0, 1012.0, np.inf],
                   labels=[4, 3, 2, 1, 0], right=False).astype(int)


def score_wind_direction(deg: pd.Series) -> pd.Series:
    """
    Scores wind direction relative to Balayan Bay / Anilao coastal topography.

    Rationale:
        - 0°-90° & 315°-360° (NNE/NW): Sheltered by Batangas mainland / Mt. Gulugod Baboy (Score 0).
        - 90°-135° (ENE/ESE): Moderate offshore sheltering (Score 1).
        - 135°-180° & 270°-315° (SE/W): Exposed open-water fetch (Score 2).
        - 180°-270° (SSW/SW Habagat): Direct, uninhibited South China Sea fetch creating dangerous chop (Score 3).
    """
    d = deg % 360
    score = pd.Series(-1, index=deg.index)  # -1 = unassigned sentinel

    cond1 = ((d >= 0) & (d <= 90)) | (d > 315)
    score = score.where(~(cond1 & (score == -1)), 0)

    cond2 = (d <= 135)
    score = score.where(~(cond2 & (score == -1)), 1)

    cond3 = (d <= 180) | (d > 270)
    score = score.where(~(cond3 & (score == -1)), 2)

    score = score.where(score != -1, 3)  # Southwest Habagat exposure
    return score.astype(int)


def compute_subscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms continuous physical telemetry into normalized discrete sub-scores (0-4).

    Parameters:
        df (pd.DataFrame): Preprocessed weather and marine dataset.

    Returns:
        pd.DataFrame: 9-column dataframe containing sub-scores for all safety parameters.
    """
    sub = pd.DataFrame(index=df.index)
    sub["wave_height"] = score_wave_height(df["hs"])
    sub["wind_speed"] = score_wind_speed(df["wind_speed"], df["wind_gust"])
    sub["ocean_current"] = score_ocean_current(df["current_speed"])
    sub["swell_height"] = score_wave_height(df["swell_height"])  # PHP delegates to wave height
    sub["wave_period"] = score_wave_period(df["tp"])
    sub["wind_wave_height"] = score_wind_wave_height(df["wind_wave_height"])
    sub["rain"] = score_rain(df["rain_rate_mm_hr"])
    sub["sea_level_pressure"] = score_sea_level_pressure(df["slp"])
    sub["wind_direction"] = score_wind_direction(df["wind_dir"])
    return sub


# ---------------------------------------------------------------------------
# Weights — matched EXACTLY to PHP's WEIGHTS constant. Sums to 1.000.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "wave_height": 0.160,
    "wind_speed": 0.150,
    "ocean_current": 0.140,
    "swell_height": 0.125,
    "wave_period": 0.115,
    "wind_wave_height": 0.100,
    "rain": 0.080,
    "sea_level_pressure": 0.070,
    "wind_direction": 0.060,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.000, matching PHP exactly"

# PHP's synergy multiplier counts hazards across a SPECIFIC 5-feature subset
SYNERGY_HAZARD_FEATURES = ["wave_height", "wind_speed", "ocean_current", "wind_direction", "wave_period"]


def compute_final_scores(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates weighted composite safety scores and applies non-linear synergy penalties.

    Rationale:
        Multiple concurrent moderate hazards (e.g. moderate wind + adverse current + opposing swell)
        compound diver fatigue exponentially compared to isolated hazards. The synergy multiplier
        escalates score by +15% for 3 concurrent hazards and +25% for 4+ concurrent hazards.

    Parameters:
        sub (pd.DataFrame): 9-variable discrete sub-scores.

    Returns:
        pd.DataFrame: Contains base_score_pct, hazard_count, synergy_multiplier, and final_score_pct.
    """
    base_score = sum(sub[feat] * WEIGHTS[feat] for feat in WEIGHTS) / 4.0 * 100

    hazard_count = (sub[SYNERGY_HAZARD_FEATURES] >= 2).sum(axis=1)
    synergy = pd.Series(1.00, index=sub.index)
    synergy = synergy.where(hazard_count < 3, 1.15)
    synergy = synergy.where(hazard_count < 4, 1.25)

    final_score = (base_score * synergy).clip(upper=100.0)
    return pd.DataFrame({
        "base_score_pct": base_score,
        "hazard_count": hazard_count,
        "synergy_multiplier": synergy,
        "final_score_pct": final_score,
    })


def score_to_tier(score_pct: pd.Series) -> pd.Series:
    """
    Maps 0-100 continuous score to the 5 official safety classifications.

    Thresholds:
        - 0.00 - 20.00%: Very Safe (0)
        - 20.01 - 40.00%: Safe (1)
        - 40.01 - 60.00%: Moderate (2)
        - 60.01 - 80.00%: High Risk (3)
        - 80.01 - 100.0%: Critical Risk (4)
    """
    bins = [-0.01, 20.0, 40.0, 60.0, 80.0, 100.0]
    return pd.cut(score_pct, bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)


# TODO: Evaluate dynamic micro-site bathymetric weight adjustments for deep-line vs shallow-reef training sites.


def main():
    print(f"Loading {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"Applying PHP-aligned scoring to {len(df)} rows...")
    print(f"Weights (matched to PHP WEIGHTS constant, sum={sum(WEIGHTS.values()):.3f}):")
    for feat, w in WEIGHTS.items():
        print(f"  {feat:20s} {w:.3f} ({w*100:.1f}%)")

    print("\n--- Diagnostic: Individual Physical Safety Threshold Breach Counts ---")
    breach_counts = individual_safety_threshold_breaches(df)
    for name, count in breach_counts.items():
        print(f"  {name:30s}: {count:5d} rows ({count/len(df)*100:.2f}%)")

    safety_threshold_breach = check_safety_thresholds(df)
    subscores = compute_subscores(df)
    scores = compute_final_scores(subscores)

    tier = score_to_tier(scores["final_score_pct"])
    tier = tier.where(~safety_threshold_breach, 4)

    labels = pd.DataFrame({
        "risk_tier": tier,
        "risk_tier_name": tier.map(dict(enumerate(TIER_NAMES))),
        "final_score_pct": scores["final_score_pct"],
        "hazard_count": scores["hazard_count"],
        "safety_threshold_triggered": safety_threshold_breach,
        "hard_gate_triggered": safety_threshold_breach,  # Backward compatibility
    }, index=df.index)

    print("\nLabel distribution:")
    counts = labels["risk_tier_name"].value_counts().reindex(TIER_NAMES, fill_value=0)
    for name, count in counts.items():
        pct = count / len(labels) * 100
        print(f"  {name:14s} {count:6d} rows ({pct:5.1f}%)")

    threshold_count = int(safety_threshold_breach.sum())
    print(f"\nTotal physical safety threshold breaches (auto-Critical): {threshold_count} rows ({threshold_count/len(df)*100:.2f}%)")

    labels.to_parquet(OUT_PATH)
    print(f"\nWrote {len(labels)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
