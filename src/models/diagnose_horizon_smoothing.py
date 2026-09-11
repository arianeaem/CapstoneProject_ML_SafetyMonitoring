"""
Diagnostic tool for investigating upstream regression-to-the-mean and variance
shrinkage across forecast horizons (1h -> 144h), and how it affects downstream
safety classifier performance.

Run from project root: python src/models/diagnose_horizon_smoothing.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

from src.validation.splits import temporal_split
from src.features.horizon_targets import HORIZONS
from src.labels.build_safety_labels import HARD_GATE, TIER_NAMES
from src.models.train_safety_classifier import (
    generate_classifier_dataset, predict_booster
)


def main():
    print("=" * 75)
    print("DIAGNOSTIC: UPSTREAM FORECASTER VARIANCE & LONG-HORIZON SPREAD")
    print(f"Horizons: {HORIZONS} hours ahead")
    print("=" * 75 + "\n")

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "training_features.parquet")
    labels = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "safety_labels.parquet")

    classifier_X, classifier_y = generate_classifier_dataset(df, labels)
    full = classifier_X.copy()
    full["target_risk_tier"] = classifier_y.values

    train, val, test = temporal_split(full)
    feature_cols = [c for c in full.columns if c != "target_risk_tier"]

    val_X = val[feature_cols]

    # Load safety classifier
    classifier = xgb.Booster()
    classifier.load_model(str(MODELS_DIR / "xgb_safety_classifier.json"))
    val_preds = predict_booster(classifier, val_X)
    val = val.copy()
    val["pred_risk_tier"] = val_preds

    # -----------------------------------------------------------------------
    # Part 1: Distribution Shrinkage Analysis by Horizon
    # -----------------------------------------------------------------------
    print("=" * 75)
    print("PART 1: FORECAST VARIANCE & PEAK SHRINKAGE ACROSS ALL 8 HORIZONS")
    print("Notice how max values and std dev evolve across horizons:")
    print("=" * 75)

    key_vars = ["pred_wind_gust", "pred_wind_speed", "pred_hs"]
    for var in key_vars:
        print(f"\nDistribution of `{var}` across horizons (Validation Set):")
        print(f"{'Horizon':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'50%':>8} {'95%':>8} {'Max':>8}")
        print("-" * 64)
        for h in HORIZONS:
            h_data = val.loc[val["horizon"] == h, var]
            print(f"{str(h) + 'h':<10} {h_data.mean():>8.2f} {h_data.std():>8.2f} {h_data.min():>8.2f} "
                  f"{h_data.median():>8.2f} {h_data.quantile(0.95):>8.2f} {h_data.max():>8.2f}")

    # -----------------------------------------------------------------------
    # Part 2: Spot-Check Long-Horizon Critical Risk Rows
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PART 2: SPOT-CHECKING TRUE CRITICAL RISK ROWS AT 48h, 72h, 96h, 144h")
    print("Checking how P90 tail risk features support hazard detection:")
    print("=" * 75)

    for h in [48, 72, 96, 144]:
        h_val = val[val["horizon"] == h]
        crit_rows = h_val[h_val["target_risk_tier"] == 4]
        print(f"\n--- Horizon {h}h: Found {len(crit_rows)} True Critical Risk Hours ---")
        if len(crit_rows) == 0:
            print("No Critical Risk rows in this horizon slice.")
            continue

        sample = crit_rows.head(3)
        for idx in sample.index:
            row = sample.loc[idx]
            pred_tier = int(row["pred_risk_tier"])
            print(f"\nTimestamp: {idx} (Forecast {h}h ahead)")
            print(f"  Ground Truth Tier: Critical Risk (4)")
            print(f"  ML Predicted Tier: {TIER_NAMES[pred_tier]} ({pred_tier})")
            print(f"  Predicted Physical Features:")
            print(f"    - Mean Gust:       {row['pred_wind_gust']:>5.2f} m/s  | P90 Gust: {row['pred_wind_gust_p90']:>5.2f} m/s (Gate: >= {HARD_GATE['wind_gust_ms']:.2f})")
            print(f"    - Mean Wind:       {row['pred_wind_speed']:>5.2f} m/s  | P90 Wind: {row['pred_wind_speed_p90']:>5.2f} m/s (Gate: >= {HARD_GATE['wind_speed_ms']:.2f})")
            print(f"    - Mean Wave(hs):   {row['pred_hs']:>5.2f} m    | P90 Wave: {row['pred_hs_p90']:>5.2f} m   (Gate: >= {HARD_GATE['wave_height_m']:.2f})")

    print("\n" + "=" * 75)
    print("Diagnostic complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()
