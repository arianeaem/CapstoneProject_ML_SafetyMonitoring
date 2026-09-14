"""
Final, one-time evaluation of the complete multi-horizon forecasting pipeline
(wave/wind/current forecasters -> safety classifier -> deterministic safety thresholds) on `test` —
the 15% chronological split that has been deliberately untouched through every
step of this build.

Evaluates multi-horizon performance across all 8 horizons (1h, 6h, 12h, 24h, 48h, 72h, 96h, 144h),
both for the ML classifier alone and with the deterministic safety thresholds layer applied.

Run from the project root: python src/models/final_evaluation.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

from src.validation.splits import temporal_split
from src.features.horizon_targets import HORIZONS
from src.models.eval_plots import plot_confusion_matrix
from src.models.train_safety_classifier import (
    generate_classifier_dataset, asymmetric_cost_score, predict_booster,
    TIER_NAMES, COST_MATRIX,
)
from src.serve.safety_thresholds import apply_safety_thresholds


def load_classifier() -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(MODELS_DIR / "xgb_safety_classifier.json"))
    return booster


def main():
    print("=" * 70)
    print("FINAL COMBINED MULTI-HORIZON EVALUATION ON TEST SET")
    print("RUN ONCE — HELD-OUT CHRONOLOGICAL TEST SPLIT")
    print(f"Horizons: {HORIZONS} hours ahead")
    print("=" * 70 + "\n")

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "training_features.parquet")
    labels = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "safety_labels.parquet")

    classifier_X, classifier_y = generate_classifier_dataset(df, labels)

    full = classifier_X.copy()
    full["target_risk_tier"] = classifier_y.values

    train, val, test = temporal_split(full)  # only `test` is used from here on
    feature_cols = [c for c in full.columns if c != "target_risk_tier"]

    print(f"Test split: {len(test)} rows ({test.index.min()} to {test.index.max()})")
    print("This temporal range has been held out across all tuning and model selections.\n")

    test_X = test[feature_cols]
    true_tiers = [int(x) for x in test["target_risk_tier"].values]

    # Step 1: Classifier prediction from forecaster outputs
    print("Running safety classifier inference on forecaster predictions...")
    classifier = load_classifier()
    ml_preds = predict_booster(classifier, test_X)

    # Step 2: Safety thresholds layer applied on top of predicted telemetry
    print("Applying deterministic physical safety thresholds on test predictions...")
    final_tiers: list[int] = []
    threshold_triggered_count = 0

    for i in range(len(test)):
        row = test_X.iloc[i]
        telemetry = {
            "wind_speed": float(row["pred_wind_speed"]),
            "wind_gust": float(row["pred_wind_gust"]),
            "hs": float(row["pred_hs"]),
            "swell_height": float(row["pred_swell_height"]),
            "current_speed": float(row["pred_current_speed"]),
            "rain_rate_mm_hr": float(row["rain_rate_mm_hr_lag0h"]),
            "slp": float(row["pred_slp"]),
        }
        result = apply_safety_thresholds(int(ml_preds[i]), telemetry, pagasa=None)
        final_tiers.append(int(result["final_tier"]))
        if result["safety_threshold_triggered"]:
            threshold_triggered_count += 1

    print(f"Safety thresholds triggered on {threshold_triggered_count} / {len(test)} test rows "
          f"({threshold_triggered_count/len(test)*100:.2f}%)\n")

    # --- Report both stages: ML alone, and ML + safety thresholds combined ---
    for stage_name, preds in [("ML Classifier Alone", ml_preds),
                               ("Full System (ML + Safety Thresholds)", final_tiers)]:
        print("=" * 70)
        print(f"STAGE: {stage_name}")
        print("=" * 70)
        weighted_f1 = float(f1_score(true_tiers, preds, average="weighted"))
        cost = asymmetric_cost_score(true_tiers, preds)
        print(f"Weighted F1:          {weighted_f1:.4f}  (PRD target >= 0.92)")
        print(f"Asymmetric Cost Score: {cost:.4f} (lower is safer)")
        print("\nPer-class report:")
        print(classification_report(true_tiers, preds, target_names=TIER_NAMES,
                                      labels=[0, 1, 2, 3, 4], zero_division=0))

        critical_indices = [i for i, y in enumerate(true_tiers) if y == 4]
        critical_total = len(critical_indices)
        if critical_total > 0:
            critical_preds = [preds[i] for i in critical_indices]
            fnr_any = float(sum(1 for p in critical_preds if p != 4) / critical_total)
            fnr_to_safe = float(sum(1 for p in critical_preds if p in (0, 1)) / critical_total)
            print(f"True Critical Risk rows in test: {critical_total}")
            print(f"  Misclassified as ANYTHING else: {fnr_any*100:.1f}%")
            print(f"  Misclassified specifically as Very Safe/Safe: {fnr_to_safe*100:.1f}%")
        else:
            print("No Critical Risk rows in test set.")
        print()

    # Per-horizon test breakdown for the full system
    print("=" * 70)
    print("FULL SYSTEM TEST PERFORMANCE BY FORECAST HORIZON:")
    print("=" * 70)
    for h in HORIZONS:
        mask = (test["horizon"] == h).values
        if np.sum(mask) > 0:
            h_true = [true_tiers[i] for i in range(len(true_tiers)) if mask[i]]
            h_preds = [final_tiers[i] for i in range(len(final_tiers)) if mask[i]]
            h_f1 = float(f1_score(h_true, h_preds, average="weighted"))
            h_cost = asymmetric_cost_score(h_true, h_preds)
            print(f"  Horizon {h:>3}h: Weighted F1 = {h_f1:.4f} | Asymm Cost = {h_cost:.4f} (n={np.sum(mask)})")

    cm_final = confusion_matrix(true_tiers, final_tiers, labels=[0, 1, 2, 3, 4])
    plot_confusion_matrix(cm_final, TIER_NAMES, "final_system_test_evaluation")
    print("\nSaved final confusion matrix to reports/figures/final_system_test_evaluation_confusion_matrix.png")
    print("=" * 70)
    print("Final evaluation script ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
