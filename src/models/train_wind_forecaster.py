"""
Trains multi-horizon XGBoost forecaster for wind and atmospheric variables:
(wind_speed, wind_gust, wind_dir_sin, wind_dir_cos, slp).
Takes horizon as an input feature and predicts target at t+H (H in [1, 6, 12, 24, 48, 72, 96, 144] hours).

Wind direction is predicted via sin/cos decomposition to handle the 0°/360° circular boundary,
and reconstructed as degrees via atan2 during evaluation.

Run from project root: python src/models/train_wind_forecaster.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from src.validation.splits import temporal_split, walk_forward_folds
from src.features.lagged_features import build_lagged_features
from src.features.horizon_targets import build_stacked_dataset, HORIZONS

MODEL_NAME = "xgb_wind_forecaster"
TARGETS = ["wind_speed", "wind_gust", "wind_dir_sin", "wind_dir_cos", "slp"]


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae_score(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def circular_angular_diff(y_true_deg, y_pred_deg):
    """Computes shortest angular difference in degrees accounting for 360 wrap-around."""
    diff = np.abs(y_true_deg - y_pred_deg) % 360
    return np.minimum(diff, 360 - diff)


def main():
    print("=" * 70)
    print("TRAINING MULTI-HORIZON WIND & SLP FORECASTER MODELS")
    print(f"Horizons: {HORIZONS} hours ahead")
    print(f"Targets: {TARGETS}")
    print("=" * 70 + "\n")

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "training_features.parquet")
    print(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    print("Generating lagged/rolling historical features...")
    lagged = build_lagged_features(df)

    print("Constructing multi-horizon forward targets and stacking...")
    stacked = build_stacked_dataset(df, lagged)
    print(f"Stacked multi-horizon dataset: {stacked.shape[0]} rows, {stacked.shape[1]} columns\n")

    train, val, test = temporal_split(stacked)  # test untouched
    print(f"Train split: {len(train)} rows ({train.index.min()} to {train.index.max()})")
    print(f"Val split:   {len(val)} rows ({val.index.min()} to {val.index.max()})")
    print(f"Test split:  {len(test)} rows (held out)\n")

    feature_cols = [c for c in stacked.columns if not c.startswith("target_")]
    print(f"Input feature count: {len(feature_cols)} (includes horizon as an input feature)\n")

    all_metrics = {}
    val_predictions = {}

    for target in TARGETS:
        target_col = f"target_{target}"
        print("-" * 70)
        print(f"Tuning and Training {MODEL_NAME} for {target_col}...")
        print("-" * 70)

        def objective(trial):
            params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "random_state": 42,
            }
            fold_scores = []
            for fold_train, fold_val in walk_forward_folds(train, n_splits=3, gap_hours=48):
                model = xgb.XGBRegressor(**params)
                model.fit(fold_train[feature_cols], fold_train[target_col], verbose=False)
                preds = model.predict(fold_val[feature_cols])
                fold_scores.append(rmse_score(fold_val[target_col], preds))
            return float(np.mean(fold_scores))

        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15)

        best_params = study.best_params
        best_params.update({
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        })
        print(f"\nBest Optuna Hyperparameters for {target}: {best_params}")

        model = xgb.XGBRegressor(**best_params)
        model.fit(train[feature_cols], train[target_col], verbose=False)

        model_path = MODELS_DIR / f"{MODEL_NAME}_{target}.json"
        model.save_model(str(model_path))
        print(f"Saved model to {model_path}")

        # Per-horizon validation evaluation
        val_preds = model.predict(val[feature_cols])
        val_predictions[target] = val_preds
        target_metrics = {"hyperparameters": best_params, "horizons": {}}

        print(f"\nValidation Performance by Horizon for {target}:")
        for h in HORIZONS:
            mask = (val["horizon"] == h).values
            if np.sum(mask) > 0:
                y_true = val.loc[mask, target_col].values
                y_p = val_preds[mask]
                h_rmse = rmse_score(y_true, y_p)
                h_mae = mae_score(y_true, y_p)
                target_metrics["horizons"][str(h)] = {"rmse": h_rmse, "mae": h_mae, "n": int(np.sum(mask))}
                print(f"  Horizon {h:>3}h: RMSE = {h_rmse:.4f}, MAE = {h_mae:.4f} (n={np.sum(mask)})")

        all_metrics[target] = target_metrics
        print()

    # Reconstruct circular wind_dir from sin and cos predictions
    if "wind_dir_sin" in val_predictions and "wind_dir_cos" in val_predictions and "target_wind_dir" in val.columns:
        pred_sin = val_predictions["wind_dir_sin"]
        pred_cos = val_predictions["wind_dir_cos"]
        pred_wind_dir = (np.degrees(np.arctan2(pred_sin, pred_cos))) % 360

        print("-" * 70)
        print("Reconstructed Circular Wind Direction (degrees) Performance by Horizon:")
        print("-" * 70)
        wind_dir_metrics = {"horizons": {}}
        for h in HORIZONS:
            mask = (val["horizon"] == h).values
            if np.sum(mask) > 0:
                true_deg = val.loc[mask, "target_wind_dir"].values
                pred_deg = pred_wind_dir[mask]
                angular_diffs = circular_angular_diff(true_deg, pred_deg)
                deg_mae = float(np.mean(angular_diffs))
                deg_rmse = float(np.sqrt(np.mean(angular_diffs ** 2)))
                wind_dir_metrics["horizons"][str(h)] = {"circular_mae_deg": deg_mae, "circular_rmse_deg": deg_rmse, "n": int(np.sum(mask))}
                print(f"  Horizon {h:>3}h: Circular MAE = {deg_mae:.2f}°, Circular RMSE = {deg_rmse:.2f}° (n={np.sum(mask)})")

        all_metrics["wind_dir_reconstructed_deg"] = wind_dir_metrics
        print()

    metrics_path = MODELS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved all metrics to {metrics_path}")
    print("\n" + "=" * 70)
    print("Wind & SLP Forecaster training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
