"""
Trains multi-horizon XGBoost forecaster for ocean current variables:
(current_u, current_v).
Takes horizon as an input feature and predicts target at t+H (H in [1, 6, 12, 24, 48, 72, 96, 144] hours).

Current speed and direction are derived vectorially from predicted (current_u, current_v):
  current_speed = sqrt(u^2 + v^2)
  current_dir = atan2(v, u) in degrees [0, 360)
and evaluated against ground-truth speed and circular direction.

Run from project root: python src/models/train_current_forecaster.py
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

MODEL_NAME = "xgb_current_forecaster"
TARGETS = ["current_u", "current_v"]


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
    print("TRAINING MULTI-HORIZON CURRENT FORECASTER MODELS")
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
                print(f"  Horizon {h:>3}h: RMSE = {h_rmse:.4f} m/s, MAE = {h_mae:.4f} m/s (n={np.sum(mask)})")

        all_metrics[target] = target_metrics
        print()

    # Derive speed and direction from predicted (u, v)
    pred_u = val_predictions["current_u"]
    pred_v = val_predictions["current_v"]
    pred_speed = np.sqrt(pred_u ** 2 + pred_v ** 2)
    pred_dir = (np.degrees(np.arctan2(pred_v, pred_u))) % 360

    print("-" * 70)
    print("Derived Current Speed & Direction Validation Performance by Horizon:")
    print("-" * 70)
    derived_speed_metrics = {"horizons": {}}
    derived_dir_metrics = {"horizons": {}}

    for h in HORIZONS:
        mask = (val["horizon"] == h).values
        if np.sum(mask) > 0:
            true_u = val.loc[mask, "target_current_u"].values
            true_v = val.loc[mask, "target_current_v"].values
            true_speed = np.sqrt(true_u ** 2 + true_v ** 2)
            true_dir = (np.degrees(np.arctan2(true_v, true_u))) % 360

            h_pred_speed = pred_speed[mask]
            h_pred_dir = pred_dir[mask]

            speed_rmse = rmse_score(true_speed, h_pred_speed)
            speed_mae = mae_score(true_speed, h_pred_speed)
            derived_speed_metrics["horizons"][str(h)] = {"rmse": speed_rmse, "mae": speed_mae, "n": int(np.sum(mask))}

            dir_diffs = circular_angular_diff(true_dir, h_pred_dir)
            dir_mae = float(np.mean(dir_diffs))
            dir_rmse = float(np.sqrt(np.mean(dir_diffs ** 2)))
            derived_dir_metrics["horizons"][str(h)] = {"circular_mae_deg": dir_mae, "circular_rmse_deg": dir_rmse, "n": int(np.sum(mask))}

            print(f"  Horizon {h:>3}h:")
            print(f"    Speed:     RMSE = {speed_rmse:.4f} m/s, MAE = {speed_mae:.4f} m/s")
            print(f"    Direction: Circular MAE = {dir_mae:.2f}°, Circular RMSE = {dir_rmse:.2f}°")

    all_metrics["derived_current_speed"] = derived_speed_metrics
    all_metrics["derived_current_dir"] = derived_dir_metrics

    metrics_path = MODELS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved all metrics to {metrics_path}")
    print("\n" + "=" * 70)
    print("Current Forecaster training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
