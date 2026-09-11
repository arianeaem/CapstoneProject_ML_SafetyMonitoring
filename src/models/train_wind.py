"""
Trains xgb_wind_regressor: predicts wind_speed, wind_gust, delta_p_3h (plain
regression, same pattern as train_wave.py) plus wind_dir (handled specially,
see below, since it's a circular quantity).

Design decisions specific to wind (different from train_wave.py):

1. wind_u / wind_v excluded from inputs entirely. Unlike the wave case, these
   are DETERMINISTIC sources of wind_speed and wind_dir (simple trigonometry:
   speed = sqrt(u^2+v^2), dir = atan2(v,u)) — leaving them in would let the
   model just learn that formula instead of anything meteorologically useful,
   producing a fake perfect score that means nothing.

2. hs, tp, swell_height, wind_wave_height (wave outputs) also excluded. Per
   the PRD's architecture, wave/wind/current regressors are parallel Stage-1
   models feeding into the same Stage-2 safety classifier, not chained to each
   other. Wind causes waves, not the reverse — using wave state to predict
   wind would be circular in the pipeline and physically backwards.

3. wind_dir is circular (0 deg and 359 deg are 1 degree apart, not 359 apart).
   A plain regression on raw degrees would treat crossing that boundary as a
   huge error. Standard fix used here: train on sin/cos of the angle as two
   auxiliary regressions, reconstruct the angle via atan2 at evaluation time,
   and score with proper circular distance — this satisfies the PRD's "cyclic
   angular loss" requirement without a custom XGBoost objective.

4. PRD's explicit acceptance threshold here is wind_speed MAE <= 1.2 m/s
   (note: MAE, not RMSE — different from hs's RMSE threshold in train_wave.py).

Run from the project root: python src\\models\\train_wind.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


from src.validation.splits import load_training_features, temporal_split, walk_forward_folds
from src.models.eval_plots import (
    plot_pred_vs_actual, plot_residuals_over_time, plot_feature_importance,
    plot_walk_forward_scores, plot_actual_vs_predicted_overlay,
)

MODEL_NAME = "xgb_wind_regressor"
TARGETS = ["wind_speed", "wind_gust", "delta_p_3h"]  # wind_dir handled separately below
PRIMARY_TARGET = "wind_speed"

WAVE_TARGETS = ["hs", "tp", "swell_height", "wind_wave_height"]
EXCLUDED_FEATURES = ["wind_u", "wind_v", "wind_current_alignment"] + WAVE_TARGETS

# Targets that get their own independent Optuna search instead of reusing
# PRIMARY_TARGET's hyperparameters. Empty by default — add a target here if
# its validation results look underfit, same as tp was added in train_wave.py
# after the first run revealed it needed separate tuning.
RETUNE_TARGETS = []

ACCEPTANCE_THRESHOLDS_MAE = {"wind_speed": 1.2}  # m/s, MAE — PRD Section 3.3


def get_feature_columns(df):
    excluded = set(TARGETS) | {"wind_dir"} | set(EXCLUDED_FEATURES)
    return [c for c in df.columns if c not in excluded]


def tune_hyperparameters(train_df, features, target_col, n_trials=40):
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
        }
        fold_rmses = []
        for fold_train, fold_val in walk_forward_folds(train_df):
            model = xgb.XGBRegressor(**params)
            model.fit(fold_train[features], fold_train[target_col], verbose=False)
            preds = model.predict(fold_val[features])
            fold_rmses.append(rmse_score(fold_val[target_col], preds))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize", study_name=f"{MODEL_NAME}_{target_col}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def collect_fold_scores(train_df, features, target_col, params):
    fold_rmses = []
    for fold_train, fold_val in walk_forward_folds(train_df):
        model = xgb.XGBRegressor(**params)
        model.fit(fold_train[features], fold_train[target_col], verbose=False)
        preds = model.predict(fold_val[features])
        fold_rmses.append(rmse_score(fold_val[target_col], preds))
    return fold_rmses


def evaluate(y_true, y_pred, target_col, mae_threshold=None):
    rmse = rmse_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    status = ""
    if mae_threshold is not None:
        status = "PASS" if mae <= mae_threshold else "MISS"
        status = f"  [{status} — MAE threshold {mae_threshold}]"
    print(f"  {target_col:14s} RMSE={rmse:.4f}  MAE={mae:.4f}  bias={bias:+.4f}{status}")
    return {"rmse": rmse, "mae": mae, "bias": bias}


def circular_angle_error(y_true_deg, y_pred_deg):
    """Shortest circular distance in degrees, e.g. true=359, pred=2 -> 3 deg, not 357."""
    diff = np.abs(y_true_deg - y_pred_deg) % 360
    return np.minimum(diff, 360 - diff)


def evaluate_wind_dir(y_true_deg, pred_sin, pred_cos):
    pred_deg = (np.degrees(np.arctan2(pred_sin, pred_cos))) % 360
    errors = circular_angle_error(y_true_deg, pred_deg)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    print(f"  {'wind_dir':14s} circular RMSE={rmse:.2f} deg  circular MAE={mae:.2f} deg")
    return pred_deg, {"circular_rmse_deg": rmse, "circular_mae_deg": mae}


def main():
    df = load_training_features()
    train, val, test = temporal_split(df)  # test is not touched anywhere below
    features = get_feature_columns(df)

    print(f"Training {MODEL_NAME} on {len(features)} features")
    print(f"Excluded (self-leaky or cross-stage): {EXCLUDED_FEATURES}")
    print(f"Tuning against primary target '{PRIMARY_TARGET}' with 40 Optuna trials...")
    best_params, best_cv_rmse = tune_hyperparameters(train, features, PRIMARY_TARGET, n_trials=40)
    best_params.update({"objective": "reg:squarederror", "tree_method": "hist", "random_state": 42})
    print(f"Best walk-forward CV RMSE ({PRIMARY_TARGET}): {best_cv_rmse:.4f}")
    reused_for = [t for t in TARGETS if t not in RETUNE_TARGETS] + ["wind_dir_sin", "wind_dir_cos"]
    print(f"Reusing these hyperparameters for {reused_for}: {best_params}\n")

    fold_scores = collect_fold_scores(train, features, PRIMARY_TARGET, best_params)
    plot_walk_forward_scores(fold_scores, PRIMARY_TARGET, MODEL_NAME)

    target_params = {t: best_params for t in TARGETS}
    for target_col in RETUNE_TARGETS:
        print(f"Retuning independently for '{target_col}'...")
        retuned_params, retuned_cv_rmse = tune_hyperparameters(train, features, target_col, n_trials=40)
        retuned_params.update({"objective": "reg:squarederror", "tree_method": "hist", "random_state": 42})
        print(f"Best walk-forward CV RMSE ({target_col}): {retuned_cv_rmse:.4f}\n")
        target_params[target_col] = retuned_params
        rf_scores = collect_fold_scores(train, features, target_col, retuned_params)
        plot_walk_forward_scores(rf_scores, target_col, MODEL_NAME)

    print("Final validation metrics (val set, touched once per target):")
    metrics = {}

    # Plain regression targets: wind_speed, wind_gust, delta_p_3h
    for target_col in TARGETS:
        params = target_params[target_col]
        model = xgb.XGBRegressor(**params)
        model.fit(train[features], train[target_col], verbose=False)

        val_preds = model.predict(val[features])
        threshold = ACCEPTANCE_THRESHOLDS_MAE.get(target_col)
        metrics[target_col] = evaluate(val[target_col].values, val_preds, target_col, threshold)

        plot_pred_vs_actual(val[target_col].values, val_preds, target_col, MODEL_NAME)
        plot_residuals_over_time(val.index, val[target_col].values, val_preds, target_col, MODEL_NAME)
        plot_feature_importance(model, features, target_col, MODEL_NAME)
        plot_actual_vs_predicted_overlay(val.index, val[target_col].values, val_preds, target_col, MODEL_NAME)

        model.save_model(str(MODELS_DIR / f"{MODEL_NAME}_{target_col}.json"))

    # Circular target: wind_dir, via sin/cos decomposition
    train_dir_sin = np.sin(np.radians(train["wind_dir"]))
    train_dir_cos = np.cos(np.radians(train["wind_dir"]))
    val_dir_sin = np.sin(np.radians(val["wind_dir"]))
    val_dir_cos = np.cos(np.radians(val["wind_dir"]))

    sin_model = xgb.XGBRegressor(**best_params)
    sin_model.fit(train[features], train_dir_sin, verbose=False)
    cos_model = xgb.XGBRegressor(**best_params)
    cos_model.fit(train[features], train_dir_cos, verbose=False)

    pred_sin = sin_model.predict(val[features])
    pred_cos = cos_model.predict(val[features])
    pred_deg, dir_metrics = evaluate_wind_dir(val["wind_dir"].values, pred_sin, pred_cos)
    metrics["wind_dir"] = dir_metrics

    # Reuse the pred-vs-actual / overlay plots on the reconstructed angle — note
    # in the figure that a wraparound (e.g. true 359 -> pred 2) will show as a
    # visual outlier even though the circular metrics above score it correctly;
    # that's a plotting artifact, not a real error, and is fine to caveat verbally.
    plot_pred_vs_actual(val["wind_dir"].values, pred_deg, "wind_dir", MODEL_NAME)
    plot_feature_importance(sin_model, features, "wind_dir_sin", MODEL_NAME)

    sin_model.save_model(str(MODELS_DIR / f"{MODEL_NAME}_wind_dir_sin.json"))
    cos_model.save_model(str(MODELS_DIR / f"{MODEL_NAME}_wind_dir_cos.json"))

    with open(str(MODELS_DIR / f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump({"hyperparameters": best_params, "validation_metrics": metrics}, f, indent=2)

    print(f"\nSaved models to {MODELS_DIR}/{MODEL_NAME}_<target>.json")
    print("test set untouched — reserved for the final combined evaluation across all 4 models.")


if __name__ == "__main__":
    main()
