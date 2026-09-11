"""
Trains xgb_current_regressor: predicts current_u and current_v (zonal/meridional
surface current velocity) as direct vector regression, matching the PRD's
"Vector Regression" design for this model.

Design decisions specific to current (different from train_wave.py / train_wind.py):

1. current_speed / current_dir excluded from inputs. Since current_u and
   current_v ARE the training targets, the raw current_speed/current_dir
   columns are deterministic trig functions of them (speed = sqrt(u^2+v^2),
   dir = atan2(v,u)) — leaving them in would leak the targets, same failure
   mode as wind_u/wind_v in train_wind.py, just mirrored.

2. wind_current_alignment excluded. It's computed from current_dir, which is
   itself derived from current_u/current_v — leaks for the same reason as #1.

3. Wind features (wind_u, wind_v, wind_speed, wind_gust, wind_dir) are KEPT
   as inputs, unlike the wave regressor's exclusion of wind->wave causality
   concerns being irrelevant here. Wind-driven surface currents (Ekman
   transport) are a real, physically legitimate predictive relationship.

4. hs, tp, swell_height, wind_wave_height (wave outputs) excluded — same
   parallel Stage-1 architecture reasoning as train_wind.py.

5. No separate current_speed/current_dir models are trained. Instead,
   current_speed and current_dir are DERIVED from the current_u/current_v
   predictions at evaluation time (sqrt and atan2), then scored — current_dir
   with the same circular distance handling used for wind_dir. This matches
   the PRD's vector-regression framing and avoids training + tuning two
   redundant extra models.

6. No PRD acceptance threshold exists for current (the PRD's Section 3.3
   table only specifies wave, wind, and the now-dropped tide thresholds) —
   metrics are reported without a fabricated PASS/MISS, same treatment as tp.

Run from the project root: python src\\models\\train_current.py
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

MODEL_NAME = "xgb_current_regressor"
TARGETS = ["current_u", "current_v"]
PRIMARY_TARGET = "current_u"

WAVE_TARGETS = ["hs", "tp", "swell_height", "wind_wave_height"]
EXCLUDED_FEATURES = ["current_speed", "current_dir", "wind_current_alignment"] + WAVE_TARGETS

# Same on-demand pattern as train_wave.py / train_wind.py: empty until a
# target's validation results show it needs independent tuning.
RETUNE_TARGETS = []


def get_feature_columns(df):
    excluded = set(TARGETS) | set(EXCLUDED_FEATURES)
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


def evaluate(y_true, y_pred, target_col):
    rmse = rmse_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    print(f"  {target_col:14s} RMSE={rmse:.4f}  MAE={mae:.4f}  bias={bias:+.4f}")
    return {"rmse": rmse, "mae": mae, "bias": bias}


def circular_angle_error(y_true_deg, y_pred_deg):
    diff = np.abs(y_true_deg - y_pred_deg) % 360
    return np.minimum(diff, 360 - diff)


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
    reused_for = [t for t in TARGETS if t not in RETUNE_TARGETS]
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
    val_preds_by_target = {}

    for target_col in TARGETS:
        params = target_params[target_col]
        model = xgb.XGBRegressor(**params)
        model.fit(train[features], train[target_col], verbose=False)

        val_preds = model.predict(val[features])
        val_preds_by_target[target_col] = val_preds
        metrics[target_col] = evaluate(val[target_col].values, val_preds, target_col)

        plot_pred_vs_actual(val[target_col].values, val_preds, target_col, MODEL_NAME)
        plot_residuals_over_time(val.index, val[target_col].values, val_preds, target_col, MODEL_NAME)
        plot_feature_importance(model, features, target_col, MODEL_NAME)
        plot_actual_vs_predicted_overlay(val.index, val[target_col].values, val_preds, target_col, MODEL_NAME)

        model.save_model(str(MODELS_DIR / f"{MODEL_NAME}_{target_col}.json"))

    # Derived, not separately trained: current_speed and current_dir from the
    # current_u/current_v predictions above.
    pred_u, pred_v = val_preds_by_target["current_u"], val_preds_by_target["current_v"]
    pred_speed = np.sqrt(pred_u ** 2 + pred_v ** 2)
    pred_dir = (np.degrees(np.arctan2(pred_v, pred_u))) % 360

    metrics["current_speed"] = evaluate(val["current_speed"].values, pred_speed, "current_speed")
    plot_pred_vs_actual(val["current_speed"].values, pred_speed, "current_speed", MODEL_NAME)
    plot_actual_vs_predicted_overlay(val.index, val["current_speed"].values, pred_speed, "current_speed", MODEL_NAME)

    dir_errors = circular_angle_error(val["current_dir"].values, pred_dir)
    dir_mae = float(np.mean(dir_errors))
    dir_rmse = float(np.sqrt(np.mean(dir_errors ** 2)))
    print(f"  {'current_dir':14s} circular RMSE={dir_rmse:.2f} deg  circular MAE={dir_mae:.2f} deg")
    metrics["current_dir"] = {"circular_rmse_deg": dir_rmse, "circular_mae_deg": dir_mae}
    plot_pred_vs_actual(val["current_dir"].values, pred_dir, "current_dir", MODEL_NAME)

    with open(str(MODELS_DIR / f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump({"hyperparameters": best_params, "validation_metrics": metrics}, f, indent=2)

    print(f"\nSaved models to {MODELS_DIR}/{MODEL_NAME}_<target>.json")
    print("current_speed/current_dir are derived metrics, not separately saved models.")
    print("test set untouched — reserved for the final combined evaluation across all 4 models.")


if __name__ == "__main__":
    main()
