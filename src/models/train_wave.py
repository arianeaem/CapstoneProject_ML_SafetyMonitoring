"""
Trains xgb_wave_regressor: 4 independent single-output XGBoost models predicting
hs, tp, swell_height, wind_wave_height (implemented as 4 parallel models sharing one
feature set and one tuned hyperparameter set for high computational efficiency).

Tuning Design: Optuna tunes against `hs` (40 trials) since significant wave height is the
primary safety-gating variable; the resulting hyperparameters are reused for the
other wave targets, with `tp` receiving independent fine-tuning where error distribution warrants.

Run from the project root: python src\\models\\train_wave.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Running this file directly (python src\models\train_wave.py) only puts
# src\models on Python's import path, not the project root — so "from src....."
# below would fail with ModuleNotFoundError without this. Inserting the project
# root explicitly makes it work regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Same project-root anchoring as the sys.path fix above — "models/" as a plain
# relative string breaks (or silently writes to the wrong place) depending on
# which folder the script happens to be run from. This guarantees it always
# resolves to <project_root>/models regardless of cwd.
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def rmse_score(y_true, y_pred):
    """sklearn removed mean_squared_error(..., squared=False) in 1.6+ (you're on 1.9.0) —
    compute RMSE manually so this doesn't break across scikit-learn versions."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


from src.validation.splits import load_training_features, temporal_split, walk_forward_folds
from src.models.eval_plots import (
    plot_pred_vs_actual, plot_residuals_over_time, plot_feature_importance,
    plot_walk_forward_scores, plot_actual_vs_predicted_overlay,
)

MODEL_NAME = "xgb_wave_regressor"
TARGETS = ["hs", "tp", "swell_height", "wind_wave_height"]
PRIMARY_TARGET = "hs"  # the one Optuna actually tunes against

# Excluded on purpose: these are DERIVED FROM the wave targets themselves
# (wave_steepness = f(hs, tp), swell_ratio = f(swell_height, hs), wave_power = f(hs, tp)).
# Including them as inputs here would be target leakage — the model could reconstruct
# hs/tp almost trivially from a feature that already encodes them. These three are
# legitimate features for the safety classifier, which sits downstream of
# these outputs, but not for the models that produce hs/tp/swell_height themselves.
LEAKY_FEATURES = ["wave_steepness", "swell_ratio", "wave_power"]

ACCEPTANCE_THRESHOLDS = {"hs": 0.15}  # m, RMSE operational acceptance target threshold


def get_feature_columns(df):
    excluded = set(TARGETS) | set(LEAKY_FEATURES)
    return [c for c in df.columns if c not in excluded]


def tune_hyperparameters(train_df, features, target, n_trials=40):
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
            model.fit(fold_train[features], fold_train[target], verbose=False)
            preds = model.predict(fold_val[features])
            fold_rmses.append(rmse_score(fold_val[target], preds))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize", study_name=f"{MODEL_NAME}_{target}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def collect_fold_scores(train_df, features, target, params):
    """Re-run walk-forward CV once with the final params, purely to capture
    per-fold RMSE for the walk-forward figure — training itself already happened
    during tuning, this is just for the plot."""
    fold_rmses = []
    for fold_train, fold_val in walk_forward_folds(train_df):
        model = xgb.XGBRegressor(**params)
        model.fit(fold_train[features], fold_train[target], verbose=False)
        preds = model.predict(fold_val[features])
        fold_rmses.append(rmse_score(fold_val[target], preds))
    return fold_rmses


def evaluate(y_true, y_pred, target):
    rmse = rmse_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    threshold = ACCEPTANCE_THRESHOLDS.get(target)
    status = ""
    if threshold is not None:
        status = "PASS" if rmse <= threshold else "MISS"
        status = f"  [{status} — threshold {threshold} m]"
    print(f"  {target:18s} RMSE={rmse:.4f}  MAE={mae:.4f}  bias={bias:+.4f}{status}")
    return {"rmse": rmse, "mae": mae, "bias": bias}


# Targets that get their own independent Optuna search instead of reusing
# PRIMARY_TARGET's hyperparameters. Added after diagnosing that tp's error
# distribution (a tight ~6s mode with real tails in both directions) needs
# more model capacity than hs's tuned params provide — hs/swell_height/
# wind_wave_height all validated well under the shared params, so only tp
# gets the extra tuning cost.
RETUNE_TARGETS = ["tp"]


def main():
    df = load_training_features()
    train, val, test = temporal_split(df)  # test is not touched anywhere below
    features = get_feature_columns(df)

    print(f"Training {MODEL_NAME} on {len(features)} features, excluding leaky: {LEAKY_FEATURES}")
    print(f"Tuning against primary target '{PRIMARY_TARGET}' with 40 Optuna trials...")
    best_params, best_cv_rmse = tune_hyperparameters(train, features, PRIMARY_TARGET, n_trials=40)
    best_params.update({"objective": "reg:squarederror", "tree_method": "hist", "random_state": 42})
    print(f"Best walk-forward CV RMSE ({PRIMARY_TARGET}): {best_cv_rmse:.4f}")
    print(f"Reusing these hyperparameters for {[t for t in TARGETS if t not in RETUNE_TARGETS]}: {best_params}\n")

    fold_scores = collect_fold_scores(train, features, PRIMARY_TARGET, best_params)
    plot_walk_forward_scores(fold_scores, PRIMARY_TARGET, MODEL_NAME)

    # Independent tuning pass for targets flagged in RETUNE_TARGETS
    target_params = {t: best_params for t in TARGETS}
    for target in RETUNE_TARGETS:
        print(f"Retuning independently for '{target}' (shared params underfit its tails)...")
        retuned_params, retuned_cv_rmse = tune_hyperparameters(train, features, target, n_trials=40)
        retuned_params.update({"objective": "reg:squarederror", "tree_method": "hist", "random_state": 42})
        print(f"Best walk-forward CV RMSE ({target}): {retuned_cv_rmse:.4f}")
        print(f"Retuned hyperparameters for {target}: {retuned_params}\n")
        target_params[target] = retuned_params

        target_fold_scores = collect_fold_scores(train, features, target, retuned_params)
        plot_walk_forward_scores(target_fold_scores, target, MODEL_NAME)

    print("Final validation metrics (val set, touched once per target):")
    metrics = {}
    for target in TARGETS:
        params = target_params[target]
        model = xgb.XGBRegressor(**params)
        model.fit(train[features], train[target], verbose=False)

        val_preds = model.predict(val[features])
        metrics[target] = evaluate(val[target].values, val_preds, target)

        plot_pred_vs_actual(val[target].values, val_preds, target, MODEL_NAME)
        plot_residuals_over_time(val.index, val[target].values, val_preds, target, MODEL_NAME)
        plot_feature_importance(model, features, target, MODEL_NAME)
        plot_actual_vs_predicted_overlay(val.index, val[target].values, val_preds, target, MODEL_NAME)

        model.save_model(str(MODELS_DIR / f"{MODEL_NAME}_{target}.json"))

    with open(str(MODELS_DIR / f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump({"hyperparameters": best_params, "validation_metrics": metrics}, f, indent=2)

    print(f"\nSaved 4 target models to {MODELS_DIR}/{MODEL_NAME}_<target>.json")
    print(f"Saved figures to reports/figures/{MODEL_NAME}_*.png")
    print("test set untouched — reserved for the final combined evaluation across all 4 models.")


if __name__ == "__main__":
    main()
