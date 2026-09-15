"""
Trains multi-horizon xgb_safety_classifier: predicts a 5-tier risk class
(Very Safe -> Safe -> Moderate -> High Risk -> Critical Risk) across all forecast
horizons (1h, 6h, 12h, 24h, 48h, 72h, 96h, 144h) from the OUTPUTS of the three
already-trained multi-horizon physics forecasters (wave, wind, current) plus
tail-risk / P90 uncertainty bounds.

TAIL RISK & LONG-HORIZON UTILITY:
At longer horizons (48h-144h), point predictions naturally regress toward the mean
under uncertainty. To provide genuine safety utility at multi-day horizons without
false-positive suppression, the classifier receives both expected mean physics and
upper-quantile tail uncertainty bounds (P90: pred + 1.645 * sigma_H), allowing it
to evaluate severe storm risks even when mean forecasts are smoothed.

Run from the project root: python src/models/train_safety_classifier.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from src.validation.splits import temporal_split, walk_forward_folds
from src.features.lagged_features import build_lagged_features
from src.features.horizon_targets import build_stacked_dataset, HORIZONS
from src.models.eval_plots import plot_confusion_matrix

MODEL_NAME = "xgb_safety_classifier"
TIER_NAMES = ["Very Safe", "Safe", "Moderate", "High Risk", "Critical Risk"]
KMH_TO_MS = 1000 / 3600

# Class safety weights prioritizing higher-severity risks
CLASS_SAFETY_WEIGHTS = {0: 1.0, 1: 1.0, 2: 2.5, 3: 5.0, 4: 10.0}

# Full 5x5 cost matrix, rows=true, cols=predicted — used for optimization & reporting
COST_MATRIX = np.array([
    [0,   1,  2,  4,  6],
    [1,   0,  2,  4,  6],
    [5,   3,  0,  1,  3],
    [20, 15,  5,  0,  2],
    [100, 80, 40, 10,  0],
])

WAVE_TARGETS = ["hs", "tp", "swell_height", "wind_wave_height"]
WIND_TARGETS = ["wind_speed", "wind_gust", "slp"]
CURRENT_TARGETS = ["current_u", "current_v"]

# Estimated standard error growth per horizon for P90 tail risk bounds
SIGMA_GROWTH = {
    "wind_gust": {1: 0.8, 6: 1.5, 12: 2.0, 24: 2.5, 48: 3.2, 72: 3.8, 96: 4.2, 144: 4.8},
    "wind_speed": {1: 0.6, 6: 1.1, 12: 1.5, 24: 1.9, 48: 2.4, 72: 2.9, 96: 3.3, 144: 3.8},
    "hs": {1: 0.05, 6: 0.10, 12: 0.14, 24: 0.18, 48: 0.24, 72: 0.28, 96: 0.32, 144: 0.38},
}


# TODO: Implement Split Conformal Prediction sets to generate formal statistical confidence intervals for multi-horizon risk tiers.


def load_regressor(name: str) -> xgb.XGBRegressor:
    """
    Loads a trained XGBoost physics regressor from the models directory.

    Parameters:
        name (str): Model filename prefix (e.g. 'xgb_wave_forecaster_hs').

    Returns:
        xgb.XGBRegressor: Initialized inference model.
    """
    model = xgb.XGBRegressor(n_jobs=-1)
    model.load_model(str(MODELS_DIR / f"{name}.json"))
    return model


def generate_classifier_dataset(df: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Constructs the multi-horizon meta-feature dataset for the safety classifier.

    Pipeline Rationale:
        1. Two-Stage Stacked Architecture: The safety classifier does not read raw historical weather
           directly; instead, it consumes the predictions of the 11 specialized physics regressors.
        2. Horizon Expansion: Duplicates feature blocks across all 8 operational horizons (1h to 144h).
        3. P90 Uncertainty Augmentation: Computes upper-tail 90th percentile bounds ($P90 = \mu + 1.645 \cdot \sigma_H$)
           to ensure extreme squall risks remain visible even when multi-day point forecasts smooth out.

    Parameters:
        df (pd.DataFrame): Base preprocessed historical features.
        labels (pd.DataFrame): Canonical ground-truth risk tiers.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (classifier_feature_matrix, target_risk_labels).
    """
    print("Generating lagged/rolling historical features...")
    lagged = build_lagged_features(df)

    print(f"Constructing multi-horizon stacked dataset for horizons: {HORIZONS}...")
    stacked_blocks = []
    for h in HORIZONS:
        block = lagged.copy()
        block["horizon"] = h
        block["target_risk_tier"] = labels["risk_tier"].shift(-h)
        stacked_blocks.append(block)

    stacked = pd.concat(stacked_blocks, axis=0).dropna().sort_index()
    print(f"Stacked multi-horizon rows: {len(stacked)}")

    forecaster_feature_cols = [c for c in stacked.columns if c != "target_risk_tier"]

    print("Loading trained multi-horizon forecasters and generating predicted physics features...")
    preds = {}

    # Wave predictions
    for target in WAVE_TARGETS:
        model = load_regressor(f"xgb_wave_forecaster_{target}")
        preds[f"pred_{target}"] = model.predict(stacked[forecaster_feature_cols])

    # Wind predictions
    for target in WIND_TARGETS:
        model = load_regressor(f"xgb_wind_forecaster_{target}")
        preds[f"pred_{target}"] = model.predict(stacked[forecaster_feature_cols])

    sin_model = load_regressor("xgb_wind_forecaster_wind_dir_sin")
    cos_model = load_regressor("xgb_wind_forecaster_wind_dir_cos")
    pred_sin = sin_model.predict(stacked[forecaster_feature_cols])
    pred_cos = cos_model.predict(stacked[forecaster_feature_cols])
    preds["pred_wind_dir"] = (np.degrees(np.arctan2(pred_sin, pred_cos))) % 360

    # Current predictions
    u_model = load_regressor("xgb_current_forecaster_current_u")
    v_model = load_regressor("xgb_current_forecaster_current_v")
    pred_u = u_model.predict(stacked[forecaster_feature_cols])
    pred_v = v_model.predict(stacked[forecaster_feature_cols])
    preds["pred_current_u"] = pred_u
    preds["pred_current_v"] = pred_v
    preds["pred_current_speed"] = np.sqrt(pred_u ** 2 + pred_v ** 2)
    preds["pred_current_dir"] = (np.degrees(np.arctan2(pred_v, pred_u))) % 360

    # P90 Tail risk / upper quantile features
    h_arr = stacked["horizon"].values
    gust_sigma = np.array([SIGMA_GROWTH["wind_gust"].get(int(h), 3.0) for h in h_arr])
    wind_sigma = np.array([SIGMA_GROWTH["wind_speed"].get(int(h), 2.5) for h in h_arr])
    hs_sigma = np.array([SIGMA_GROWTH["hs"].get(int(h), 0.25) for h in h_arr])

    p90_wind_gust = preds["pred_wind_gust"] + 1.645 * gust_sigma
    p90_wind_speed = preds["pred_wind_speed"] + 1.645 * wind_sigma
    p90_hs = preds["pred_hs"] + 1.645 * hs_sigma

    # Assemble classifier inputs (aligned by integer position to prevent Cartesian join)
    classifier_X = pd.DataFrame({
        "horizon": stacked["horizon"].values,
        "pred_hs": preds["pred_hs"],
        "pred_hs_p90": p90_hs,
        "pred_tp": preds["pred_tp"],
        "pred_swell_height": preds["pred_swell_height"],
        "pred_wind_wave_height": preds["pred_wind_wave_height"],
        "pred_wind_speed": preds["pred_wind_speed"],
        "pred_wind_speed_p90": p90_wind_speed,
        "pred_wind_gust": preds["pred_wind_gust"],
        "pred_wind_gust_p90": p90_wind_gust,
        "pred_slp": preds["pred_slp"],
        "pred_wind_dir": preds["pred_wind_dir"],
        "pred_current_u": preds["pred_current_u"],
        "pred_current_v": preds["pred_current_v"],
        "pred_current_speed": preds["pred_current_speed"],
        "pred_current_dir": preds["pred_current_dir"],
        "rain_rate_mm_hr_lag0h": stacked["rain_rate_mm_hr_lag0h"].values,
    }, index=stacked.index)

    classifier_y = pd.Series(stacked["target_risk_tier"].values.astype(int),
                             index=stacked.index, name="target_risk_tier")
    print(f"Generated {classifier_X.shape[1]} classifier input features for {len(classifier_X)} rows.\n")
    return classifier_X, classifier_y


def asymmetric_cost_score(y_true, y_pred) -> float:
    """Fast vectorized asymmetric safety penalty computation."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(COST_MATRIX[y_t, y_p]))


def fit_booster(params: dict, train_X, train_y) -> xgb.Booster:
    """Uses low-level DMatrix/xgb.train API with all CPU threads."""
    params = dict(params)
    num_boost_round = params.pop("n_estimators")
    weights = compute_sample_weight(class_weight=CLASS_SAFETY_WEIGHTS, y=train_y)
    dtrain = xgb.DMatrix(train_X, label=train_y, weight=weights, nthread=-1)
    return xgb.train(params, dtrain, num_boost_round=num_boost_round)


def predict_booster(booster: xgb.Booster, X) -> np.ndarray:
    """Fast vectorized argmax over multi:softprob per-class probabilities."""
    probs = booster.predict(xgb.DMatrix(X, nthread=-1))
    return np.argmax(probs, axis=1)


def tune_hyperparameters(train_X, train_y, n_trials=15):
    """Tunes hyperparameters using multi-threaded walk-forward cross-validation."""
    optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": 5,
            "tree_method": "hist",
            "nthread": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "seed": 42,
        }
        combined = train_X.copy()
        combined["target_risk_tier"] = train_y.values

        fold_costs = []
        for fold_train, fold_val in walk_forward_folds(combined, n_splits=3, gap_hours=48):
            fold_train_X = fold_train.drop(columns=["target_risk_tier"])
            fold_train_y = fold_train["target_risk_tier"]
            fold_val_X = fold_val.drop(columns=["target_risk_tier"])
            fold_val_y = fold_val["target_risk_tier"]

            booster = fit_booster(params, fold_train_X, fold_train_y)
            preds = predict_booster(booster, fold_val_X)
            fold_costs.append(asymmetric_cost_score(fold_val_y.values, preds))
        return float(np.mean(fold_costs))

    study = optuna.create_study(direction="minimize", study_name=f"{MODEL_NAME}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main():
    print("=" * 70)
    print("TRAINING MULTI-HORIZON XGB_SAFETY_CLASSIFIER")
    print(f"Horizons: {HORIZONS} hours ahead")
    print(f"Risk Tiers: {TIER_NAMES}")
    print("=" * 70 + "\n")

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "training_features.parquet")
    labels = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "safety_labels.parquet")

    classifier_X, classifier_y = generate_classifier_dataset(df, labels)

    full = classifier_X.copy()
    full["target_risk_tier"] = classifier_y.values

    train, val, test = temporal_split(full)  # test held out
    feature_cols = [c for c in full.columns if c != "target_risk_tier"]

    print(f"Train split: {len(train)} rows ({train.index.min()} to {train.index.max()})")
    print(f"Val split:   {len(val)} rows ({val.index.min()} to {val.index.max()})")
    print(f"Test split:  {len(test)} rows (held out)\n")
    print(f"Classifier feature set ({len(feature_cols)}): {feature_cols}\n")

    print("Tuning with 15 Optuna trials (multi-threaded, hist), minimizing asymmetric cost score...")
    train_X, train_y = train[feature_cols], train["target_risk_tier"]
    best_params, best_cv_cost = tune_hyperparameters(train_X, train_y, n_trials=15)
    best_params.update({
        "objective": "multi:softprob",
        "num_class": 5,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 42,
    })
    print(f"\nBest walk-forward CV asymmetric cost: {best_cv_cost:.4f} (lower is safer)")
    print(f"Final hyperparameters: {best_params}\n")

    print("Fitting final classifier booster on full training split...")
    booster = fit_booster(best_params, train_X, train_y)

    val_X, val_y = val[feature_cols], val["target_risk_tier"]
    val_preds = predict_booster(booster, val_X)

    weighted_f1 = float(f1_score(val_y, val_preds, average="weighted"))
    cost = asymmetric_cost_score(val_y.values, val_preds)
    cm = confusion_matrix(val_y, val_preds, labels=[0, 1, 2, 3, 4])

    print("\n" + "=" * 70)
    print("OVERALL VALIDATION RESULTS:")
    print("=" * 70)
    print(f"  Weighted F1:           {weighted_f1:.4f}  (Target Benchmark >= 0.92)")
    print(f"  Asymmetric Cost Score: {cost:.4f} (lower is safer)")
    print("\nPer-class report:")
    print(classification_report(val_y, val_preds, target_names=TIER_NAMES, zero_division=0))

    # Per-horizon evaluation breakdown across all 8 horizons
    horizon_metrics = {}
    print("=" * 70)
    print("VALIDATION PERFORMANCE BY FORECAST HORIZON:")
    print("=" * 70)
    for h in HORIZONS:
        mask = (val["horizon"] == h).values
        if np.sum(mask) > 0:
            h_val_y = val_y.iloc[mask].values
            h_val_preds = val_preds[mask]
            h_f1 = float(f1_score(h_val_y, h_val_preds, average="weighted"))
            h_cost = asymmetric_cost_score(h_val_y, h_val_preds)
            h_cm = confusion_matrix(h_val_y, h_val_preds, labels=[0, 1, 2, 3, 4]).tolist()

            h_crit_idx = [i for i, y in enumerate(h_val_y) if y == 4]
            h_crit_total = len(h_crit_idx)
            h_crit_fnr = float(sum(1 for i in h_crit_idx if h_val_preds[i] != 4) / h_crit_total) if h_crit_total > 0 else 0.0

            horizon_metrics[str(h)] = {
                "weighted_f1": h_f1,
                "asymmetric_cost_score": h_cost,
                "critical_total": h_crit_total,
                "critical_fnr": h_crit_fnr,
                "confusion_matrix": h_cm,
                "n": int(np.sum(mask)),
            }
            print(f"  Horizon {h:>3}h: Weighted F1 = {h_f1:.4f} | Asymm Cost = {h_cost:.4f} | Critical FNR = {h_crit_fnr*100:.1f}% (n={np.sum(mask)})")

    val_y_list = [int(x) for x in val_y.values]
    critical_indices = [i for i, y in enumerate(val_y_list) if y == 4]
    critical_total = len(critical_indices)
    if critical_total > 0:
        critical_preds = [val_preds[i] for i in critical_indices]
        fnr_any = float(sum(1 for p in critical_preds if p != 4) / critical_total)
        fnr_to_safe = float(sum(1 for p in critical_preds if p in (0, 1)) / critical_total)
        print(f"\nCritical Risk in val set: {critical_total} rows")
        print(f"  Misclassified as ANYTHING else: {fnr_any*100:.1f}%")
        print(f"  Misclassified specifically as Very Safe/Safe: {fnr_to_safe*100:.1f}%")
    else:
        print("\nWARNING: zero Critical Risk rows in val set.")
        fnr_any, fnr_to_safe = None, None

    plot_confusion_matrix(cm, TIER_NAMES, MODEL_NAME)
    model_path = MODELS_DIR / f"{MODEL_NAME}.json"
    booster.save_model(str(model_path))

    metrics = {
        "hyperparameters": best_params,
        "weighted_f1": weighted_f1,
        "asymmetric_cost_score": cost,
        "confusion_matrix": cm.tolist(),
        "critical_total_in_val": critical_total,
        "critical_fnr_any": fnr_any,
        "critical_fnr_to_safe": fnr_to_safe,
        "horizons": horizon_metrics,
    }
    metrics_path = MODELS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to reports/figures/{MODEL_NAME}_confusion_matrix.png")
    print("\ntest set untouched — reserved for final evaluation.")


if __name__ == "__main__":
    main()
