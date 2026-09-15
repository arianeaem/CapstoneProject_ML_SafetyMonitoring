"""
Baseline Comparison Suite: ML Forecasters vs Physical Baselines.
Compares multi-horizon ML forecaster models against two standard baselines:
1. Persistence Baseline: predicted value(t+H) = value(t) (lag0h observation)
2. Climatological Baseline: predicted value(t+H) = historical mean for (hour-of-day, day-of-year)
   fit strictly on the `train` split only to prevent lookahead leakage.
   For circular directions (wind_dir, current_dir), circular mean is computed via
   sin/cos component averaging (vector mean), not naive degree averaging.

Evaluates across all 5 forecast horizons: 1h, 6h, 24h (1-day), 72h (3-day), 168h (7-day).
Produces formatted summary tables and skill-decay metrics.

Run from project root: python src/models/baselines.py
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from src.validation.splits import temporal_split
from src.features.lagged_features import build_lagged_features
from src.features.horizon_targets import build_stacked_dataset, HORIZONS

VARIABLES = [
    # Wave variables
    ("hs", "xgb_wave_forecaster_hs", "linear"),
    ("tp", "xgb_wave_forecaster_tp", "linear"),
    ("swell_height", "xgb_wave_forecaster_swell_height", "linear"),
    ("wind_wave_height", "xgb_wave_forecaster_wind_wave_height", "linear"),
    # Wind & atmospheric variables
    ("wind_speed", "xgb_wind_forecaster_wind_speed", "linear"),
    ("wind_gust", "xgb_wind_forecaster_wind_gust", "linear"),
    ("slp", "xgb_wind_forecaster_slp", "linear"),
    # Current vector variables
    ("current_u", "xgb_current_forecaster_current_u", "linear"),
    ("current_v", "xgb_current_forecaster_current_v", "linear"),
]


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae_score(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def circular_angular_diff(y_true_deg, y_pred_deg):
    """Computes shortest angular difference in degrees accounting for 360 wrap-around."""
    diff = np.abs(y_true_deg - y_pred_deg) % 360
    return np.minimum(diff, 360 - diff)


def circular_rmse_score(y_true_deg, y_pred_deg) -> float:
    diffs = circular_angular_diff(y_true_deg, y_pred_deg)
    return float(np.sqrt(np.mean(diffs ** 2)))


def circular_mae_score(y_true_deg, y_pred_deg) -> float:
    diffs = circular_angular_diff(y_true_deg, y_pred_deg)
    return float(np.mean(diffs))


def persistence_prediction(stacked_df: pd.DataFrame, var: str) -> pd.Series:
    """predicted value(t+H) = value(t) — the naive forecast. Reads the lag0h
    column built in Step 1 (the current/most-recent observed value), rather
    than re-deriving it — same value, single source of truth."""
    return stacked_df[f"{var}_lag0h"]


def fit_climatology(train_df: pd.DataFrame, var: str) -> pd.Series:
    """Fits a (hour-of-day, day-of-year) -> historical mean lookup table
    using ONLY the training split. This is the actual leakage guard — a
    version that computed climatology from the full dataset (train+val+test
    combined) would let future information leak into the val/test baseline."""
    ts = pd.to_datetime(train_df.index)
    return train_df.groupby([ts.hour, ts.dayofyear])[var].mean()


def apply_climatology(target_index: pd.DatetimeIndex, climatology_table: pd.Series, global_mean: float) -> np.ndarray:
    """Broadcasts a climatology table FIT ON TRAIN ONLY onto any other split
    (val or test), looking up each row's own hour/day-of-year. Never re-fits
    on the split being evaluated."""
    keys = list(zip(target_index.hour, target_index.dayofyear))
    return np.array([climatology_table.get(k, global_mean) for k in keys])


def fit_circular_climatology(train_df: pd.DataFrame, sin_col: str, cos_col: str) -> pd.Series:
    """Computes circular mean direction per (hour, day-of-year) via sin/cos averaging on train split only."""
    ts = pd.to_datetime(train_df.index)
    mean_sin = train_df.groupby([ts.hour, ts.dayofyear])[sin_col].mean()
    mean_cos = train_df.groupby([ts.hour, ts.dayofyear])[cos_col].mean()
    mean_dir = (np.degrees(np.arctan2(mean_sin, mean_cos))) % 360
    return mean_dir


def compare_to_baselines(model_rmse: dict, persistence_rmse: dict, climatology_rmse: dict, units: dict) -> pd.DataFrame:
    """model_rmse, persistence_rmse, climatology_rmse: dicts keyed by (variable, horizon)."""
    rows = []
    for key in model_rmse:
        var, h = key
        m_rmse = model_rmse[key]
        p_rmse = persistence_rmse[key]
        c_rmse = climatology_rmse[key]
        unit = units.get(var, "")
        rows.append({
            "variable": var,
            "unit": unit,
            "horizon_h": h,
            "model_rmse": round(m_rmse, 4),
            "persistence_rmse": round(p_rmse, 4),
            "climatology_rmse": round(c_rmse, 4),
            "beats_persistence": m_rmse < p_rmse,
            "beats_climatology": m_rmse < c_rmse,
            "pct_improvement_over_persistence": round((p_rmse - m_rmse) / p_rmse * 100, 2),
            "pct_improvement_over_climatology": round((c_rmse - m_rmse) / c_rmse * 100, 2),
        })
    return pd.DataFrame(rows)


def plot_skill_decay(df_results: pd.DataFrame):
    """Generates comparison skill-decay curves for defense reporting."""
    unique_vars = df_results["variable"].unique()
    n_vars = len(unique_vars)
    fig, axes = plt.subplots(int(np.ceil(n_vars / 3)), 3, figsize=(16, 14))
    axes = axes.flatten()

    for i, var in enumerate(unique_vars):
        ax = axes[i]
        sub = df_results[df_results["variable"] == var].sort_values("horizon_h")
        horizons = sub["horizon_h"].values
        unit = sub["unit"].iloc[0]

        ax.plot(horizons, sub["model_rmse"].values, marker="o", color="#0284c7", linewidth=2.5, label="XGBoost Forecaster")
        ax.plot(horizons, sub["persistence_rmse"].values, marker="s", linestyle="--", color="#dc2626", label="Persistence")
        ax.plot(horizons, sub["climatology_rmse"].values, marker="^", linestyle=":", color="#16a34a", label="Climatology")

        ax.set_title(f"{var.upper()} Forecast Error ({unit}) vs Horizon", fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon (hours)", fontsize=10)
        ax.set_ylabel(f"RMSE ({unit})" if unit else "RMSE", fontsize=10)
        ax.set_xticks(HORIZONS)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = FIGURES_DIR / "forecaster_baseline_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved skill decay curves to {plot_path}")


def main():
    print("=" * 70)
    print("BASELINE COMPARISON BENCHMARK SUITE")
    print("Models vs Persistence vs Climatology across all 5 Horizons")
    print("Includes Linear Regressors + Circular Wind & Current Directions")
    print("=" * 70 + "\n")

    raw_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "training_features.parquet")
    lagged = build_lagged_features(raw_df)
    stacked = build_stacked_dataset(raw_df, lagged)

    train_stacked, val_stacked, test_stacked = temporal_split(stacked)
    train_raw, val_raw, test_raw = temporal_split(raw_df)

    feature_cols = [c for c in stacked.columns if not c.startswith("target_")]

    model_rmse = {}
    persistence_rmse = {}
    climatology_rmse = {}
    units = {
        "hs": "m", "tp": "s", "swell_height": "m", "wind_wave_height": "m",
        "wind_speed": "m/s", "wind_gust": "m/s", "slp": "hPa",
        "current_u": "m/s", "current_v": "m/s",
        "wind_dir_deg": "deg", "current_dir_deg": "deg",
    }

    print(f"1. Evaluating 9 Linear Target Variables on validation split ({len(val_stacked)} rows)...\n")

    # Store predictions for circular reconstruction
    predictions_cache = {}

    for var, model_filename, _ in VARIABLES:
        target_col = f"target_{var}"
        model_path = MODELS_DIR / f"{model_filename}.json"

        if not model_path.exists():
            print(f"WARNING: Model file {model_path} not found! Skipping {var}...")
            continue

        model = xgb.XGBRegressor()
        model.load_model(str(model_path))

        val_preds = model.predict(val_stacked[feature_cols])
        predictions_cache[var] = val_preds

        # Persistence prediction
        pers_preds = persistence_prediction(val_stacked, var).values

        # Fit Climatology strictly on train split
        clim_table = fit_climatology(train_raw, var)
        global_mean = float(train_raw[var].mean())
        val_ts = pd.to_datetime(val_stacked.index)
        clim_preds = apply_climatology(val_ts, clim_table, global_mean)

        y_true_all = val_stacked[target_col].values

        for h in HORIZONS:
            mask = (val_stacked["horizon"] == h).values
            if np.sum(mask) == 0:
                continue

            y_t = y_true_all[mask]
            m_p = val_preds[mask]
            p_p = pers_preds[mask]
            c_p = clim_preds[mask]

            key = (var, h)
            model_rmse[key] = rmse_score(y_t, m_p)
            persistence_rmse[key] = rmse_score(y_t, p_p)
            climatology_rmse[key] = rmse_score(y_t, c_p)

    # -----------------------------------------------------------------------
    # 2. Circular Wind Direction Evaluation (Degrees, Circular Mean Climatology)
    # -----------------------------------------------------------------------
    sin_path = MODELS_DIR / "xgb_wind_forecaster_wind_dir_sin.json"
    cos_path = MODELS_DIR / "xgb_wind_forecaster_wind_dir_cos.json"

    if sin_path.exists() and cos_path.exists() and "target_wind_dir" in val_stacked.columns:
        print("2. Evaluating Circular Wind Direction (wind_dir_deg)...")
        sin_model = xgb.XGBRegressor()
        sin_model.load_model(str(sin_path))
        cos_model = xgb.XGBRegressor()
        cos_model.load_model(str(cos_path))

        pred_sin = sin_model.predict(val_stacked[feature_cols])
        pred_cos = cos_model.predict(val_stacked[feature_cols])
        pred_wind_dir = (np.degrees(np.arctan2(pred_sin, pred_cos))) % 360

        # Persistence for wind_dir (observed wind_dir at time t)
        if "wind_dir" in raw_df.columns:
            pers_wind_dir = raw_df.loc[val_stacked.index, "wind_dir"].values
        elif "wind_dir_lag0h" in val_stacked.columns:
            pers_wind_dir = val_stacked["wind_dir_lag0h"].values
        else:
            pers_wind_dir = (np.degrees(np.arctan2(val_stacked["wind_v_lag0h"].values, val_stacked["wind_u_lag0h"].values))) % 360

        # Circular Climatology fit on train (sin/cos component means)
        train_raw_sin = np.sin(np.radians(train_raw["wind_dir"]))
        train_raw_cos = np.cos(np.radians(train_raw["wind_dir"]))
        train_df_wind = pd.DataFrame({"sin": train_raw_sin, "cos": train_raw_cos}, index=train_raw.index)
        clim_wind_table = fit_circular_climatology(train_df_wind, "sin", "cos")
        global_wind_mean = float((np.degrees(np.arctan2(train_raw_sin.mean(), train_raw_cos.mean()))) % 360)
        val_ts = pd.to_datetime(val_stacked.index)
        clim_wind_preds = apply_climatology(val_ts, clim_wind_table, global_wind_mean)

        y_true_wind_dir = val_stacked["target_wind_dir"].values

        for h in HORIZONS:
            mask = (val_stacked["horizon"] == h).values
            if np.sum(mask) == 0:
                continue

            y_t = y_true_wind_dir[mask]
            m_p = pred_wind_dir[mask]
            p_p = pers_wind_dir[mask]
            c_p = clim_wind_preds[mask]

            key = ("wind_dir_deg", h)
            model_rmse[key] = circular_rmse_score(y_t, m_p)
            persistence_rmse[key] = circular_rmse_score(y_t, p_p)
            climatology_rmse[key] = circular_rmse_score(y_t, c_p)

    # -----------------------------------------------------------------------
    # 3. Circular Current Direction Evaluation (Degrees, Vector Mean Climatology)
    # -----------------------------------------------------------------------
    if "current_u" in predictions_cache and "current_v" in predictions_cache:
        print("3. Evaluating Circular Current Direction (current_dir_deg)...")
        pred_u = predictions_cache["current_u"]
        pred_v = predictions_cache["current_v"]
        pred_current_dir = (np.degrees(np.arctan2(pred_v, pred_u))) % 360

        # Persistence for current_dir
        u_lag0 = val_stacked["current_u_lag0h"].values
        v_lag0 = val_stacked["current_v_lag0h"].values
        pers_current_dir = (np.degrees(np.arctan2(v_lag0, u_lag0))) % 360

        # Circular Climatology fit on train (vector component means)
        clim_current_table = fit_circular_climatology(train_raw, "current_u", "current_v")
        global_current_mean = float((np.degrees(np.arctan2(train_raw["current_v"].mean(), train_raw["current_u"].mean()))) % 360)
        val_ts = pd.to_datetime(val_stacked.index)
        clim_current_preds = apply_climatology(val_ts, clim_current_table, global_current_mean)

        true_u_all = val_stacked["target_current_u"].values
        true_v_all = val_stacked["target_current_v"].values
        y_true_curr_dir = (np.degrees(np.arctan2(true_v_all, true_u_all))) % 360

        for h in HORIZONS:
            mask = (val_stacked["horizon"] == h).values
            if np.sum(mask) == 0:
                continue

            y_t = y_true_curr_dir[mask]
            m_p = pred_current_dir[mask]
            p_p = pers_current_dir[mask]
            c_p = clim_current_preds[mask]

            key = ("current_dir_deg", h)
            model_rmse[key] = circular_rmse_score(y_t, m_p)
            persistence_rmse[key] = circular_rmse_score(y_t, p_p)
            climatology_rmse[key] = circular_rmse_score(y_t, c_p)

    # Build comparison summary table
    df_comparison = compare_to_baselines(model_rmse, persistence_rmse, climatology_rmse, units)

    print("\n" + "-" * 85)
    print("MANDATORY BASELINE COMPARISON RESULTS SUMMARY (VAL SPLIT)")
    print("-" * 85)
    print(df_comparison.to_string(index=False))

    # Save outputs
    csv_path = REPORTS_DIR / "baseline_comparison_summary.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nSaved comparison summary table to {csv_path}")

    metrics_json_path = MODELS_DIR / "baseline_comparison_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(df_comparison.to_dict(orient="records"), f, indent=2)
    print(f"Saved JSON metrics to {metrics_json_path}")

    # Generate and save figures
    if len(df_comparison) > 0:
        plot_skill_decay(df_comparison)

    print("\n" + "=" * 70)
    print("Baseline Comparison Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
