"""
Shared evaluation figures for the physics regressors (wave, wind, current) and safety classifier.
Import these from every train_*.py script instead of re-implementing plotting
per model — keeps the figures visually consistent across the defense deck.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# eval_plots.py lives at <project_root>/src/models/eval_plots.py — anchor to
# project root explicitly rather than a cwd-relative path, since relative
# "reports/figures" silently creates a WRONG folder (e.g. src/models/reports/figures)
# if this is ever run from a different working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = str(_PROJECT_ROOT / "reports" / "figures")


def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_pred_vs_actual(y_true, y_pred, target_name: str, model_name: str):
    """Scatter of predicted vs actual on the validation set, with a y=x reference line.
    The single most important figure — a tight diagonal cluster is the headline evidence."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, color="#003049")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#c1121f", label="y = x (perfect prediction)")
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"{model_name}: Predicted vs Actual — {target_name} (validation set)")
    ax.legend()
    out_path = f"{FIGURES_DIR}/{model_name}_{target_name}_pred_vs_actual.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_residuals_over_time(timestamps, y_true, y_pred, target_name: str, model_name: str):
    """Residual (pred - actual) over the validation period. Catches error clustering
    around specific events (e.g. a storm month) that a scatter plot would hide."""
    _ensure_dir()
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, residuals, color="#669bbc", linewidth=0.8)
    ax.axhline(0, color="#780000", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Residual (predicted - actual {target_name})")
    ax.set_title(f"{model_name}: Residuals over time — {target_name}")
    out_path = f"{FIGURES_DIR}/{model_name}_{target_name}_residuals_over_time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importance(model, feature_names, target_name: str, model_name: str, top_n: int = 15):
    """XGBoost gain-based feature importance — usually the first thing a panelist asks about."""
    _ensure_dir()
    raw_importances = [float(x) for x in model.feature_importances_]
    pairs = sorted(zip(raw_importances, feature_names), key=lambda x: x[0], reverse=True)[:top_n]
    pairs.reverse()

    top_scores = [p[0] for p in pairs]
    top_names = [str(p[1]) for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_names, top_scores, color="#780000")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"{model_name}: Top {top_n} Feature Importances — {target_name}")
    out_path = f"{FIGURES_DIR}/{model_name}_{target_name}_feature_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_walk_forward_scores(fold_scores: list, target_name: str, model_name: str):
    """RMSE per walk-forward fold — demonstrates the temporal CV strategy was actually
    followed, and surfaces any single fold that behaves unusually (e.g. a typhoon season)."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = list(range(len(fold_scores)))
    ax.bar(folds, fold_scores, color="#669bbc")
    ax.axhline(np.mean(fold_scores), color="#c1121f", linestyle="--",
               label=f"mean = {np.mean(fold_scores):.4f}")
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("RMSE")
    ax.set_xticks(folds)
    ax.set_title(f"{model_name}: Walk-Forward CV RMSE per Fold — {target_name}")
    ax.legend()
    out_path = f"{FIGURES_DIR}/{model_name}_{target_name}_walkforward_folds.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_actual_vs_predicted_overlay(timestamps, y_true, y_pred, target_name: str, model_name: str,
                                      window_days: int = 30):
    """Actual vs predicted as two overlaid lines, over one representative window rather
    than the full validation set — the easiest figure to read at a glance in a slide."""
    _ensure_dir()
    df = pd.DataFrame({"time": timestamps, "actual": y_true, "predicted": y_pred}).set_index("time")
    window = df.iloc[: window_days * 24] if len(df) > window_days * 24 else df

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window.index, window["actual"], label="Actual", color="#003049", linewidth=1.2)
    ax.plot(window.index, window["predicted"], label="Predicted", color="#c1121f",
            linewidth=1.2, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel(target_name)
    ax.set_title(f"{model_name}: Actual vs Predicted — {target_name} "
                 f"(first {window_days} days of validation)")
    ax.legend()
    out_path = f"{FIGURES_DIR}/{model_name}_{target_name}_overlay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm, class_names, model_name: str):
    """5x5 confusion matrix heatmap for the safety classifier. Rows = true tier,
    columns = predicted tier — the row for Critical Risk is the one that matters
    most: mass should sit on the diagonal, not spread into the low-risk columns."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name}: Confusion Matrix (validation set)")
    cm_max = float(cm.max())
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if float(cm[i, j]) > cm_max / 2 else "black")
    fig.colorbar(im, ax=ax, label="count")
    out_path = f"{FIGURES_DIR}/{model_name}_confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
