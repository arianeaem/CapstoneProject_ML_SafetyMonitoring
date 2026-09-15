"""
Temporal Split & Walk-Forward Cross-Validation Harness.
Guarantees strictly chronological splits to prevent data leakage in time-series forecasting.
"""

from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / ".venv").exists() or (current / "data").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def load_training_features() -> pd.DataFrame:
    """Loads the validated, frozen training features parquet."""
    root = _find_project_root()
    path = root / "data" / "processed" / "training_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Training features not found at {path}. Run build_features.py first.")
    return pd.read_parquet(path)


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time-indexed dataset into chronological Train (70%), Validation (15%), and Test (15%) subsets.

    Time-Series Scientific Rationale:
        Random k-fold cross-validation or data shuffling is strictly forbidden in weather and ocean
        forecasting. In maritime environments, atmospheric variables possess strong multi-day auto-correlations;
        random shuffling would leak future storm signatures into past training folds, generating artificially
        inflated accuracy scores that collapse in production.

    Parameters:
        df (pd.DataFrame): Time-indexed dataset sorted in ascending chronological order.
        train_frac (float): Fraction of initial records allocated to model training (default 0.70).
        val_frac (float): Fraction allocated to hyperparameter tuning & early stopping (default 0.15).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def walk_forward_folds(df: pd.DataFrame, n_splits: int = 5, gap_hours: int = 48):
    """
    Generates expanding-window walk-forward cross-validation splits with an embargo gap.

    Methodological Rationale:
        The 48-hour embargo gap between train and validation splits ensures that multi-day lagged
        features (e.g., 24h/48h autoregressive momentum) from the training window cannot overlap
        or artificially inform validation predictions.

    Parameters:
        df (pd.DataFrame): Sorted time-series feature matrix.
        n_splits (int): Number of expanding walk-forward folds.
        gap_hours (int): Embargo buffer length in hours (default 48 hours).

    Yields:
        tuple[pd.DataFrame, pd.DataFrame]: (fold_train_df, fold_val_df) pairs.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_hours)
    for tr_idx, val_idx in tscv.split(df):
        yield df.iloc[tr_idx], df.iloc[val_idx]

