"""
Temporal Split & Walk-Forward Cross-Validation Harness (PRD Day 5).
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


def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Chronological 3-way split: Train (70%), Validation (15%), Test (15%).
    Never shuffles data to prevent temporal lookahead leakage.
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
    Walk-forward time-series split with an embargo gap between train and test.
    Yields (fold_train_df, fold_val_df) pairs ready for model fitting and evaluation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_hours)
    for tr_idx, val_idx in tscv.split(df):
        yield df.iloc[tr_idx], df.iloc[val_idx]
