"""
Builds processed training features from interim collocated data.
Reads: data/interim/collocated.parquet
Writes: data/processed/training_features.parquet

Usage: python src/features/build_features.py
"""

from pathlib import Path
import pandas as pd

try:
    from physics import compute_marine_physics_features
except ImportError:
    from src.features.physics import compute_marine_physics_features


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / ".venv").exists() or (current / "data").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def build_training_features() -> pd.DataFrame:
    """
    Constructs the canonical feature table from collocated raw meteorological and oceanographic records.

    Physics Feature Engineering:
        1. Cyclical Temporal Transforms: Encodes diurnal (hour_sin/cos) and seasonal (doy_sin/cos) cycles.
        2. Non-Linear Wave Dynamics: Calculates wave steepness ($Hs / L$) and swell-to-total-energy ratio.
        3. Barometric Tendency ($\Delta P_{3h}$): Captures rapid 3-hour atmospheric pressure drops.
        4. Vector Current & Wind Components: Decomposes scalar speeds and directions into orthogonal $u$ and $v$ vectors.

    Returns:
        pd.DataFrame: Cleaned feature table saved to `data/processed/training_features.parquet`.
    """
    root = _find_project_root()
    interim_path = root / "data" / "interim" / "collocated.parquet"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "training_features.parquet"

    print(f"Loading {interim_path}...")
    df_raw = pd.read_parquet(interim_path)
    print(f"  Raw collocated shape: {df_raw.shape}")

    # Compute physics and cyclical features
    df_feat = compute_marine_physics_features(df_raw)

    # Drop leading NaNs created by .shift(3) on delta_p_3h
    df_clean = df_feat.dropna()
    leading_nans = len(df_feat) - len(df_clean)
    print(f"  Dropped {leading_nans} leading NaN rows from shift(3)")

    assert len(df_clean.dropna()) == len(df_clean), "Unexpected NaNs remaining in training features!"
    assert len(df_clean) == len(df_raw) - 3, f"Expected {len(df_raw) - 3} rows, got {len(df_clean)}"

    df_clean.to_parquet(out_path)
    print(f"wrote {len(df_clean)} rows, {df_clean.shape[1]} columns -> {out_path}")
    print(f"Columns ({df_clean.shape[1]}): {list(df_clean.columns)}")
    return df_clean


# TODO: Evaluate adding tidal height harmonics (M2/S2 tidal constituents) when Batangas tide gauge telemetry is integrated.



if __name__ == "__main__":
    build_training_features()
