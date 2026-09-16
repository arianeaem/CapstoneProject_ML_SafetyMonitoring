"""
Exports all trained XGBoost multi-horizon forecasters and regressors to ONNX format
for sub-5ms CPU inference in the FastAPI serving pipeline.

Dependencies:
    pip install skl2onnx onnxmltools onnxruntime --break-system-packages

Models Exported:
    - Multi-Horizon Wave Forecasters (hs, tp, swell_height, wind_wave_height)
    - Multi-Horizon Wind & SLP Forecasters (wind_speed, wind_gust, slp, wind_dir_sin, wind_dir_cos)
    - Multi-Horizon Current Forecasters (current_u, current_v)
    - Spatial Regressors (wave, wind, current)

Parity Verification:
    Every exported model is verified against its original XGBoost prediction on a sample batch.
    Max absolute error must satisfy tolerance (< 1e-3).

Run from project root:
    python src/serve/export_onnx.py
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
try:
    from onnxconverter_common.data_types import FloatTensorType
except ImportError:
    try:
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

from src.features.lagged_features import build_lagged_features
from src.features.horizon_targets import build_stacked_dataset

FORECASTER_MODELS = [
    "xgb_wave_forecaster_hs",
    "xgb_wave_forecaster_tp",
    "xgb_wave_forecaster_swell_height",
    "xgb_wave_forecaster_wind_wave_height",
    "xgb_wind_forecaster_wind_speed",
    "xgb_wind_forecaster_wind_gust",
    "xgb_wind_forecaster_slp",
    "xgb_wind_forecaster_wind_dir_sin",
    "xgb_wind_forecaster_wind_dir_cos",
    "xgb_current_forecaster_current_u",
    "xgb_current_forecaster_current_v",
]


def export_xgboost_to_onnx(model_name: str, feature_names: list, sample_X: pd.DataFrame) -> bool:
    """Loads a saved XGBoost JSON model, converts to ONNX, and verifies parity."""
    json_path = MODELS_DIR / f"{model_name}.json"
    if not json_path.exists():
        print(f"  [SKIPPED] {model_name}.json not found in {MODELS_DIR}")
        return False

    model = xgb.XGBRegressor()
    model.load_model(str(json_path))

    # Reset feature names to default positional identifiers for onnxmltools compatibility
    model.get_booster().feature_names = None

    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)

    out_path = ONNX_DIR / f"{model_name}.onnx"
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Parity verification: Compare original XGBoost output with ONNX runtime output
    sample_values = sample_X[feature_names].values.astype(np.float32)
    original_preds = model.predict(sample_X[feature_names])

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_raw = session.run(None, {"input": sample_values})[0]
    onnx_preds = np.asarray(onnx_raw, dtype=np.float32).flatten()

    max_diff = float(np.max(np.abs(original_preds - onnx_preds)))
    mean_diff = float(np.mean(np.abs(original_preds - onnx_preds)))
    ok = max_diff < 1e-3

    status = "OK" if ok else "MISMATCH"
    print(f"  [{status}] {model_name:38s} | Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f}")
    return ok


def benchmark_latency(onnx_path: Path, n_features: int, n_calls: int = 300) -> float:
    """Measures single-sample inference latency in milliseconds."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, n_features).astype(np.float32)
    
    # Warmup
    for _ in range(20):
        session.run(None, {"input": dummy})
        
    start = time.perf_counter()
    for _ in range(n_calls):
        session.run(None, {"input": dummy})
    elapsed_ms = (time.perf_counter() - start) / n_calls * 1000.0
    return elapsed_ms


def main():
    print("=" * 80)
    print("EXPORTING MULTI-HORIZON XGBOOST FORECASTERS TO ONNX")
    print("Target Serving Architecture: Sub-5ms CPU ONNX Runtime")
    print("=" * 80 + "\n")

    raw_path = PROJECT_ROOT / "data" / "processed" / "training_features.parquet"
    if not raw_path.exists():
        raw_path = PROJECT_ROOT / "data" / "processed" / "historical_11_physics_hourly.parquet"

    print(f"1. Ingesting dataset from {raw_path}...")
    df = pd.read_parquet(raw_path)
    print(f"   Loaded {len(df)} historical hourly observations.")

    print("\n2. Building multi-horizon lagged feature matrix...")
    lagged = build_lagged_features(df)
    stacked = build_stacked_dataset(df, lagged)
    
    feature_cols = [c for c in stacked.columns if not c.startswith("target_")]
    print(f"   Total Input Features per sample: {len(feature_cols)} (132 lags + 1 horizon column)")

    # Save exact positional feature names to manifest
    features_manifest_path = ONNX_DIR / "forecaster_features.json"
    with open(features_manifest_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"   Saved feature manifest to {features_manifest_path}")

    # Create verification sample
    sample_df = stacked.sample(n=min(200, len(stacked)), random_state=42)

    print(f"\n3. Converting {len(FORECASTER_MODELS)} Forecasters to ONNX and Verifying Parity...")
    results = []
    for model_name in FORECASTER_MODELS:
        success = export_xgboost_to_onnx(model_name, feature_cols, sample_df)
        results.append((model_name, success))

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print("-" * 80)
    print(f"Parity Summary: {passed}/{total} forecasters successfully exported and verified.")

    print("\n4. Inference Latency Benchmark (Target SLA Budget: < 5.0 ms/call):")
    print("-" * 80)
    for model_name, ok in results:
        if ok:
            onnx_file = ONNX_DIR / f"{model_name}.onnx"
            latency = benchmark_latency(onnx_file, len(feature_cols))
            status = "PASS" if latency < 5.0 else "WARN"
            print(f"  [{status}] {model_name:38s} | Latency: {latency:.3f} ms/call")

    print("\n" + "=" * 80)
    print(f"Export Complete. ONNX artifacts stored in: {ONNX_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
