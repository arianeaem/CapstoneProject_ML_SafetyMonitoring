"""
Exports all trained models (wave, wind, current regressors + safety
classifier — 12 individual XGBoost models total) to ONNX, for the FastAPI
inference service.

Requires onnxmltools (NOT yet installed — see requirements.txt note below).
Feature counts are NOT hardcoded — each model's input shape is derived from
the same feature-column functions used during training (imported directly
from train_safety_classifier.py), so this can't silently drift out of sync
with what each model actually expects.

Every exported model is verified against its original XGBoost predictions on
a small sample before being trusted — an ONNX conversion that "succeeds" but
produces different numbers is worse than one that fails loudly, since it
would silently corrupt every downstream prediction in the live service.

Run from the project root: python src\\serve\\export_onnx.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxconverter_common.data_types import FloatTensorType
# The classifier operator does a stricter isinstance check than the regressor
# operator does, and only accepts onnxmltools' OWN FloatTensorType class, not
# onnxconverter_common's — they're usually interchangeable but not here. All
# 11 regressor exports worked fine with the import above; only the classifier
# needs this second one.
from onnxmltools.convert.common.data_types import FloatTensorType as ClassifierFloatTensorType
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

from src.validation.splits import load_training_features
from src.models.train_safety_classifier import (
    wave_feature_columns, wind_feature_columns, current_feature_columns,
    WAVE_TARGETS, WIND_TARGETS,
)

# name -> (feature-column function, is a raw Booster rather than XGBRegressor)
WAVE_MODELS = [f"xgb_wave_regressor_{t}" for t in WAVE_TARGETS]
WIND_MODELS = [f"xgb_wind_regressor_{t}" for t in WIND_TARGETS] + \
              ["xgb_wind_regressor_wind_dir_sin", "xgb_wind_regressor_wind_dir_cos"]
CURRENT_MODELS = ["xgb_current_regressor_current_u", "xgb_current_regressor_current_v"]
CLASSIFIER_MODEL = "xgb_safety_classifier"


def export_regressor(name: str, features: list, sample_X: pd.DataFrame) -> bool:
    """Loads a saved XGBRegressor, converts to ONNX, verifies parity. Returns True on success."""
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / f"{name}.json"))

    # onnxmltools' converter only understands XGBoost's default f0/f1/f2... feature
    # naming — it errors on real pandas column names (which is what got stored,
    # since training used named DataFrames). Resetting to None makes the booster
    # dump with generic names; this does not change predictions at all, since
    # they're already positional (column order), not name-based.
    model.get_booster().feature_names = None

    initial_type = [("input", FloatTensorType([None, len(features)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)

    out_path = ONNX_DIR / f"{name}.onnx"
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Verify: same predictions from the original model and the exported ONNX file
    original_preds = model.predict(sample_X[features])
    session = ort.InferenceSession(str(out_path))
    onnx_raw = session.run(None, {"input": np.asarray(sample_X[features].values, dtype=np.float32)})[0]
    onnx_preds = np.asarray(onnx_raw, dtype=np.float32).flatten()

    max_diff = float(np.max(np.abs(original_preds - onnx_preds)))
    ok = max_diff < 1e-3  # small float tolerance, not exact bit-for-bit
    status = "OK" if ok else "MISMATCH"
    print(f"  [{status}] {name}: max prediction diff = {max_diff:.6f}")
    return ok


def export_classifier(features: list, sample_X: pd.DataFrame) -> bool:
    """Classifier is a raw xgb.Booster (see train_safety_classifier.py's docstring
    for why — XGBClassifier's sklearn wrapper broke on rare-class folds).
    convert_xgboost is documented to support raw Boosters directly, same call
    pattern as the regressors — but this is the one conversion path in this
    script that hasn't been run against your actual environment, so it's
    wrapped defensively rather than assumed to just work."""
    booster = xgb.Booster()
    booster.load_model(str(MODELS_DIR / f"{CLASSIFIER_MODEL}.json"))
    booster.feature_names = None  # same fix as export_regressor — see its comment

    initial_type = [("input", ClassifierFloatTensorType([None, len(features)]))]
    try:
        onnx_model = convert_xgboost(booster, initial_types=initial_type)
    except Exception as e:
        print(f"  [FAILED] {CLASSIFIER_MODEL} conversion raised: {type(e).__name__}: {e}")
        print(f"  This is the one export path in this script not yet verified end-to-end.")
        print(f"  Try instead: load the same file into a fresh xgb.XGBClassifier(n_classes=5) "
              f"via .load_model() and pass THAT object to convert_xgboost — the sklearn "
              f"wrapper is more commonly tested with onnxmltools than a raw Booster. "
              f"Loading (not fitting) a booster into either API is safe; only .fit() had "
              f"the earlier contiguous-class bug, not .load_model()/.predict().")
        return False

    out_path = ONNX_DIR / f"{CLASSIFIER_MODEL}.onnx"
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    dtest = xgb.DMatrix(sample_X[features])
    original_probs = np.asarray(booster.predict(dtest), dtype=np.float32)
    original_preds = [int(np.argmax(row)) for row in original_probs]

    session = ort.InferenceSession(str(out_path))
    onnx_out = session.run(None, {"input": np.asarray(sample_X[features].values, dtype=np.float32)})
    # multi:softprob ONNX output is typically (labels, probabilities) — take probabilities
    onnx_raw_probs = onnx_out[1] if len(onnx_out) > 1 else onnx_out[0]
    onnx_probs = np.asarray(onnx_raw_probs, dtype=np.float32)
    onnx_preds = [int(np.argmax(row)) for row in onnx_probs]

    matches = sum(1 for o, p in zip(original_preds, onnx_preds) if o == p)
    ok = matches == len(sample_X)
    status = "OK" if ok else "MISMATCH"
    print(f"  [{status}] {CLASSIFIER_MODEL}: {matches}/{len(sample_X)} predicted classes match")
    return ok


def benchmark_latency(onnx_path: Path, n_features: int, n_calls: int = 200) -> float:
    """Average single-row inference time in ms — PRD target is <5ms per model call."""
    session = ort.InferenceSession(str(onnx_path))
    dummy = np.asarray(np.random.rand(1, n_features), dtype=np.float32)
    import time
    start = time.perf_counter()
    for _ in range(n_calls):
        session.run(None, {"input": dummy})
    elapsed_ms = (time.perf_counter() - start) / n_calls * 1000
    return elapsed_ms


def main():
    print(f"Loading a sample of rows to verify export parity...")
    df = load_training_features()
    sample = df.sample(n=min(100, len(df)), random_state=42)

    wave_feats = wave_feature_columns(df)
    wind_feats = wind_feature_columns(df)
    current_feats = current_feature_columns(df)

    # Save the EXACT training-time feature order for each model. ONNX models
    # are purely positional — they have no concept of column names, only
    # column order. Serving-time code must use these exact saved lists, never
    # recompute feature selection against a freshly-built live DataFrame and
    # assume its column order happens to match. A silent order mismatch would
    # produce confidently wrong predictions with no error at all.
    import json
    with open(ONNX_DIR / "wave_regressor_features.json", "w") as f:
        json.dump(wave_feats, f, indent=2)
    with open(ONNX_DIR / "wind_regressor_features.json", "w") as f:
        json.dump(wind_feats, f, indent=2)
    with open(ONNX_DIR / "current_regressor_features.json", "w") as f:
        json.dump(current_feats, f, indent=2)

    print(f"\nExporting {len(WAVE_MODELS)} wave regressor models ({len(wave_feats)} features each)...")
    wave_results = [export_regressor(name, wave_feats, sample) for name in WAVE_MODELS]

    print(f"\nExporting {len(WIND_MODELS)} wind regressor models ({len(wind_feats)} features each)...")
    wind_results = [export_regressor(name, wind_feats, sample) for name in WIND_MODELS]

    print(f"\nExporting {len(CURRENT_MODELS)} current regressor models ({len(current_feats)} features each)...")
    current_results = [export_regressor(name, current_feats, sample) for name in CURRENT_MODELS]

    print(f"\nExporting safety classifier (12 pred_* features)...")
    # Classifier's features are the 12 "pred_*" columns generated from regressor
    # outputs, not raw environmental columns — see train_safety_classifier.py.
    # For export/latency purposes only the count and dtype matter, so build a
    # representative sample directly rather than re-running all 3 regressors here.
    classifier_feature_names = [
        "pred_hs", "pred_tp", "pred_swell_height", "pred_wind_wave_height",
        "pred_wind_speed", "pred_wind_gust", "pred_delta_p_3h", "pred_wind_dir",
        "pred_current_u", "pred_current_v", "pred_current_speed", "pred_current_dir",
    ]
    classifier_sample = pd.DataFrame(
        np.random.rand(len(sample), len(classifier_feature_names)),
        columns=classifier_feature_names, index=sample.index,
    )
    with open(ONNX_DIR / "classifier_features.json", "w") as f:
        json.dump(classifier_feature_names, f, indent=2)
    classifier_result = export_classifier(classifier_feature_names, classifier_sample)

    all_results = wave_results + wind_results + current_results + [classifier_result]
    total, passed = len(all_results), sum(all_results)
    print(f"\n{passed}/{total} models exported and verified successfully.")

    if passed < total:
        print("WARNING: one or more models failed parity verification — do NOT use those "
              ".onnx files in the FastAPI service until this is resolved.")

    print("\nLatency benchmark (target: <5ms per call, PRD Section 9.1):")
    sample_paths = [
        (ONNX_DIR / f"{WAVE_MODELS[0]}.onnx", len(wave_feats)),
        (ONNX_DIR / f"{WIND_MODELS[0]}.onnx", len(wind_feats)),
        (ONNX_DIR / f"{CURRENT_MODELS[0]}.onnx", len(current_feats)),
        (ONNX_DIR / f"{CLASSIFIER_MODEL}.onnx", len(classifier_feature_names)),
    ]
    for path, n_feat in sample_paths:
        if path.exists():
            latency = benchmark_latency(path, n_feat)
            status = "PASS" if latency < 5.0 else "MISS"
            print(f"  [{status}] {path.stem}: {latency:.3f} ms/call")

    print(f"\nExported models saved to {ONNX_DIR}")


if __name__ == "__main__":
    main()
