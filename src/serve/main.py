"""
FastAPI inference service. Wraps the 12 ONNX models (wave/wind/current
regressors + safety classifier) and the deterministic hard-gate layer behind
one HTTP endpoint — this is the boundary WeatherForecastService.php and
WeatherSafetyService.php call into from Laravel.

INPUT DESIGN — worth understanding, not just accepting: the request does NOT
include raw wave data (hs, tp, swell_height, wind_wave_height). Every
regressor's feature set was built to predict wave/wind/current state FROM
current + wind + pressure + rain + time — never from wave data itself (wave
is only ever a target, never an input, anywhere in this pipeline). So the
live boundary conditions Laravel already pulls from CMEMS/ECMWF/GFS (current,
wind, pressure, rain) are the entire input; wave state is always something
this service PRODUCES, never something the caller needs to already know.

Run: uvicorn src.serve.main:app --host 127.0.0.1 --port 8001
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_DIR = PROJECT_ROOT / "models" / "onnx"

from src.models.train_safety_classifier import WAVE_TARGETS, WIND_TARGETS
from src.serve.hard_gate import apply_hard_gate, TIER_NAMES

app = FastAPI(title="Camp FreedivePH Weather Safety Inference Service")

# ---------------------------------------------------------------------------
# Load all 12 ONNX sessions ONCE at startup, not per-request — this is what
# keeps inference in the sub-millisecond range confirmed during export.
#
# FEATURE ORDER: loaded from the JSON manifests export_onnx.py saves at
# export time, NOT recomputed via wave_feature_columns()/etc against a live
# request DataFrame. ONNX models are purely positional — a freshly-built
# DataFrame's column order has no guaranteed relationship to what the model
# was trained on, and a mismatch would silently produce wrong predictions
# with no error. The saved manifest is the single source of truth for order.
# ---------------------------------------------------------------------------
SESSIONS = {}
FEATURE_ORDER = {}


@app.on_event("startup")
def load_models():
    model_names = (
        [f"xgb_wave_regressor_{t}" for t in WAVE_TARGETS]
        + [f"xgb_wind_regressor_{t}" for t in WIND_TARGETS]
        + ["xgb_wind_regressor_wind_dir_sin", "xgb_wind_regressor_wind_dir_cos"]
        + ["xgb_current_regressor_current_u", "xgb_current_regressor_current_v"]
        + ["xgb_safety_classifier"]
    )
    missing = [n for n in model_names if not (ONNX_DIR / f"{n}.onnx").exists()]
    if missing:
        raise RuntimeError(f"Missing ONNX models: {missing} — run src/serve/export_onnx.py first")

    for name in model_names:
        SESSIONS[name] = ort.InferenceSession(str(ONNX_DIR / f"{name}.onnx"))

    import json
    for key, filename in [("wave", "wave_regressor_features.json"),
                           ("wind", "wind_regressor_features.json"),
                           ("current", "current_regressor_features.json"),
                           ("classifier", "classifier_features.json")]:
        manifest_path = ONNX_DIR / filename
        if not manifest_path.exists():
            raise RuntimeError(f"Missing feature manifest: {manifest_path} — re-run export_onnx.py")
        with open(manifest_path) as f:
            FEATURE_ORDER[key] = json.load(f)

    print(f"Loaded {len(SESSIONS)} ONNX models and {len(FEATURE_ORDER)} feature-order manifests.")


def run_onnx(session_name: str, X: np.ndarray) -> np.ndarray:
    session = SESSIONS[session_name]
    result = session.run(None, {"input": X.astype(np.float32)})[0]
    return result.flatten()


def run_classifier_onnx(X: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """The classifier's ONNX export can return its outputs in either order
    ([labels, probabilities] or [probabilities, labels] depending on
    onnxmltools version — export_onnx.py's own verification step had to try
    output[1] as a fallback for exactly this reason. Rather than assume a
    fixed index here too, find whichever output array's last dimension
    actually matches n_classes — robust regardless of ordering."""
    session = SESSIONS["xgb_safety_classifier"]
    outputs = session.run(None, {"input": X.astype(np.float32)})
    for out in outputs:
        arr = np.array(out)
        if arr.ndim == 2 and arr.shape[1] == n_classes:
            return arr
    raise RuntimeError(
        f"Could not find a ({len(X)}, {n_classes})-shaped probability output among "
        f"the classifier's ONNX outputs — got shapes {[np.array(o).shape for o in outputs]}. "
        f"This needs a manual look before the service can be trusted in production."
    )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class HourlyReading(BaseModel):
    timestamp: str = Field(..., description="ISO 8601, e.g. 2026-09-01T00:00:00")
    current_u: float
    current_v: float
    current_speed: float
    current_dir: float
    wind_u: float
    wind_v: float
    wind_speed: float
    wind_gust: float
    wind_dir: float
    slp: float
    rain_rate_mm_hr: float


class PagasaAdvisory(BaseModel):
    tcws_signal: int = 0
    gale_warning: bool = False
    tsunami_warning: bool = False


class ForecastRequest(BaseModel):
    readings: List[HourlyReading] = Field(
        ..., min_items=4,
        description="At least 4 consecutive hourly readings — delta_p_3h needs a 3-hour lag, "
                    "so the first 3 rows in any batch can't produce a prediction. Not a "
                    "practical limit for a real 16-day (384-hour) forecast batch."
    )
    pagasa: Optional[PagasaAdvisory] = None


class HourlyPrediction(BaseModel):
    timestamp: str
    predicted_hs: float
    predicted_tp: float
    predicted_swell_height: float
    predicted_wind_wave_height: float
    predicted_wind_speed: float
    predicted_wind_gust: float
    predicted_wind_dir: float
    predicted_delta_p_3h: float
    predicted_current_u: float
    predicted_current_v: float
    predicted_current_speed: float
    predicted_current_dir: float
    ml_risk_tier: str
    final_risk_tier: str
    hard_gate_triggered: bool
    override_reasons: List[str]


class ForecastResponse(BaseModel):
    predictions: List[HourlyPrediction]
    skipped_leading_rows: int


# ---------------------------------------------------------------------------
# Feature engineering — matches training EXACTLY (same formulas used in
# build_features.py / train_wind.py). Kept self-contained here rather than
# importing from an external module whose structure isn't guaranteed stable,
# since these specific formulas are simple and already well-established.
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])

    df["delta_p_3h"] = df["slp"].diff(3)
    df["wind_current_alignment"] = np.minimum(
        np.abs(df["wind_dir"] - df["current_dir"]) % 360,
        360 - (np.abs(df["wind_dir"] - df["current_dir"]) % 360),
    )
    hour = ts.dt.hour
    doy = ts.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(SESSIONS)}


@app.post("/forecast/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    if len(SESSIONS) == 0:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    raw = pd.DataFrame([r.model_dump() for r in request.readings])
    df = engineer_features(raw)

    valid = df.dropna(subset=["delta_p_3h"]).reset_index(drop=True)
    skipped = len(df) - len(valid)
    if len(valid) == 0:
        raise HTTPException(status_code=400,
                             detail="No rows have a valid delta_p_3h — send more consecutive hourly readings")

    # STAGED EXECUTION ORDER — wave regressor must run FIRST, not in parallel
    # with wind/current. Reason (found via the missing_cols check below,
    # after the first real test run failed loudly rather than silently):
    # wave_steepness and swell_ratio are engineered from hs/tp/swell_height,
    # excluded from the WAVE regressor's own inputs (correctly — that would
    # be target leakage), but NOT excluded from wind/current's inputs — those
    # two models legitimately learned to use them as real predictive
    # features during training. Since the live request never contains raw
    # wave data (wave state is always an output here, never an input), these
    # two columns can only be computed AFTER the wave regressor produces its
    # own predictions. This corrects an earlier claim in train_wind.py's
    # docstring that the three regressors are fully parallel/independent —
    # they're not quite: wind and current depend on wave-DERIVED features,
    # even though they don't depend on raw wave values directly.
    wave_feats = FEATURE_ORDER["wave"]
    missing_wave = [c for c in wave_feats if c not in valid.columns]
    if missing_wave:
        raise HTTPException(
            status_code=500,
            detail=f"Server error: wave regressor features missing from request-derived "
                    f"columns: {missing_wave} — engineer_features() has drifted out of sync "
                    f"with the training-time feature set, not a client error."
        )

    X_wave = valid[wave_feats].values
    preds = {}
    for target in WAVE_TARGETS:
        preds[target] = run_onnx(f"xgb_wave_regressor_{target}", X_wave)

    # Now that predicted hs/tp/swell_height exist, derive the same two
    # engineered features wind/current were trained on — same formulas as
    # build_features.py, applied to PREDICTIONS instead of ground truth,
    # exactly as the model expects at serving time.
    g = 9.80665
    valid["wave_steepness"] = (2 * np.pi * preds["hs"]) / (g * preds["tp"] ** 2)
    valid["swell_ratio"] = preds["swell_height"] / (preds["hs"] + 1e-5)

    wind_feats = FEATURE_ORDER["wind"]
    current_feats = FEATURE_ORDER["current"]
    missing_cols = [c for c in wind_feats + current_feats if c not in valid.columns]
    if missing_cols:
        raise HTTPException(
            status_code=500,
            detail=f"Server error: wind/current regressor features still missing after "
                    f"deriving wave_steepness/swell_ratio: {missing_cols} — needs a manual look, "
                    f"not a client error."
        )

    X_wind = valid[wind_feats].values
    X_current = valid[current_feats].values

    for target in WIND_TARGETS:
        preds[target] = run_onnx(f"xgb_wind_regressor_{target}", X_wind)
    pred_sin = run_onnx("xgb_wind_regressor_wind_dir_sin", X_wind)
    pred_cos = run_onnx("xgb_wind_regressor_wind_dir_cos", X_wind)
    preds["wind_dir"] = (np.degrees(np.arctan2(pred_sin, pred_cos))) % 360
    preds["current_u"] = run_onnx("xgb_current_regressor_current_u", X_current)
    preds["current_v"] = run_onnx("xgb_current_regressor_current_v", X_current)
    preds["current_speed"] = np.sqrt(preds["current_u"] ** 2 + preds["current_v"] ** 2)
    preds["current_dir"] = (np.degrees(np.arctan2(preds["current_v"], preds["current_u"]))) % 360

    pred_by_name = {
        "pred_hs": preds["hs"], "pred_tp": preds["tp"],
        "pred_swell_height": preds["swell_height"], "pred_wind_wave_height": preds["wind_wave_height"],
        "pred_wind_speed": preds["wind_speed"], "pred_wind_gust": preds["wind_gust"],
        "pred_delta_p_3h": preds["delta_p_3h"], "pred_wind_dir": preds["wind_dir"],
        "pred_current_u": preds["current_u"], "pred_current_v": preds["current_v"],
        "pred_current_speed": preds["current_speed"], "pred_current_dir": preds["current_dir"],
    }
    classifier_feature_order = FEATURE_ORDER["classifier"]
    classifier_features = np.column_stack([pred_by_name[name] for name in classifier_feature_order])
    classifier_probs = run_classifier_onnx(classifier_features)
    ml_preds = np.argmax(classifier_probs, axis=1)

    pagasa_dict = request.pagasa.model_dump() if request.pagasa else None

    results = []
    for i in range(len(valid)):
        telemetry = {
            "wind_speed": float(preds["wind_speed"][i]),
            "wind_gust": float(preds["wind_gust"][i]),
            "hs": float(preds["hs"][i]),
            "swell_height": float(preds["swell_height"][i]),
            "current_speed": float(preds["current_speed"][i]),
            "rain_rate_mm_hr": float(valid.loc[i, "rain_rate_mm_hr"]),
            "slp": float(valid.loc[i, "slp"]),
        }
        gate_result = apply_hard_gate(int(ml_preds[i]), telemetry, pagasa_dict)

        results.append(HourlyPrediction(
            timestamp=str(valid.loc[i, "timestamp"]),
            predicted_hs=float(preds["hs"][i]),
            predicted_tp=float(preds["tp"][i]),
            predicted_swell_height=float(preds["swell_height"][i]),
            predicted_wind_wave_height=float(preds["wind_wave_height"][i]),
            predicted_wind_speed=float(preds["wind_speed"][i]),
            predicted_wind_gust=float(preds["wind_gust"][i]),
            predicted_wind_dir=float(preds["wind_dir"][i]),
            predicted_delta_p_3h=float(preds["delta_p_3h"][i]),
            predicted_current_u=float(preds["current_u"][i]),
            predicted_current_v=float(preds["current_v"][i]),
            predicted_current_speed=float(preds["current_speed"][i]),
            predicted_current_dir=float(preds["current_dir"][i]),
            ml_risk_tier=TIER_NAMES[int(ml_preds[i])],
            final_risk_tier=gate_result["final_tier_name"],
            hard_gate_triggered=gate_result["hard_gate_triggered"],
            override_reasons=gate_result["override_reasons"],
        ))

    return ForecastResponse(predictions=results, skipped_leading_rows=skipped)
