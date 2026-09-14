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

from src.models.train_safety_classifier import WAVE_TARGETS
from src.serve.safety_thresholds import apply_safety_thresholds, evaluate_operational_safety, TIER_NAMES
from src.serve.currents_cache import get_live_currents_forecast

WIND_REGRESSOR_TARGETS = ["wind_speed", "wind_gust", "delta_p_3h"]

app = FastAPI(
    title="Camp FreedivePH Weather Safety & Booking Assessment Service",
    description="Microservice providing marine physics forecasting, 9-variable PHP score alignment, "
                "deterministic safety threshold overrides, and operational horizon cutoff assessments "
                "for Laravel booking management.",
    version="2.0.0",
)

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
        + [f"xgb_wind_regressor_{t}" for t in WIND_REGRESSOR_TARGETS]
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
    current_u: Optional[float] = None
    current_v: Optional[float] = None
    current_speed: Optional[float] = None
    current_dir: Optional[float] = None
    wind_u: Optional[float] = None
    wind_v: Optional[float] = None
    wind_speed: float
    wind_gust: float
    wind_dir: float
    slp: float
    rain_rate_mm_hr: float = 0.0


class PagasaAdvisory(BaseModel):
    tcws_signal: int = 0
    gale_warning: bool = False
    tsunami_warning: bool = False


class ForecastRequest(BaseModel):
    readings: List[HourlyReading] = Field(
        ..., min_items=1,
        description="Hourly readings for inference."
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
    safety_threshold_triggered: bool = False
    hard_gate_triggered: bool = False  # Backward-compatible alias
    override_reasons: List[str]


class ForecastResponse(BaseModel):
    predictions: List[HourlyPrediction]
    skipped_leading_rows: int


# --- Booking Assessment Schemas (Laravel Boundary Interface) ---
class BookingAssessmentRequest(BaseModel):
    planned_date: str = Field(..., description="Date of dive session (YYYY-MM-DD), e.g. '2026-09-15'")
    dive_start: str = Field("08:00", description="Start time of dive session (HH:MM), e.g. '08:00'")
    dive_end: str = Field("12:00", description="End time of dive session (HH:MM), e.g. '12:00'")
    boundary_weather: List[HourlyReading] = Field(
        ..., min_items=1,
        description="Hourly atmospheric readings from Open-Meteo covering the session."
    )
    pagasa: Optional[PagasaAdvisory] = None
    site_name: Optional[str] = Field("Anilao, Mabini, Batangas", description="Dive site location")


class HourlyAssessmentDetail(BaseModel):
    timestamp: str
    hour: int
    horizon_hours: int
    operational_status: str
    is_safety_verdict_active: bool
    displayed_tier: Optional[int]
    displayed_tier_name: str
    ml_raw_tier: int
    ml_raw_tier_name: str
    final_tier: int
    final_tier_name: str
    safety_threshold_triggered: bool = False
    hard_gate_triggered: bool = False  # Backward-compatible alias
    override_reasons: List[str]
    advisory_message: str
    current_source: str
    predicted_hs: float
    predicted_tp: float
    predicted_swell_height: float
    predicted_wind_wave_height: float
    predicted_wind_speed: float
    predicted_wind_gust: float
    predicted_wind_dir: float
    predicted_current_speed: float
    predicted_current_dir: float
    rain_rate_mm_hr: float
    slp: float


class WorstHourSummary(BaseModel):
    timestamp: str
    hour: int
    horizon_hours: int
    final_tier: int
    final_tier_name: str
    safety_threshold_triggered: bool = False
    hard_gate_triggered: bool = False  # Backward-compatible alias
    override_reasons: List[str]
    primary_hazard: str
    advisory_message: str


class BookingAssessmentResponse(BaseModel):
    planned_date: str
    dive_start: str
    dive_end: str
    session_duration_hours: int
    min_horizon_hours: int
    max_horizon_hours: int
    overall_operational_status: str
    overall_recommendation: str  # "GO", "PROVISIONAL_GO", "CAUTION_ADVANCED_ONLY", "HIGH_RISK_NO_GO", "NO_GO"
    is_authoritative_go: bool
    displayed_risk_tier: Optional[int]
    displayed_risk_name: str
    safety_threshold_triggered: bool = False
    overall_safety_threshold_triggered: bool = False
    overall_hard_gate_triggered: bool = False  # Backward-compatible alias
    worst_hour: WorstHourSummary
    hourly_assessments: List[HourlyAssessmentDetail]
    generated_at: str


# ---------------------------------------------------------------------------
# Feature engineering — matches training EXACTLY
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])

    if "delta_p_3h" not in df.columns or df["delta_p_3h"].isna().all():
        df["delta_p_3h"] = df["slp"].diff(3).bfill().fillna(0.0)

    # Compute wind components if missing
    if "wind_u" not in df.columns or df["wind_u"].isna().all():
        rad = np.radians(df["wind_dir"])
        df["wind_u"] = -df["wind_speed"] * np.sin(rad)
        df["wind_v"] = -df["wind_speed"] * np.cos(rad)

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
    return {
        "status": "ok",
        "service": "Camp FreedivePH Weather Safety Assessment Service",
        "models_loaded": len(SESSIONS),
        "onnx_sessions": list(SESSIONS.keys()),
    }


def _run_inference_pipeline(raw_df: pd.DataFrame, pagasa_dict: Optional[dict]):
    """Internal core inference execution shared across endpoints."""
    # Check if ocean currents are missing or unpopulated; if so, inject from CMEMS cache/climatology
    needs_currents = (
        "current_u" not in raw_df.columns
        or raw_df["current_u"].isna().any()
        or (raw_df["current_u"].fillna(0.0) == 0.0).all()
    )

    if needs_currents:
        ts_index = pd.DatetimeIndex(pd.to_datetime(raw_df["timestamp"]))
        currents_df = get_live_currents_forecast(ts_index)
        raw_df["current_u"] = currents_df["current_u"].values
        raw_df["current_v"] = currents_df["current_v"].values
        raw_df["current_speed"] = currents_df["current_speed"].values
        raw_df["current_dir"] = currents_df["current_dir"].values
        raw_df["current_source"] = currents_df["current_source"].values
    else:
        if "current_speed" not in raw_df.columns or raw_df["current_speed"].isna().any():
            raw_df["current_speed"] = np.sqrt(raw_df["current_u"]**2 + raw_df["current_v"]**2)
        if "current_dir" not in raw_df.columns or raw_df["current_dir"].isna().any():
            raw_df["current_dir"] = (np.degrees(np.arctan2(raw_df["current_v"], raw_df["current_u"]))) % 360.0
        raw_df["current_source"] = "payload_provided"

    valid = engineer_features(raw_df)

    # 1. Wave Regressors
    wave_feats = FEATURE_ORDER["wave"]
    X_wave = valid[wave_feats].values
    preds = {}
    for target in WAVE_TARGETS:
        preds[target] = run_onnx(f"xgb_wave_regressor_{target}", X_wave)

    # 2. Derive wave-dependent physics features
    g = 9.80665
    valid["wave_steepness"] = (2 * np.pi * preds["hs"]) / (g * np.maximum(preds["tp"], 0.5) ** 2)
    valid["swell_ratio"] = preds["swell_height"] / (preds["hs"] + 1e-5)

    # 3. Wind and Current Regressors
    wind_feats = FEATURE_ORDER["wind"]
    current_feats = FEATURE_ORDER["current"]
    X_wind = valid[wind_feats].values
    X_current = valid[current_feats].values

    for target in WIND_REGRESSOR_TARGETS:
        preds[target] = run_onnx(f"xgb_wind_regressor_{target}", X_wind)
    pred_sin = run_onnx("xgb_wind_regressor_wind_dir_sin", X_wind)
    pred_cos = run_onnx("xgb_wind_regressor_wind_dir_cos", X_wind)
    preds["wind_dir"] = (np.degrees(np.arctan2(pred_sin, pred_cos))) % 360.0
    preds["current_u"] = run_onnx("xgb_current_regressor_current_u", X_current)
    preds["current_v"] = run_onnx("xgb_current_regressor_current_v", X_current)
    preds["current_speed"] = np.sqrt(preds["current_u"] ** 2 + preds["current_v"] ** 2)
    preds["current_dir"] = (np.degrees(np.arctan2(preds["current_v"], preds["current_u"]))) % 360.0

    # 4. Safety Classifier
    pred_by_name = {
        "pred_hs": preds["hs"], "pred_tp": preds["tp"],
        "pred_swell_height": preds["swell_height"], "pred_wind_wave_height": preds["wind_wave_height"],
        "pred_wind_speed": preds["wind_speed"], "pred_wind_gust": preds["wind_gust"],
        "pred_delta_p_3h": valid["delta_p_3h"].values, "pred_wind_dir": preds["wind_dir"],
        "pred_current_u": preds["current_u"], "pred_current_v": preds["current_v"],
        "pred_current_speed": preds["current_speed"], "pred_current_dir": preds["current_dir"],
    }
    classifier_feature_order = FEATURE_ORDER["classifier"]
    classifier_features = np.column_stack([pred_by_name[name] for name in classifier_feature_order])
    classifier_probs = run_classifier_onnx(classifier_features)
    ml_preds = np.argmax(classifier_probs, axis=1)

    return valid, preds, ml_preds


@app.post("/forecast/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    if len(SESSIONS) == 0:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    raw = pd.DataFrame([r.model_dump() for r in request.readings])
    pagasa_dict = request.pagasa.model_dump() if request.pagasa else None

    valid, preds, ml_preds = _run_inference_pipeline(raw, pagasa_dict)

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
        threshold_result = apply_safety_thresholds(int(ml_preds[i]), telemetry, pagasa_dict)

        results.append(HourlyPrediction(
            timestamp=str(valid.loc[i, "timestamp"]),
            predicted_hs=float(preds["hs"][i]),
            predicted_tp=float(preds["tp"][i]),
            predicted_swell_height=float(preds["swell_height"][i]),
            predicted_wind_wave_height=float(preds["wind_wave_height"][i]),
            predicted_wind_speed=float(preds["wind_speed"][i]),
            predicted_wind_gust=float(preds["wind_gust"][i]),
            predicted_wind_dir=float(preds["wind_dir"][i]),
            predicted_delta_p_3h=float(valid.loc[i, "delta_p_3h"]),
            predicted_current_u=float(preds["current_u"][i]),
            predicted_current_v=float(preds["current_v"][i]),
            predicted_current_speed=float(preds["current_speed"][i]),
            predicted_current_dir=float(preds["current_dir"][i]),
            ml_risk_tier=TIER_NAMES[int(ml_preds[i])],
            final_risk_tier=threshold_result["final_tier_name"],
            safety_threshold_triggered=threshold_result["safety_threshold_triggered"],
            hard_gate_triggered=threshold_result["hard_gate_triggered"],
            override_reasons=threshold_result["override_reasons"],
        ))

    return ForecastResponse(predictions=results, skipped_leading_rows=0)


# ===========================================================================
# /assess-booking: The Core Laravel Integration Endpoint
# ===========================================================================
@app.post("/assess-booking", response_model=BookingAssessmentResponse)
def assess_booking(request: BookingAssessmentRequest):
    """
    Evaluates a planned freediving booking session for Laravel:
    1. Injects live CMEMS ocean currents (or climatological fallback).
    2. Runs multi-horizon physics forecasting for waves, winds, and currents.
    3. Evaluates 9-variable PHP-aligned safety rules & deterministic hard-gates.
    4. Enforces the 3-Tier Operational Cutoff Policy based on query horizon.
    5. Resolves the worst_hour across the dive window and returns the overall Go/No-Go verdict.
    """
    if len(SESSIONS) == 0:
        raise HTTPException(status_code=503, detail="Inference models not loaded yet")

    raw = pd.DataFrame([b.model_dump() for b in request.boundary_weather])
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="boundary_weather cannot be empty")

    pagasa_dict = request.pagasa.model_dump() if request.pagasa else None
    valid, preds, ml_preds = _run_inference_pipeline(raw, pagasa_dict)

    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)

    # Parse session time boundaries (e.g. 08:00 to 12:00 on planned_date)
    try:
        start_hour_int = int(request.dive_start.split(":")[0])
        end_hour_int = int(request.dive_end.split(":")[0])
    except Exception:
        start_hour_int, end_hour_int = 8, 12

    all_hourly_details: List[HourlyAssessmentDetail] = []
    session_hourly_details: List[HourlyAssessmentDetail] = []

    for i in range(len(valid)):
        ts_dt = pd.to_datetime(valid.loc[i, "timestamp"])
        hour_int = ts_dt.hour

        # Calculate forecast horizon in hours relative to current time
        if ts_dt.tzinfo is None:
            ts_utc = ts_dt.replace(tzinfo=timezone.utc)
        else:
            ts_utc = ts_dt.astimezone(timezone.utc)

        horizon_hours = max(1, int((ts_utc - now_utc).total_seconds() / 3600.0))

        telemetry = {
            "wind_speed": float(preds["wind_speed"][i]),
            "wind_gust": float(preds["wind_gust"][i]),
            "delta_p_3h": float(valid.loc[i, "delta_p_3h"]) if "delta_p_3h" in valid.columns else 0.0,
            "hs": float(preds["hs"][i]),
            "swell_height": float(preds["swell_height"][i]),
            "current_speed": float(preds["current_speed"][i]),
            "rain_rate_mm_hr": float(valid.loc[i, "rain_rate_mm_hr"]),
            "slp": float(valid.loc[i, "slp"]),
        }

        # Apply deterministic safety threshold and operational cutoff policy
        op_result = evaluate_operational_safety(horizon_hours, int(ml_preds[i]), telemetry, pagasa_dict)

        detail = HourlyAssessmentDetail(
            timestamp=str(valid.loc[i, "timestamp"]),
            hour=hour_int,
            horizon_hours=horizon_hours,
            operational_status=op_result["operational_status"],
            is_safety_verdict_active=op_result["is_safety_verdict_active"],
            displayed_tier=op_result["displayed_tier"],
            displayed_tier_name=op_result["displayed_tier_name"],
            ml_raw_tier=int(ml_preds[i]),
            ml_raw_tier_name=TIER_NAMES[int(ml_preds[i])],
            final_tier=op_result["displayed_tier"] if op_result["displayed_tier"] is not None else int(op_result["ml_raw_prediction"]),
            final_tier_name=TIER_NAMES[int(op_result["ml_raw_prediction"])],
            safety_threshold_triggered=op_result["safety_threshold_triggered"],
            hard_gate_triggered=op_result["hard_gate_triggered"],
            override_reasons=op_result["override_reasons"],
            advisory_message=op_result["advisory_message"],
            current_source=str(valid.loc[i, "current_source"]),
            predicted_hs=round(float(preds["hs"][i]), 3),
            predicted_tp=round(float(preds["tp"][i]), 2),
            predicted_swell_height=round(float(preds["swell_height"][i]), 3),
            predicted_wind_wave_height=round(float(preds["wind_wave_height"][i]), 3),
            predicted_wind_speed=round(float(preds["wind_speed"][i]), 2),
            predicted_wind_gust=round(float(preds["wind_gust"][i]), 2),
            predicted_wind_dir=round(float(preds["wind_dir"][i]), 1),
            predicted_current_speed=round(float(preds["current_speed"][i]), 3),
            predicted_current_dir=round(float(preds["current_dir"][i]), 1),
            rain_rate_mm_hr=round(float(valid.loc[i, "rain_rate_mm_hr"]), 2),
            slp=round(float(valid.loc[i, "slp"]), 2),
        )

        all_hourly_details.append(detail)
        # Check if hour belongs to the planned dive session window
        if start_hour_int <= hour_int <= end_hour_int:
            session_hourly_details.append(detail)

    # Use session hours if present, otherwise evaluate all submitted hours
    target_hours = session_hourly_details if len(session_hourly_details) > 0 else all_hourly_details

    # --- Identify the WORST HOUR in the session ---
    # Sort key: 1. safety_threshold_triggered (True first), 2. final_tier (highest first), 3. predicted_hs, 4. predicted_wind_speed
    worst = max(
        target_hours,
        key=lambda h: (1 if h.safety_threshold_triggered else 0, h.final_tier, h.predicted_hs, h.predicted_wind_speed)
    )

    primary_hazard = worst.override_reasons[0] if worst.safety_threshold_triggered else f"Peak Risk: {worst.final_tier_name}"

    worst_summary = WorstHourSummary(
        timestamp=worst.timestamp,
        hour=worst.hour,
        horizon_hours=worst.horizon_hours,
        final_tier=worst.final_tier,
        final_tier_name=worst.final_tier_name,
        safety_threshold_triggered=worst.safety_threshold_triggered,
        hard_gate_triggered=worst.hard_gate_triggered,
        override_reasons=worst.override_reasons,
        primary_hazard=primary_hazard,
        advisory_message=worst.advisory_message,
    )

    # --- Overall Session Verdict ---
    any_threshold_breach = any(h.safety_threshold_triggered for h in target_hours)
    max_tier = max(h.final_tier for h in target_hours)
    min_horizon = min(h.horizon_hours for h in target_hours)
    max_horizon = max(h.horizon_hours for h in target_hours)

    # Determine dominant operational status across session
    if max_horizon <= 1:
        overall_op_status = "TACTICAL_CLEARANCE"
    elif max_horizon <= 24:
        overall_op_status = "PROVISIONAL_TREND_OUTLOOK"
    else:
        overall_op_status = "EXTENDED_TREND_OUTLOOK"

    # Determine 5-tier recommendation directly matching platform safety classifications
    if any_threshold_breach or max_tier == 4:
        overall_recommendation = "Critical Risk"
        operational_action = "NO_GO"
        is_authoritative_go = False
    elif max_tier == 3:
        overall_recommendation = "High Risk"
        operational_action = "HIGH_RISK_NO_GO"
        is_authoritative_go = False
    elif max_tier == 2:
        overall_recommendation = "Moderate"
        operational_action = "CAUTION_ADVANCED_ONLY"
        is_authoritative_go = False
    elif max_tier == 1:
        overall_recommendation = "Safe"
        operational_action = "PROVISIONAL_GO" if overall_op_status != "TACTICAL_CLEARANCE" else "GO"
        is_authoritative_go = (overall_op_status == "TACTICAL_CLEARANCE")
    else:
        overall_recommendation = "Very Safe"
        operational_action = "PROVISIONAL_GO" if overall_op_status != "TACTICAL_CLEARANCE" else "GO"
        is_authoritative_go = (overall_op_status == "TACTICAL_CLEARANCE")

    displayed_risk_tier = worst.displayed_tier
    displayed_risk_name = worst.displayed_tier_name

    return BookingAssessmentResponse(
        planned_date=request.planned_date,
        dive_start=request.dive_start,
        dive_end=request.dive_end,
        session_duration_hours=len(target_hours),
        min_horizon_hours=min_horizon,
        max_horizon_hours=max_horizon,
        overall_operational_status=overall_op_status,
        overall_recommendation=overall_recommendation,
        is_authoritative_go=is_authoritative_go,
        displayed_risk_tier=displayed_risk_tier,
        displayed_risk_name=displayed_risk_name,
        safety_threshold_triggered=any_threshold_breach,
        overall_safety_threshold_triggered=any_threshold_breach,
        overall_hard_gate_triggered=any_threshold_breach,
        worst_hour=worst_summary,
        hourly_assessments=target_hours,
        generated_at=now_utc.isoformat(),
    )
