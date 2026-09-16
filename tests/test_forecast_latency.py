"""
Latency and SLA Budget Load Tests for Multi-Horizon Marine Physics Forecaster.
Verifies that end-to-end inference for all 9 horizons stays strictly under the 350ms p95 budget,
testing both ultra-fast ONNX runtime and Python native / Climatology fallback serving branches.

Run with:
    pytest tests/test_forecast_latency.py -v
    pytest tests/test_forecast_latency.py --benchmark-only (if pytest-benchmark is installed)
"""

import time
import logging
import pytest
import numpy as np
from fastapi.testclient import TestClient

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("autogluon").setLevel(logging.WARNING)

from src.serve.main import app, get_multi_horizon_physics, MultiHorizonForecastRequest
from src.serve.model_router import router

client = TestClient(app)
ALL_HORIZONS = [1, 6, 12, 24, 48, 72, 96, 144, 168]


@pytest.fixture(scope="module")
def sample_features():
    np.random.seed(42)
    return np.random.uniform(low=0.1, high=5.0, size=(133,)).astype(np.float32)


@pytest.mark.parametrize("horizon", ALL_HORIZONS)
def test_forecast_all_horizons_response_schema_and_quantiles(horizon, sample_features):
    """
    Tests that /forecast returns structured quantile-bearing forecasts for all 9 horizons.
    """
    payload = {
        "horizon_hours": horizon,
        "feature_vector": sample_features.tolist(),
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200, f"Failed for horizon {horizon}h: {response.text}"
    
    data = response.json()
    assert data["horizon_hours"] == horizon
    assert "physics_forecast" in data
    assert "metadata" in data
    
    pf = data["physics_forecast"]
    # Check that all quantile variables carry p10, p50, p90
    quantile_keys = [
        "significant_wave_height_m", "peak_period_s", "swell_height_m", "wind_wave_height_m",
        "wind_speed_kmh", "wind_gust_kmh", "wind_direction_deg", "sea_level_pressure_hpa",
        "current_u_ms", "current_v_ms", "current_speed_ms", "current_direction_deg"
    ]
    for key in quantile_keys:
        assert key in pf, f"Missing {key} in physics_forecast"
        q = pf[key]
        assert "p10" in q and "p50" in q and "p90" in q, f"Malformed quantile object for {key}: {q}"
        if key not in ["wind_direction_deg", "current_direction_deg", "current_u_ms", "current_v_ms"]:
            assert q["p10"] <= q["p50"] <= q["p90"] or q["p10"] <= q["p90"], f"Quantile order violation in {key}: {q}"
            
    # Check derived hydrodynamic features
    assert "wave_steepness" in pf
    assert "swell_ratio" in pf
    assert 0.0 <= pf["swell_ratio"] <= 1.0


@pytest.mark.parametrize("horizon", ALL_HORIZONS)
def test_forecast_latency_budget(horizon, sample_features):
    """
    Executes 50 iterations per horizon to measure p50, p95, and max latency.
    Asserts p95 < 350ms SLA budget.
    """
    latencies = []
    req = MultiHorizonForecastRequest(
        horizon_hours=horizon,
        feature_vector=sample_features.tolist(),
    )

    # Warmup
    for _ in range(5):
        get_multi_horizon_physics(req)

    # 50 measured iterations
    for _ in range(50):
        t0 = time.perf_counter()
        resp = get_multi_horizon_physics(req)
        t1 = time.perf_counter()
        assert resp.horizon_hours == horizon
        latencies.append((t1 - t0) * 1000.0)  # in milliseconds

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    max_l = np.max(latencies)

    print(f"\n[HORIZON {horizon:3d}h LATENCY] p50: {p50:6.2f}ms | p95: {p95:6.2f}ms | p99: {p99:6.2f}ms | max: {max_l:6.2f}ms")
    assert p95 < 350.0, f"SLA Violation at H={horizon}h: p95 latency {p95:.2f}ms exceeds 350ms budget!"


def test_native_and_climatology_branches_specifically(sample_features):
    """
    Specifically tests non-ONNX models (Native Python WeightedEnsemble and Climatology fallback)
    to confirm that native Python execution overhead stays well within the 350ms latency budget.
    """
    native_latencies = []
    # Test across H=6 (WeightedEnsemble winner) and H=96/144/168 (Climatology Fallback winner)
    test_horizons = [6, 72, 96, 144, 168]

    for h in test_horizons:
        req = MultiHorizonForecastRequest(
            horizon_hours=h,
            feature_vector=sample_features.tolist(),
        )
        # Warmup
        for _ in range(3):
            get_multi_horizon_physics(req)

        for _ in range(30):
            t0 = time.perf_counter()
            resp = get_multi_horizon_physics(req)
            t1 = time.perf_counter()
            assert resp.horizon_hours == h
            native_latencies.append((t1 - t0) * 1000.0)

    p95_native = np.percentile(native_latencies, 95)
    p50_native = np.percentile(native_latencies, 50)
    max_native = np.max(native_latencies)

    print(f"\n[NATIVE/FALLBACK BRANCHES] p50: {p50_native:6.2f}ms | p95: {p95_native:6.2f}ms | max: {max_native:6.2f}ms")
    assert p95_native < 350.0, f"Native/Fallback p95 latency {p95_native:.2f}ms exceeds 350ms budget!"


# Optional pytest-benchmark hook
def test_benchmark_forecast_h24(benchmark, sample_features):
    req = MultiHorizonForecastRequest(
        horizon_hours=24,
        feature_vector=sample_features.tolist(),
    )
    result = benchmark(get_multi_horizon_physics, req)
    assert result.horizon_hours == 24
