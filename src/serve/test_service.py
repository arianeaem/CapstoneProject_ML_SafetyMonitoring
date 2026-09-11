"""
End-to-end test against the RUNNING FastAPI service (not a mock) — this is
the first script in the whole project that exercises the actual live
serving path, rather than historical batch data. Requires the service to
already be running (uvicorn src.serve.main:app --host 127.0.0.1 --port 8001)
in another terminal before this is run.

Run from the project root: python src\\serve\\test_service.py
"""

import json
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    raise SystemExit("Missing 'requests' package — run: pip install requests")

BASE_URL = "http://127.0.0.1:8001"

CALM: dict[str, object] = {
    "current_u": 0.05, "current_v": 0.05, "current_speed": 0.07, "current_dir": 45.0,
    "wind_u": 2.0, "wind_v": 2.0, "wind_speed": 2.8, "wind_gust": 3.5, "wind_dir": 45.0,
    "slp": 1012.0, "rain_rate_mm_hr": 0.0,
}


def make_readings(n_hours: int, overrides: dict | None = None, override_index: int = -1) -> list:
    """Builds n_hours of consecutive calm readings, one hour apart, optionally
    applying `overrides` to a single reading at `override_index` (negative
    indices count from the end, same as normal Python list indexing)."""
    start = datetime(2026, 9, 1, 0, 0, 0)
    actual_override_idx = override_index if override_index >= 0 else n_hours + override_index
    readings = []
    for i in range(n_hours):
        reading: dict[str, object] = dict(CALM)
        reading["timestamp"] = (start + timedelta(hours=i)).isoformat()
        if overrides and i == actual_override_idx:
            reading.update(overrides)
        readings.append(reading)
    return readings


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if not condition and detail else ""))
    return condition


def test_health():
    print("Test 1: /health")
    r = requests.get(f"{BASE_URL}/health")
    ok = check("service responds 200", r.status_code == 200)
    if ok:
        body = r.json()
        check("reports 12 models loaded", body.get("models_loaded") == 12, str(body))
    print()


def test_calm_batch():
    print("Test 2: calm 8-hour batch — checks leading-row skip and baseline sanity")
    readings = make_readings(8)
    r = requests.post(f"{BASE_URL}/forecast/predict", json={"readings": readings})
    check("request succeeds", r.status_code == 200, r.text[:300])
    if r.status_code != 200:
        print()
        return

    body = r.json()
    check("skipped_leading_rows == 3 (delta_p_3h needs a 3-hour lag)",
          body["skipped_leading_rows"] == 3, f"got {body['skipped_leading_rows']}")
    check("returns 5 predictions (8 sent - 3 skipped)",
          len(body["predictions"]) == 5, f"got {len(body['predictions'])}")

    first = body["predictions"][0]
    check("calm conditions do not trigger the hard-gate",
          not first["hard_gate_triggered"], json.dumps(first, indent=2))
    check("ml_risk_tier is a valid tier name",
          first["ml_risk_tier"] in ["Very Safe", "Safe", "Moderate", "High Risk", "Critical Risk"],
          first["ml_risk_tier"])
    print(f"  Sample prediction: hs={first['predicted_hs']:.3f}m, "
          f"wind_speed={first['predicted_wind_speed']:.2f}m/s, "
          f"risk={first['final_risk_tier']}")
    print()


def test_hard_gate_trigger():
    print("Test 3: extreme rain_rate on the last hour — should force Critical Risk")
    # rain_rate_mm_hr and slp are the two hard-gate fields that pass straight
    # through from the request (never predicted by any regressor) — unlike
    # wind_gust, which IS a wind-regressor target and therefore can't be
    # forced by overriding the raw request field (that field only feeds
    # OTHER models as a legitimate input; it never controls the wind
    # regressor's own predicted wind_gust, which is what the gate actually
    # checks). 30.0 mm/hr is above the 25.0 mm/hr hard-gate threshold.
    readings = make_readings(8, overrides={"rain_rate_mm_hr": 30.0}, override_index=-1)
    r = requests.post(f"{BASE_URL}/forecast/predict", json={"readings": readings})
    check("request succeeds", r.status_code == 200, r.text[:300])
    if r.status_code != 200:
        print()
        return

    body = r.json()
    last = body["predictions"][-1]
    check("hard_gate_triggered is True on the extreme-rain hour",
          last["hard_gate_triggered"], json.dumps(last, indent=2))
    check("final_risk_tier is Critical Risk",
          last["final_risk_tier"] == "Critical Risk", last["final_risk_tier"])
    check("override_reasons mentions the rain breach",
          any("rain" in reason.lower() for reason in last["override_reasons"]),
          str(last["override_reasons"]))

    earlier = body["predictions"][0]
    check("an earlier, still-calm hour in the SAME request is not affected",
          not earlier["hard_gate_triggered"], json.dumps(earlier, indent=2))
    print()


def test_pagasa_override():
    print("Test 4: PAGASA TCWS signal 3 — applies to the WHOLE batch, not one hour")
    readings = make_readings(8)
    payload = {"readings": readings, "pagasa": {"tcws_signal": 3}}
    r = requests.post(f"{BASE_URL}/forecast/predict", json=payload)
    check("request succeeds", r.status_code == 200, r.text[:300])
    if r.status_code != 200:
        print()
        return

    body = r.json()
    all_critical = all(p["final_risk_tier"] == "Critical Risk" for p in body["predictions"])
    check("EVERY hour in the batch is forced to Critical Risk (not just one)",
          all_critical, str([p["final_risk_tier"] for p in body["predictions"]]))
    check("override_reasons mentions the TCWS signal",
          any("TCWS" in r for r in body["predictions"][0]["override_reasons"]),
          str(body["predictions"][0]["override_reasons"]))
    print()


def test_assess_booking_calm():
    print("Test 5: /assess-booking with calm session (08:00 - 12:00) — Laravel integration")
    start = datetime(2026, 9, 15, 6, 0, 0)
    boundary_weather = []
    for i in range(8):  # 06:00 to 13:00
        boundary_weather.append({
            "timestamp": (start + timedelta(hours=i)).isoformat(),
            "wind_speed": 3.0 + 0.1 * i,
            "wind_gust": 4.5 + 0.1 * i,
            "wind_dir": 90.0,
            "slp": 1012.0,
            "rain_rate_mm_hr": 0.0,
        })

    payload = {
        "planned_date": "2026-09-15",
        "dive_start": "08:00",
        "dive_end": "12:00",
        "boundary_weather": boundary_weather,
    }

    r = requests.post(f"{BASE_URL}/assess-booking", json=payload)
    check("request succeeds with HTTP 200", r.status_code == 200, r.text[:300])
    if r.status_code != 200:
        print()
        return

    body = r.json()
    check("returns session_duration_hours == 5 (08:00 to 12:00 inclusive)",
          body["session_duration_hours"] == 5, f"got {body['session_duration_hours']}")
    check("worst_hour exists and is populated",
          "worst_hour" in body and body["worst_hour"]["hour"] >= 8, str(body.get("worst_hour")))
    check("overall_recommendation is valid",
          body["overall_recommendation"] in ["GO", "PROVISIONAL_GO", "CAUTION_ADVANCED_ONLY", "HIGH_RISK_NO_GO", "NO_GO"],
          body["overall_recommendation"])
    check("currents automatically populated from cache/climatology",
          all("current_source" in h for h in body["hourly_assessments"]))
    print(f"  Overall: {body['overall_recommendation']} ({body['overall_operational_status']}), "
          f"Worst Hour: {body['worst_hour']['timestamp']} ({body['worst_hour']['final_tier_name']})")
    print()


def test_assess_booking_hard_gate():
    print("Test 6: /assess-booking with storm wind breach at 10:00 — forces NO_GO")
    start = datetime(2026, 9, 15, 6, 0, 0)
    boundary_weather = []
    for i in range(8):  # 06:00 to 13:00
        hour_ts = start + timedelta(hours=i)
        wind = 12.0 if hour_ts.hour == 10 else 3.0  # 12 m/s = 43.2 km/h (> 38 km/h hard gate)
        boundary_weather.append({
            "timestamp": hour_ts.isoformat(),
            "wind_speed": wind,
            "wind_gust": wind * 1.3,
            "wind_dir": 90.0,
            "slp": 1012.0,
            "rain_rate_mm_hr": 0.0,
        })

    payload = {
        "planned_date": "2026-09-15",
        "dive_start": "08:00",
        "dive_end": "12:00",
        "boundary_weather": boundary_weather,
    }

    r = requests.post(f"{BASE_URL}/assess-booking", json=payload)
    check("request succeeds with HTTP 200", r.status_code == 200, r.text[:300])
    if r.status_code != 200:
        print()
        return

    body = r.json()
    check("overall_hard_gate_triggered is True", body["overall_hard_gate_triggered"] is True)
    check("overall_recommendation is NO_GO", body["overall_recommendation"] == "NO_GO", body["overall_recommendation"])
    check("worst_hour identifies 10:00 storm breach",
          body["worst_hour"]["hour"] == 10 and body["worst_hour"]["hard_gate_triggered"] is True,
          str(body["worst_hour"]))
    print(f"  Identified Worst Hour: {body['worst_hour']['timestamp']} — {body['worst_hour']['primary_hazard']}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("LIVE SERVICE END-TO-END TEST — requires uvicorn already running")
    print("=" * 70 + "\n")
    test_health()
    test_calm_batch()
    test_hard_gate_trigger()
    test_pagasa_override()
    test_assess_booking_calm()
    test_assess_booking_hard_gate()
    print("=" * 70)
    print("Done. Review any [FAIL] lines above before trusting this service.")
    print("=" * 70)

