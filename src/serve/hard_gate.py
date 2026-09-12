"""
Deterministic hard-gate layer and Operational Horizon Policy.
Sits directly on top of xgb_safety_classifier's output: any breach of a physical
threshold or an active PAGASA storm signal forces Critical Risk regardless of what
the classifier predicted.

OPERATIONAL HORIZON POLICY (PRD Section 8 & Empirical Finding):
- H = 1h (Tactical Departure Window):
    * ML Safety Classifier is ACTIVE: "TACTICAL_CLEARANCE"
    * High model fidelity (Empirical Critical FNR = 4.5%, Precision = 96.8%).
    * Authoritative Go/No-Go dockside departure clearance.
- 1h < H <= 24h (Provisional Planning Window: 6h, 12h, 24h):
    * Discrete ML safety tier is SUPPRESSED: "PROVISIONAL_TREND_OUTLOOK"
    * Empirical Critical FNR is 40.9% - 54.5% (approx. coin-flip reliability due
      to early-stage MSE variance smoothing).
    * Surfacing a discrete "Safe" badge with an abstract caution icon creates false
      reassurance; therefore, discrete classification is suppressed and the UI surfaces
      the empirical ~45% miss rate, raw physics, P90 tail risk bounds, and active
      hard-gate backstop for advance planning only. Final clearance deferred to T-1h.
- H > 24h (Extended Window: 48h, 72h, 96h, 144h):
    * Discrete ML safety tier is SUPPRESSED: "EXTENDED_TREND_OUTLOOK"
    * Empirical Critical FNR is 82.0% - 100.0% (climatological mean-regression).
    * Surfaces raw physical trajectory, P90 tail bounds, and active hard-gate backstop.

Thresholds are imported from build_safety_labels.py, NOT retyped here.

Run from the project root: python src/serve/hard_gate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.labels.build_safety_labels import HARD_GATE, KMH_TO_MS  # noqa: E402

TIER_NAMES = ["Very Safe", "Safe", "Moderate", "High Risk", "Critical Risk"]
TIER_CRITICAL = 4
TIER_NO_CONSTRAINT = 0  # what the hard-gate contributes to max() when nothing is breached

TACTICAL_GO_NO_GO_HORIZON_HOURS = 1
PROVISIONAL_CUTOFF_HORIZON_HOURS = 24


def check_physical_breach(telemetry: dict) -> tuple[bool, list[str]]:
    """telemetry keys expected: wind_speed, wind_gust (m/s), delta_p_3h (hPa), hs, swell_height (m),
    current_speed (m/s), rain_rate_mm_hr (mm/hr), slp (hPa). Same units as the
    trained forecasters' outputs."""
    reasons = []

    if telemetry.get("wind_speed", 0.0) >= HARD_GATE["wind_speed_ms"]:
        reasons.append(f"sustained wind {telemetry['wind_speed']:.1f} m/s >= "
                        f"{HARD_GATE['wind_speed_ms']:.2f} m/s limit")
    if telemetry.get("wind_gust", 0.0) >= HARD_GATE["wind_gust_ms"]:
        reasons.append(f"wind gust {telemetry['wind_gust']:.1f} m/s >= "
                        f"{HARD_GATE['wind_gust_ms']:.2f} m/s limit")
    if telemetry.get("hs", 0.0) >= HARD_GATE["wave_height_m"]:
        reasons.append(f"wave height {telemetry['hs']:.2f} m >= {HARD_GATE['wave_height_m']} m limit")
    if telemetry.get("swell_height", 0.0) >= HARD_GATE["swell_height_m"]:
        reasons.append(f"swell height {telemetry['swell_height']:.2f} m >= "
                        f"{HARD_GATE['swell_height_m']} m limit")
    if telemetry.get("current_speed", 0.0) >= HARD_GATE["current_ms"]:
        reasons.append(f"current {telemetry['current_speed']:.2f} m/s >= {HARD_GATE['current_ms']} m/s limit")
    if telemetry.get("rain_rate_mm_hr", 0.0) >= HARD_GATE["rain_mm_hr"]:
        reasons.append(f"rain rate {telemetry['rain_rate_mm_hr']:.1f} mm/hr >= "
                        f"{HARD_GATE['rain_mm_hr']} mm/hr limit")
    if telemetry.get("slp", 1013.25) <= HARD_GATE["pressure_hpa"]:
        reasons.append(f"pressure {telemetry['slp']:.1f} hPa <= {HARD_GATE['pressure_hpa']} hPa limit")

    # Context-Aware Compound Precursor Check:
    # A rapid 3-hour barometric drop (>= 2.5 hPa / 3h) requires companion storm indicators
    # (Squall Gusts >= 38.0 km/h [10.56 m/s] OR Rain Rate >= 15.0 mm/hr) to trigger an emergency breach.
    # Normal tropical diurnal solar heating drops (1.5 - 2.0 hPa) on calm sunny days do NOT trigger a false alarm.
    delta_p = telemetry.get("delta_p_3h", 0.0)
    pressure_drop = abs(delta_p) if delta_p < 0 else delta_p
    wind_gust_ms = telemetry.get("wind_gust", 0.0)
    rain_rate = telemetry.get("rain_rate_mm_hr", 0.0)
    squall_gust_threshold_ms = 38.0 * KMH_TO_MS  # 10.56 m/s (38 km/h)
    storm_rain_threshold_mm = 15.0  # 15.0 mm/hr

    if pressure_drop >= 2.5 and (wind_gust_ms >= squall_gust_threshold_ms or rain_rate >= storm_rain_threshold_mm):
        reasons.append(
            f"rapid barometric drop ({pressure_drop:.1f} hPa/3h) with accompanying squalls/rain "
            f"({wind_gust_ms * 3.6:.1f} km/h gusts, {rain_rate:.1f} mm/hr rain)"
        )

    return len(reasons) > 0, reasons


def check_pagasa_override(pagasa: dict | None = None) -> tuple[bool, list[str]]:
    """Evaluates active PAGASA cyclone signals, gale warnings, and tsunami alerts."""
    if pagasa is None:
        return False, []

    reasons = []
    if pagasa.get("tcws_signal", 0) >= 3:
        reasons.append(f"PAGASA TCWS Signal #{pagasa['tcws_signal']} active")
    if pagasa.get("gale_warning", False):
        reasons.append("PAGASA Gale Warning in effect")
    if pagasa.get("tsunami_warning", False):
        reasons.append("Active tsunami warning")
    return len(reasons) > 0, reasons


def apply_hard_gate(ml_prediction: int, telemetry: dict, pagasa: dict | None = None) -> dict:
    """
    Core Deterministic Hard-Gate:
    Final Risk = max(ml_prediction, hard_gate_result, pagasa_override).
    A breach can only ever push the tier UP toward Critical.

    This function is 100% pure physical/regulatory logic and has no concept
    of forecast horizons or confidence-based suppression.
    """
    physical_breach, physical_reasons = check_physical_breach(telemetry)
    pagasa_breach, pagasa_reasons = check_pagasa_override(pagasa)

    gate_tier = TIER_CRITICAL if (physical_breach or pagasa_breach) else TIER_NO_CONSTRAINT
    final_tier = max(ml_prediction, gate_tier)

    all_reasons = physical_reasons + pagasa_reasons
    return {
        "final_tier": final_tier,
        "final_tier_name": TIER_NAMES[final_tier],
        "ml_prediction": ml_prediction,
        "ml_prediction_name": TIER_NAMES[ml_prediction],
        "hard_gate_triggered": physical_breach or pagasa_breach,
        "override_reasons": all_reasons if all_reasons else ["within all physical/advisory limits"],
    }


def evaluate_operational_safety(
    horizon_hours: int,
    ml_prediction: int,
    telemetry: dict,
    pagasa: dict | None = None
) -> dict:
    """
    Operational Decision Function Enforcing the 3-Tier Horizon Cutoff Policy.

    - Band 1 (H = 1h): "TACTICAL_CLEARANCE"
      Surfaces the authoritative active ML safety verdict + hard gate.
      Empirical Critical FNR = 4.5%, Precision = 96.8%.

    - Band 2 (1h < H <= 24h, i.e. 6h, 12h, 24h): "PROVISIONAL_TREND_OUTLOOK"
      SUPPRESSES discrete ML safety tier (displayed_tier: null).
      Empirical Critical FNR is 40.9% - 54.5% (~45% miss rate). Surfaces raw physics,
      P90 tail risk, and active hard-gate backstop for tentative planning.

    - Band 3 (H > 24h, i.e. 48h, 72h, 96h, 144h): "EXTENDED_TREND_OUTLOOK"
      SUPPRESSES discrete ML safety tier (displayed_tier: null).
      Empirical Critical FNR is 82.0% - 100.0% due to climatological mean-regression.
      Surfaces physical trajectory, P90 tail risk, and active hard-gate backstop.
    """
    hard_gate_result = apply_hard_gate(ml_prediction, telemetry, pagasa)

    if horizon_hours <= TACTICAL_GO_NO_GO_HORIZON_HOURS:
        operational_status = "TACTICAL_CLEARANCE"
        is_safety_verdict_active = True
        displayed_tier = hard_gate_result["final_tier"]
        displayed_tier_name = hard_gate_result["final_tier_name"]
        advisory_message = (
            "Real-time tactical clearance (1h). High model fidelity (Critical FNR: 4.5%, "
            "Precision: 96.8%). Directly authorizes boat departure / dive dispatch."
        )
    elif horizon_hours <= PROVISIONAL_CUTOFF_HORIZON_HOURS:
        operational_status = "PROVISIONAL_TREND_OUTLOOK"
        is_safety_verdict_active = False
        displayed_tier = None  # Discrete tier suppressed
        displayed_tier_name = "SUPPRESSED_PROVISIONAL_TREND"
        advisory_message = (
            f"Provisional planning outlook ({horizon_hours}h ahead). Discrete safety tier is SUPPRESSED "
            "because models at this range historically miss ~45% of dangerous conditions (FNR: 40.9%–54.5%) "
            "due to early MSE variance smoothing. Displaying raw physics, P90 tail bounds, and hard-gate alerts "
            "for tentative planning; formal safety clearance is strictly evaluated at T-1h."
        )
    else:
        operational_status = "EXTENDED_TREND_OUTLOOK"
        is_safety_verdict_active = False
        displayed_tier = None  # Discrete tier suppressed
        displayed_tier_name = "SUPPRESSED_FOR_EXTENDED_HORIZON"
        advisory_message = (
            f"Extended macro outlook ({horizon_hours}h ahead). Discrete safety tier is SUPPRESSED "
            "due to climatological mean-regression (Critical FNR: 82%–100%). Displaying physical trajectory, "
            "P90 tail risk, and hard-gate alerts for advance trip scheduling; re-evaluate as conditions "
            "approach T-24h and T-1h."
        )

    return {
        "horizon_hours": horizon_hours,
        "operational_status": operational_status,
        "is_safety_verdict_active": is_safety_verdict_active,
        "displayed_tier": displayed_tier,
        "displayed_tier_name": displayed_tier_name,
        "ml_raw_prediction": ml_prediction,
        "ml_raw_prediction_name": TIER_NAMES[ml_prediction],
        "hard_gate_triggered": hard_gate_result["hard_gate_triggered"],
        "override_reasons": hard_gate_result["override_reasons"],
        "advisory_message": advisory_message,
        "telemetry": telemetry,
    }


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
def _run_tests():
    print("Running hard-gate and operational cutoff test suite...\n")
    failures = []
    total_checks = [0]

    def check(name, condition, message):
        total_checks[0] += 1
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            failures.append(f"{name}: {message}")

    calm = {"wind_speed": 3.0, "wind_gust": 4.0, "hs": 0.3, "swell_height": 0.2,
            "current_speed": 0.1, "rain_rate_mm_hr": 0.0, "slp": 1012.0}

    # 1. Fully calm conditions, ML says Very Safe -> stays Very Safe, no override
    r = apply_hard_gate(0, calm)
    check("calm conditions, ML=Very Safe -> stays Very Safe",
          r["final_tier"] == 0 and not r["hard_gate_triggered"], f"got {r}")

    # 2. Hard-gate must NEVER lower a higher ML prediction, even if physically calm
    r = apply_hard_gate(3, calm)
    check("calm conditions, ML=High Risk -> gate does not suppress ML's own caution",
          r["final_tier"] == 3, f"got {r}")

    # 3-9: each individual threshold, one at a time, all others calm
    breach_cases = {
        "wind_speed": {**calm, "wind_speed": HARD_GATE["wind_speed_ms"] + 0.1},
        "wind_gust": {**calm, "wind_gust": HARD_GATE["wind_gust_ms"] + 0.1},
        "hs": {**calm, "hs": HARD_GATE["wave_height_m"] + 0.1},
        "swell_height": {**calm, "swell_height": HARD_GATE["swell_height_m"] + 0.1},
        "current_speed": {**calm, "current_speed": HARD_GATE["current_ms"] + 0.1},
        "rain_rate_mm_hr": {**calm, "rain_rate_mm_hr": HARD_GATE["rain_mm_hr"] + 1.0},
        "slp": {**calm, "slp": HARD_GATE["pressure_hpa"] - 1.0},
    }
    for name, telemetry in breach_cases.items():
        r = apply_hard_gate(0, telemetry)
        check(f"{name} breach alone forces Critical Risk (ML said Very Safe)",
              r["final_tier"] == 4 and r["hard_gate_triggered"], f"got {r}")

    # 10. PAGASA TCWS Signal #3 override
    r_tcws = apply_hard_gate(0, calm, pagasa={"tcws_signal": 3})
    check("PAGASA TCWS #3 override forces Critical Risk",
          r_tcws["final_tier"] == 4 and r_tcws["hard_gate_triggered"], f"got {r_tcws}")

    # 11. PAGASA Gale Warning override
    r_gale = apply_hard_gate(0, calm, pagasa={"gale_warning": True})
    check("PAGASA Gale Warning override forces Critical Risk",
          r_gale["final_tier"] == 4 and r_gale["hard_gate_triggered"], f"got {r_gale}")

    # 12. Operational Cutoff Policy: 1h -> Tactical Clearance Active
    op1 = evaluate_operational_safety(1, 0, calm)
    check("Horizon 1h -> TACTICAL_CLEARANCE with active displayed tier",
          op1["operational_status"] == "TACTICAL_CLEARANCE" and op1["is_safety_verdict_active"] and op1["displayed_tier"] == 0,
          f"got {op1}")

    # 13. Operational Cutoff Policy: 6h -> PROVISIONAL_TREND_OUTLOOK with SUPPRESSED discrete tier
    op6 = evaluate_operational_safety(6, 1, calm)
    check("Horizon 6h -> PROVISIONAL_TREND_OUTLOOK with SUPPRESSED discrete tier",
          op6["operational_status"] == "PROVISIONAL_TREND_OUTLOOK" and not op6["is_safety_verdict_active"] and op6["displayed_tier"] is None,
          f"got {op6}")

    # 14. Operational Cutoff Policy: 24h -> PROVISIONAL_TREND_OUTLOOK with SUPPRESSED discrete tier
    op24 = evaluate_operational_safety(24, 1, calm)
    check("Horizon 24h -> PROVISIONAL_TREND_OUTLOOK with SUPPRESSED discrete tier",
          op24["operational_status"] == "PROVISIONAL_TREND_OUTLOOK" and not op24["is_safety_verdict_active"] and op24["displayed_tier"] is None,
          f"got {op24}")

    # 15. Operational Cutoff Policy: 72h -> EXTENDED_TREND_OUTLOOK with SUPPRESSED discrete tier
    op72 = evaluate_operational_safety(72, 0, calm)
    check("Horizon 72h -> EXTENDED_TREND_OUTLOOK with SUPPRESSED discrete tier",
          op72["operational_status"] == "EXTENDED_TREND_OUTLOOK" and not op72["is_safety_verdict_active"] and op72["displayed_tier"] is None,
          f"got {op72}")

    # 16. Operational Cutoff Policy: 72h with Hard-Gate breach -> Hard-gate still flags breach
    breach_wind = {**calm, "wind_speed": HARD_GATE["wind_speed_ms"] + 1.0}
    op72_breach = evaluate_operational_safety(72, 0, breach_wind)
    check("Horizon 72h with storm wind -> hard_gate_triggered is True even when discrete tier is suppressed",
          op72_breach["hard_gate_triggered"] is True and "sustained wind" in op72_breach["override_reasons"][0],
          f"got {op72_breach}")

    # 17. Intermediate & Non-Grid Horizon Range Checks (e.g. H=2h, H=12h, H=30h)
    op2 = evaluate_operational_safety(2, 0, calm)
    check("Intermediate Horizon 2h -> PROVISIONAL_TREND_OUTLOOK (1h < H <= 24h interval)",
          op2["operational_status"] == "PROVISIONAL_TREND_OUTLOOK" and not op2["is_safety_verdict_active"] and op2["displayed_tier"] is None,
          f"got {op2}")

    op12 = evaluate_operational_safety(12, 0, calm)
    check("Intermediate Horizon 12h -> PROVISIONAL_TREND_OUTLOOK (1h < H <= 24h interval)",
          op12["operational_status"] == "PROVISIONAL_TREND_OUTLOOK" and not op12["is_safety_verdict_active"] and op12["displayed_tier"] is None,
          f"got {op12}")

    op30 = evaluate_operational_safety(30, 0, calm)
    check("Intermediate Horizon 30h -> EXTENDED_TREND_OUTLOOK (H > 24h interval)",
          op30["operational_status"] == "EXTENDED_TREND_OUTLOOK" and not op30["is_safety_verdict_active"] and op30["displayed_tier"] is None,
          f"got {op30}")

    # 18. Diurnal solar heating barometric drop on calm sunny day (no companion storm indicators) -> NO false alarm breach
    calm_diurnal_drop = {**calm, "delta_p_3h": -2.8, "wind_gust": 4.0, "rain_rate_mm_hr": 0.0}
    r_diurnal = apply_hard_gate(0, calm_diurnal_drop)
    check("Diurnal solar pressure drop (-2.8 hPa) on calm sunny day does NOT trigger false alarm breach",
          r_diurnal["final_tier"] == 0 and not r_diurnal["hard_gate_triggered"], f"got {r_diurnal}")

    # 19. Severe storm precursor drop WITH companion squall gusts (>= 38 km/h / 10.56 m/s) -> Hard-gate breach
    storm_drop_squall = {**calm, "delta_p_3h": -2.8, "wind_gust": 11.0, "rain_rate_mm_hr": 0.0}
    r_storm_squall = apply_hard_gate(0, storm_drop_squall)
    check("Storm precursor pressure drop (-2.8 hPa) with companion squall gusts (11.0 m/s) forces Critical Risk",
          r_storm_squall["final_tier"] == 4 and r_storm_squall["hard_gate_triggered"] and "rapid barometric drop" in r_storm_squall["override_reasons"][0],
          f"got {r_storm_squall}")

    # 20. Severe storm precursor drop WITH companion heavy rain (>= 15 mm/hr) -> Hard-gate breach
    storm_drop_rain = {**calm, "delta_p_3h": -2.8, "wind_gust": 5.0, "rain_rate_mm_hr": 16.0}
    r_storm_rain = apply_hard_gate(0, storm_drop_rain)
    check("Storm precursor pressure drop (-2.8 hPa) with companion heavy rain (16.0 mm/hr) forces Critical Risk",
          r_storm_rain["final_tier"] == 4 and r_storm_rain["hard_gate_triggered"] and "rapid barometric drop" in r_storm_rain["override_reasons"][0],
          f"got {r_storm_rain}")

    print(f"\n{total_checks[0] - len(failures)} / {total_checks[0]} tests passed")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
            sys.exit(1)
    else:
        print("All hard-gate and operational cutoff tests passed.")


if __name__ == "__main__":
    _run_tests()

