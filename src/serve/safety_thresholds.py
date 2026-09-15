"""
Deterministic Physical Safety Limits & Thresholds layer and Operational Horizon Policy.
Sits directly on top of xgb_safety_classifier's output: any breach of a physical
threshold or an active PAGASA storm signal forces Critical Risk regardless of what
the classifier predicted.

OPERATIONAL HORIZON POLICY (Safety Framework & Empirical Horizon Validation):
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
      safety threshold backstop for advance planning only. Final clearance deferred to T-1h.
- H > 24h (Extended Window: 48h, 72h, 96h, 144h):
    * Discrete ML safety tier is SUPPRESSED: "EXTENDED_TREND_OUTLOOK"
    * Empirical Critical FNR is 82.0% - 100.0% (climatological mean-regression).
    * Surfaces raw physical trajectory, P90 tail bounds, and active safety threshold backstop.

Thresholds are imported from build_safety_labels.py, NOT retyped here.

Run from the project root: python src/serve/safety_thresholds.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.labels.build_safety_labels import SAFETY_THRESHOLDS, HARD_GATE, KMH_TO_MS  # noqa: E402

TIER_NAMES = ["Very Safe", "Safe", "Moderate", "High Risk", "Critical Risk"]
TIER_CRITICAL = 4
TIER_NO_CONSTRAINT = 0  # what the safety threshold contributes to max() when nothing is breached

TACTICAL_GO_NO_GO_HORIZON_HOURS = 1
PROVISIONAL_CUTOFF_HORIZON_HOURS = 24


def check_physical_breach(telemetry: dict) -> tuple[bool, list[str]]:
    """
    Evaluates individual telemetry variables against deterministic safety thresholds.

    Business Logic / Meteorological Rationale:
        1. Single-Variable Hard Limits: If sustained winds, gusts, wave heights, swells,
           currents, rain rate, or minimum barometric pressure exceed safe operating
           parameters, the session is instantly unsafe regardless of model confidence.
        2. Context-Aware Compound Precursor Check:
           A rapid 3-hour barometric drop (>= 2.5 hPa / 3h) is an established cyclone/storm
           indicator. However, in tropical latitudes, semi-diurnal solar atmospheric tides
           (the S2 oscillation) naturally drop pressure by 1.5 - 2.0 hPa every afternoon.
           To prevent daily false alarms on calm sunny days, a rapid barometric drop only
           triggers an emergency breach if accompanied by squall gusts (>= 38 km/h) or
           heavy rain (>= 15 mm/hr).

    Parameters:
        telemetry (dict): Dictionary of physical readings:
            - wind_speed (float): Sustained wind in m/s.
            - wind_gust (float): Peak gust in m/s.
            - hs (float): Significant wave height in meters.
            - swell_height (float): Swell height in meters.
            - current_speed (float): Ocean current speed in m/s.
            - rain_rate_mm_hr (float): Precipitation rate in mm/hr.
            - slp (float): Sea level pressure in hPa.
            - delta_p_3h (float, optional): 3-hour barometric tendency in hPa.

    Returns:
        tuple[bool, list[str]]: (breached, list of human-readable breach explanations).
    """
    reasons = []

    limits = SAFETY_THRESHOLDS if "SAFETY_THRESHOLDS" in globals() else HARD_GATE

    if telemetry.get("wind_speed", 0.0) >= limits["wind_speed_ms"]:
        reasons.append(f"sustained wind {telemetry['wind_speed']:.1f} m/s >= "
                        f"{limits['wind_speed_ms']:.2f} m/s limit")
    if telemetry.get("wind_gust", 0.0) >= limits["wind_gust_ms"]:
        reasons.append(f"wind gust {telemetry['wind_gust']:.1f} m/s >= "
                        f"{limits['wind_gust_ms']:.2f} m/s limit")
    if telemetry.get("hs", 0.0) >= limits["wave_height_m"]:
        reasons.append(f"wave height {telemetry['hs']:.2f} m >= {limits['wave_height_m']} m limit")
    if telemetry.get("swell_height", 0.0) >= limits["swell_height_m"]:
        reasons.append(f"swell height {telemetry['swell_height']:.2f} m >= "
                        f"{limits['swell_height_m']} m limit")
    if telemetry.get("current_speed", 0.0) >= limits["current_ms"]:
        reasons.append(f"current {telemetry['current_speed']:.2f} m/s >= {limits['current_ms']} m/s limit")
    if telemetry.get("rain_rate_mm_hr", 0.0) >= limits["rain_mm_hr"]:
        reasons.append(f"rain rate {telemetry['rain_rate_mm_hr']:.1f} mm/hr >= "
                        f"{limits['rain_mm_hr']} mm/hr limit")
    if telemetry.get("slp", 1013.25) <= limits["pressure_hpa"]:
        reasons.append(f"pressure {telemetry['slp']:.1f} hPa <= {limits['pressure_hpa']} hPa limit")

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
    """
    Evaluates official PAGASA cyclone signals, gale warnings, and tsunami alerts.

    Business Logic:
        Government regulatory marine advisories supersede mathematical forecasting models.
        TCWS Signal #3+, Gale Warnings, and Tsunami Warnings mandate immediate cessation
        of all watercraft operations and freediving training.

    Parameters:
        pagasa (dict | None): Optional dictionary containing:
            - tcws_signal (int): Tropical Cyclone Wind Signal (0-5).
            - gale_warning (bool): Active Coast Guard gale warning flag.
            - tsunami_warning (bool): Active PHIVOLCS tsunami advisory flag.

    Returns:
        tuple[bool, list[str]]: (breached, list of active advisory statements).
    """
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


def apply_safety_thresholds(ml_prediction: int, telemetry: dict, pagasa: dict | None = None) -> dict:
    """
    Core Deterministic Safety Threshold Override.

    Mathematical Logic:
        Final Risk Tier = max(ml_prediction, safety_threshold_tier, pagasa_override_tier).
        A physical threshold or advisory breach can only ever escalate the risk tier
        toward Critical Risk (Tier 4); it can never downgrade a conservative ML classification.

    Parameters:
        ml_prediction (int): Raw integer class (0=Very Safe .. 4=Critical Risk) from xgb_safety_classifier.
        telemetry (dict): Physical weather and oceanographic variables.
        pagasa (dict | None): Active PAGASA advisories.

    Returns:
        dict: Standardized safety evaluation dictionary containing:
            - final_tier (int): Post-override integer risk tier (0-4).
            - final_tier_name (str): Label (e.g. 'Critical Risk').
            - ml_prediction (int): Pre-override classifier prediction.
            - ml_prediction_name (str): Pre-override classifier label.
            - safety_threshold_triggered (bool): True if any physical or advisory limit breached.
            - hard_gate_triggered (bool): Backward-compatible alias.
            - override_reasons (list[str]): Detailed explanations of all triggered constraints.
    """
    physical_breach, physical_reasons = check_physical_breach(telemetry)
    pagasa_breach, pagasa_reasons = check_pagasa_override(pagasa)

    threshold_tier = TIER_CRITICAL if (physical_breach or pagasa_breach) else TIER_NO_CONSTRAINT
    final_tier = max(ml_prediction, threshold_tier)

    all_reasons = physical_reasons + pagasa_reasons
    triggered = physical_breach or pagasa_breach
    return {
        "final_tier": final_tier,
        "final_tier_name": TIER_NAMES[final_tier],
        "ml_prediction": ml_prediction,
        "ml_prediction_name": TIER_NAMES[ml_prediction],
        "safety_threshold_triggered": triggered,
        "hard_gate_triggered": triggered,  # Backward-compatible alias
        "override_reasons": all_reasons if all_reasons else ["within all physical/advisory limits"],
    }


# Backwards compatibility alias
apply_hard_gate = apply_safety_thresholds


def evaluate_operational_safety(
    horizon_hours: int,
    ml_prediction: int,
    telemetry: dict,
    pagasa: dict | None = None
) -> dict:
    """
    Operational Decision Function Enforcing the 3-Tier Horizon Cutoff Policy.

    - Band 1 (H = 1h): "TACTICAL_CLEARANCE"
      Surfaces the authoritative active ML safety verdict + safety thresholds.
      Empirical Critical FNR = 4.5%, Precision = 96.8%.

    - Band 2 (1h < H <= 24h, i.e. 6h, 12h, 24h): "PROVISIONAL_TREND_OUTLOOK"
      SUPPRESSES discrete ML safety tier (displayed_tier: null).
      Empirical Critical FNR is 40.9% - 54.5% (~45% miss rate). Surfaces raw physics,
      P90 tail risk, and active safety limits for tentative planning.

    - Band 3 (H > 24h, i.e. 48h, 72h, 96h, 144h): "EXTENDED_TREND_OUTLOOK"
      SUPPRESSES discrete ML safety tier (displayed_tier: null).
      Empirical Critical FNR is 82.0% - 100.0% due to climatological mean-regression.
      Surfaces physical trajectory, P90 tail risk, and active safety limits.
    """
    threshold_result = apply_safety_thresholds(ml_prediction, telemetry, pagasa)

    if horizon_hours <= TACTICAL_GO_NO_GO_HORIZON_HOURS:
        operational_status = "TACTICAL_CLEARANCE"
        is_safety_verdict_active = True
        displayed_tier = threshold_result["final_tier"]
        displayed_tier_name = threshold_result["final_tier_name"]
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
            "because models at this range historically miss ~45% of dangerous conditions (FNR: 40.9%-54.5%) "
            "due to early MSE variance smoothing. Displaying raw physics, P90 tail bounds, and safety limit alerts "
            "for tentative planning; formal safety clearance is strictly evaluated at T-1h."
        )
    else:
        operational_status = "EXTENDED_TREND_OUTLOOK"
        is_safety_verdict_active = False
        displayed_tier = None  # Discrete tier suppressed
        displayed_tier_name = "SUPPRESSED_FOR_EXTENDED_HORIZON"
        advisory_message = (
            f"Extended macro outlook ({horizon_hours}h ahead). Discrete safety tier is SUPPRESSED "
            "due to climatological mean-regression (Critical FNR: 82%-100%). Displaying physical trajectory, "
            "P90 tail risk, and safety limit alerts for advance trip scheduling; re-evaluate as conditions "
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
        "safety_threshold_triggered": threshold_result["safety_threshold_triggered"],
        "hard_gate_triggered": threshold_result["hard_gate_triggered"],  # Backward-compatible alias
        "override_reasons": threshold_result["override_reasons"],
        "advisory_message": advisory_message,
        "telemetry": telemetry,
    }


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
def _run_tests():
    print("Running safety threshold and operational cutoff test suite...\n")
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

    # 1. Calm conditions -> no override
    r = apply_safety_thresholds(0, calm)
    check("calm -> final_tier is 0", r["final_tier"] == 0, f"got {r}")
    check("calm -> not triggered", not r["safety_threshold_triggered"], f"got {r}")

    # 2. Safety thresholds must NEVER lower a higher ML prediction, even if physically calm
    r = apply_safety_thresholds(3, calm)
    check("ML=3 calm -> final_tier stays 3", r["final_tier"] == 3, f"got {r}")
    check("ML=3 calm -> not triggered", not r["safety_threshold_triggered"], f"got {r}")

    # 3. Sustained wind breach
    w_breach = dict(calm, wind_speed=11.0)  # > 10.56 m/s
    r = apply_safety_thresholds(0, w_breach)
    check("wind breach -> final_tier is 4", r["final_tier"] == 4, f"got {r}")
    check("wind breach -> triggered is True", r["safety_threshold_triggered"] is True, f"got {r}")

    # 4. Wind gust breach
    g_breach = dict(calm, wind_gust=14.0)  # > 13.33 m/s
    r = apply_safety_thresholds(1, g_breach)
    check("gust breach -> final_tier is 4", r["final_tier"] == 4, f"got {r}")

    # 5. Wave height breach
    hs_breach = dict(calm, hs=2.0)
    r = apply_safety_thresholds(0, hs_breach)
    check("hs breach -> final_tier is 4", r["final_tier"] == 4, f"got {r}")

    # 6. Swell height breach
    sw_breach = dict(calm, swell_height=1.9)
    r = apply_safety_thresholds(0, sw_breach)
    check("swell breach -> final_tier is 4", r["final_tier"] == 4, f"got {r}")

    # 7. Current speed breach
    c_breach = dict(calm, current_speed=0.9)
    r = apply_safety_thresholds(0, c_breach)
    check("current breach -> final_tier is 4", r["final_tier"] == 4, f"got {r}")

    # 8. Rain rate breach
    r_breach = dict(calm, rain_rate_mm_hr=26.0)
    r = apply_safety_thresholds(0, r_breach)
    check("rain breach -> final_tier is 4", r["final_tier"] == 4, f"got {r}")

    # 9. Pressure breach
    p_breach = dict(calm, slp=995.0)
    r = apply_safety_thresholds(0, p_breach)
    check("pressure breach -> final_tier is 4", r["final_tier"] == 4, f"got {r}")

    # 10. PAGASA signal 3
    r = apply_safety_thresholds(0, calm, {"tcws_signal": 3})
    check("PAGASA signal 3 -> final_tier is 4", r["final_tier"] == 4, f"got {r}")
    check("PAGASA signal 3 -> triggered is True", r["safety_threshold_triggered"] is True, f"got {r}")

    # 11. PAGASA gale warning
    r = apply_safety_thresholds(0, calm, {"gale_warning": True})
    check("PAGASA gale -> final_tier is 4", r["final_tier"] == 4, f"got {r}")

    # 12. PAGASA tsunami warning
    r = apply_safety_thresholds(0, calm, {"tsunami_warning": True})
    check("tsunami -> final_tier is 4", r["final_tier"] == 4, f"got {r}")

    # 13. PAGASA signal 1 / 2 alone do NOT trigger critical override
    r = apply_safety_thresholds(0, calm, {"tcws_signal": 2})
    check("PAGASA signal 2 -> final_tier is 0 (no critical override)", r["final_tier"] == 0, f"got {r}")
    check("PAGASA signal 2 -> not triggered", not r["safety_threshold_triggered"], f"got {r}")

    # 14. Operational Cutoff Policy: 1h Horizon -> Tactical clearance active
    op1 = evaluate_operational_safety(1, 0, calm)
    check("1h horizon -> TACTICAL_CLEARANCE", op1["operational_status"] == "TACTICAL_CLEARANCE", f"got {op1}")
    check("1h horizon -> is_safety_verdict_active is True", op1["is_safety_verdict_active"] is True, f"got {op1}")
    check("1h horizon -> displayed_tier is 0", op1["displayed_tier"] == 0, f"got {op1}")

    # 15. Operational Cutoff Policy: 6h Horizon -> Discrete tier suppressed
    op6 = evaluate_operational_safety(6, 0, calm)
    check("6h horizon -> PROVISIONAL_TREND_OUTLOOK", op6["operational_status"] == "PROVISIONAL_TREND_OUTLOOK", f"got {op6}")
    check("6h horizon -> is_safety_verdict_active is False", op6["is_safety_verdict_active"] is False, f"got {op6}")
    check("6h horizon -> displayed_tier is None", op6["displayed_tier"] is None, f"got {op6}")

    # 16. Operational Cutoff Policy: 72h with breach -> flags breach
    op72_breach = evaluate_operational_safety(72, 0, w_breach)
    check("72h with breach -> triggered is True", op72_breach["safety_threshold_triggered"] is True, f"got {op72_breach}")
    check("72h horizon -> EXTENDED_TREND_OUTLOOK", op72_breach["operational_status"] == "EXTENDED_TREND_OUTLOOK", f"got {op72_breach}")
    check("72h horizon -> displayed_tier is None (suppressed)", op72_breach["displayed_tier"] is None, f"got {op72_breach}")

    # 17. Diurnal barometric pressure drop WITHOUT squall gusts or rain -> No breach
    diurnal_drop = dict(calm, delta_p_3h=-2.6, wind_gust=5.0, rain_rate_mm_hr=0.0)
    r_diurnal = apply_safety_thresholds(0, diurnal_drop)
    check("diurnal drop (-2.6 hPa) with calm wind -> no breach",
          not r_diurnal["safety_threshold_triggered"], f"got {r_diurnal}")

    # 18. Severe storm precursor drop WITH companion squall gusts (>= 38 km/h / 10.56 m/s) -> Breach
    storm_drop_gust = dict(calm, delta_p_3h=-2.6, wind_gust=11.0, rain_rate_mm_hr=0.0)
    r_storm_gust = apply_safety_thresholds(0, storm_drop_gust)
    check("storm precursor drop (-2.6 hPa) + squall gust (11 m/s) -> breach",
          r_storm_gust["safety_threshold_triggered"] is True and r_storm_gust["final_tier"] == 4, f"got {r_storm_gust}")

    # 19. Severe storm precursor drop WITH companion heavy rain (>= 15 mm/hr) -> Breach
    storm_drop_rain = dict(calm, delta_p_3h=-2.6, wind_gust=5.0, rain_rate_mm_hr=16.0)
    r_storm_rain = apply_safety_thresholds(0, storm_drop_rain)
    check("storm precursor drop (-2.6 hPa) + storm rain (16 mm/hr) -> breach",
          r_storm_rain["safety_threshold_triggered"] is True and r_storm_rain["final_tier"] == 4, f"got {r_storm_rain}")

    print(f"\n{total_checks[0]} checks run.")
    if failures:
        print(f"FAILED {len(failures)} checks:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All safety threshold and operational cutoff tests passed.")


if __name__ == "__main__":
    _run_tests()
