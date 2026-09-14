import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.serve.safety_thresholds import (  # noqa: F401
    SAFETY_THRESHOLDS,
    HARD_GATE,
    KMH_TO_MS,
    TIER_NAMES,
    TIER_CRITICAL,
    TIER_NO_CONSTRAINT,
    TACTICAL_GO_NO_GO_HORIZON_HOURS,
    PROVISIONAL_CUTOFF_HORIZON_HOURS,
    check_physical_breach,
    check_pagasa_override,
    apply_safety_thresholds,
    apply_hard_gate,
    evaluate_operational_safety,
)

if __name__ == "__main__":
    from src.serve.safety_thresholds import _run_tests
    _run_tests()
