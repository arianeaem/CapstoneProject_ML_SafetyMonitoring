"""
Shared constants for every ingestion script.
Every source resamples to the same hourly index (target_hourly_index()) before
writing to data/interim/.

Paths are anchored to the project root via pathlib, NOT relative strings — this
is what lets every script run correctly regardless of which folder you're
standing in when you invoke `python`. Do not revert this back to plain
"data/raw" style strings; that breaks the moment a script is run from a
subfolder like src/ingest instead of the project root.
"""

from pathlib import Path
import pandas as pd


def _find_project_root() -> Path:
    """Climbs up from this file's location looking for the .venv folder, which
    only exists once, at the real project root — this works correctly even if
    this exact config.py file gets copied somewhere else by accident, unlike a
    hardcoded parent.parent.parent count."""
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / ".venv").exists():
            return current
        current = current.parent
    raise RuntimeError(
        "Could not locate project root (looked for a .venv folder within 6 levels up). "
        "Make sure you're running this from inside the project, and that only one "
        "config.py exists (check for an accidental duplicate at the project root)."
    )


PROJECT_ROOT = _find_project_root()

# Balayan Bay / Verde Island Passage bounding box (Anilao/Mabini dive sites)
LON_MIN, LON_MAX = 120.7, 121.1
LAT_MIN, LAT_MAX = 13.5, 14.0
SITE_LAT, SITE_LON = 13.7481, 120.9408  # exact dive-site point for post-collocation extraction

# 3-year historical observation window (2022-01-01 to 2024-12-31)
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
INTERIM_DIR = str(PROJECT_ROOT / "data" / "interim")


def target_hourly_index():
    """The one canonical hourly index every interim file must match exactly.
    resample() alone only spans a source's own first/last raw timestamp, which
    silently truncates at dataset-specific boundaries — reindexing onto this
    fixed index catches and fixes that before it reaches collocation."""
    return pd.date_range(f"{START_DATE} 00:00:00", f"{END_DATE} 23:00:00", freq="1h")
