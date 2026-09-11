"""
Source: TPXO9 Atlas / FES2014 harmonic constituents, calibrated against NAMRIA
        Batangas Port / Puerto Galera tide-gauge records.
Pulls:  nothing over an API — this is computed, not downloaded. NAMRIA gauge
        history is the one manual acquisition step here (request from NAMRIA
        or use a published tide-gauge archive for the same station).
"""

import pandas as pd
from utide import solve, reconstruct

from config import START_DATE, END_DATE, SITE_LAT, RAW_DIR, INTERIM_DIR, target_hourly_index

GAUGE_CSV = f"{RAW_DIR}/namria_batangas_tide_gauge.csv"  # columns: time, water_level_m
OUT_INTERIM = f"{INTERIM_DIR}/tides.parquet"


def fit_harmonics():
    gauge = pd.read_csv(GAUGE_CSV, parse_dates=["time"]).set_index("time")

    coef = solve(
        t=gauge.index.values,
        u=gauge["water_level_m"].values,
        lat=SITE_LAT,
        method="ols",
        conf_int="linear",
    )
    return coef


def to_interim(coef):
    hourly_index = target_hourly_index()
    recon = reconstruct(t=hourly_index.values, coef=coef)

    df = pd.DataFrame({"time": hourly_index, "tide_height": recon["h"]}).set_index("time")
    df["tidal_rate"] = df["tide_height"].diff()  # dh/dt, also computed again in features.py —
                                                   # kept here too so raw interim files are self-describing
    df.to_parquet(OUT_INTERIM)
    print(f"wrote {len(df)} hourly rows -> {OUT_INTERIM}")


if __name__ == "__main__":
    coefficients = fit_harmonics()
    to_interim(coefficients)