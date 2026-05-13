"""
Tyre Degradation XGBoost Training Script
==========================================
Drop at: scripts/ml/train_tyre_deg.py
Run:     python scripts/ml/train_tyre_deg.py

Trains an XGBoost model that predicts tyre degradation rate (seconds/lap)
from features available in FastF1 session data. Replaces the static
_TYRE_DEG lookup table in prediction_engine.py with a data-driven model
trained on 3 years of real race data.

Dependencies:  fastf1  xgboost  scikit-learn  pandas  numpy  joblib
Install:       pip install fastf1 xgboost scikit-learn pandas numpy joblib

Output files (written to data/models/):
  tyre_deg_xgb.joblib      — trained XGBoost regressor
  tyre_deg_meta.joblib     — encoders, feature list, circuit/driver maps
  tyre_deg_report.json     — training report (MAE, feature importance, etc.)

How it integrates with prediction_engine.py:
  The TyreDegModel class in tyre_deg_model.py (also in this folder) wraps
  the saved model and is imported by prediction_engine.py. When the model
  file exists, it overrides the static _TYRE_DEG constants. When it doesn't
  exist, the engine falls back to the static table (no code change needed).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

YEARS   = [2022, 2023, 2024]
CACHE   = Path("data/raw/fastf1_cache")
OUT_DIR = Path("data/models")

# Races to collect (covers all track surface types, temperatures, and layouts)
RACE_ROUNDS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan",
    "Miami", "Monaco", "Spain", "Canada",
    "Austria", "Britain", "Hungary", "Belgium",
    "Netherlands", "Italy", "Singapore", "United States",
    "Mexico", "Brazil", "Las Vegas", "Abu Dhabi",
]

# Feature columns used for training (must match tyre_deg_model.py)
FEATURE_COLS = [
    "compound_id",    # encoded compound (S=0 M=1 H=2 I=3 W=4)
    "fresh_tyre",     # 1=new set, 0=used set
    "tyre_life",      # laps on this compound in this stint
    "stint_num",      # which stint (0=first, 1=second, 2=third)
    "abs_lap",        # absolute lap number in the race
    "track_temp",     # track surface temperature °C
    "air_temp",       # ambient air temperature °C
    "fuel_load_kg",   # estimated remaining fuel (110 - abs_lap * 1.5)
    "driver_id",      # label-encoded driver code
    "circuit_id",     # label-encoded circuit/GP name
]

COMPOUND_ID_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
                   "INTERMEDIATE": 3, "WET": 4, "UNKNOWN": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def _safe_seconds(val) -> float | None:
    """Convert Timedelta to float seconds safely."""
    if val is None:
        return None
    try:
        import pandas as pd
        if pd.isna(val):
            return None
        return float(val.total_seconds())
    except Exception:
        return None


def collect_race_laps(year: int, gp: str) -> pd.DataFrame | None:
    """
    Load one race from FastF1 and extract per-lap features for training.

    Returns a DataFrame with FEATURE_COLS + 'deg_rate' + 'gp' columns,
    or None if the race failed to load or had insufficient data.

    Feature engineering:
      - deg_rate: (lap_time - stint_base_time) / (tyre_life - 1)
        where stint_base_time = lap 2 of the stint (lap 1 is the out-lap)
      - fuel_load_kg: estimated from lap number (110kg start, ~1.5kg/lap)
      - IsAccurate: FastF1 flag — we discard laps where this is False
      - in-laps and out-laps: discarded (first and last of each stint)
    """
    try:
        import fastf1
        from fastf1.core import DataNotLoadedError
        fastf1.Cache.enable_cache(str(CACHE))
        session = fastf1.get_session(year, gp, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        laps_df = session.laps
    except Exception as exc:
        logger.warning("Could not load %d %s: %s", year, gp, exc)
        return None

    if laps_df is None or len(laps_df) < 50:
        return None

    # Merge weather: get average track/air temp for this race
    try:
        weather = session.weather_data
        track_temp = float(weather["TrackTemp"].mean())
        air_temp   = float(weather["AirTemp"].mean())
    except Exception:
        track_temp = 30.0
        air_temp   = 20.0

    rows = []
    drivers = laps_df["Driver"].unique()

    for driver in drivers:
        d_laps = (laps_df[laps_df["Driver"] == driver]
                  .sort_values("LapNumber")
                  .reset_index(drop=True))

        stints = d_laps["Stint"].unique()
        for stint_num_idx, stint_id in enumerate(sorted(stints)):
            stint_laps = d_laps[d_laps["Stint"] == stint_id].reset_index(drop=True)
            if len(stint_laps) < 4:
                continue

            compound_raw = str(stint_laps.iloc[0].get("Compound", "UNKNOWN")).upper()
            compound_id  = COMPOUND_ID_MAP.get(compound_raw, 1)
            fresh_tyre   = int(bool(stint_laps.iloc[0].get("FreshTyre", True)))

            # Base lap time: lap 2 of stint (out-lap warmed up, not an in-lap)
            base_lt = _safe_seconds(stint_laps.iloc[1].get("LapTime"))
            if base_lt is None or base_lt < 60:
                continue

            for sl_idx, (_, row) in enumerate(stint_laps.iterrows()):
                tyre_life = int(row.get("TyreLife", sl_idx + 1))
                abs_lap   = int(row.get("LapNumber", 1))
                is_acc    = bool(row.get("IsAccurate", True))

                # Skip out-lap, in-lap, and inaccurate laps
                if sl_idx == 0 or sl_idx == len(stint_laps) - 1 or not is_acc:
                    continue

                lt = _safe_seconds(row.get("LapTime"))
                if lt is None or lt < 60 or tyre_life <= 1:
                    continue

                # deg_rate: seconds of pace loss per lap of tyre age vs stint base
                deg_rate = (lt - base_lt) / max(1, tyre_life - 1)

                # Filter out physically implausible values
                # (safety car laps, incidents, etc.)
                if deg_rate < -0.5 or deg_rate > 0.5:
                    continue

                fuel_load = max(0.0, 110.0 - abs_lap * 1.5)

                rows.append({
                    "compound_id":  compound_id,
                    "fresh_tyre":   fresh_tyre,
                    "tyre_life":    tyre_life,
                    "stint_num":    stint_num_idx,
                    "abs_lap":      abs_lap,
                    "track_temp":   track_temp,
                    "air_temp":     air_temp,
                    "fuel_load_kg": fuel_load,
                    "driver":       driver,
                    "gp":           gp,
                    "year":         year,
                    "deg_rate":     deg_rate,
                })

    if not rows:
        return None

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train():
    raise NotImplementedError("Model training function to be added in next commit.")
