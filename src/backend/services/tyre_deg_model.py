"""
Tyre Degradation ML Model — Inference Wrapper
==============================================
Drop at: src/backend/services/tyre_deg_model.py

Wraps the trained XGBoost model and exposes a single predict() method
that the prediction engine calls instead of the static _TYRE_DEG lookup.

Integration contract with prediction_engine.py:
  - Returns a float: predicted deg rate in seconds per lap
  - Falls back to the static table when the model is unavailable
  - Never raises — always returns a usable value

Usage in prediction_engine.py (build_snapshot and simulators):
    from src.backend.services.tyre_deg_model import TyreDegModel
    _tyre_model = TyreDegModel()   # module-level singleton

    # In _observed_deg_rate or wherever a deg rate is needed:
    rate = _tyre_model.predict(
        compound="S",
        tyre_life=15,
        track_temp=42.0,
        air_temp=32.0,
        fuel_load_kg=50.0,
        driver_code="VER",
        circuit="Bahrain",
        fresh_tyre=True,
        stint_num=1,
        abs_lap=40,
    )
    # Falls back to static table if model not loaded

Static fallback values (same as current _TYRE_DEG in prediction_engine.py):
    S: 0.10 s/lap  M: 0.06  H: 0.04  I: 0.07  W: 0.05
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Static fallback — mirrors _TYRE_DEG in prediction_engine.py exactly
_STATIC_DEG: dict[str, float] = {
    "S": 0.10,
    "M": 0.06,
    "H": 0.04,
    "I": 0.07,
    "W": 0.05,
    "U": 0.07,
}

_COMPOUND_ID: dict[str, int] = {
    "S": 0, "SOFT": 0,
    "M": 1, "MEDIUM": 1,
    "H": 2, "HARD": 2,
    "I": 3, "INTERMEDIATE": 3,
    "W": 4, "WET": 4,
}

# Feature column order — must match train_tyre_deg.py FEATURE_COLS exactly
_FEATURE_COLS = [
    "compound_id",
    "fresh_tyre",
    "tyre_life",
    "stint_num",
    "abs_lap",
    "track_temp",
    "air_temp",
    "fuel_load_kg",
    "driver_id",
    "circuit_id",
]


class TyreDegModel:
    """
    Singleton wrapper around the trained XGBoost tyre degradation model.

    Instantiate once at module level in prediction_engine.py:
        _tyre_model = TyreDegModel()

    The model loads lazily on first predict() call. If the model file
    doesn't exist (e.g. not yet trained), predict() returns the static
    fallback value silently — zero disruption to the engine.

    Thread safety: joblib model.predict() is thread-safe for inference.
    """

    def __init__(self, model_dir: Path | None = None):
        # Resolve model directory
        if model_dir is None:
            from src.backend.config import settings
            model_dir = settings.models_dir
        self._model_path   = Path(model_dir) / "tyre_deg_xgb.joblib"
        self._meta_path    = Path(model_dir) / "tyre_deg_meta.joblib"
        self._model        = None
        self._meta: dict   = {}
        self._loaded       = False
        self._load_failed  = False

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def _try_load(self) -> bool:
        """Attempt to load model + meta. Marks _load_failed on any error."""
        if self._loaded:
            return True
        if self._load_failed:
            return False
        try:
            import joblib
            self._model = joblib.load(self._model_path)
            self._meta  = joblib.load(self._meta_path)
            self._loaded = True
            logger.info("TyreDegModel loaded from %s", self._model_path)
            return True
        except FileNotFoundError:
            logger.debug(
                "TyreDegModel not found at %s — using static fallback",
                self._model_path,
            )
            self._load_failed = True
            return False
        except Exception as exc:
            logger.warning("TyreDegModel load failed: %s — using static fallback", exc)
            self._load_failed = True
            return False

    # ── Feature encoding ─────────────────────────────────────────────────────

    def _encode_driver(self, driver_code: str) -> int:
        """Encode driver code to integer. Unknown drivers get index 0."""
        driver_list = self._meta.get("driver_list", [])
        code = str(driver_code).upper()
        try:
            return driver_list.index(code)
        except ValueError:
            return 0   # unknown driver → treat as neutral

    def _encode_circuit(self, circuit: str) -> int:
        """Encode circuit name to integer. Unknown circuits get index 0."""
        circuit_list = self._meta.get("circuit_list", [])
        c = str(circuit).strip()
        # Try exact match first, then partial match
        for i, name in enumerate(circuit_list):
            if name.lower() == c.lower():
                return i
        for i, name in enumerate(circuit_list):
            if c.lower() in name.lower() or name.lower() in c.lower():
                return i
        return 0

    def _encode_compound(self, compound: str) -> int:
        return _COMPOUND_ID.get(str(compound).upper(), 1)

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True if the model is loaded and ready for inference."""
        return self._try_load()

    def predict(
        self,
        *,
        compound: str,
        tyre_life: int,
        track_temp: float = 30.0,
        air_temp: float = 20.0,
        fuel_load_kg: float = 60.0,
        driver_code: str = "UNK",
        circuit: str = "Unknown",
        fresh_tyre: bool = True,
        stint_num: int = 0,
        abs_lap: int = 1,
    ) -> float:
        """
        Predict tyre degradation rate in seconds per lap.

        Args:
            compound:     Tyre compound letter or full name (S/M/H/I/W)
            tyre_life:    Laps on current compound
            track_temp:   Track surface temperature °C (default 30)
            air_temp:     Ambient air temperature °C (default 20)
            fuel_load_kg: Estimated remaining fuel kg (default 60)
            driver_code:  Three-letter driver code (default UNK)
            circuit:      Circuit / GP name (default Unknown)
            fresh_tyre:   True if this is a new unused set
            stint_num:    Which stint (0=first, 1=second, etc.)
            abs_lap:      Absolute race lap number

        Returns:
            Predicted deg rate in s/lap. Always >= 0.001.
            Falls back to static table if model unavailable.
        """
        if not self._try_load():
            return _STATIC_DEG.get(str(compound).upper()[:1], 0.06)

        feature_vec = [[
            self._encode_compound(compound),     # compound_id
            int(fresh_tyre),                     # fresh_tyre
            max(1, int(tyre_life)),              # tyre_life
            max(0, int(stint_num)),              # stint_num
            max(1, int(abs_lap)),                # abs_lap
            float(track_temp),                   # track_temp
            float(air_temp),                     # air_temp
            max(0.0, float(fuel_load_kg)),       # fuel_load_kg
            self._encode_driver(driver_code),    # driver_id
            self._encode_circuit(circuit),       # circuit_id
        ]]

        try:
            pred = float(self._model.predict(feature_vec)[0])
            # Clamp to physically plausible range
            return round(max(0.001, min(pred, 0.50)), 5)
        except Exception as exc:
            logger.warning("TyreDegModel.predict failed: %s", exc)
            return _STATIC_DEG.get(str(compound).upper()[:1], 0.06)

    def predict_batch(self, rows: list[dict[str, Any]]) -> list[float]:
        """
        Batch predict for multiple drivers/laps at once.

        Args:
            rows: list of dicts, each with the same keys as predict().

        Returns:
            list of float deg rates, same order as input.
        """
        if not self._try_load():
            return [
                _STATIC_DEG.get(str(r.get("compound", "M")).upper()[:1], 0.06)
                for r in rows
            ]

        try:
            feature_matrix = [
                [
                    self._encode_compound(r.get("compound", "M")),
                    int(r.get("fresh_tyre", True)),
                    max(1, int(r.get("tyre_life", 1))),
                    max(0, int(r.get("stint_num", 0))),
                    max(1, int(r.get("abs_lap", 1))),
                    float(r.get("track_temp", 30.0)),
                    float(r.get("air_temp", 20.0)),
                    max(0.0, float(r.get("fuel_load_kg", 60.0))),
                    self._encode_driver(r.get("driver_code", "UNK")),
                    self._encode_circuit(r.get("circuit", "Unknown")),
                ]
                for r in rows
            ]
            preds = self._model.predict(feature_matrix)
            return [round(max(0.001, min(float(p), 0.50)), 5) for p in preds]
        except Exception as exc:
            logger.warning("TyreDegModel.predict_batch failed: %s", exc)
            return [
                _STATIC_DEG.get(str(r.get("compound", "M")).upper()[:1], 0.06)
                for r in rows
            ]

    def describe(self) -> dict[str, Any]:
        """Return model metadata for logging / API responses."""
        if not self._try_load():
            return {"status": "unavailable", "source": "static_fallback"}
        return {
            "status":       "loaded",
            "model_path":   str(self._model_path),
            "feature_cols": _FEATURE_COLS,
            "n_estimators": getattr(self._model, "n_estimators", None),
            "source":       "xgboost",
        }
