"""
Runtime ML inference utilities for race event scoring.

Current scope:
- Overtake probability scoring using a trained XGBoost model
  saved at data/models/overtake_detector_v1.joblib
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.backend.config import settings

logger = logging.getLogger(__name__)

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


OVERTAKE_FEATURE_ORDER = [
    "relative_distance",
    "speed_diff",
    "gap_change_2s",
    "throttle_a",
    "brake_a",
    "speed_a",
    "drs_a",
]

_OVERTAKE_MODEL: Any | None = None
_OVERTAKE_MODEL_LOAD_ATTEMPTED = False


def _overtake_model_path() -> Path:
    return settings.models_dir / "overtake_detector_v1.joblib"


def _load_overtake_model() -> Any | None:
    global _OVERTAKE_MODEL
    global _OVERTAKE_MODEL_LOAD_ATTEMPTED

    if _OVERTAKE_MODEL_LOAD_ATTEMPTED:
        return _OVERTAKE_MODEL
    _OVERTAKE_MODEL_LOAD_ATTEMPTED = True

    if not settings.overtake_ml_enabled:
        logger.info("Overtake ML inference disabled by configuration")
        return None
    if joblib is None:
        logger.warning("joblib not available; overtake ML inference disabled")
        return None

    model_path = _overtake_model_path()
    if not model_path.exists():
        logger.info("No overtake model found at %s; using rules-only mode", model_path)
        return None

    try:
        _OVERTAKE_MODEL = joblib.load(model_path)
        logger.info("Loaded overtake ML model from %s", model_path)
    except Exception as exc:
        logger.warning("Failed to load overtake ML model (%s); using rules-only mode", exc)
        _OVERTAKE_MODEL = None

    return _OVERTAKE_MODEL


def predict_overtake_probability(features: dict[str, float]) -> dict[str, Any]:
    """
    Returns probability result dict.
    """
    model = _load_overtake_model()
    if model is None:
        return {"probability": None, "source": "rules_only"}

    try:
        row = [float(features.get(k, 0.0)) for k in OVERTAKE_FEATURE_ORDER]
        x = np.array([row], dtype=float)
        proba = float(model.predict_proba(x)[0][1])
        return {"probability": max(0.0, min(1.0, proba)), "source": "hybrid_ml"}
    except Exception as exc:
        logger.debug("Overtake ML inference failed: %s", exc)
        return {"probability": None, "source": "rules_only"}

