"""
F1 Visualization System — Coordinate & Track Validation

Validates that processed coordinates are in meters and that the computed
track length matches the official lap distance within tolerance.
"""
import logging
from src.backend.schemas import TrackPoint

logger = logging.getLogger(__name__)

# Official track lengths (meters) — source: FIA circuit data
OFFICIAL_TRACK_LENGTHS: dict[str, float] = {
    "Monza": 5793.0,
    "Silverstone": 5891.0,
    "Spa-Francorchamps": 7004.0,
    "Monaco": 3337.0,
    "Bahrain": 5412.0,
    "Jeddah": 6174.0,
    "Melbourne": 5278.0,
    "Suzuka": 5807.0,
    "Singapore": 4940.0,
    "Interlagos": 4309.0,
    "COTA": 5513.0,
    "Zandvoort": 4259.0,
    "Hungaroring": 4381.0,
    "Imola": 4909.0,
    "Barcelona": 4657.0,
    "Red Bull Ring": 4318.0,
    "Baku": 6003.0,
    "Las Vegas": 6201.0,
    "Lusail": 5419.0,
    "Shanghai": 5451.0,
    "Miami": 5412.0,
    "Abu Dhabi": 5281.0,
}


def compute_track_perimeter(track: list[TrackPoint]) -> float:
    """Compute closed-loop perimeter of track centerline in meters."""
    if len(track) < 2:
        return 0.0
    import numpy as np
    total = 0.0
    for i in range(1, len(track)):
        dx = track[i].x - track[i - 1].x
        dy = track[i].y - track[i - 1].y
        total += np.sqrt(dx ** 2 + dy ** 2)
    # Close loop
    dx = track[0].x - track[-1].x
    dy = track[0].y - track[-1].y
    total += np.sqrt(dx ** 2 + dy ** 2)
    return float(total)


def validate_track_length(
    track: list[TrackPoint],
    track_name: str,
    tolerance: float = 0.05,
) -> dict:
    """
    Compare computed track length to official.

    Returns a dict with validation results:
        {
            "computed_m": 5801.2,
            "official_m": 5793.0,
            "delta_pct": 0.14,
            "passed": True,
            "warning": None
        }
    """
    computed = compute_track_perimeter(track)

    # Try to find official length (fuzzy match)
    official = None
    for name, length in OFFICIAL_TRACK_LENGTHS.items():
        if name.lower() in track_name.lower() or track_name.lower() in name.lower():
            official = length
            break

    result = {
        "computed_m": round(computed, 1),
        "official_m": official,
        "delta_pct": None,
        "passed": True,
        "warning": None,
    }

    if official is None:
        result["warning"] = f"No official track length found for '{track_name}'"
        logger.warning(result["warning"])
        return result

    delta_pct = abs(computed - official) / official * 100
    result["delta_pct"] = round(delta_pct, 2)

    if delta_pct > tolerance * 100:
        result["passed"] = False
        result["warning"] = (
            f"Track length mismatch: computed {computed:.1f}m vs official {official:.1f}m "
            f"(delta {delta_pct:.1f}% > {tolerance * 100:.0f}% tolerance)"
        )
        logger.warning(result["warning"])
    else:
        logger.info(
            "Track length OK: computed %.1fm vs official %.1fm (delta %.2f%%)",
            computed, official, delta_pct,
        )

    return result
