"""
F1 Visualization System — Track Geometry Router

Provides circuit track geometry (centerline, inner/outer boundaries, DRS zones)
extracted from FastF1 telemetry data.
"""
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import ORJSONResponse

from src.backend.config import settings
from src.backend.services.fetcher import fetch_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["track"])


def _build_track_geometry(session, track_width: float = 200.0) -> dict:
    """
    Build track geometry from the fastest lap telemetry.

    Replicates the logic from example/f1-race-replay-main's
    build_track_from_example_lap() function.
    """
    # Try qualifying first for DRS data, then fall back to race fastest lap
    example_lap = None

    try:
        fastest = session.laps.pick_fastest()
        if fastest is not None:
            tel = fastest.get_telemetry()
            if tel is not None and "X" in tel.columns and "Y" in tel.columns and len(tel) > 10:
                example_lap = tel
    except Exception as e:
        logger.warning("Could not get fastest lap telemetry: %s", e)

    if example_lap is None:
        raise ValueError("No valid lap telemetry found for track geometry")

    x = example_lap["X"].values.astype(float)
    y = example_lap["Y"].values.astype(float)

    # Compute tangent normals
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2)
    norm[norm == 0] = 1.0
    dx /= norm
    dy /= norm

    # Perpendicular normals
    nx = -dy
    ny = dx

    # Inner / outer boundary offsets
    half_w = track_width / 2
    x_outer = x + nx * half_w
    y_outer = y + ny * half_w
    x_inner = x - nx * half_w
    y_inner = y - ny * half_w

    # World bounds
    all_x = np.concatenate([x, x_inner, x_outer])
    all_y = np.concatenate([y, y_inner, y_outer])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())

    # DRS zones
    drs_zones = []
    if "DRS" in example_lap.columns:
        drs_vals = example_lap["DRS"].values
        drs_start = None
        for i, val in enumerate(drs_vals):
            if val in [10, 12, 14]:
                if drs_start is None:
                    drs_start = i
            else:
                if drs_start is not None:
                    drs_zones.append({"start_idx": int(drs_start), "end_idx": int(i - 1)})
                    drs_start = None
        if drs_start is not None:
            drs_zones.append({"start_idx": int(drs_start), "end_idx": int(len(drs_vals) - 1)})

    # Downsample for JSON efficiency — keep every Nth point
    n_points = len(x)
    step = max(1, n_points // 500)  # ~500 points max

    def downsample(arr):
        sampled = arr[::step].tolist()
        # Always include the last point to close the loop
        if len(arr) > 0:
            sampled.append(float(arr[-1]))
        return [round(v, 1) for v in sampled]

    return {
        "centerline_x": downsample(x),
        "centerline_y": downsample(y),
        "inner_x": downsample(x_inner),
        "inner_y": downsample(y_inner),
        "outer_x": downsample(x_outer),
        "outer_y": downsample(y_outer),
        "drs_zones": drs_zones,
        "bounds": {
            "x_min": round(x_min, 1),
            "x_max": round(x_max, 1),
            "y_min": round(y_min, 1),
            "y_max": round(y_max, 1),
        },
        "full_outer_x": [round(v, 1) for v in x_outer.tolist()],
        "full_outer_y": [round(v, 1) for v in y_outer.tolist()],
        "point_count": n_points,
        "downsample_step": step,
    }


@router.get("/track/{session_id}")
async def get_track(session_id: str):
    """
    Get track geometry for a session.

    Returns centerline, inner/outer boundaries, DRS zones, and bounds,
    all as arrays of coordinates suitable for Canvas drawing.
    """
    # Check geometry cache (separate from exporter track.json list of points)
    cache_path = settings.processed_dir / session_id / "track_geometry.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        return ORJSONResponse(content={"session_id": session_id, **data})

    # Backward compatibility for old geometry caches stored in track.json.
    legacy_cache_path = settings.processed_dir / session_id / "track.json"
    if legacy_cache_path.exists():
        try:
            with open(legacy_cache_path) as f:
                legacy_data = json.load(f)
            if isinstance(legacy_data, dict) and "centerline_x" in legacy_data:
                return ORJSONResponse(content={"session_id": session_id, **legacy_data})
        except Exception:
            pass

    # Parse session_id: "{year}_{gp}_{type}"
    # e.g. "2024_Bahrain_Grand_Prix_R" → year=2024, gp="Bahrain Grand Prix", type="R"
    # Split from left to get year, from right to get session type
    first_underscore = session_id.find("_")
    last_underscore = session_id.rfind("_")
    if first_underscore == -1 or last_underscore == -1 or first_underscore == last_underscore:
        raise HTTPException(status_code=400, detail=f"Invalid session_id format: {session_id}")

    year_str = session_id[:first_underscore]
    session_type = session_id[last_underscore + 1:]
    gp = session_id[first_underscore + 1:last_underscore].replace("_", " ")

    try:
        year = int(year_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid year: {year_str}")

    try:
        session = fetch_session(year, gp, session_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error loading session for track geometry")
        raise HTTPException(status_code=500, detail=f"Session load error: {e}")

    try:
        geometry = _build_track_geometry(session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Cache for future requests
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(geometry, f)

    return ORJSONResponse(content={"session_id": session_id, **geometry})
