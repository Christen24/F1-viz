"""
F1 Visualization System — Master Timeline Resampler

Creates a single synchronized timeline for all drivers at a fixed frame rate.
Uses cubic spline interpolation for positions and linear for numeric channels.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import fastf1

from src.backend.config import settings
from src.backend.schemas import (
    DriverFrame, DriverInfo, SessionEvent, SessionMetadata,
    TimeFrame, TrackPoint,
)

logger = logging.getLogger(__name__)

# F1 2024 team colors (hex)
TEAM_COLORS: dict[str, str] = {
    "Red Bull Racing": "#3671C6",
    "Red Bull": "#3671C6",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "AlphaTauri": "#6692FF",
    "RB": "#6692FF",
    "Alfa Romeo": "#C92D4B",
    "Kick Sauber": "#52E252",
    "Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
}


def _get_team_color(team_name: str) -> str:
    """Look up team color, fall back to gray."""
    for key, color in TEAM_COLORS.items():
        if key.lower() in team_name.lower():
            return color
    return "#888888"


def extract_driver_infos(session: fastf1.core.Session) -> list[DriverInfo]:
    """Extract driver metadata from session results."""
    drivers = []
    results = session.results
    if results is None or len(results) == 0:
        return drivers

    for _, row in results.iterrows():
        drivers.append(DriverInfo(
            code=str(row.get("Abbreviation", "UNK")),
            name=f"{row.get('FirstName', '')} {row.get('LastName', '')}".strip(),
            team=str(row.get("TeamName", "Unknown")),
            team_color=_get_team_color(str(row.get("TeamName", ""))),
            number=int(row.get("DriverNumber", 0)),
        ))
    return drivers


def extract_track_outline(session: fastf1.core.Session) -> list[TrackPoint]:
    """
    Extract track centerline from the fastest lap's positional telemetry.
    Coordinates from FastF1 are in 1/10 of a meter — we divide by 10.
    """
    try:
        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry()
    except Exception:
        logger.warning("Could not get fastest lap telemetry for track outline")
        return []

    if tel is None or len(tel) == 0:
        return []

    # FastF1 X/Y are in 1/10 meter units
    x_vals = tel["X"].values / 10.0
    y_vals = tel["Y"].values / 10.0

    # Downsample to ~500 points for a lightweight spline
    total = len(x_vals)
    step = max(1, total // 500)
    points = []
    for i in range(0, total, step):
        points.append(TrackPoint(x=float(x_vals[i]), y=float(y_vals[i])))

    return points


def compute_track_length(track: list[TrackPoint]) -> float:
    """Compute perimeter length of track from point list, in meters."""
    if len(track) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(track)):
        dx = track[i].x - track[i - 1].x
        dy = track[i].y - track[i - 1].y
        total += np.sqrt(dx ** 2 + dy ** 2)
    # Close the loop
    dx = track[0].x - track[-1].x
    dy = track[0].y - track[-1].y
    total += np.sqrt(dx ** 2 + dy ** 2)
    return float(total)


def _extract_driver_telemetry(session: fastf1.core.Session, driver_code: str) -> Optional[pd.DataFrame]:
    """
    Extract and concatenate telemetry for one driver across all laps.
    Returns a DataFrame with columns:
        SessionTime_s, X, Y, Distance, Speed, Throttle, Brake, nGear, DRS,
        LapNumber, Compound, TyreLife
    """
    driver_laps = session.laps.pick_drivers(driver_code)
    if driver_laps is None or len(driver_laps) == 0:
        return None

    all_frames = []
    for _, lap in driver_laps.iterrows():
        try:
            tel = lap.get_telemetry()
        except Exception:
            continue

        if tel is None or len(tel) == 0:
            continue

        df = pd.DataFrame()
        # Convert SessionTime (timedelta) to float seconds
        if "SessionTime" in tel.columns:
            df["SessionTime_s"] = tel["SessionTime"].dt.total_seconds()
        elif "Time" in tel.columns:
            df["SessionTime_s"] = tel["Time"].dt.total_seconds()
        else:
            continue

        # Position: FastF1 uses 1/10 meter → convert to meters
        df["X"] = tel["X"].values / 10.0 if "X" in tel.columns else 0.0
        df["Y"] = tel["Y"].values / 10.0 if "Y" in tel.columns else 0.0

        # Telemetry channels
        df["Distance"] = tel["Distance"].values if "Distance" in tel.columns else 0.0
        df["Speed"] = tel["Speed"].values if "Speed" in tel.columns else 0.0
        df["Throttle"] = tel["Throttle"].values if "Throttle" in tel.columns else 0.0
        df["Brake"] = tel["Brake"].astype(float).values if "Brake" in tel.columns else 0.0
        df["nGear"] = tel["nGear"].values if "nGear" in tel.columns else 0
        df["DRS"] = tel["DRS"].values if "DRS" in tel.columns else 0

        # Lap-level metadata
        df["LapNumber"] = int(lap.get("LapNumber", 0))
        df["Compound"] = str(lap.get("Compound", "UNKNOWN"))
        df["TyreLife"] = int(lap.get("TyreLife", 0)) if pd.notna(lap.get("TyreLife")) else 0

        all_frames.append(df)

    if not all_frames:
        return None

    combined = pd.concat(all_frames, ignore_index=True)
    combined.sort_values("SessionTime_s", inplace=True)
    combined.drop_duplicates(subset=["SessionTime_s"], keep="first", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    return combined


def _interpolate_driver_to_timeline(
    driver_df: pd.DataFrame,
    master_times: np.ndarray,
) -> list[DriverFrame]:
    """
    Resample a single driver's telemetry to the master timeline using:
    - Cubic spline for X, Y positions
    - Linear interpolation for all other channels
    - Vectorized numpy operations (no per-frame Python loop)
    """
    t_orig = driver_df["SessionTime_s"].values
    if len(t_orig) < 50:
        # Skip drivers with very little data (early retirement / no telemetry)
        return []

    # Clamp master times to driver's data range
    t_min, t_max = t_orig[0], t_orig[-1]

    # Cubic spline for position (smooth paths)
    try:
        cs_x = CubicSpline(t_orig, driver_df["X"].values, extrapolate=False)
        cs_y = CubicSpline(t_orig, driver_df["Y"].values, extrapolate=False)
    except Exception:
        cs_x = cs_y = None

    # Linear interpolation for numeric channels (vectorized)
    def lin_interp(col: str, default: float = 0.0) -> np.ndarray:
        vals = driver_df[col].values if col in driver_df.columns else np.full(len(t_orig), default)
        return np.interp(master_times, t_orig, vals, left=default, right=default)

    x_interp = cs_x(master_times) if cs_x else np.interp(master_times, t_orig, driver_df["X"].values)
    y_interp = cs_y(master_times) if cs_y else np.interp(master_times, t_orig, driver_df["Y"].values)

    # Replace NaN from cubic spline extrapolation with linear fallback
    nan_mask = np.isnan(x_interp) | np.isnan(y_interp)
    if nan_mask.any():
        x_lin = np.interp(master_times, t_orig, driver_df["X"].values)
        y_lin = np.interp(master_times, t_orig, driver_df["Y"].values)
        x_interp[nan_mask] = x_lin[nan_mask]
        y_interp[nan_mask] = y_lin[nan_mask]

    dist_interp = lin_interp("Distance")
    speed_interp = lin_interp("Speed")
    throttle_interp = lin_interp("Throttle")
    brake_interp = lin_interp("Brake")
    gear_interp = np.round(lin_interp("nGear")).astype(int)
    drs_interp = lin_interp("DRS") >= 10

    # Lap number: forward-fill (nearest previous)
    lap_interp = np.interp(master_times, t_orig, driver_df["LapNumber"].values).astype(int)

    # In-range mask (vectorized)
    in_range = (master_times >= t_min) & (master_times <= t_max)

    # Nearest compound via vectorized searchsorted
    idx_all = np.searchsorted(t_orig, master_times, side="right") - 1
    idx_all = np.clip(idx_all, 0, len(t_orig) - 1)

    compounds_arr = driver_df["Compound"].values
    tyre_life_arr = driver_df["TyreLife"].values

    # Build frames in bulk
    frames = []
    for i in range(len(master_times)):
        is_in = bool(in_range[i])
        compound = str(compounds_arr[idx_all[i]]) if is_in else "UNKNOWN"
        tyre_age = int(tyre_life_arr[idx_all[i]]) if is_in else 0

        frames.append(DriverFrame(
            x=float(x_interp[i]),
            y=float(y_interp[i]),
            distance=float(dist_interp[i]),
            speed=float(speed_interp[i]),
            throttle=float(throttle_interp[i]),
            brake=float(brake_interp[i]),
            gear=int(gear_interp[i]),
            drs=bool(drs_interp[i]),
            tyre_compound=compound,
            tyre_age=tyre_age,
            lap=int(lap_interp[i]),
            _synthetic=not is_in,
        ))

    return frames


def resample_session(
    session: fastf1.core.Session,
    frame_rate: Optional[float] = None,
) -> tuple[list[TimeFrame], list[DriverInfo], list[TrackPoint], float, float]:
    """
    Resample the entire session to a master timeline.

    Returns:
        (frames, driver_infos, track_outline, start_time, end_time)
    """
    hz = frame_rate or settings.default_frame_rate
    logger.info("Resampling session at %.1f Hz", hz)

    # Extract driver infos
    driver_infos = extract_driver_infos(session)
    driver_codes = [d.code for d in driver_infos]

    # Extract track outline
    track = extract_track_outline(session)

    # Extract per-driver telemetry
    driver_data: dict[str, pd.DataFrame] = {}
    for code in driver_codes:
        df = _extract_driver_telemetry(session, code)
        if df is not None and len(df) > 0:
            driver_data[code] = df
            logger.info("  %s: %d telemetry samples, t=[%.1f, %.1f]",
                        code, len(df), df["SessionTime_s"].min(), df["SessionTime_s"].max())

    if not driver_data:
        raise ValueError("No telemetry data found for any driver")

    # Compute master timeline envelope
    global_start = min(df["SessionTime_s"].min() for df in driver_data.values())
    global_end = max(df["SessionTime_s"].max() for df in driver_data.values())
    dt = 1.0 / hz
    master_times = np.arange(global_start, global_end, dt)
    logger.info("Master timeline: %.1f → %.1f (%.0f frames at %.1f Hz)",
                global_start, global_end, len(master_times), hz)

    # Resample each driver to master timeline
    driver_frames: dict[str, list[DriverFrame]] = {}
    for code, df in driver_data.items():
        frames = _interpolate_driver_to_timeline(df, master_times)
        if frames:
            driver_frames[code] = frames
            logger.info("  %s: %d resampled frames", code, len(frames))

    # Assemble synchronized TimeFrame list
    time_frames = []
    for i, t in enumerate(master_times):
        drivers_at_t = {}
        for code, frames in driver_frames.items():
            if i < len(frames):
                drivers_at_t[code] = frames[i]

        # Compute race order at each frame from lap progress and lap distance.
        # This gives a stable P1/P2/... signal for UI overlays.
        progress = []
        for code, frame in drivers_at_t.items():
            lap_num = max(int(frame.lap), 0)
            dist = float(frame.distance)
            progress.append((code, lap_num, dist))

        progress.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for pos, (code, _, _) in enumerate(progress, start=1):
            drivers_at_t[code].position = pos

        time_frames.append(TimeFrame(
            t=float(t),
            drivers=drivers_at_t,
        ))

    return time_frames, driver_infos, track, float(global_start), float(global_end)
