"""
F1 Visualization System — Lap Summary Service

Computes per-lap race summaries from FastF1 session.laps data.
This is the primary data source for the Lap Playback Engine.

Performance: Uses vectorized pandas operations on session.laps DataFrame
instead of per-row get_car_data() calls.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backend.config import settings
from src.backend.schemas import LapSummary, LapEvent

logger = logging.getLogger(__name__)

# Compound letter mapping
_COMPOUND_MAP = {
    "SOFT": "S", "MEDIUM": "M", "HARD": "H",
    "INTERMEDIATE": "I", "WET": "W", "UNKNOWN": "U",
    "TEST_UNKNOWN": "U",
}


def _safe_td_seconds(val, default: float = 0.0) -> float:
    """Convert a pandas Timedelta to seconds safely."""
    if pd.isna(val):
        return default
    try:
        return val.total_seconds()
    except (AttributeError, TypeError):
        return default


def _compound_letter(compound) -> str:
    """Map FastF1 compound name to single letter."""
    if not compound or pd.isna(compound):
        return "U"
    return _COMPOUND_MAP.get(str(compound).upper(), str(compound)[0].upper())


def compute_lap_summaries(session) -> list[LapSummary]:
    """
    Compute per-lap summaries from a loaded FastF1 session.

    Two-pass approach for speed:
      1. Core data from session.laps DataFrame (vectorized, fast)
      2. Telemetry aggregates from a single session.car_data load (batch)

    Args:
        session: A loaded fastf1.core.Session object.

    Returns:
        List of LapSummary objects, one per race lap.
    """
    laps_df = session.laps
    if laps_df is None or len(laps_df) == 0:
        logger.warning("No laps data available")
        return []

    total_laps = int(laps_df["LapNumber"].max())
    drivers = laps_df["Driver"].unique().tolist()

    logger.info("Computing lap summaries: %d laps, %d drivers", total_laps, len(drivers))

    # ===== PASS 1: Core data (fast — pure DataFrame ops) =====
    # Pre-compute cumulative times per driver using vectorized ops
    cum_times: dict[str, float] = {}
    stint_starts: dict[str, int] = {}
    prev_compounds: dict[str, str] = {}

    results: list[LapSummary] = []

    for lap_num in range(1, total_laps + 1):
        lap_rows = laps_df[laps_df["LapNumber"] == lap_num]
        if lap_rows.empty:
            continue

        positions: dict[str, int] = {}
        lap_times_dict: dict[str, float] = {}
        tyres: dict[str, str] = {}
        tyre_ages: dict[str, int] = {}
        pit_stops: list[LapEvent] = []
        events: list[LapEvent] = []
        sector_dict: dict[str, list[float]] = {}
        driver_cum_times: dict[str, float] = {}

        for _, row in lap_rows.iterrows():
            code = str(row["Driver"])
            lap_time_s = _safe_td_seconds(row.get("LapTime"))
            lap_times_dict[code] = round(lap_time_s, 3) if lap_time_s > 0 else 0.0

            # Cumulative time for position calculation
            cum_times[code] = cum_times.get(code, 0.0) + lap_time_s
            driver_cum_times[code] = cum_times[code]

            # Tyre compound
            compound = _compound_letter(row.get("Compound", "U"))
            tyres[code] = compound

            # Track tyre age via stint
            life = row.get("TyreLife")
            if life is not None and not pd.isna(life):
                tyre_ages[code] = int(life)
            else:
                if code not in prev_compounds or prev_compounds[code] != compound:
                    stint_starts[code] = lap_num
                    prev_compounds[code] = compound
                tyre_ages[code] = lap_num - stint_starts.get(code, lap_num) + 1

            # Pit stop detection from PitInTime / PitOutTime
            pit_in = row.get("PitInTime")
            pit_out = row.get("PitOutTime")
            has_pit_in = pit_in is not None and not pd.isna(pit_in)
            if has_pit_in:
                pit_dur = 0.0
                has_pit_out = pit_out is not None and not pd.isna(pit_out)
                if has_pit_out:
                    pit_dur = _safe_td_seconds(pit_out - pit_in)
                pit_stops.append(LapEvent(
                    type="pit_stop",
                    actor=code,
                    details={
                        "pit_duration": round(pit_dur, 1),
                        "compound": compound,
                        "lap": lap_num
                    },
                ))

            # Sector times
            s1 = _safe_td_seconds(row.get("Sector1Time"))
            s2 = _safe_td_seconds(row.get("Sector2Time"))
            s3 = _safe_td_seconds(row.get("Sector3Time"))
            if s1 > 0 or s2 > 0 or s3 > 0:
                sector_dict[code] = [round(s1, 3), round(s2, 3), round(s3, 3)]

        # ── Positions ──────────────────────────────────────────────────────
        # Prefer FastF1's official Position column (set by the FIA timing system).
        # Fall back to cumulative-lap-time ranking only when Position is absent.
        has_official_position = "Position" in lap_rows.columns

        for _, row in lap_rows.iterrows():
            code = str(row["Driver"])
            if has_official_position:
                pos_val = row.get("Position")
                if pos_val is not None and not pd.isna(pos_val):
                    positions[code] = int(pos_val)

        if not positions:
            # No official positions — rank by cumulative lap time (ascending)
            sorted_drivers = sorted(
                ((c, t) for c, t in driver_cum_times.items() if t > 0),
                key=lambda x: x[1],
            )
            for rank, (code, _) in enumerate(sorted_drivers, 1):
                positions[code] = rank
            sorted_drivers_list = sorted_drivers
        else:
            # Build sorted_drivers_list by official position for gap computation
            sorted_drivers_list = sorted(positions.items(), key=lambda x: x[1])

        leader = sorted_drivers_list[0][0] if sorted_drivers_list else ""

        # ── Gaps ───────────────────────────────────────────────────────────
        # Use FastF1's GapToLeader if available; otherwise delta of cum times.
        gaps: dict[str, float] = {}
        has_gap_col = "GapToLeader" in lap_rows.columns

        for _, row in lap_rows.iterrows():
            code = str(row["Driver"])
            if has_gap_col:
                g = row.get("GapToLeader")
                if g is not None and not pd.isna(g):
                    try:
                        gaps[code] = round(float(_safe_td_seconds(g) if hasattr(g, 'total_seconds') else g), 3)
                    except (TypeError, ValueError):
                        pass

        if not gaps:
            # Fallback: use difference in cumulative times from the leader
            leader_time = driver_cum_times.get(leader, 0.0)
            for code, ct in driver_cum_times.items():
                gap = ct - leader_time
                gaps[code] = round(max(0.0, gap), 3)

        # Overtake detection
        if results:
            prev = results[-1]
            for code, pos in positions.items():
                prev_pos = prev.positions.get(code)
                if prev_pos is not None and pos < prev_pos:
                    for other_code, other_pos in positions.items():
                        if other_code != code:
                            other_prev = prev.positions.get(other_code)
                            if other_prev is not None and other_prev == pos:
                                events.append(LapEvent(
                                    type="overtake", actor=code, victim=other_code,
                                ))
                                break

        results.append(LapSummary(
            lap=lap_num,
            leader=leader,
            positions=positions,
            lap_times=lap_times_dict,
            gaps=gaps,
            tyres=tyres,
            tyre_ages=tyre_ages,
            pit_stops=pit_stops,
            events=events,
            sector_times=sector_dict,
        ))

    # ===== PASS 2: Telemetry aggregates (batch) =====
    # Try to get all car data once and aggregate per driver per lap
    try:
        _enrich_with_telemetry(session, laps_df, results)
    except Exception as e:
        logger.warning("Could not enrich with telemetry: %s", e)

    logger.info("Computed %d lap summaries for %d drivers", len(results), len(drivers))
    return results


def _enrich_with_telemetry(session, laps_df, results: list[LapSummary]):
    """
    Batch-enrich lap summaries with telemetry aggregates.

    Uses FastF1's per-lap get_car_data() which handles time slicing
    internally. Limits to top-10 drivers for performance.
    """
    drivers = laps_df["Driver"].unique()
    lap_index = {r.lap: r for r in results}

    # Limit to first 10 drivers to keep processing time reasonable
    drivers_to_process = list(drivers)[:10]
    enriched_count = 0

    for driver in drivers_to_process:
        code = str(driver)
        driver_laps = laps_df[laps_df["Driver"] == driver].sort_values("LapNumber")

        for _, lap_row in driver_laps.iterrows():
            lap_num = int(lap_row["LapNumber"])
            summary = lap_index.get(lap_num)
            if not summary:
                continue

            try:
                # Use FastF1's built-in per-lap car data slicing
                single_lap_df = driver_laps[driver_laps["LapNumber"] == lap_num]
                car_data = single_lap_df.get_car_data()
                if car_data is None or len(car_data) == 0:
                    continue
            except Exception:
                continue

            # Speed
            if "Speed" in car_data.columns:
                speeds = car_data["Speed"].dropna().values
                if len(speeds) > 0:
                    summary.avg_speed[code] = round(float(np.nanmean(speeds)), 1)
                    summary.max_speed[code] = round(float(np.nanmax(speeds)), 1)
                    enriched_count += 1

            # Throttle
            if "Throttle" in car_data.columns:
                throttle = car_data["Throttle"].dropna().values
                if len(throttle) > 0:
                    summary.throttle_pct[code] = round(float(np.nanmean(throttle)), 1)

            # Brake
            if "Brake" in car_data.columns:
                brake = car_data["Brake"].dropna().values.astype(float)
                if len(brake) > 0:
                    summary.brake_pct[code] = round(float(np.nanmean(brake)) * 100, 1)

            # DRS
            if "DRS" in car_data.columns:
                drs_vals = car_data["DRS"].dropna().values
                if len(drs_vals) > 0:
                    drs_open = np.sum(drs_vals > 10)
                    summary.drs_pct[code] = round(float(drs_open / len(drs_vals)) * 100, 1)

    logger.info("Enriched %d telemetry data points for %d drivers", enriched_count, len(drivers_to_process))


def get_lap_summaries(session_id: str) -> list[LapSummary] | None:
    """Load cached lap summaries, or return None if not cached."""
    laps_path = settings.processed_dir / session_id / "laps.json"
    if not laps_path.exists():
        return None

    with open(laps_path) as f:
        data = json.load(f)

    return [LapSummary(**lap) for lap in data]


def save_lap_summaries(session_id: str, summaries: list[LapSummary]) -> None:
    """Cache lap summaries to disk as laps.json."""
    out_dir = settings.processed_dir / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    laps_path = out_dir / "laps.json"

    with open(laps_path, "w") as f:
        json.dump([s.model_dump() for s in summaries], f)

    logger.info("Saved %d lap summaries to %s", len(summaries), laps_path)
