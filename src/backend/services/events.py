"""
F1 Visualization System — Rule-based Event Detection

Detects overtakes, pit stops, fastest laps, and computes highlight scores.
"""
import logging
from typing import Optional

from src.backend.config import settings
from src.backend.schemas import SessionData, SessionEvent, TimeFrame
from src.backend.services.ml_inference import predict_overtake_probability

logger = logging.getLogger(__name__)


def detect_events(session_data: SessionData) -> list[SessionEvent]:
    """
    Run all rule-based event detectors on a processed session.

    Returns list of SessionEvent sorted by timestamp.
    """
    events = []
    events.extend(_detect_overtakes(session_data.frames))
    events.extend(_detect_pit_stops(session_data.frames))
    events.extend(_detect_fastest_laps(session_data.frames, session_data.metadata.frame_rate))
    events.sort(key=lambda e: e.t)

    # Compute highlight scores
    _score_highlights(events)

    logger.info("Detected %d events (%d overtakes, %d pits, %d fastest laps)",
                len(events),
                sum(1 for e in events if e.event_type == "overtake"),
                sum(1 for e in events if e.event_type == "pit_stop"),
                sum(1 for e in events if e.event_type == "fastest_lap"))

    return events


def _detect_overtakes(frames: list[TimeFrame]) -> list[SessionEvent]:
    """
    Detect overtakes by monitoring position order changes.

    An overtake is flagged when:
    1. Driver A is behind Driver B at time t₁
    2. Driver A is ahead of Driver B at time t₂ (t₂ - t₁ < 5s)
    3. Both drivers were within 50m distance at some point in that window
    """
    events = []

    if len(frames) < 2:
        return events

    # Build per-frame position rankings based on distance traveled
    prev_rankings: dict[str, float] = {}
    dt = max(0.05, float(frames[1].t - frames[0].t))
    estimated_hz = max(1.0, 1.0 / dt)
    lookback_n = max(1, int(round(2.0 * estimated_hz)))
    blend = min(1.0, max(0.0, float(settings.overtake_ml_blend_weight)))

    for i, frame in enumerate(frames):
        if not frame.drivers:
            continue

        # Sort drivers by distance (descending = leading)
        rankings = {}
        for code, df in frame.drivers.items():
            rankings[code] = df.distance

        if prev_rankings:
            # Check for position swaps
            for code_a, dist_a in rankings.items():
                for code_b, dist_b in rankings.items():
                    if code_a >= code_b:
                        continue

                    prev_a = prev_rankings.get(code_a)
                    prev_b = prev_rankings.get(code_b)

                    if prev_a is None or prev_b is None:
                        continue

                    # Was A behind B? Is A now ahead of B?
                    was_behind = prev_a < prev_b
                    now_ahead = dist_a > dist_b

                    if was_behind and now_ahead:
                        # Check proximity (within 100m at swap point)
                        proximity = abs(dist_a - dist_b)
                        if proximity < 100:
                            frame_2s_ago = frames[max(0, i - lookback_n)]
                            drivers_2s = frame_2s_ago.drivers
                            a_2s = drivers_2s.get(code_a)
                            b_2s = drivers_2s.get(code_b)
                            prev_rel_2s = 0.0
                            if a_2s is not None and b_2s is not None:
                                prev_rel_2s = float(a_2s.distance - b_2s.distance)
                            rel_now = float(dist_a - dist_b)
                            gap_change_2s = rel_now - prev_rel_2s
                            speed_a = float(frame.drivers[code_a].speed)
                            speed_b = float(frame.drivers[code_b].speed)
                            ml_features = {
                                "relative_distance": rel_now,
                                "speed_diff": speed_a - speed_b,
                                "gap_change_2s": float(gap_change_2s),
                                "throttle_a": float(frame.drivers[code_a].throttle),
                                "brake_a": float(frame.drivers[code_a].brake),
                                "speed_a": speed_a,
                                "drs_a": 1.0 if bool(frame.drivers[code_a].drs) else 0.0,
                            }
                            base_conf = min(1.0, 100 / max(proximity, 1))
                            ml_res = predict_overtake_probability(ml_features)
                            ml_prob = ml_res.get("probability")
                            confidence = base_conf
                            source = ml_res.get("source", "rule")
                            if ml_prob is not None:
                                confidence = (1.0 - blend) * base_conf + blend * ml_prob
                                if ml_prob < float(settings.overtake_ml_min_probability):
                                    # Keep deterministic detection, but down-weight low-confidence swaps.
                                    confidence = min(confidence, ml_prob)

                            details = {
                                "victim": code_b,
                                "proximity_m": round(proximity, 1),
                                "speed_overtaker": round(speed_a, 1),
                                "speed_victim": round(speed_b, 1),
                                "gap_change_2s": round(float(gap_change_2s), 2),
                            }
                            if ml_prob is not None:
                                details["ml_probability"] = round(float(ml_prob), 4)

                            events.append(SessionEvent(
                                t=frame.t,
                                event_type="overtake",
                                driver=code_a,
                                details=details,
                                confidence=float(max(0.0, min(1.0, confidence))),
                                source=source,
                            ))

        prev_rankings = rankings

    # Deduplicate overtakes (same pair within 10s)
    deduped = []
    for event in events:
        victim = event.details.get("victim", "")
        is_dup = any(
            e.driver == event.driver
            and e.details.get("victim") == victim
            and abs(e.t - event.t) < 10.0
            for e in deduped
        )
        if not is_dup:
            deduped.append(event)

    return deduped


def _detect_pit_stops(frames: list[TimeFrame]) -> list[SessionEvent]:
    """
    Detect pit stops by monitoring speed drops and position changes.

    A pit stop is flagged when a driver's speed drops below 100 km/h
    for more than 2 seconds during the race.
    """
    events = []

    if len(frames) < 2:
        return events

    # Track low-speed windows per driver
    low_speed_start: dict[str, float] = {}

    for frame in frames:
        for code, df in frame.drivers.items():
            if df.speed < 100:
                if code not in low_speed_start:
                    low_speed_start[code] = frame.t
            else:
                if code in low_speed_start:
                    duration = frame.t - low_speed_start[code]
                    if 15 < duration < 60:  # Reasonable pit stop duration
                        events.append(SessionEvent(
                            t=low_speed_start[code],
                            event_type="pit_stop",
                            driver=code,
                            details={
                                "duration_s": round(duration, 1),
                                "exit_time": frame.t,
                            },
                            confidence=0.9,
                            source="rule",
                        ))
                    del low_speed_start[code]

    return events


def _detect_fastest_laps(
    frames: list[TimeFrame],
    frame_rate: float,
) -> list[SessionEvent]:
    """
    Detect fastest laps by tracking lap-start times per driver.
    """
    events = []

    # Track per-driver lap boundaries
    driver_lap_start: dict[str, tuple[int, float]] = {}  # code -> (lap_num, start_time)
    lap_times: dict[str, list[tuple[int, float]]] = {}  # code -> [(lap_num, lap_time)]

    for frame in frames:
        for code, df in frame.drivers.items():
            current_lap = df.lap
            if current_lap <= 0:
                continue

            if code not in driver_lap_start:
                driver_lap_start[code] = (current_lap, frame.t)
                lap_times[code] = []
                continue

            prev_lap, prev_t = driver_lap_start[code]
            if current_lap > prev_lap:
                lap_time = frame.t - prev_t
                if 60 < lap_time < 200:  # Sanity check: 1-3.5 minutes
                    lap_times.setdefault(code, []).append((prev_lap, lap_time))
                driver_lap_start[code] = (current_lap, frame.t)

    # Find overall fastest lap
    all_laps = []
    for code, laps in lap_times.items():
        for lap_num, lt in laps:
            all_laps.append((code, lap_num, lt))

    if all_laps:
        all_laps.sort(key=lambda x: x[2])
        fastest = all_laps[0]
        events.append(SessionEvent(
            t=0,  # We'll approximate
            event_type="fastest_lap",
            driver=fastest[0],
            details={
                "lap_number": fastest[1],
                "lap_time_s": round(fastest[2], 3),
            },
            highlight_score=0.8,
            confidence=0.95,
            source="rule",
        ))

    return events


def _score_highlights(events: list[SessionEvent]) -> None:
    """
    Assign highlight scores to events based on heuristics.
    Modifies events in-place.
    """
    for event in events:
        if event.event_type == "overtake":
            # Higher score for closer overtakes and higher speed
            proximity = event.details.get("proximity_m", 50)
            speed = event.details.get("speed_overtaker", 200)
            event.highlight_score = min(1.0,
                (100 - proximity) / 100 * 0.5 + speed / 350 * 0.5
            )
        elif event.event_type == "pit_stop":
            event.highlight_score = 0.3
        elif event.event_type == "fastest_lap":
            event.highlight_score = 0.8
