"""
Race Intelligence Platform — Insight Generation

Template-based natural language insights from telemetry and event data.
Covers: strategy, tire degradation, overtakes, pace analysis.
"""
import logging
from src.backend.schemas import SessionData, SessionEvent, InsightItem

logger = logging.getLogger(__name__)


def generate_insights(session_data: SessionData) -> list[InsightItem]:
    """Generate natural language insights from session data."""
    insights: list[InsightItem] = []

    insights.extend(_strategy_insights(session_data))
    insights.extend(_pace_insights(session_data))
    insights.extend(_overtake_insights(session_data))
    insights.extend(_tire_insights(session_data))

    # Sort by confidence descending
    insights.sort(key=lambda i: i.confidence, reverse=True)

    logger.info("Generated %d insights for session %s",
                len(insights), session_data.metadata.session_id)
    return insights


def _strategy_insights(session_data: SessionData) -> list[InsightItem]:
    """Detect undercut/overcut strategies from pit stop timing."""
    insights = []
    pit_events = [e for e in session_data.events if e.event_type == "pit_stop"]

    if len(pit_events) < 2:
        return insights

    # Group pits by proximity (within 3 laps = potential undercut/overcut)
    for i, pit_a in enumerate(pit_events):
        for pit_b in pit_events[i + 1:]:
            time_gap = abs(pit_a.t - pit_b.t)
            if time_gap < 90:  # Within ~1 lap
                earlier = pit_a if pit_a.t < pit_b.t else pit_b
                later = pit_b if pit_a.t < pit_b.t else pit_a
                insights.append(InsightItem(
                    category="strategy",
                    text=f"{earlier.driver} pitted {time_gap:.0f}s before {later.driver} — "
                         f"potential undercut attempt.",
                    confidence=0.75,
                    related_drivers=[earlier.driver, later.driver],
                ))

    return insights[:5]  # Cap


def _pace_insights(session_data: SessionData) -> list[InsightItem]:
    """Analyze pace trends across the race."""
    insights = []
    frames = session_data.frames
    if len(frames) < 10:
        return insights

    # Find fastest and slowest average speeds per driver
    driver_speeds: dict[str, list[float]] = {}
    for frame in frames:
        for code, df in frame.drivers.items():
            if df.speed > 50:  # Filter pit lane
                driver_speeds.setdefault(code, []).append(df.speed)

    if driver_speeds:
        avg_speeds = {code: sum(s) / len(s) for code, s in driver_speeds.items() if s}
        if avg_speeds:
            fastest = max(avg_speeds, key=avg_speeds.get)  # type: ignore
            slowest = min(avg_speeds, key=avg_speeds.get)  # type: ignore
            insights.append(InsightItem(
                category="pace",
                text=f"{fastest} had the highest average race speed at "
                     f"{avg_speeds[fastest]:.1f} km/h. {slowest} was slowest "
                     f"at {avg_speeds[slowest]:.1f} km/h.",
                confidence=0.9,
                related_drivers=[fastest, slowest],
            ))

    return insights


def _overtake_insights(session_data: SessionData) -> list[InsightItem]:
    """Summarize overtaking patterns."""
    insights = []
    overtakes = [e for e in session_data.events if e.event_type == "overtake"]

    if not overtakes:
        return insights

    # Most active overtaker
    overtaker_counts: dict[str, int] = {}
    for ot in overtakes:
        overtaker_counts[ot.driver] = overtaker_counts.get(ot.driver, 0) + 1

    top_overtaker = max(overtaker_counts, key=overtaker_counts.get)  # type: ignore
    count = overtaker_counts[top_overtaker]

    insights.append(InsightItem(
        category="overtake",
        text=f"{top_overtaker} was the most aggressive driver with {count} "
             f"overtaking moves during the race.",
        confidence=0.85,
        related_drivers=[top_overtaker],
    ))

    insights.append(InsightItem(
        category="overtake",
        text=f"Total of {len(overtakes)} overtakes detected in this session.",
        confidence=0.95,
    ))

    return insights


def _tire_insights(session_data: SessionData) -> list[InsightItem]:
    """Analyze tire compound usage and stint lengths."""
    insights = []
    frames = session_data.frames
    if len(frames) < 10:
        return insights

    # Track compound changes per driver
    driver_stints: dict[str, list[str]] = {}
    for frame in frames:
        for code, df in frame.drivers.items():
            comp = df.tyre_compound
            if comp and comp != "U":
                stints = driver_stints.setdefault(code, [])
                if not stints or stints[-1] != comp:
                    stints.append(comp)

    # Find drivers with most stops
    if driver_stints:
        most_stops_driver = max(driver_stints, key=lambda c: len(driver_stints[c]))
        stop_count = len(driver_stints[most_stops_driver]) - 1
        if stop_count > 0:
            strategy = " → ".join(driver_stints[most_stops_driver])
            insights.append(InsightItem(
                category="tire",
                text=f"{most_stops_driver} used a {stop_count + 1}-stop strategy: {strategy}.",
                confidence=0.85,
                related_drivers=[most_stops_driver],
            ))

    return insights
