"""
F1 Visualization System — Session Fetcher

Wraps FastF1 to fetch and cache session data with rate limiting.
"""
import logging
import time
from pathlib import Path

import fastf1

from src.backend.config import settings

logger = logging.getLogger(__name__)


def enable_cache() -> None:
    """Configure FastF1 cache directory."""
    cache_dir = str(settings.fastf1_cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    logger.info("FastF1 cache enabled at %s", cache_dir)


def fetch_session(year: int, gp: str, session_type: str) -> fastf1.core.Session:
    """
    Fetch a session from FastF1 with caching and rate-limit delay.

    Args:
        year: Season year (e.g., 2024)
        gp: Grand Prix name or round number (e.g., 'Monza', 14)
        session_type: Session identifier ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S', 'SQ')

    Returns:
        Loaded FastF1 Session object with telemetry and laps.

    Raises:
        ValueError: If session cannot be found or loaded.
    """
    enable_cache()

    logger.info("Fetching session: %d %s %s", year, gp, session_type)
    time.sleep(settings.fastf1_rate_limit_delay)  # polite delay

    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(
            telemetry=True,
            laps=True,
            weather=False,
            messages=False,
        )
    except Exception as e:
        raise ValueError(
            f"Failed to load session {year} {gp} {session_type}: {e}"
        ) from e

    driver_count = len(session.drivers)
    lap_count = len(session.laps) if session.laps is not None else 0
    logger.info(
        "Session loaded: %d drivers, %d total laps",
        driver_count, lap_count,
    )

    return session


def get_session_id(year: int, gp: str, session_type: str) -> str:
    """Generate a canonical session ID string."""
    gp_clean = gp.replace(" ", "_").replace(".", "")
    return f"{year}_{gp_clean}_{session_type}"
