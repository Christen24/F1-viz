"""
FastF1 Endpoint Smoke Test

Validates that FastF1 can load a known session. Run as part of CI.
Usage: python scripts/check_fastf1.py
"""
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check():
    try:
        import fastf1
        logger.info("FastF1 version: %s", fastf1.__version__)

        fastf1.Cache.enable_cache("data/raw/fastf1_cache")

        # Try to load a minimal session (without telemetry for speed)
        session = fastf1.get_session(2023, "Monza", "R")
        session.load(telemetry=False, laps=False, weather=False, messages=False)

        logger.info("✓ FastF1 session loaded: %s %s", session.event["EventName"], session.name)
        logger.info("✓ Drivers: %s", list(session.drivers)[:5])
        return True

    except Exception as e:
        logger.error("✗ FastF1 check failed: %s", e)
        return False


if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
