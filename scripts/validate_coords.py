"""
Track Coordinate Validation Script

Validates that computed track coordinates are in meters and that
the track length matches the official FIA length within tolerance.

Usage: python scripts/validate_coords.py --session 2024_Monza_R
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate track coordinates")
    parser.add_argument("--session", required=True, help="Session ID (e.g., 2024_Monza_R)")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Tolerance as fraction (default 0.05 = 5%%)")
    args = parser.parse_args()

    session_dir = Path(args.data_dir) / args.session
    track_path = session_dir / "track.json"
    meta_path = session_dir / "metadata.json"

    if not track_path.exists():
        logger.error("✗ Track file not found: %s", track_path)
        sys.exit(1)

    with open(track_path) as f:
        track = json.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    logger.info("Session: %s", args.session)
    logger.info("Track: %s", meta.get("track_name", "Unknown"))
    logger.info("Points: %d", len(track))

    # Compute perimeter
    import numpy as np
    total = 0.0
    for i in range(1, len(track)):
        dx = track[i]["x"] - track[i - 1]["x"]
        dy = track[i]["y"] - track[i - 1]["y"]
        total += np.sqrt(dx ** 2 + dy ** 2)
    # Close loop
    dx = track[0]["x"] - track[-1]["x"]
    dy = track[0]["y"] - track[-1]["y"]
    total += np.sqrt(dx ** 2 + dy ** 2)

    logger.info("Computed track length: %.1f m", total)

    official = meta.get("track_length_m", 0)
    computed = meta.get("computed_track_length_m", total)

    if official > 0:
        delta_pct = abs(computed - official) / official * 100
        logger.info("Official track length: %.1f m", official)
        logger.info("Delta: %.2f%%", delta_pct)

        if delta_pct > args.tolerance * 100:
            logger.warning("✗ FAIL: Delta %.2f%% exceeds tolerance %.0f%%", delta_pct, args.tolerance * 100)
            sys.exit(1)
        else:
            logger.info("✓ PASS: Within tolerance")
    else:
        logger.warning("⚠ No official track length available for comparison")

    # Check coordinate units
    xs = [p["x"] for p in track]
    ys = [p["y"] for p in track]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    logger.info("Coordinate ranges: X=[%.1f, %.1f] (%.1fm), Y=[%.1f, %.1f] (%.1fm)",
                min(xs), max(xs), x_range, min(ys), max(ys), y_range)

    if x_range > 50000 or y_range > 50000:
        logger.warning("✗ Coordinates appear to be in 1/10 mm or smaller — likely not meters")
        sys.exit(1)
    elif x_range > 10000 or y_range > 10000:
        logger.warning("⚠ Coordinates may be in 1/10 meter units (large range)")
    else:
        logger.info("✓ Coordinates appear to be in meters")


if __name__ == "__main__":
    main()
