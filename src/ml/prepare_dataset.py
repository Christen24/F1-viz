"""
F1 Visualization System — ML Dataset Preparation

Reads processed session data and generates feature matrices for ML training.
Labels overtakes using heuristic rules (position order changes).

Usage:
    python src/ml/prepare_dataset.py --session 2024_Monza_R [--seed 42]
"""
import argparse
import csv
import gzip
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_session_frames(session_dir: Path) -> list[dict]:
    """Load all frames from chunked gzipped JSON files."""
    chunks_dir = session_dir / "chunks"
    if not chunks_dir.exists():
        raise FileNotFoundError(f"No chunks directory at {chunks_dir}")

    all_frames = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.json.gz")):
        import orjson
        with gzip.open(chunk_file, "rb") as f:
            chunk_data = orjson.loads(f.read())
        all_frames.extend(chunk_data["frames"])

    logger.info("Loaded %d frames from %s", len(all_frames), session_dir.name)
    return all_frames


def compute_features(frames: list[dict]) -> list[dict]:
    """
    Compute per-window feature vectors for overtake detection.

    Features per (t, driver_a, driver_b) pair:
    - relative_distance: dist_a - dist_b
    - speed_diff: speed_a - speed_b
    - gap_change_2s: change in relative distance over 2 seconds
    - throttle_var_a: throttle variance of driver_a over 2s window
    - brake_var_a: brake variance of driver_a over 2s window
    - drs_a: whether driver_a has DRS
    """
    feature_rows = []
    frame_rate = 4  # assumed

    for i in range(8, len(frames)):  # need at least 2s lookback
        frame = frames[i]
        frame_2s_ago = frames[max(0, i - 8)]  # 8 frames back at 4Hz = 2s

        drivers_now = frame.get("drivers", frame.get("d", {}))
        drivers_2s = frame_2s_ago.get("drivers", frame_2s_ago.get("d", {}))
        t = frame["t"]

        driver_codes = list(drivers_now.keys())

        for a_idx, code_a in enumerate(driver_codes):
            for code_b in driver_codes[a_idx + 1:]:
                da = drivers_now[code_a]
                db = drivers_now[code_b]

                # Get distance values (handle both compact and expanded keys)
                dist_a = da.get("distance", da.get("d", 0))
                dist_b = db.get("distance", db.get("d", 0))
                speed_a = da.get("speed", da.get("s", 0))
                speed_b = db.get("speed", db.get("s", 0))
                throttle_a = da.get("throttle", da.get("th", 0))
                brake_a = da.get("brake", da.get("br", 0))
                drs_a = da.get("drs", False)

                # Relative distance
                rel_dist = dist_a - dist_b

                # 2s ago
                da_prev = drivers_2s.get(code_a, {})
                db_prev = drivers_2s.get(code_b, {})
                prev_rel_dist = (
                    da_prev.get("distance", da_prev.get("d", 0)) -
                    db_prev.get("distance", db_prev.get("d", 0))
                )

                gap_change = rel_dist - prev_rel_dist

                # Only include pairs that are close enough to be relevant
                if abs(rel_dist) > 200:
                    continue

                feature_rows.append({
                    "t": t,
                    "driver_a": code_a,
                    "driver_b": code_b,
                    "relative_distance": round(rel_dist, 2),
                    "speed_diff": round(speed_a - speed_b, 1),
                    "gap_change_2s": round(gap_change, 2),
                    "throttle_a": round(throttle_a, 1),
                    "brake_a": round(brake_a, 2),
                    "speed_a": round(speed_a, 1),
                    "drs_a": int(drs_a),
                })

    logger.info("Computed %d feature rows", len(feature_rows))
    return feature_rows


def label_overtakes_heuristic(frames: list[dict]) -> list[dict]:
    """
    Generate heuristic overtake labels from position order changes.

    An overtake is labeled when:
    1. Driver A's distance was < Driver B's at time t
    2. Driver A's distance is > Driver B's at time t + delta
    3. Both drivers were within 100m
    """
    labels = []

    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        prev_drivers = prev_frame.get("drivers", prev_frame.get("d", {}))
        curr_drivers = curr_frame.get("drivers", curr_frame.get("d", {}))

        for code_a in curr_drivers:
            for code_b in curr_drivers:
                if code_a >= code_b:
                    continue

                if code_a not in prev_drivers or code_b not in prev_drivers:
                    continue

                pa = prev_drivers[code_a]
                pb = prev_drivers[code_b]
                ca = curr_drivers[code_a]
                cb = curr_drivers[code_b]

                dist_prev_a = pa.get("distance", pa.get("d", 0))
                dist_prev_b = pb.get("distance", pb.get("d", 0))
                dist_curr_a = ca.get("distance", ca.get("d", 0))
                dist_curr_b = cb.get("distance", cb.get("d", 0))

                was_behind = dist_prev_a < dist_prev_b
                now_ahead = dist_curr_a > dist_curr_b
                proximity = abs(dist_curr_a - dist_curr_b)

                if was_behind and now_ahead and proximity < 100:
                    labels.append({
                        "t": curr_frame["t"],
                        "overtaker": code_a,
                        "overtaken": code_b,
                        "proximity_m": round(proximity, 1),
                        "label": 1,
                    })

    # Deduplicate (same pair within 10s)
    deduped = []
    for label in labels:
        is_dup = any(
            l["overtaker"] == label["overtaker"]
            and l["overtaken"] == label["overtaken"]
            and abs(l["t"] - label["t"]) < 10
            for l in deduped
        )
        if not is_dup:
            deduped.append(label)

    logger.info("Labeled %d heuristic overtakes", len(deduped))
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Prepare ML dataset from processed session")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed" / args.session
    labeled_dir = data_dir / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    if not processed_dir.exists():
        logger.error("Session not found: %s", processed_dir)
        sys.exit(1)

    # Load frames
    frames = load_session_frames(processed_dir)

    # Generate features
    features = compute_features(frames)

    # Save features
    features_path = labeled_dir / f"{args.session}_features.csv"
    if features:
        with open(features_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features[0].keys())
            writer.writeheader()
            writer.writerows(features)
        logger.info("Saved features to %s", features_path)

    # Generate heuristic labels
    labels = label_overtakes_heuristic(frames)

    labels_path = labeled_dir / f"{args.session}_heuristic_overtakes.csv"
    if labels:
        with open(labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=labels[0].keys())
            writer.writeheader()
            writer.writerows(labels)
        logger.info("Saved labels to %s", labels_path)

    # Save metadata
    meta_path = labeled_dir / f"{args.session}_features_meta.json"
    meta = {
        "session": args.session,
        "feature_count": len(features),
        "label_count": len(labels),
        "seed": args.seed,
        "created_at": datetime.utcnow().isoformat(),
        "columns": list(features[0].keys()) if features else [],
        "synthetic_columns": ["drs_a"],  # Mark provenance
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Update manifest
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"version": "1.0.0", "datasets": {}, "models": {}}

    manifest["datasets"][f"{args.session}_features"] = {
        "type": "features",
        "rows": len(features),
        "labels": len(labels),
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("✓ Dataset prepared for %s", args.session)


if __name__ == "__main__":
    main()
