"""
Offline artifact generator.

For each already-processed session in data/processed/, generate any missing:
  - track_geometry.json  (from track.json points)
  - track_replay.json    (compact per-frame replay payload)
  - insights.json        (derived from laps.json)

Does NOT call FastF1. Reads ONLY from already-present JSON/chunk files.
"""
import gzip
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import orjson

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_artifacts")

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backend.config import settings

PROCESSED = settings.processed_dir


# ── track_geometry.json ──────────────────────────────────────────────────────
def _build_geometry_from_track_json(track_data: list[dict], track_width: float = 200.0) -> dict:
    x = np.array([p["x"] for p in track_data], dtype=float)
    y = np.array([p["y"] for p in track_data], dtype=float)

    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2)
    norm[norm == 0] = 1.0
    dx /= norm
    dy /= norm
    nx = -dy
    ny = dx

    half_w = track_width / 2
    x_outer = x + nx * half_w
    y_outer = y + ny * half_w
    x_inner = x - nx * half_w
    y_inner = y - ny * half_w

    all_x = np.concatenate([x, x_inner, x_outer])
    all_y = np.concatenate([y, y_inner, y_outer])

    step = max(1, len(x) // 500)

    def ds(arr):
        sampled = arr[::step].tolist()
        if len(arr) > 0:
            sampled.append(float(arr[-1]))
        return [round(v, 1) for v in sampled]

    return {
        "centerline_x": ds(x),
        "centerline_y": ds(y),
        "inner_x": ds(x_inner),
        "inner_y": ds(y_inner),
        "outer_x": ds(x_outer),
        "outer_y": ds(y_outer),
        "drs_zones": [],
        "bounds": {
            "x_min": round(float(all_x.min()), 1),
            "x_max": round(float(all_x.max()), 1),
            "y_min": round(float(all_y.min()), 1),
            "y_max": round(float(all_y.max()), 1),
        },
        "point_count": len(x),
        "downsample_step": step,
    }


def generate_track_geometry(session_dir: Path) -> bool:
    out_path = session_dir / "track_geometry.json"
    if out_path.exists():
        return False

    track_path = session_dir / "track.json"
    if not track_path.exists():
        logger.warning("%s: track.json missing, skipping track_geometry", session_dir.name)
        return False

    with open(track_path) as f:
        track_data = json.load(f)

    if not isinstance(track_data, list) or len(track_data) < 10:
        logger.warning("%s: track.json too small or wrong format", session_dir.name)
        return False

    geometry = _build_geometry_from_track_json(track_data)
    with open(out_path, "w") as f:
        json.dump(geometry, f)
    logger.info("%s: wrote track_geometry.json (%d bytes)", session_dir.name, out_path.stat().st_size)
    return True


# ── track_replay.json ─────────────────────────────────────────────────────────
def _compute_lap_boundaries(all_frames: list, total_laps: int) -> list[float]:
    if not all_frames:
        return []
    boundaries = [all_frames[0]["t"]]
    for lap_num in range(2, total_laps + 1):
        t_start = next(
            (
                f["t"]
                for f in all_frames
                if any(d.get("lap", 0) >= lap_num for d in f["drivers"].values())
            ),
            all_frames[-1]["t"],
        )
        boundaries.append(t_start)
    boundaries.append(all_frames[-1]["t"])
    return boundaries


def generate_track_replay(session_dir: Path) -> bool:
    out_path = session_dir / "track_replay.json"
    if out_path.exists():
        return False

    meta_path = session_dir / "metadata.json"
    track_path = session_dir / "track.json"
    chunks_dir = session_dir / "chunks"

    if not meta_path.exists() or not track_path.exists() or not chunks_dir.exists():
        logger.warning("%s: prerequisite files missing, skipping track_replay", session_dir.name)
        return False

    with open(meta_path) as f:
        meta = json.load(f)
    with open(track_path) as f:
        raw_track = json.load(f)

    if not isinstance(raw_track, list):
        logger.warning("%s: track.json not a list, skipping track_replay", session_dir.name)
        return False

    # Load all frames from chunks (compact representation only)
    all_frames_raw = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.json.gz")):
        with gzip.open(chunk_file, "rb") as f:
            chunk = orjson.loads(f.read())
        for fd in chunk["frames"]:
            all_frames_raw.append(fd)

    if not all_frames_raw:
        logger.warning("%s: no chunk frames found, skipping track_replay", session_dir.name)
        return False

    total_laps = meta.get("total_laps", 0)
    start_t = float(all_frames_raw[0]["t"])

    lap_boundaries = _compute_lap_boundaries(all_frames_raw, total_laps)
    lap_boundaries_rel = [max(0.0, round(float(t - start_t), 3)) for t in lap_boundaries]

    duration = round(float(all_frames_raw[-1]["t"] - start_t), 3)

    # Build compact frames  — only d, s, pos per driver
    frames_out = []
    for fd in all_frames_raw:
        positions = {}
        for code, d in fd["drivers"].items():
            positions[code] = {
                "d": round(float(d.get("distance", 0)), 2),
                "s": round(float(d.get("speed", 0)), 1),
                "pos": int(d.get("position", 0)),
            }
        frames_out.append({
            "t": round(float(fd["t"] - start_t), 3),
            "positions": positions,
        })

    track_points = [{"x": p["x"], "y": p["y"]} for p in raw_track]
    track_length = meta.get("computed_track_length_m") or meta.get("track_length_m") or 0.0

    drivers = [
        {"id": d["code"], "color": d.get("team_color", "#888888")}
        for d in meta.get("drivers", [])
    ]

    result = {
        "session_id": session_dir.name,
        "track_points": track_points,
        "track_length": round(float(track_length), 3),
        "drivers": drivers,
        "lap_boundaries": lap_boundaries_rel,
        "duration": duration,
        "frame_rate": meta.get("frame_rate", 2.0),
        "frames": frames_out,
    }

    with open(out_path, "w") as f:
        json.dump(result, f)

    logger.info("%s: wrote track_replay.json (%d bytes, %d frames)",
                session_dir.name, out_path.stat().st_size, len(frames_out))
    return True


# ── insights.json ─────────────────────────────────────────────────────────────
def generate_insights(session_dir: Path) -> bool:
    out_path = session_dir / "insights.json"
    if out_path.exists():
        return False

    laps_path = session_dir / "laps.json"
    if not laps_path.exists():
        logger.warning("%s: laps.json missing, skipping insights", session_dir.name)
        return False

    with open(laps_path) as f:
        summaries = json.load(f)

    if not summaries:
        result = {"session_id": session_dir.name, "insights": []}
        with open(out_path, "w") as f:
            json.dump(result, f)
        return True

    insights = []
    last = summaries[-1]
    leader = last.get("leader", "")
    total_laps = len(summaries)

    if leader:
        insights.append({"category": "race", "text": f"{leader} won after {total_laps} laps", "priority": 1})

    overtakes = sum(
        len([e for e in s.get("events", []) if e.get("type") == "overtake"])
        for s in summaries
    )
    if overtakes:
        insights.append({"category": "action", "text": f"{overtakes} overtakes detected", "priority": 2})

    pits = sum(len(s.get("pit_stops", [])) for s in summaries)
    if pits:
        insights.append({"category": "strategy", "text": f"{pits} pit stops performed", "priority": 2})

    # Fastest lap
    best_lap_time = None
    best_driver = None
    for s in summaries:
        for drv, t in (s.get("lap_times") or {}).items():
            if t and t > 0:
                if best_lap_time is None or t < best_lap_time:
                    best_lap_time = t
                    best_driver = drv
    if best_driver and best_lap_time:
        mins = int(best_lap_time // 60)
        secs = best_lap_time % 60
        insights.append({
            "category": "pace",
            "text": f"Fastest lap: {best_driver} — {mins}:{secs:06.3f}",
            "priority": 2,
        })

    result = {"session_id": session_dir.name, "insights": insights}
    with open(out_path, "w") as f:
        json.dump(result, f)
    logger.info("%s: wrote insights.json (%d bytes, %d insights)",
                session_dir.name, out_path.stat().st_size, len(insights))
    return True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    sessions = sorted([
        p for p in PROCESSED.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    ])
    logger.info("Found %d sessions", len(sessions))

    for session_dir in sessions:
        logger.info("── Processing: %s", session_dir.name)
        generate_track_geometry(session_dir)
        generate_track_replay(session_dir)
        generate_insights(session_dir)

    # Final size report
    print("\n── Artifact sizes ──")
    for session_dir in sessions:
        print(f"\n{session_dir.name}:")
        for fname in ["track_geometry.json", "track_replay.json", "insights.json"]:
            p = session_dir / fname
            if p.exists():
                size_kb = p.stat().st_size / 1024
                print(f"  {fname}: {size_kb:.1f} KB")
            else:
                print(f"  {fname}: MISSING")


if __name__ == "__main__":
    main()
