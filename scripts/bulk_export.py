"""
F1 Bulk Exporter — scripts/bulk_export.py

Generates ALL required artifacts for every completed race from 2020 to latest.
Runs offline after export; backend never needs FastF1 for historical races.

Usage:
    python scripts/bulk_export.py [--years 2020-2025] [--workers 4] [--force]
"""
import argparse
import gzip
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import orjson

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fastf1
from src.backend.config import settings
from src.backend.services.fetcher import get_session_id
from src.backend.services.laps import compute_lap_summaries, save_lap_summaries
from src.backend.services.exporter import process_and_export_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bulk_export")

REQUIRED_FILES = [
    "metadata.json", "events.json", "laps.json", "retirements.json",
    "track.json", "track_geometry.json", "track_replay.json", "insights.json",
]

PROCESSED = settings.processed_dir

# ── helpers ───────────────────────────────────────────────────────────────────

def _is_complete(session_id: str) -> bool:
    d = PROCESSED / session_id
    return d.is_dir() and all((d / f).exists() for f in REQUIRED_FILES)


def _discover_races(year: int) -> list[dict]:
    """Return list of {gp, event_name} for completed Race rounds in a season."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        logger.error("Cannot fetch schedule for %d: %s", year, e)
        return []

    now = datetime.now(tz=timezone.utc)
    races = []
    for _, row in schedule.iterrows():
        rnd = int(row.get("RoundNumber", 0))
        if rnd <= 0:
            continue
        event_date = row.get("EventDate")
        try:
            if hasattr(event_date, "tzinfo") and event_date.tzinfo is None:
                import pandas as pd
                event_date = pd.Timestamp(event_date).tz_localize("UTC")
            if event_date > now:
                continue  # future race
        except Exception:
            pass
        name = str(row.get("EventName", "")).strip()
        if not name:
            continue
        races.append({"year": year, "gp": name, "round": rnd})
    return races


# ── artifact generators (no FastF1 needed) ────────────────────────────────────

def _build_geometry(track_data: list[dict]) -> dict:
    x = np.array([p["x"] for p in track_data], dtype=float)
    y = np.array([p["y"] for p in track_data], dtype=float)
    dx = np.gradient(x); dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2); norm[norm == 0] = 1.0
    dx /= norm; dy /= norm
    nx = -dy; ny = dx
    hw = 100.0
    xo = x + nx * hw; yo = y + ny * hw
    xi = x - nx * hw; yi = y - ny * hw
    all_x = np.concatenate([x, xi, xo]); all_y = np.concatenate([y, yi, yo])
    step = max(1, len(x) // 500)

    def ds(arr):
        s = arr[::step].tolist()
        if len(arr): s.append(float(arr[-1]))
        return [round(v, 1) for v in s]

    return {
        "centerline_x": ds(x), "centerline_y": ds(y),
        "inner_x": ds(xi), "inner_y": ds(yi),
        "outer_x": ds(xo), "outer_y": ds(yo),
        "drs_zones": [],
        "bounds": {
            "x_min": round(float(all_x.min()), 1), "x_max": round(float(all_x.max()), 1),
            "y_min": round(float(all_y.min()), 1), "y_max": round(float(all_y.max()), 1),
        },
        "point_count": len(x), "downsample_step": step,
    }


def _build_track_replay(session_dir: Path, meta: dict) -> dict:
    """Build compact track_replay from chunks. Target < 5 MB via downsampling."""
    chunks_dir = session_dir / "chunks"
    all_frames = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.json.gz")):
        with gzip.open(chunk_file, "rb") as f:
            chunk = orjson.loads(f.read())
        all_frames.extend(chunk["frames"])

    if not all_frames:
        return {}

    # Downsample to reduce file size — keep every Nth frame targeting ~5 MB
    # At ~800 bytes/frame uncompressed, 5MB ≈ 6250 frames
    total = len(all_frames)
    step = max(1, math.ceil(total / 6000))
    sampled = all_frames[::step]

    total_laps = meta.get("total_laps", 0)
    start_t = float(sampled[0]["t"])

    # Lap boundaries from sampled frames
    lap_boundaries = [start_t]
    for lap_num in range(2, total_laps + 1):
        t_start = next(
            (f["t"] for f in sampled if any(d.get("lap", 0) >= lap_num for d in f["drivers"].values())),
            sampled[-1]["t"],
        )
        lap_boundaries.append(t_start)
    lap_boundaries.append(sampled[-1]["t"])
    lap_boundaries_rel = [max(0.0, round(float(t - start_t), 3)) for t in lap_boundaries]

    frames_out = []
    for fd in sampled:
        positions = {}
        for code, d in fd["drivers"].items():
            positions[code] = {
                "d": round(float(d.get("distance", 0)), 1),
                "s": round(float(d.get("speed", 0)), 0),
                "pos": int(d.get("position", 0)),
            }
        frames_out.append({"t": round(float(fd["t"] - start_t), 2), "positions": positions})

    return {
        "session_id": session_dir.name,
        "track_points": [],  # populated by caller from track.json
        "track_length": round(float(meta.get("computed_track_length_m") or meta.get("track_length_m") or 0), 1),
        "drivers": [{"id": d["code"], "color": d.get("team_color", "#888")} for d in meta.get("drivers", [])],
        "lap_boundaries": lap_boundaries_rel,
        "duration": round(float(sampled[-1]["t"] - start_t), 2),
        "frame_rate": meta.get("frame_rate", 2.0),
        "frames": frames_out,
    }


def _build_insights(laps: list[dict], session_id: str) -> dict:
    insights = []
    if not laps:
        return {"session_id": session_id, "insights": insights}

    last = laps[-1]
    leader = last.get("leader", "")
    if leader:
        insights.append({"category": "race", "text": f"{leader} won after {len(laps)} laps", "priority": 1})

    overtakes = sum(len([e for e in s.get("events", []) if e.get("type") == "overtake"]) for s in laps)
    if overtakes:
        insights.append({"category": "action", "text": f"{overtakes} overtakes detected", "priority": 2})

    pits = sum(len(s.get("pit_stops", [])) for s in laps)
    if pits:
        insights.append({"category": "strategy", "text": f"{pits} pit stops performed", "priority": 2})

    best_t, best_drv = None, None
    for s in laps:
        for drv, t in (s.get("lap_times") or {}).items():
            if t and t > 0 and (best_t is None or t < best_t):
                best_t, best_drv = t, drv
    if best_drv:
        m = int(best_t // 60); s_val = best_t % 60
        insights.append({"category": "pace", "text": f"Fastest lap: {best_drv} — {m}:{s_val:06.3f}", "priority": 2})

    return {"session_id": session_id, "insights": insights}


def _build_retirements(laps: list[dict], session_id: str) -> list[dict]:
    """Derive retirements from cached laps.json — no FastF1 needed."""
    if not laps:
        return []
    total_laps = len(laps)
    driver_last_lap: dict[str, int] = {}
    for s in laps:
        lap_no = s.get("lap", 0)
        for code in (s.get("positions") or {}).keys():
            c = str(code).upper()
            driver_last_lap[c] = max(driver_last_lap.get(c, 0), lap_no)

    retirements = []
    for code, last_lap in driver_last_lap.items():
        if last_lap > 0 and last_lap < total_laps:
            retirements.append({
                "driver": code, "name": code, "team": "Unknown",
                "retired_lap": last_lap, "reason": "Retired (inferred)",
            })
    retirements.sort(key=lambda r: r["retired_lap"])
    return retirements


def generate_secondary_artifacts(session_id: str) -> list[str]:
    """Generate track_geometry, track_replay, insights, retirements from existing exports."""
    session_dir = PROCESSED / session_id
    generated = []

    meta_path = session_dir / "metadata.json"
    track_path = session_dir / "track.json"
    laps_path = session_dir / "laps.json"

    if not meta_path.exists():
        return generated

    with open(meta_path) as f:
        meta = json.load(f)

    # track_geometry.json
    geo_path = session_dir / "track_geometry.json"
    if not geo_path.exists() and track_path.exists():
        try:
            with open(track_path) as f:
                track_data = json.load(f)
            if isinstance(track_data, list) and len(track_data) >= 10:
                geo = _build_geometry(track_data)
                with open(geo_path, "w") as f:
                    json.dump(geo, f)
                generated.append("track_geometry.json")
        except Exception as e:
            logger.warning("%s: track_geometry failed: %s", session_id, e)

    # track_replay.json
    replay_path = session_dir / "track_replay.json"
    chunks_dir = session_dir / "chunks"
    if not replay_path.exists() and track_path.exists() and chunks_dir.exists():
        try:
            replay = _build_track_replay(session_dir, meta)
            if replay:
                with open(track_path) as f:
                    raw_track = json.load(f)
                if isinstance(raw_track, list):
                    replay["track_points"] = [{"x": p["x"], "y": p["y"]} for p in raw_track]
                with open(replay_path, "w") as f:
                    json.dump(replay, f, separators=(",", ":"))
                generated.append(f"track_replay.json ({replay_path.stat().st_size // 1024} KB)")
        except Exception as e:
            logger.warning("%s: track_replay failed: %s", session_id, e)

    # laps.json (if missing)
    if not laps_path.exists():
        logger.warning("%s: laps.json missing — cannot generate insights/retirements without FastF1", session_id)
        return generated

    with open(laps_path) as f:
        laps = json.load(f)

    # insights.json
    insights_path = session_dir / "insights.json"
    if not insights_path.exists():
        try:
            result = _build_insights(laps, session_id)
            with open(insights_path, "w") as f:
                json.dump(result, f)
            generated.append("insights.json")
        except Exception as e:
            logger.warning("%s: insights failed: %s", session_id, e)

    # retirements.json
    ret_path = session_dir / "retirements.json"
    if not ret_path.exists():
        try:
            ret = _build_retirements(laps, session_id)
            with open(ret_path, "w") as f:
                json.dump(ret, f)
            generated.append("retirements.json")
        except Exception as e:
            logger.warning("%s: retirements failed: %s", session_id, e)

    return generated


def export_one(task: dict) -> dict:
    """Worker function — export a single race session and all artifacts."""
    year = task["year"]
    gp = task["gp"]
    session_id = get_session_id(year, gp, "R")
    t0 = time.time()

    result = {"session_id": session_id, "status": "unknown", "elapsed": 0.0, "error": None}

    try:
        if _is_complete(session_id):
            result["status"] = "skipped"
            result["elapsed"] = time.time() - t0
            return result

        # Run primary export (fetches FastF1, builds chunks/metadata/events/track)
        logger.info("Exporting: %s", session_id)
        session_data = process_and_export_session(year, gp, "R")

        # Compute & save laps.json
        laps_path = PROCESSED / session_id / "laps.json"
        if not laps_path.exists():
            summaries = compute_lap_summaries(
                # process_and_export_session returned a SessionData, not fastf1 session
                # We need to use the ff1 session — re-fetch lightly
                __import__("src.backend.services.fetcher", fromlist=["fetch_session"]).fetch_session(year, gp, "R")
            )
            if summaries:
                save_lap_summaries(session_id, summaries)

        # Generate secondary artifacts from what's now on disk
        generate_secondary_artifacts(session_id)

        if _is_complete(session_id):
            result["status"] = "exported"
        else:
            missing = [f for f in REQUIRED_FILES if not (PROCESSED / session_id / f).exists()]
            result["status"] = "partial"
            result["error"] = f"missing: {missing}"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error("FAILED %s: %s", session_id, e)

    result["elapsed"] = round(time.time() - t0, 1)
    return result


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1 Bulk Exporter")
    parser.add_argument("--years", default="2020-2025", help="Year range e.g. 2020-2025 or 2024")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    parser.add_argument("--force", action="store_true", help="Re-export even if complete")
    parser.add_argument("--secondary-only", action="store_true",
                        help="Only generate secondary artifacts for existing exports (no FastF1 calls)")
    args = parser.parse_args()

    # Parse year range
    if "-" in args.years:
        start_y, end_y = args.years.split("-")
        years = list(range(int(start_y), int(end_y) + 1))
    else:
        years = [int(args.years)]

    t_total = time.time()

    if args.secondary_only:
        # Quick mode: generate missing artifacts for sessions already exported
        sessions = sorted([p.name for p in PROCESSED.iterdir() if p.is_dir() and not p.name.startswith("_")])
        print(f"\nSecondary-only mode: {len(sessions)} existing sessions\n")
        for sid in sessions:
            gen = generate_secondary_artifacts(sid)
            if gen:
                print(f"  {sid}: generated {gen}")
            else:
                print(f"  {sid}: nothing to generate")
        return

    # Discover all races across all years
    all_tasks = []
    for year in years:
        races = _discover_races(year)
        logger.info("Year %d: %d completed races discovered", year, len(races))
        all_tasks.extend(races)

    total = len(all_tasks)
    print(f"\nBulk export: {total} races across {len(years)} seasons | workers={args.workers}\n")

    results = {"exported": [], "skipped": [], "partial": [], "failed": []}
    done = 0

    # Use sequential execution to avoid FastF1 multiprocessing issues with file locks
    for task in all_tasks:
        done += 1
        gp_label = f"{task['year']} R{task['round']:02d} {task['gp']}"
        print(f"  [{done:3d}/{total}] {gp_label}", end=" ... ", flush=True)
        r = export_one(task)
        status = r["status"]
        results[status if status in results else "failed"].append(r)
        elapsed = r["elapsed"]
        err = f" ({r['error']})" if r.get("error") else ""
        print(f"{status.upper()} ({elapsed}s){err}")

    # ── Final report ──────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - t_total, 1)
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE — {total_elapsed}s total")
    print(f"  Exported : {len(results['exported'])}")
    print(f"  Skipped  : {len(results['skipped'])}")
    print(f"  Partial  : {len(results['partial'])}")
    print(f"  Failed   : {len(results['failed'])}")

    # Dataset size
    all_processed = [p for p in PROCESSED.iterdir() if p.is_dir() and not p.name.startswith("_")]
    total_bytes = sum(f.stat().st_size for p in all_processed for f in p.rglob("*") if f.is_file())
    replay_sizes = [
        (p / "track_replay.json").stat().st_size
        for p in all_processed
        if (p / "track_replay.json").exists()
    ]
    print(f"\nDataset size    : {total_bytes / 1024**3:.2f} GB ({total_bytes / 1024**2:.0f} MB)")
    print(f"Sessions on disk: {len(all_processed)}")
    if replay_sizes:
        avg_kb = sum(replay_sizes) / len(replay_sizes) / 1024
        max_kb = max(replay_sizes) / 1024
        print(f"Avg track_replay: {avg_kb:.0f} KB")
        print(f"Max track_replay: {max_kb:.0f} KB")

    if results["failed"]:
        print("\nFailed sessions:")
        for r in results["failed"]:
            print(f"  {r['session_id']}: {r['error']}")

    if results["partial"]:
        print("\nPartial sessions (missing some artifacts):")
        for r in results["partial"]:
            print(f"  {r['session_id']}: {r['error']}")

    # Validation
    print("\nValidation:")
    incomplete = []
    for p in sorted(all_processed):
        missing = [f for f in REQUIRED_FILES if not (p / f).exists()]
        if missing:
            incomplete.append((p.name, missing))
    if incomplete:
        print(f"  INCOMPLETE ({len(incomplete)} sessions):")
        for name, miss in incomplete:
            print(f"    {name}: missing {miss}")
    else:
        print(f"  ALL {len(all_processed)} sessions complete")

    print("="*60)


if __name__ == "__main__":
    main()
