"""
F1 Visualization System — Session Exporter

Exports processed session data to chunked JSON with gzip support.
Manages dataset manifest for versioning.
"""
import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import orjson

import fastf1

from src.backend.config import settings
from src.backend.schemas import (
    SessionData, SessionMetadata, SessionEvent, TelemetryChunk,
    TimeFrame, TrackPoint, DriverInfo,
)
from src.backend.services.fetcher import fetch_session, get_session_id
from src.backend.services.resampler import resample_session, compute_track_length
from src.backend.services.validator import validate_track_length

logger = logging.getLogger(__name__)


def _serialize_frames(frames: list[TimeFrame]) -> list[dict]:
    """Convert TimeFrame list to dictionaries for JSON serialization."""
    result = []
    for frame in frames:
        drivers_dict = {}
        for code, df in frame.drivers.items():
            drivers_dict[code] = {
                "x": round(df.x, 2),
                "y": round(df.y, 2),
                "distance": round(df.distance, 1),
                "speed": round(df.speed, 1),
                "throttle": round(df.throttle, 1),
                "brake": round(df.brake, 2),
                "gear": df.gear,
                "drs": df.drs,
                "tyreCompound": df.tyre_compound,
                "tyreAge": df.tyre_age,
                "lap": df.lap,
                "position": df.position,
                "_synthetic": df._synthetic if hasattr(df, '_synthetic') else False,
            }

        events_list = []
        for ev in frame.events:
            events_list.append({
                "t": ev.t,
                "type": ev.event_type,
                "driver": ev.driver,
                "details": ev.details,
                "highlightScore": ev.highlight_score,
                "confidence": ev.confidence,
                "source": ev.source,
            })

        result.append({
            "t": round(frame.t, 3),
            "drivers": drivers_dict,
            "events": events_list,
        })
    return result


def process_and_export_session(
    year: int,
    gp: str,
    session_type: str,
    frame_rate: Optional[float] = None,
) -> SessionData:
    """
    Full pipeline: fetch → resample → validate → export.

    Returns SessionData and caches to disk.
    """
    session_id = get_session_id(year, gp, session_type)
    output_dir = settings.processed_dir / session_id
    metadata_path = output_dir / "metadata.json"

    # Check if already processed
    if metadata_path.exists():
        logger.info("Session %s already processed, loading from cache", session_id)
        try:
            return _load_cached_session(output_dir)
        except Exception as e:
            # Cache may be stale/incomplete (e.g., metadata-only file). Fall through
            # to a full re-export path.
            logger.warning("Cached session %s is invalid, rebuilding (%s)", session_id, e)
            try:
                metadata_path.unlink()
            except Exception:
                pass

    # Fetch from FastF1
    session = fetch_session(year, gp, session_type)

    # Resample to master timeline
    hz = frame_rate or settings.default_frame_rate
    frames, driver_infos, track, start_time, end_time = resample_session(session, hz)

    # Validate track
    track_name = str(session.event.get("EventName", gp)) if hasattr(session, 'event') else gp
    validation = validate_track_length(track, track_name)
    computed_length = validation["computed_m"]

    # Get official track length
    official_length = 0.0
    try:
        circuit = session.get_circuit_info() if hasattr(session, 'get_circuit_info') else None
    except Exception:
        circuit = None

    if validation["official_m"]:
        official_length = validation["official_m"]

    # Determine total laps
    total_laps = 0
    if session.laps is not None and len(session.laps) > 0:
        total_laps = int(session.laps["LapNumber"].max())

    # Build metadata
    metadata = SessionMetadata(
        session_id=session_id,
        year=year,
        gp=gp,
        session_type=session_type,
        track_name=track_name,
        track_length_m=official_length,
        computed_track_length_m=computed_length,
        drivers=driver_infos,
        total_laps=total_laps,
        start_time=start_time,
        end_time=end_time,
        frame_rate=hz,
        chunk_duration_s=settings.chunk_duration_s,
        total_frames=len(frames),
    )

    session_data = SessionData(
        metadata=metadata,
        track=track,
        frames=frames,
        events=[],
    )

    # Save to disk
    _save_session(session_data, output_dir)

    # Update manifest
    _update_manifest(session_id, metadata)

    logger.info("Session %s exported: %d frames, track validation: %s",
                session_id, len(frames),
                "PASS" if validation["passed"] else f"WARN ({validation['warning']})")

    return session_data


def _save_session(session_data: SessionData, output_dir: Path) -> None:
    """Save session data to disk as chunked JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_dict = session_data.metadata.model_dump()
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta_dict, f, indent=2, default=str)

    # Save track
    track_data = [{"x": p.x, "y": p.y} for p in session_data.track]
    with open(output_dir / "track.json", "w") as f:
        json.dump(track_data, f)

    # Save frames in chunks
    chunk_size = int(session_data.metadata.frame_rate * session_data.metadata.chunk_duration_s)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    frames = session_data.frames
    chunk_index = 0
    for i in range(0, len(frames), chunk_size):
        chunk_frames = frames[i:i + chunk_size]
        serialized = _serialize_frames(chunk_frames)

        chunk_data = {
            "chunk_index": chunk_index,
            "start": chunk_frames[0].t if chunk_frames else 0,
            "duration": session_data.metadata.chunk_duration_s,
            "frame_count": len(chunk_frames),
            "frames": serialized,
        }

        # Save gzipped
        chunk_bytes = orjson.dumps(chunk_data)
        with gzip.open(chunks_dir / f"chunk_{chunk_index:04d}.json.gz", "wb") as f:
            f.write(chunk_bytes)

        chunk_index += 1

    # Save events
    events_data = [e.model_dump() for e in session_data.events]
    with open(output_dir / "events.json", "w") as f:
        json.dump(events_data, f, indent=2)

    logger.info("Saved %d chunks to %s", chunk_index, chunks_dir)


def _load_cached_session(output_dir: Path) -> SessionData:
    """Load a previously cached session from disk."""
    with open(output_dir / "metadata.json") as f:
        meta_dict = json.load(f)
    metadata = SessionMetadata(**meta_dict)

    with open(output_dir / "track.json") as f:
        track_list = json.load(f)
    track = [TrackPoint(**p) for p in track_list]

    # Load all chunks
    chunks_dir = output_dir / "chunks"
    frames = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.json.gz")):
        with gzip.open(chunk_file, "rb") as f:
            chunk_data = orjson.loads(f.read())

        for frame_dict in chunk_data["frames"]:
            drivers = {}
            for code, d in frame_dict["drivers"].items():
                from src.backend.schemas import DriverFrame
                drivers[code] = DriverFrame(
                    x=d["x"], y=d["y"],
                    distance=d.get("distance", 0),
                    speed=d.get("speed", 0),
                    throttle=d.get("throttle", 0),
                    brake=d.get("brake", 0),
                    gear=d.get("gear", 0),
                    drs=d.get("drs", False),
                    tyre_compound=d.get("tyreCompound", "U"),
                    tyre_age=d.get("tyreAge", 0),
                    lap=d.get("lap", 0),
                    position=d.get("position", 0),
                    _synthetic=d.get("_synthetic", False),
                )
            frames.append(TimeFrame(
                t=frame_dict["t"],
                drivers=drivers,
                events=[],
            ))

    # Load events
    events = []
    events_path = output_dir / "events.json"
    if events_path.exists():
        with open(events_path) as f:
            events_data = json.load(f)
        events = [SessionEvent(**e) for e in events_data]

    return SessionData(
        metadata=metadata,
        track=track,
        frames=frames,
        events=events,
    )


def get_telemetry_chunk(
    session_id: str,
    start: float,
    duration: float,
) -> Optional[TelemetryChunk]:
    """Load a specific time chunk from cached session data."""
    output_dir = settings.processed_dir / session_id
    if not output_dir.exists():
        return None

    with open(output_dir / "metadata.json") as f:
        meta_dict = json.load(f)

    hz = meta_dict["frame_rate"]
    chunk_dur = meta_dict["chunk_duration_s"]

    # Find which chunk files overlap [start, start+duration]
    chunks_dir = output_dir / "chunks"
    result_frames = []

    for chunk_file in sorted(chunks_dir.glob("chunk_*.json.gz")):
        with gzip.open(chunk_file, "rb") as f:
            chunk_data = orjson.loads(f.read())

        for frame_dict in chunk_data["frames"]:
            t = frame_dict["t"]
            if start <= t < start + duration:
                from src.backend.schemas import DriverFrame
                drivers = {}
                for code, d in frame_dict["drivers"].items():
                    drivers[code] = DriverFrame(
                        x=d["x"], y=d["y"],
                        distance=d.get("distance", 0),
                        speed=d.get("speed", 0),
                        throttle=d.get("throttle", 0),
                        brake=d.get("brake", 0),
                        gear=d.get("gear", 0),
                        drs=d.get("drs", False),
                        tyre_compound=d.get("tyreCompound", "U"),
                        tyre_age=d.get("tyreAge", 0),
                        lap=d.get("lap", 0),
                        position=d.get("position", 0),
                        _synthetic=d.get("_synthetic", False),
                    )
                result_frames.append(TimeFrame(
                    t=t,
                    drivers=drivers,
                ))

    if not result_frames:
        return None

    return TelemetryChunk(
        session_id=session_id,
        start=start,
        duration=duration,
        frame_rate=hz,
        frames=result_frames,
    )


def load_session_data(session_id: str) -> SessionData | None:
    """Load a cached session by ID. Returns None if not found."""
    output_dir = settings.processed_dir / session_id
    if not output_dir.exists() or not (output_dir / "metadata.json").exists():
        return None
    return _load_cached_session(output_dir)


def _update_manifest(session_id: str, metadata: SessionMetadata) -> None:
    """Update the dataset manifest with a new processed session."""
    manifest_path = settings.data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"version": "1.0.0", "datasets": {}, "models": {}}

    manifest["datasets"][session_id] = {
        "year": metadata.year,
        "gp": metadata.gp,
        "session_type": metadata.session_type,
        "total_frames": metadata.total_frames,
        "frame_rate": metadata.frame_rate,
        "drivers": len(metadata.drivers),
        "processed_at": datetime.utcnow().isoformat(),
    }
    manifest["last_updated"] = datetime.utcnow().isoformat()

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
