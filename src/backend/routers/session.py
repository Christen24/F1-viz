"""
F1 Visualization System — Session Router

API endpoints for session metadata, chunked telemetry, and events.
"""
import gzip
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.backend.config import settings
from src.backend.services.exporter import (
    process_and_export_session,
    get_telemetry_chunk,
    load_session_data,
)
from src.backend.services.events import detect_events
from src.backend.services.fetcher import get_session_id, fetch_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["session"])

# Calendar cache: year → event list
_CALENDAR_CACHE: dict[int, list[dict]] = {}


class PrecomputeItem(BaseModel):
    year: int = Field(ge=2018, le=2030)
    gp: str = Field(min_length=2)
    session: str = Field(default="R", pattern="^(R|Q|FP1|FP2|FP3|S|SQ|SS)$")


class PrecomputeRequest(BaseModel):
    items: list[PrecomputeItem] = Field(min_length=1, max_length=100)
    force_rebuild: bool = False
    warm_chat_cache: bool = True


def _parse_session_id(session_id: str) -> tuple[int, str, str]:
    """Parse "{year}_{gp}_{session}" ids back into query parts."""
    first_underscore = session_id.find("_")
    last_underscore = session_id.rfind("_")
    if first_underscore == -1 or last_underscore == -1 or first_underscore == last_underscore:
        raise HTTPException(status_code=400, detail=f"Invalid session_id format: {session_id}")

    year_str = session_id[:first_underscore]
    session_type = session_id[last_underscore + 1:]
    gp = session_id[first_underscore + 1:last_underscore].replace("_", " ")

    try:
        year = int(year_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid year in session_id: {year_str}") from exc

    return year, gp, session_type


def _compute_lap_boundaries(frames, total_laps: int) -> list[float]:
    """Estimate leader lap boundaries from synchronized telemetry frames."""
    if not frames:
        return []

    boundaries = [float(frames[0].t)]
    for lap_num in range(2, total_laps + 1):
        lap_start = next(
            (
                float(frame.t)
                for frame in frames
                if any(driver.lap >= lap_num for driver in frame.drivers.values())
            ),
            float(frames[-1].t),
        )
        boundaries.append(lap_start)

    boundaries.append(float(frames[-1].t))
    return boundaries


def _has_complete_export(session_id: str) -> bool:
    """True when full replay artifacts exist (not just lightweight metadata)."""
    output_dir = settings.processed_dir / session_id
    metadata_path = output_dir / "metadata.json"
    track_path = output_dir / "track.json"
    chunks_dir = output_dir / "chunks"
    if not metadata_path.exists() or not track_path.exists() or not chunks_dir.exists():
        return False
    if not any(chunks_dir.glob("chunk_*.json.gz")):
        return False

    # Validate track cache shape. Exporter track.json must be a list of points.
    try:
        import json
        with open(track_path) as f:
            track_payload = json.load(f)
        if not isinstance(track_payload, list):
            return False
    except Exception:
        return False

    return True


def _load_or_build_session_data(session_id: str):
    """
    Ensure we have full exported session data for replay.

    `/api/session` may create metadata-only caches; those are insufficient for
    track replay and can break `load_session_data`.
    """
    bad_cached_export = False

    if _has_complete_export(session_id):
        try:
            loaded = load_session_data(session_id)
            if loaded is not None:
                return loaded
        except Exception as e:
            logger.warning("Corrupt cached export for %s, rebuilding (%s)", session_id, e)
            bad_cached_export = True

    year, gp, session_type = _parse_session_id(session_id)
    output_dir = settings.processed_dir / session_id
    metadata_path = output_dir / "metadata.json"

    # Remove stale metadata if cache is incomplete or failed to load; this forces
    # process_and_export_session() down the full re-export path.
    if metadata_path.exists() and (bad_cached_export or not _has_complete_export(session_id)):
        try:
            metadata_path.unlink()
        except Exception:
            pass

    return process_and_export_session(year, gp, session_type)


@router.get("/calendar/{year}")
async def get_calendar(year: int):
    """
    Return the race calendar for a season, fetched live from FastF1.
    
    Includes sprint weekend flags and available sessions per event.
    Cached in-memory after first fetch.
    """
    if year in _CALENDAR_CACHE:
        return {"year": year, "events": _CALENDAR_CACHE[year]}

    try:
        import fastf1
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        raise HTTPException(502, f"Could not fetch calendar for {year}: {e}")

    events = []
    for _, row in schedule.iterrows():
        rnd = int(row.get("RoundNumber", 0))
        if rnd <= 0:
            continue

        fmt = str(row.get("EventFormat", "conventional")).lower()
        is_sprint = "sprint" in fmt
        name = row.get("EventName", "Unknown")
        country = row.get("Country", "")

        sessions = [
            {"value": "R", "label": "Race"},
            {"value": "Q", "label": "Qualifying"},
        ]
        if is_sprint:
            sessions.extend([
                {"value": "S", "label": "Sprint"},
                {"value": "SQ", "label": "Sprint Qualifying"},
            ])

        events.append({
            "round": rnd,
            "event_name": name,
            "country": country,
            "is_sprint": is_sprint,
            "sessions": sessions,
        })

    _CALENDAR_CACHE[year] = events
    return {"year": year, "events": events}


@router.get("/session")
async def get_session(
    year: int = Query(..., ge=2018, le=2030),
    gp: str = Query(..., min_length=2),
    session: str = Query("R", pattern="^(R|Q|FP1|FP2|FP3|S|SQ|SS)$"),
):
    """
    Fetch or load a session. Returns metadata only (lightweight).

    Skips the heavy telemetry frame pipeline — only fetches metadata,
    driver info, and laps from FastF1. Computes lap summaries inline.
    """
    from src.backend.services.laps import get_lap_summaries, compute_lap_summaries, save_lap_summaries
    from src.backend.services.video import resolve_video_source

    session_id = get_session_id(year, gp, session)
    output_dir = settings.processed_dir / session_id
    metadata_path = output_dir / "metadata.json"

    # Fast path: cached metadata exists
    if metadata_path.exists():
        logger.info("Session %s cached, loading metadata only", session_id)
        import json
        with open(metadata_path) as f:
            meta = json.load(f)

        # Load track if available
        track_path = output_dir / "track.json"
        if track_path.exists():
            with open(track_path) as f:
                cached_track = json.load(f)
                if isinstance(cached_track, list):
                    meta["track"] = cached_track
                elif isinstance(cached_track, dict):
                    xs = cached_track.get("centerline_x")
                    ys = cached_track.get("centerline_y")
                    if isinstance(xs, list) and isinstance(ys, list):
                        n = min(len(xs), len(ys))
                        meta["track"] = [{"x": xs[i], "y": ys[i]} for i in range(n)]
                    else:
                        meta["track"] = []

        # Add video source
        video = resolve_video_source(year, gp, session)
        meta["video_source"] = video.model_dump()

        # Ensure lap summaries are cached (compute if needed)
        if not get_lap_summaries(session_id):
            try:
                ff1_session = fetch_session(year, gp, session)
                summaries = compute_lap_summaries(ff1_session)
                if summaries:
                    save_lap_summaries(session_id, summaries)
            except Exception as e:
                logger.warning("Could not compute lap summaries from cache: %s", e)

        return ORJSONResponse(content=meta)

    # Cold path: fetch from FastF1 (lightweight — metadata + laps only)
    try:
        ff1_session = fetch_session(year, gp, session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error fetching session")
        raise HTTPException(status_code=500, detail=f"Fetch error: {e}")

    # Extract metadata without heavy processing
    output_dir.mkdir(parents=True, exist_ok=True)

    # Driver info
    driver_infos = []
    try:
        for drv_num in ff1_session.drivers:
            drv = ff1_session.get_driver(drv_num)
            driver_infos.append({
                "code": str(drv.get("Abbreviation", f"D{drv_num}")),
                "name": str(drv.get("FullName", f"Driver {drv_num}")),
                "team": str(drv.get("TeamName", "Unknown")),
                "team_color": f"#{drv.get('TeamColor', '888888')}",
                "number": int(drv_num),
            })
    except Exception as e:
        logger.warning("Error extracting driver info: %s", e)

    # Basic metadata
    total_laps = 0
    if ff1_session.laps is not None and len(ff1_session.laps) > 0:
        total_laps = int(ff1_session.laps["LapNumber"].max())

    track_name = gp
    try:
        track_name = str(ff1_session.event.get("EventName", gp)) if hasattr(ff1_session, 'event') else gp
    except Exception:
        pass

    meta = {
        "session_id": session_id,
        "year": year,
        "gp": gp,
        "session_type": session,
        "track_name": track_name,
        "track_length_m": 0,
        "computed_track_length_m": 0,
        "drivers": driver_infos,
        "total_laps": total_laps,
        "start_time": 0,
        "end_time": 0,
        "frame_rate": settings.default_frame_rate,
        "chunk_duration_s": settings.chunk_duration_s,
        "total_frames": 0,
        "track": [],
        "event_count": 0,
    }

    # Save metadata for future fast-path
    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Compute and cache lap summaries inline (most important data)
    try:
        summaries = compute_lap_summaries(ff1_session)
        if summaries:
            save_lap_summaries(session_id, summaries)
    except Exception as e:
        logger.warning("Error computing lap summaries: %s", e)

    # Add video source
    video = resolve_video_source(year, gp, session)
    meta["video_source"] = video.model_dump()

    return ORJSONResponse(content=meta)


@router.get("/session/{session_id}/telemetry")
async def get_telemetry(
    session_id: str,
    start: float = Query(0.0, description="Start time in seconds"),
    duration: float = Query(30.0, ge=1.0, le=120.0, description="Chunk duration in seconds"),
):
    """
    Get a chunk of telemetry frames (gzipped).
    All drivers are synchronized to the same time values.
    """
    chunk = get_telemetry_chunk(session_id, start, duration)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"No data for {session_id} at t={start}")

    # Serialize to compact JSON
    import orjson
    frames_data = []
    for frame in chunk.frames:
        drivers_dict = {}
        for code, df in frame.drivers.items():
            drivers_dict[code] = {
                "x": round(df.x, 2),
                "y": round(df.y, 2),
                "d": round(df.distance, 1),
                "s": round(df.speed, 1),
                "th": round(df.throttle, 1),
                "br": round(df.brake, 2),
                "g": df.gear,
                "drs": df.drs,
                "tc": df.tyre_compound,
                "ta": df.tyre_age,
                "l": df.lap,
                "syn": df._synthetic if hasattr(df, '_synthetic') else False,
            }
        frames_data.append({
            "t": round(frame.t, 3),
            "d": drivers_dict,
        })

    response_data = {
        "session_id": chunk.session_id,
        "start": chunk.start,
        "duration": chunk.duration,
        "frame_rate": chunk.frame_rate,
        "frame_count": len(frames_data),
        "frames": frames_data,
    }

    # Gzip the response
    json_bytes = orjson.dumps(response_data)
    if settings.gzip_responses:
        compressed = gzip.compress(json_bytes)
        return Response(
            content=compressed,
            media_type="application/json",
            headers={"Content-Encoding": "gzip"},
        )

    return ORJSONResponse(content=response_data)


@router.get("/session/{session_id}/events")
async def get_events(session_id: str):
    """Get all detected events for a session.
    
    Returns events from events.json if available,
    otherwise derives events from lap summary data.
    """
    import json
    events_path = settings.processed_dir / session_id / "events.json"
    
    if events_path.exists():
        with open(events_path) as f:
            events = json.load(f)
        return ORJSONResponse(content={"session_id": session_id, "events": events})

    # Derive events from lap summary data
    from src.backend.services.laps import get_lap_summaries
    summaries = get_lap_summaries(session_id)
    if not summaries:
        raise HTTPException(status_code=404, detail=f"No events found for session {session_id}")

    events = []
    for s in summaries:
        for ev in s.events:
            events.append({
                "type": ev.type,
                "lap": s.lap,
                "actor": ev.actor,
                "victim": ev.victim or "",
                "description": ev.details.get("description", "") if ev.details else "",
                "highlight_score": 0.5,
            })
        for ps in s.pit_stops:
            driver = ps.get("driver", "") if isinstance(ps, dict) else getattr(ps, "driver", "")
            events.append({
                "type": "pit_stop",
                "lap": s.lap,
                "actor": driver,
                "description": f"{driver} pit stop",
                "highlight_score": 0.6,
            })

    return ORJSONResponse(content={"session_id": session_id, "events": events})


@router.get("/sessions")
async def list_sessions():
    """List all processed sessions from the manifest."""
    import json
    manifest_path = settings.data_dir / "manifest.json"
    if not manifest_path.exists():
        return {"sessions": []}

    with open(manifest_path) as f:
        manifest = json.load(f)

    return {"sessions": manifest.get("datasets", {})}


@router.post("/precompute")
async def precompute_sessions(req: PrecomputeRequest):
    """
    Precompute and persist full replay artifacts ahead of runtime requests.

    This allows "store beforehand" workflows:
    - fetch FastF1 once
    - export chunks/track/metadata
    - optionally warm local chat RAG cache
    """
    from src.backend.services.local_rag import warm_local_rag_cache

    results = []
    for item in req.items:
        session_id = get_session_id(item.year, item.gp, item.session)
        status = "processed"
        error = None

        try:
            if _has_complete_export(session_id) and not req.force_rebuild:
                status = "cached"
            else:
                process_and_export_session(item.year, item.gp, item.session)
                status = "processed"

            warmed = False
            if req.warm_chat_cache:
                try:
                    warmed = warm_local_rag_cache(session_id)
                except Exception:
                    warmed = False

            results.append(
                {
                    "session_id": session_id,
                    "status": status,
                    "chat_cache_warmed": warmed,
                }
            )
        except Exception as exc:
            logger.exception("Precompute failed for %s", session_id)
            error = str(exc)
            results.append(
                {
                    "session_id": session_id,
                    "status": "failed",
                    "error": error,
                    "chat_cache_warmed": False,
                }
            )

    failed = [r for r in results if r["status"] == "failed"]
    return ORJSONResponse(
        content={
            "requested": len(req.items),
            "completed": len(req.items) - len(failed),
            "failed": len(failed),
            "results": results,
        }
    )


@router.get("/session/{session_id}/highlights")
async def get_highlights(session_id: str, top_n: int = Query(10, ge=1, le=50)):
    """Get top-N highlight events for a session, sorted by score."""
    import json
    events_path = settings.processed_dir / session_id / "events.json"
    
    events = []
    if events_path.exists():
        with open(events_path) as f:
            events = json.load(f)
    else:
        # Derive from lap summaries
        from src.backend.services.laps import get_lap_summaries
        summaries = get_lap_summaries(session_id)
        if summaries:
            for s in summaries:
                for ev in s.events:
                    events.append({
                        "type": ev.type,
                        "lap": s.lap,
                        "actor": ev.actor,
                        "description": ev.details.get("description", "") if ev.details else "",
                        "highlight_score": 0.7,
                    })

    events.sort(key=lambda e: e.get("highlight_score", 0), reverse=True)
    highlights = events[:top_n]

    return ORJSONResponse(content={
        "session_id": session_id,
        "count": len(highlights),
        "highlights": highlights,
    })


@router.get("/session/{session_id}/insights")
async def get_insights(session_id: str):
    """Get AI-generated natural language insights for a session."""
    try:
        from src.backend.services.insights import generate_insights
        from src.backend.services.exporter import load_session_data
        session_data = load_session_data(session_id)
        if session_data is not None:
            insights = generate_insights(session_data)
            return ORJSONResponse(content={
                "session_id": session_id,
                "insights": [i.model_dump() for i in insights],
            })
    except Exception:
        pass

    # Fallback: generate basic insights from lap summaries
    from src.backend.services.laps import get_lap_summaries
    summaries = get_lap_summaries(session_id)
    if not summaries:
        return ORJSONResponse(content={"session_id": session_id, "insights": []})

    basic_insights = []
    leader = summaries[-1].leader if summaries else ""
    total_laps = len(summaries)
    if leader:
        basic_insights.append({"category": "race", "text": f"{leader} leads after {total_laps} laps", "priority": 1})

    overtakes = sum(len([e for e in s.events if e.type == 'overtake']) for s in summaries)
    if overtakes > 0:
        basic_insights.append({"category": "action", "text": f"{overtakes} overtakes detected across the race", "priority": 2})

    pits = sum(len(s.pit_stops) for s in summaries)
    if pits > 0:
        basic_insights.append({"category": "strategy", "text": f"{pits} pit stops performed", "priority": 2})

    return ORJSONResponse(content={"session_id": session_id, "insights": basic_insights})


@router.get("/session/{session_id}/laps")
async def get_laps(session_id: str):
    """
    Get per-lap race summaries for the Lap Playback Engine.

    Returns the full array of lap summaries computed from FastF1 session.laps.
    Cached as laps.json after first computation.
    """
    from src.backend.services.laps import get_lap_summaries, compute_lap_summaries, save_lap_summaries
    from src.backend.services.fetcher import fetch_session

    # Try cached first
    cached = get_lap_summaries(session_id)
    if cached is not None:
        return ORJSONResponse(content={
            "session_id": session_id,
            "total_laps": len(cached),
            "laps": [s.model_dump() for s in cached],
        })

    # Need to compute — load the FastF1 session
    # Parse session_id: "{year}_{gp}_{type}"
    year, gp, session_type = _parse_session_id(session_id)

    try:
        session = fetch_session(year, gp, session_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error loading session for laps")
        raise HTTPException(status_code=500, detail=f"Session load error: {e}")

    summaries = compute_lap_summaries(session)
    if not summaries:
        raise HTTPException(status_code=404, detail="No lap data available for this session")

    # Cache for future requests
    save_lap_summaries(session_id, summaries)

    return ORJSONResponse(content={
        "session_id": session_id,
        "total_laps": len(summaries),
        "laps": [s.model_dump() for s in summaries],
    })


@router.get("/session/{session_id}/track-replay")
async def get_track_replay(session_id: str):
    """
    Return a lightweight replay payload for the track map.

    Payload is intentionally minimal:
    - static track points
    - driver colors
    - per-frame distance along track
    - lap boundaries for syncing lap playback to telemetry time
    """
    try:
        session_data = _load_or_build_session_data(session_id)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error preparing track replay data")
        raise HTTPException(status_code=500, detail=f"Track replay error: {e}")

    track_points = [{"x": p.x, "y": p.y} for p in session_data.track]
    track_length = (
        session_data.metadata.computed_track_length_m
        or session_data.metadata.track_length_m
        or 0.0
    )
    lap_boundaries = _compute_lap_boundaries(session_data.frames, session_data.metadata.total_laps)

    frames = []
    start_t = float(session_data.frames[0].t) if session_data.frames else 0.0
    for frame in session_data.frames:
        positions = {}
        for code, driver in frame.drivers.items():
            positions[code] = {
                "d": round(float(driver.distance), 2),
                "s": round(float(driver.speed), 1),
                "pos": int(driver.position) if driver.position else 0,
            }
        frames.append({
            "t": round(float(frame.t - start_t), 3),
            "positions": positions,
        })

    lap_boundaries_rel = [max(0.0, round(float(t - start_t), 3)) for t in lap_boundaries]
    duration = (
        round(float(session_data.frames[-1].t - start_t), 3)
        if session_data.frames
        else round(float(session_data.metadata.end_time - session_data.metadata.start_time), 3)
    )

    drivers = [
        {"id": driver.code, "color": driver.team_color}
        for driver in session_data.metadata.drivers
    ]

    return ORJSONResponse(content={
        "session_id": session_id,
        "track_points": track_points,
        "track_length": round(float(track_length), 3),
        "drivers": drivers,
        "lap_boundaries": lap_boundaries_rel,
        "duration": duration,
        "frame_rate": session_data.metadata.frame_rate,
        "frames": frames,
    })
