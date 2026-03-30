"""
Local JSON-backed chat answers for race stats, with no external LLM dependency.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.backend.config import settings


def _pick_session_dir(session_id: str | None) -> Path | None:
    processed = settings.processed_dir
    if session_id:
        p = processed / session_id
        return p if p.exists() else None

    dirs = [d for d in processed.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not dirs:
        return None
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0]


@lru_cache(maxsize=8)
def _load_session_json(session_dir_str: str) -> dict[str, Any]:
    session_dir = Path(session_dir_str)
    with open(session_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(session_dir / "laps.json", encoding="utf-8") as f:
        laps = json.load(f)
    events_path = session_dir / "events.json"
    events = []
    if events_path.exists():
        with open(events_path, encoding="utf-8") as f:
            events = json.load(f)
    return {"metadata": metadata, "laps": laps, "events": events}


def _driver_codes(metadata: dict[str, Any]) -> set[str]:
    return {str(d.get("code", "")).upper() for d in metadata.get("drivers", []) if d.get("code")}


def _driver_name_map(metadata: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in metadata.get("drivers", []):
        code = str(d.get("code", "")).upper()
        name = str(d.get("name", "")).strip()
        if code and name:
            out[code] = name
    return out


def _driver_label(code: str, name_map: dict[str, str]) -> str:
    name = name_map.get(code.upper())
    return f"{code} ({name})" if name else code


def _find_driver_in_query(query: str, metadata: dict[str, Any], fallback: str | None = None) -> str | None:
    q = query.upper()
    for code in _driver_codes(metadata):
        if code and code in q:
            return code
    return fallback


def _top_positions(positions: dict[str, Any], top_n: int = 3) -> list[tuple[str, int]]:
    items = []
    for code, pos in positions.items():
        try:
            items.append((str(code), int(pos)))
        except Exception:
            continue
    items.sort(key=lambda x: x[1])
    return items[:top_n]


def answer_from_json(
    *,
    query: str,
    session_id: str | None,
    live_context: dict[str, Any] | None,
) -> dict[str, Any]:
    session_dir = _pick_session_dir(session_id)
    if session_dir is None:
        return {
            "answer": "No processed race JSON is available yet. Load a race once to create local JSON data.",
            "sources": [],
        }

    payload = _load_session_json(str(session_dir))
    metadata = payload["metadata"]
    laps: list[dict[str, Any]] = payload["laps"]
    if not laps:
        return {"answer": "No lap summary data found for this session.", "sources": []}

    q = (query or "").strip().lower()
    live = live_context or {}
    current_lap = int(live.get("current_lap") or len(laps) or 1)
    current_lap = max(1, min(current_lap, len(laps)))
    lap = laps[current_lap - 1]
    leader = str(live.get("leader") or lap.get("leader") or "")
    name_map = _driver_name_map(metadata)
    session_name = f"{metadata.get('year', '')} {metadata.get('gp', '')}".strip()

    sources = [
        {
            "id": "src-1",
            "rank": 1,
            "title": "Session metadata",
            "source": str(session_dir / "metadata.json"),
            "category": "race_json",
        },
        {
            "id": "src-2",
            "rank": 2,
            "title": "Lap summaries",
            "source": str(session_dir / "laps.json"),
            "category": "race_json",
        },
    ]

    if any(k in q for k in ["leader", "leading", "p1", "first place", "who is first"]):
        top3 = _top_positions(lap.get("positions", {}), top_n=3)
        podium = ", ".join([f"P{pos} {_driver_label(code, name_map)}" for code, pos in top3]) if top3 else "positions unavailable"
        return {
            "answer": (
                f"{session_name}: current lap {current_lap}/{len(laps)}.\n"
                f"Leader: {_driver_label(leader, name_map) if leader else 'N/A'}.\n"
                f"Top 3 snapshot: {podium}."
            ),
            "sources": sources,
        }

    if any(k in q for k in ["lap summary", "summarize lap", "summary", "this lap"]):
        events = lap.get("events", []) or []
        pits = lap.get("pit_stops", []) or []
        top5 = _top_positions(lap.get("positions", {}), top_n=5)
        top5_text = ", ".join([f"P{pos} {_driver_label(code, name_map)}" for code, pos in top5]) if top5 else "N/A"
        return {
            "answer": (
                f"{session_name} lap {current_lap}/{len(laps)} summary:\n"
                f"- Leader: {_driver_label(leader, name_map) if leader else 'N/A'}\n"
                f"- Top 5: {top5_text}\n"
                f"- Overtakes/events: {len(events)}\n"
                f"- Pit stops: {len(pits)}"
            ),
            "sources": sources,
        }

    if any(k in q for k in ["driver performance", "performance", "speed", "position", "where is", "how is"]):
        target = _find_driver_in_query(query, metadata, fallback=leader or None)
        if not target:
            return {
                "answer": "I could not identify a driver code in your query. Try a code like VER, HAM, or LEC.",
                "sources": sources,
            }

        pos = (lap.get("positions", {}) or {}).get(target)
        avg_speed = (lap.get("avg_speed", {}) or {}).get(target)
        max_speed = (lap.get("max_speed", {}) or {}).get(target)
        gap = (lap.get("gaps", {}) or {}).get(target)
        tyre = (lap.get("tyres", {}) or {}).get(target)
        tyre_age = (lap.get("tyre_ages", {}) or {}).get(target)
        return {
            "answer": (
                f"{session_name} lap {current_lap}/{len(laps)} - {_driver_label(target, name_map)}\n"
                f"- Position: {pos if pos is not None else 'N/A'}\n"
                f"- Gap to leader: {gap if gap is not None else 'N/A'} s\n"
                f"- Avg speed: {avg_speed if avg_speed is not None else 'N/A'} km/h\n"
                f"- Max speed: {max_speed if max_speed is not None else 'N/A'} km/h\n"
                f"- Tyre: {tyre if tyre is not None else 'N/A'} (age {tyre_age if tyre_age is not None else 'N/A'})"
            ),
            "sources": sources,
        }

    top3 = _top_positions(lap.get("positions", {}), top_n=3)
    podium = ", ".join([f"P{pos} {_driver_label(code, name_map)}" for code, pos in top3]) if top3 else "N/A"
    return {
        "answer": (
            f"{session_name} lap {current_lap}/{len(laps)}.\n"
            f"Leader: {_driver_label(leader, name_map) if leader else 'N/A'}.\n"
            f"Top 3: {podium}.\n"
            f"Try asking: 'Who is leading?', 'Lap summary', 'VER performance', or 'HAM position'."
        ),
        "sources": sources,
    }
