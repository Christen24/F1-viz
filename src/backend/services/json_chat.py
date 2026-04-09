"""
Local JSON-backed chat answers for race stats, with no external LLM dependency.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.backend.config import settings

try:
    import fastf1  # type: ignore
except Exception:  # pragma: no cover
    fastf1 = None  # type: ignore

_WINNER_QUERY_RE = re.compile(
    r"\bwho\s+won\s+(?:the\s+)?(?P<event>.+?)\s+(?P<year>20\d{2})\b",
    flags=re.IGNORECASE,
)
_WINNER_QUERY_RE_PREFIX = re.compile(
    r"\bwho\s+won\s+(?:the\s+)?(?P<year>20\d{2})\s+(?P<event>.+)$",
    flags=re.IGNORECASE,
)

_POLE_QUERY_RE = re.compile(
    r"\bwho\s+(?:had|got|took|was\s+in)\s+(?:the\s+)?pole\b.*\s+(?P<event>.+?)\s+(?P<year>20\d{2})\b",
    flags=re.IGNORECASE,
)
_POLE_QUERY_RE_PREFIX = re.compile(
    r"\bwho\s+(?:had|got|took|was\s+in)\s+(?:the\s+)?pole\b.*\s+(?P<year>20\d{2})\s+(?P<event>.+)$",
    flags=re.IGNORECASE,
)
_POLE_QUERY_SIMPLE = re.compile(
    r"\bwho\s+(?:had|got|took|was\s+in)\s+(?:the\s+)?pole\b",
    flags=re.IGNORECASE,
)


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


def _normalize_event_query(event: str) -> str:
    e = (event or "").strip().lower()
    e = re.sub(r"\bgp\b", "grand prix", e)
    e = re.sub(r"^[\s,;:.-]*the\s+", "", e)
    e = re.sub(r"[^\w\s-]", " ", e)
    e = re.sub(r"\s+", " ", e).strip()
    return e


def _extract_winner_query(query: str) -> tuple[int, str] | None:
    raw = query or ""
    match = _WINNER_QUERY_RE_PREFIX.search(raw) or _WINNER_QUERY_RE.search(raw)
    if not match:
        return None
    year = int(match.group("year"))
    event = _normalize_event_query(match.group("event"))
    if event in {"", "the", "grand prix", "prix"}:
        return None
    if not event:
        return None
    return year, event


def _extract_pole_query(query: str) -> tuple[int, str] | None:
    raw = query or ""
    match = _POLE_QUERY_RE_PREFIX.search(raw) or _POLE_QUERY_RE.search(raw)
    if not match:
        return None
    year = int(match.group("year"))
    event = _normalize_event_query(match.group("event"))
    if not event:
        return None
    return year, event


@lru_cache(maxsize=64)
def _lookup_session_result(year: int, event_query: str, session_type: str = "R") -> dict[str, Any] | None:
    if fastf1 is None:
        return None

    try:
        fastf1.Cache.enable_cache(str(settings.fastf1_cache_dir))
    except Exception:
        pass

    schedule = fastf1.get_event_schedule(year)
    event_name = None
    query_tokens = [tok for tok in re.findall(r"[a-z]+", event_query) if tok not in {"the"}]
    if not query_tokens:
        return None

    for _, row in schedule.iterrows():
        round_number = int(row.get("RoundNumber", 0) or 0)
        event_format = str(row.get("EventFormat", "")).lower()
        if round_number <= 0 or "test" in event_format:
            continue
        candidate = str(row.get("EventName", "")).lower()
        if all(tok in candidate for tok in query_tokens):
            event_name = str(row.get("EventName"))
            break

    if event_name is None:
        # Don't guess/fallback to a random event name.
        return None

    session = fastf1.get_session(year, event_name, session_type)
    try:
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception:
        session.load()

    results = session.results
    if results is None or len(results) == 0:
        return None

    top = None
    try:
        p1 = results[results["Position"] == 1]
        if len(p1) > 0:
            top = p1.iloc[0]
    except Exception:
        top = None

    if top is None:
        try:
            top = results.sort_values("Position").iloc[0]
        except Exception:
            return None

    return {
        "year": year,
        "event_name": event_name,
        "driver": str(top.get("FullName", "")).strip() or str(top.get("Abbreviation", "")).strip(),
        "team": str(top.get("TeamName", "")).strip(),
        "source": f"fastf1://{year}/{event_name}/{session_type}",
    }


def _answer_historical_winner_query(query: str) -> dict[str, Any] | None:
    parsed = _extract_winner_query(query)
    if not parsed:
        return None
    year, event = parsed
    try:
        winner = _lookup_session_result(year, event, "R")
    except Exception:
        return None
    if not winner:
        return None
    driver = winner.get("driver") or "Unknown"
    team = winner.get("team") or "Unknown"
    event_name = winner.get("event_name") or event.title()
    return {
        "answer": f"{year} {event_name}: winner was {driver} ({team}).",
        "sources": [
            {
                "id": "src-fastf1-winner",
                "rank": 1,
                "title": f"{year} {event_name} official results",
                "source": winner.get("source", "fastf1://results"),
                "category": "historical_result",
            }
        ],
    }


def _answer_pole_query(query: str, current_metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    parsed = _extract_pole_query(query)
    year, event = 0, ""
    if parsed:
        year, event = parsed
    elif _POLE_QUERY_SIMPLE.search(query) and current_metadata:
        # Fallback to current session in context
        year = int(current_metadata.get("year", 0))
        event = str(current_metadata.get("gp", ""))
    else:
        return None

    if not year or not event:
        return None

    try:
        result = _lookup_session_result(year, event, "Q")
    except Exception:
        return None

    if not result:
        return None

    driver = result.get("driver") or "Unknown"
    team = result.get("team") or "Unknown"
    event_name = result.get("event_name") or event.title()

    return {
        "answer": f"{year} {event_name}: pole position was taken by {driver} ({team}).",
        "sources": [
            {
                "id": "src-fastf1-pole",
                "rank": 1,
                "title": f"{year} {event_name} qualifying results",
                "source": result.get("source", "fastf1://qualifying"),
                "category": "historical_result",
            }
        ],
    }


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


def _winner_code_from_lap(lap: dict[str, Any]) -> str | None:
    positions = lap.get("positions", {}) or {}
    ranked = _top_positions(positions, top_n=1)
    if ranked:
        return ranked[0][0]
    leader = str(lap.get("leader") or "").strip()
    return leader or None


def answer_from_json(
    *,
    query: str,
    session_id: str | None,
    live_context: dict[str, Any] | None,
) -> dict[str, Any]:
    winner_answer = _answer_historical_winner_query(query)
    if winner_answer is not None:
        return winner_answer

    # Move session_dir lookup earlier to allow falling back to current session for pole
    session_dir = _pick_session_dir(session_id)
    payload = None
    if session_dir:
        payload = _load_session_json(str(session_dir))

    pole_answer = _answer_pole_query(query, payload["metadata"] if payload else None)
    if pole_answer is not None:
        return pole_answer

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

    winner_intent = any(
        k in q
        for k in (
            "race winner",
            "who is the winner",
            "who's the winner",
            "whos the winner",
            "who won this race",
            "winner in this race",
            "winner of this race",
            "winner in this rac",
            "winner of this rac",
        )
    )
    if winner_intent:
        race_finished = current_lap >= len(laps)
        if race_finished:
            final_lap = laps[-1]
            winner_code = _winner_code_from_lap(final_lap)
            winner_text = _driver_label(winner_code, name_map) if winner_code else "N/A"
            return {
                "answer": (
                    f"{session_name}: race finished ({len(laps)}/{len(laps)} laps).\n"
                    f"Winner: {winner_text}."
                ),
                "sources": sources,
            }

        projected = _driver_label(leader, name_map) if leader else "N/A"
        return {
            "answer": (
                f"{session_name}: race is still in progress (lap {current_lap}/{len(laps)}).\n"
                f"Current leader: {projected}.\n"
                "Winner is not confirmed yet."
            ),
            "sources": sources,
        }

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
            f"Try asking: 'Who is leading?', 'Lap summary', 'Max Verstappen performance', or 'Lewis Hamilton position'."
        ),
        "sources": sources,
    }
