"""
Local RAG retrieval over processed race JSON files.

This provides retrieval context + source metadata that can be passed to an LLM
for answer synthesis.
"""
from __future__ import annotations

import json
import math
import collections
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.backend.config import settings
from src.backend.constants import DRIVER_CODES as _DRIVER_CODE_MAPPING

_TOKEN_RE = re.compile(r"[a-z0-9]+")



@dataclass(frozen=True)
class RagDoc:
    title: str
    source: str
    text: str
    category: str


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


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [s.strip() for s in raw if s and s.strip()]


def _is_lap_summary_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        k in q
        for k in (
            "lap summary",
            "summary of lap",
            "summarize lap",
            "this lap",
            "race summary",
        )
    )


def _bm25_score(query_tokens: set[str], doc_tokens: list[str],
                idf: dict[str, float], avg_dl: float,
                k1: float = 1.5, b: float = 0.75) -> float:
    dl = len(doc_tokens)
    tf_map = collections.Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in tf_map:
            continue
        tf = tf_map[term]
        score += idf.get(term, 0.0) * (
            tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
        )
    return score


def _format_lap_summary_from_context(context: str, live_context: dict[str, Any] | None = None) -> str:
    blocks = [b.strip() for b in (context or "").split("\n\n---\n\n") if b.strip()]
    lap_line = ""
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        body = " ".join(lines[1:])
        if " lap " in body.lower() and "top positions:" in body.lower():
            lap_line = body
            break

    if not lap_line:
        return ""

    # Example:
    # "2024 Bahrain Grand Prix lap 48: leader VER (Max ...). Top positions: P1 ... Events: 3. Pit stops: 1."
    race_part = lap_line
    leader_part = ""
    top_part = ""
    events_part = "N/A"
    pits_part = "N/A"

    if "Top positions:" in lap_line:
        left, right = lap_line.split("Top positions:", 1)
        race_part = left.strip().rstrip(".")
        top_part = right.strip()
    if "leader " in race_part:
        before_leader, after_leader = race_part.split("leader ", 1)
        race_part = before_leader.strip().rstrip(":")
        leader_part = after_leader.strip().rstrip(".")

    # Parse trailing metrics from top part.
    top_clean = top_part
    if "Events:" in top_clean:
        top_clean, ev = top_clean.split("Events:", 1)
        events_part = ev.split(".", 1)[0].strip()
        rem = ev.split(".", 1)[1] if "." in ev else ""
        if "Pit stops:" in rem:
            pits_part = rem.split("Pit stops:", 1)[1].split(".", 1)[0].strip()
    elif "Pit stops:" in top_clean:
        top_clean, ps = top_clean.split("Pit stops:", 1)
        pits_part = ps.split(".", 1)[0].strip()
    top_clean = top_clean.strip().rstrip(".")

    live = live_context or {}
    lap = live.get("current_lap")
    total = live.get("total_laps")
    live_head = ""
    if lap and total:
        live_head = f"Live race context: lap {lap}/{total}.\n"

    lines = [
        f"{race_part} summary:",
        f"- Leader: {leader_part or 'N/A'}",
        f"- Top order: {top_clean or 'N/A'}",
        f"- Events this lap: {events_part}",
        f"- Pit stops this lap: {pits_part}",
    ]
    return live_head + "\n".join(lines)


@lru_cache(maxsize=32)
def _build_docs(session_dir_str: str, _mtime: float) -> tuple[dict[str, Any], list[RagDoc], dict[str, float], float]:
    session_dir = Path(session_dir_str)

    with open(session_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(session_dir / "laps.json", encoding="utf-8") as f:
        laps: list[dict[str, Any]] = json.load(f)
    events = []
    events_path = session_dir / "events.json"
    if events_path.exists():
        with open(events_path, encoding="utf-8") as f:
            events = json.load(f)

    drivers = metadata.get("drivers", [])
    name_map = {
        str(d.get("code", "")).upper(): str(d.get("name", "")).strip()
        for d in drivers
        if d.get("code")
    }

    docs: list[RagDoc] = []
    session_name = f"{metadata.get('year', '')} {metadata.get('gp', '')}".strip()
    drivers_text = ", ".join(
        [f"{code} ({name_map.get(code, '')})".strip() for code in sorted(name_map.keys())]
    )
    docs.append(
        RagDoc(
            title="Session overview",
            source=str(session_dir / "metadata.json"),
            category="race_overview",
            text=(
                f"Session: {session_name} ({metadata.get('session_type', '')}). "
                f"Track: {metadata.get('track_name', '')}. "
                f"Total laps: {metadata.get('total_laps', len(laps))}. "
                f"Drivers: {drivers_text}."
            ),
        )
    )

    # Lap-level docs for retrieval.
    for lap in laps:
        lap_no = int(lap.get("lap", 0))
        leader = str(lap.get("leader", ""))
        positions = lap.get("positions", {}) or {}
        ranked = []
        for code, pos in positions.items():
            try:
                ranked.append((str(code), int(pos)))
            except Exception:
                continue
        ranked.sort(key=lambda x: x[1])
        top5 = ranked[:5]
        top5_text = ", ".join(
            [f"P{p} {c} ({name_map.get(c, 'Unknown')})" for c, p in top5]
        )
        pit_count = len(lap.get("pit_stops", []) or [])
        evt_count = len(lap.get("events", []) or [])

        docs.append(
            RagDoc(
                title=f"Lap {lap_no} summary",
                source=str(session_dir / "laps.json"),
                category="lap_summary",
                text=(
                    f"{session_name} lap {lap_no}: "
                    f"leader {leader} ({name_map.get(leader, 'Unknown')}). "
                    f"Top positions: {top5_text}. "
                    f"Events: {evt_count}. Pit stops: {pit_count}."
                ),
            )
        )

    # Driver-level aggregate docs.
    for code, name in name_map.items():
        pos_vals = []
        avg_speed_vals = []
        max_speed_vals = []
        best_lap = None
        for lap in laps:
            pos = (lap.get("positions", {}) or {}).get(code)
            if isinstance(pos, int):
                pos_vals.append(pos)
            spd = (lap.get("avg_speed", {}) or {}).get(code)
            if isinstance(spd, (int, float)):
                avg_speed_vals.append(float(spd))
            mspd = (lap.get("max_speed", {}) or {}).get(code)
            if isinstance(mspd, (int, float)):
                max_speed_vals.append(float(mspd))
            lt = (lap.get("lap_times", {}) or {}).get(code)
            if isinstance(lt, (int, float)) and lt > 0:
                best_lap = lt if best_lap is None else min(best_lap, float(lt))

        if not pos_vals:
            continue
        avg_pos = sum(pos_vals) / len(pos_vals)
        best_pos = min(pos_vals)
        worst_pos = max(pos_vals)
        avg_spd = (sum(avg_speed_vals) / len(avg_speed_vals)) if avg_speed_vals else 0.0
        peak_spd = max(max_speed_vals) if max_speed_vals else 0.0

        docs.append(
            RagDoc(
                title=f"Driver performance {code}",
                source=str(session_dir / "laps.json"),
                category="driver_performance",
                text=(
                    f"Driver {code} ({name}). "
                    f"For {name} ({code}), average position was {avg_pos:.2f}, best position {best_pos}, worst position {worst_pos}. "
                    f"{name} ({code}) average speed {avg_spd:.1f} km/h, max speed {peak_spd:.1f} km/h. "
                    f"Best lap time for {name} ({code}) was {best_lap if best_lap is not None else 'N/A'}."
                ),
            )
        )

    # Event feed doc
    if events:
        sample = events[:120]
        evt_lines = []
        for ev in sample:
            evt_lines.append(
                f"t={ev.get('t')} type={ev.get('event_type')} driver={ev.get('driver')} details={ev.get('details')}"
            )
        docs.append(
            RagDoc(
                title="Detected event feed",
                source=str(session_dir / "events.json"),
                category="events",
                text="\n".join(evt_lines),
            )
        )

    # IDF Precomputation
    n_docs = len(docs)
    doc_tokens_list = [_tokens(d.text) for d in docs]
    avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(1, n_docs)
    
    # Term -> number of docs containing term
    df: dict[str, int] = collections.defaultdict(int)
    for dt in doc_tokens_list:
        for term in set(dt):
            df[term] += 1
            
    idf: dict[str, float] = {}
    for term, count in df.items():
        # Standard BM25 IDF
        idf[term] = math.log((n_docs - count + 0.5) / (count + 0.5) + 1.0)

    return metadata, docs, idf, avg_dl


def retrieve_local_context(
    *,
    query: str,
    session_id: str | None,
    top_k: int = 6,
) -> dict[str, Any]:
    session_dir = _pick_session_dir(session_id)
    if session_dir is None:
        return {"context": "", "sources": [], "session_id": None}

    # Check mtime of laps.json for cache invalidation
    laps_path = session_dir / "laps.json"
    mtime = laps_path.stat().st_mtime if laps_path.exists() else 0.0

    metadata, docs, idf, avg_dl = _build_docs(str(session_dir), mtime)
    q_tokens = _tokens(query)
    q_raw = (query or "").strip().lower()

    scored = []
    for doc in docs:
        d_tokens_list = list(_TOKEN_RE.findall(doc.text.lower()))
        score = _bm25_score(q_tokens, d_tokens_list, idf, avg_dl)
        if q_raw and q_raw in doc.text.lower():
            score += 0.5  # Heavy bias for exact phrase match
        if score > 0:
            scored.append((score, doc))

    if not scored:
        # fallback to session overview + latest lap docs
        defaults = []
        for doc in docs:
            if doc.category == "race_overview":
                defaults.append((0.1, doc))
        for doc in reversed(docs):
            if doc.category == "lap_summary":
                defaults.append((0.08, doc))
            if len(defaults) >= top_k:
                break
        scored = defaults

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[: max(1, min(top_k, 20))]

    context_parts = []
    sources = []
    for i, (score, doc) in enumerate(picked, start=1):
        context_parts.append(f"[{i}] {doc.title}\n{doc.text}")
        sources.append(
            {
                "id": f"src-{i}",
                "rank": i,
                "title": doc.title,
                "source": doc.source,
                "category": doc.category,
                "score": round(float(score), 4),
            }
        )

    return {
        "session_id": metadata.get("session_id"),
        "context": "\n\n---\n\n".join(context_parts),
        "sources": sources,
    }


def answer_from_local_rag(
    *,
    query: str,
    retrieval: dict[str, Any],
    live_context: dict[str, Any] | None = None,
) -> str:
    """
    Deterministic extractive fallback when external LLM is unavailable.
    Still grounded in retrieved local context.
    """
    context = str(retrieval.get("context", "") or "")
    if not context.strip():
        return ""

    if _is_lap_summary_query(query):
        summary = _format_lap_summary_from_context(context, live_context=live_context)
        if summary:
            return summary

    q_tokens = _tokens(query)
    candidates: list[tuple[float, str]] = []
    for sentence in _sentences(context):
        s_tokens = _tokens(sentence)
        overlap = len(q_tokens & s_tokens)
        if overlap <= 0:
            continue
        score = overlap / math.sqrt(max(1, len(s_tokens)))
        if any(k in sentence.lower() for k in ("leader", "position", "speed", "lap", "pit")):
            score += 0.15
        candidates.append((score, sentence))

    if not candidates:
        # No lexical overlap -> let caller try alternate fallback paths.
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = [text for _, text in candidates[:3]]

    live = live_context or {}
    lap = live.get("current_lap")
    leader = live.get("leader")
    prefix_bits = []
    if lap:
        prefix_bits.append(f"lap {lap}")
    if leader:
        full_leader = _DRIVER_CODE_MAPPING.get(leader, leader)
        prefix_bits.append(f"leader {full_leader}")
    prefix = f"Live context ({', '.join(prefix_bits)}).\n" if prefix_bits else ""

    return f"{prefix}" + "\n".join(f"- {line}" for line in top)


def warm_local_rag_cache(session_id: str) -> bool:
    """
    Preload the local RAG document cache for a processed session.
    Returns True if cache warm-up succeeded.
    """
    session_dir = settings.processed_dir / session_id
    if not session_dir.exists():
        return False
    laps_path = session_dir / "laps.json"
    mtime = laps_path.stat().st_mtime if laps_path.exists() else 0.0
    _build_docs(str(session_dir), mtime)
    return True
