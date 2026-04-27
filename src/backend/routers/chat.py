"""
Chat router with SSE streaming for RAG responses.
LLM provider order: Ollama (local) → Gemini → OpenRouter
"""
from __future__ import annotations

import json
import logging
import os
import asyncio
import hashlib
import re
from collections import defaultdict, deque
from email.utils import parsedate_to_datetime
from pathlib import Path
import time
import sqlite3
from typing import Any, AsyncIterator, Literal

import httpx
from dotenv import dotenv_values
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.backend.config import settings
from src.backend.constants import DRIVER_CODES as _DRIVER_CODE_MAPPING
from src.backend.services.local_rag import retrieve_local_context, answer_from_local_rag
from src.backend.services.json_chat import answer_from_json
from src.backend.services.rag import f1_knowledge
from src.backend.services.wiki_fallback import retrieve_wikipedia_context

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# ── SQLite-backed rate-limit state (survives restarts) ───────────────────────
_LLM_LIMIT_DB = settings.models_dir / "rate_limit.db"


def _init_limit_db():
    if not _LLM_LIMIT_DB.parent.exists():
        _LLM_LIMIT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_LLM_LIMIT_DB), check_same_thread=False)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS limits "
            "(key TEXT PRIMARY KEY, val_float REAL, val_int INTEGER)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO limits (key, val_float, val_int) "
            "VALUES ('cooldown_until', 0.0, 0)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO limits (key, val_float, val_int) "
            "VALUES ('streak', 0.0, 0)"
        )
        conn.commit()
    finally:
        conn.close()


_init_limit_db()


def _get_limit_state() -> tuple[float, int]:
    conn = sqlite3.connect(str(_LLM_LIMIT_DB), check_same_thread=False)
    try:
        until = conn.execute(
            "SELECT val_float FROM limits WHERE key='cooldown_until'"
        ).fetchone()[0]
        streak = conn.execute(
            "SELECT val_int FROM limits WHERE key='streak'"
        ).fetchone()[0]
        return float(until), int(streak)
    except Exception:
        return 0.0, 0
    finally:
        conn.close()


def _set_limit_state(until: float, streak: int):
    conn = sqlite3.connect(str(_LLM_LIMIT_DB), check_same_thread=False)
    try:
        conn.execute(
            "UPDATE limits SET val_float=? WHERE key='cooldown_until'", (until,)
        )
        conn.execute(
            "UPDATE limits SET val_int=? WHERE key='streak'", (streak,)
        )
        conn.commit()
    finally:
        conn.close()


_LLM_COOLDOWN_BASE_SECONDS = 8.0
_LLM_COOLDOWN_MAX_SECONDS = 120.0
_ANSWER_CACHE: dict[str, tuple[float, str]] = {}
_ANSWER_CACHE_TTL_SECONDS = float(max(1, settings.chat_answer_cache_ttl_seconds))
_ANSWER_CACHE_VERSION = "v2"
_LLM_CALLS_WINDOW = 60.0
_LLM_CALLS_GLOBAL: deque[float] = deque()
_LLM_CALLS_BY_SESSION: dict[str, deque[float]] = defaultdict(deque)
_LLM_LAST_STRATEGY_CALL_BY_SESSION: dict[str, float] = {}
_GLOBAL_RAG_DISABLED_REASON: str | None = None
_HISTORICAL_WINNER_RE = re.compile(r"\bwho\s+won\b.*\b20\d{2}\b", flags=re.IGNORECASE)

# ── Ollama availability cache ────────────────────────────────────────────────
# Once Ollama fails (not running), we skip it for _OLLAMA_SKIP_WINDOW seconds
# rather than timing out on every request.
_OLLAMA_LAST_FAILURE: float = 0.0
_OLLAMA_SKIP_WINDOW: float = 30.0   # seconds to skip after a failure


def _normalize_prompt(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _cache_key(
    *,
    session_id: str | None,
    user_prompt: str,
    category: str | None,
    live_context: dict[str, Any] | None,
) -> str:
    live = live_context or {}
    base = "|".join(
        [
            _ANSWER_CACHE_VERSION,
            str(session_id or ""),
            str(category or ""),
            str(live.get("current_lap") or ""),
            str(live.get("leader") or ""),
            _normalize_prompt(user_prompt),
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    row = _ANSWER_CACHE.get(key)
    if not row:
        return None
    expires_at, value = row
    if time.time() > expires_at:
        _ANSWER_CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: str) -> None:
    _ANSWER_CACHE[key] = (time.time() + _ANSWER_CACHE_TTL_SECONDS, value)


def _prune_llm_windows(now: float) -> None:
    cutoff = now - _LLM_CALLS_WINDOW
    while _LLM_CALLS_GLOBAL and _LLM_CALLS_GLOBAL[0] < cutoff:
        _LLM_CALLS_GLOBAL.popleft()
    to_delete: list[str] = []
    for sid, calls in _LLM_CALLS_BY_SESSION.items():
        while calls and calls[0] < cutoff:
            calls.popleft()
        if not calls:
            to_delete.append(sid)
    for sid in to_delete:
        _LLM_CALLS_BY_SESSION.pop(sid, None)


def _llm_budget_check(session_id: str | None) -> tuple[bool, str | None]:
    now = time.time()
    _prune_llm_windows(now)
    if len(_LLM_CALLS_GLOBAL) >= max(1, settings.chat_llm_max_calls_per_minute):
        return False, "global_llm_budget_reached"
    sid = session_id or "__global__"
    per_session = _LLM_CALLS_BY_SESSION.get(sid)
    if per_session and len(per_session) >= max(
        1, settings.chat_llm_max_calls_per_session_per_minute
    ):
        return False, "session_llm_budget_reached"
    return True, None


def _record_llm_call(session_id: str | None) -> None:
    now = time.time()
    sid = session_id or "__global__"
    _LLM_CALLS_GLOBAL.append(now)
    _LLM_CALLS_BY_SESSION[sid].append(now)


def _strategy_llm_interval_ok(session_id: str | None) -> tuple[bool, str | None]:
    sid = session_id or "__global__"
    now = time.time()
    last = _LLM_LAST_STRATEGY_CALL_BY_SESSION.get(sid)
    min_interval = float(max(0, settings.chat_strategy_llm_min_interval_seconds))
    if last is None or min_interval <= 0:
        return True, None
    elapsed = now - last
    if elapsed >= min_interval:
        return True, None
    remaining = int(max(1, round(min_interval - elapsed)))
    return False, f"strategy_llm_throttled_{remaining}s"


def _extract_retry_after_seconds(resp: httpx.Response) -> float | None:
    raw = (
        resp.headers.get("Retry-After") or resp.headers.get("retry-after") or ""
    ).strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            try:
                dt = parsedate_to_datetime(raw)
                delta = dt.timestamp() - time.time()
                if delta > 0:
                    return delta
            except Exception:
                pass
    try:
        payload = resp.json()
    except Exception:
        payload = {}
    details = (payload.get("error") or {}).get("details") or []
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            retry_delay = str(detail.get("retryDelay") or "").strip()
            if not retry_delay:
                continue
            match = re.match(r"^\s*(\d+(?:\.\d+)?)s\s*$", retry_delay)
            if match:
                return max(0.0, float(match.group(1)))
    return None


async def _register_llm_rate_limit(resp: httpx.Response) -> float:
    cooldown_until, streak = await asyncio.to_thread(_get_limit_state)
    retry_after = _extract_retry_after_seconds(resp)
    exp_backoff = min(
        _LLM_COOLDOWN_MAX_SECONDS,
        _LLM_COOLDOWN_BASE_SECONDS * (2 ** min(streak, 5)),
    )
    cooldown_seconds = max(retry_after or 0.0, exp_backoff)
    new_until = time.time() + cooldown_seconds
    new_streak = min(streak + 1, 12)
    await asyncio.to_thread(_set_limit_state, new_until, new_streak)
    return cooldown_seconds


async def _reset_llm_backoff() -> None:
    await asyncio.to_thread(_set_limit_state, 0.0, 0)


def _should_use_llm(query: str) -> bool:
    q = _normalize_prompt(query)
    lap_summary_tokens = [
        "lap summary", "summarize lap", "summary of lap",
        "give me a lap summary", "race summary",
    ]
    simple_tokens = [
        "who is leading", "leader", "summary", "position", "speed",
        "where is", "how is", "top 3", "top 5", "pit stop",
    ]
    complex_tokens = [
        "why", "explain", "compare", "strategy", "predict", "insight",
        "analyze", "analysis", "what if",
    ]
    if _HISTORICAL_WINNER_RE.search(q):
        return False
    if any(k in q for k in lap_summary_tokens):
        return True
    if _is_global_knowledge_query(query):
        return True
    if any(k in q for k in complex_tokens):
        return True
    if any(k in q for k in simple_tokens):
        return False
    return False


def _should_use_llm_strategy(query: str) -> bool:
    q = _normalize_prompt(query)
    if any(
        k in q
        for k in (
            "what if", "best strategy", "optimize", "simulate", "probability",
            "tradeoff", "compare", "expected", "should we", "should i",
        )
    ):
        return True
    return len(q.split()) >= 6


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    session_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=20)
    season: int | None = None
    event_name: str | None = None
    category: str | None = None
    allow_llm: bool = False
    live_context: dict[str, Any] | None = None


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _merge_rag_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    context_parts: list[str] = []
    merged_sources: list[dict[str, Any]] = []
    rank = 1
    for payload in payloads:
        ctx = str(payload.get("context") or "").strip()
        if ctx:
            context_parts.append(ctx)
        for source in payload.get("sources") or []:
            item = dict(source)
            item["rank"] = rank
            item["id"] = f"src-{rank}"
            merged_sources.append(item)
            rank += 1
    return {
        "context": "\n\n---\n\n".join(context_parts),
        "sources": merged_sources,
    }


async def _retrieve_pitcrew_context(
    *,
    query: str,
    session_id: str | None,
    top_k: int,
    season: int | None,
    event_name: str | None,
    global_only: bool = False,
) -> dict[str, Any]:
    global _GLOBAL_RAG_DISABLED_REASON

    local_payload: dict[str, Any] = {"context": "", "sources": []}
    if not global_only:
        local_payload = retrieve_local_context(
            query=query,
            session_id=session_id,
            top_k=max(2, min(top_k, 6)),
        )

    global_payload: dict[str, Any] = {"context": "", "sources": []}
    if _GLOBAL_RAG_DISABLED_REASON is None:
        try:
            global_payload = await f1_knowledge(
                query=query,
                top_k=max(2, min(top_k, 4)),
                season=season,
                event_name=event_name,
            )
        except Exception as exc:
            _GLOBAL_RAG_DISABLED_REASON = str(exc)
            logger.info(
                "Global RAG disabled for this backend run: %s",
                _GLOBAL_RAG_DISABLED_REASON,
            )

    if global_only and not str(global_payload.get("context") or "").strip():
        try:
            wiki_payload = await retrieve_wikipedia_context(query)
            if str(wiki_payload.get("context") or "").strip():
                global_payload = wiki_payload
        except Exception as exc:
            logger.debug("Wikipedia live fallback failed: %s", exc)

    return _merge_rag_payloads(local_payload, global_payload)


def _is_global_knowledge_query(query: str) -> bool:
    """
    Returns True for questions that are purely about general F1 knowledge
    (biography, history, stats) and do NOT require live race context.
    These are routed to a cloud LLM with a clean prompt.
    """
    q = _normalize_prompt(query)
    if _HISTORICAL_WINNER_RE.search(q):
        return True
    global_tokens = (
        # Biography / personal
        "age", "born", "birth", "birthday", "how old",
        "nationality", "biography", "bio", "debut", "first race",
        "career", "height", "weight", "helmet",
        # Achievements
        "how many wins", "how many poles", "how many championships",
        "how many podiums", "total wins", "world champion",
        "champion season", "constructor champion",
        # Team / org facts
        "which team", "team principal", "headquarters", "founded",
        # Location
        "where is", "where was",
    )
    if any(tok in q for tok in global_tokens):
        return True
    # Historical race results (with year)
    if re.search(r"\b20\d{2}\b", q) and any(
        tok in q for tok in ("who won", "winner", "podium", "pole", "what happened", "grand prix")
    ):
        return True
    return False


def _compact_strategy_state(live_context: dict[str, Any] | None) -> dict[str, Any]:
    live = live_context or {}
    session_id = live.get("session_id")
    leader = str(live.get("leader") or "")
    current_lap = int(live.get("current_lap") or 0)
    total_laps = int(live.get("total_laps") or 0)

    positions_raw = live.get("positions") or {}
    gaps_raw = live.get("gaps") or {}
    tyres_raw = live.get("tyres") or {}
    avg_speed_raw = live.get("avg_speed") or {}

    laps_data = _load_laps_for_session(session_id)

    ordered: list[tuple[str, int]] = []
    if isinstance(positions_raw, dict):
        for code, pos in positions_raw.items():
            try:
                ordered.append((str(code), int(pos)))
            except Exception:
                continue
        ordered.sort(key=lambda row: row[1])

    top5 = []
    for code, pos in ordered[:5]:
        gap_val = gaps_raw.get(code) if isinstance(gaps_raw, dict) else None
        tyre_val = tyres_raw.get(code) if isinstance(tyres_raw, dict) else None
        speed_val = avg_speed_raw.get(code) if isinstance(avg_speed_raw, dict) else None
        top5.append(
            {
                "code": code,
                "name": _DRIVER_CODE_MAPPING.get(code, code),
                "pos": pos,
                "gap_s": round(float(gap_val), 3) if isinstance(gap_val, (int, float)) else None,
                "gap_trend_3lap_s": _gap_trend(laps_data, code),
                "tyre": str(tyre_val) if tyre_val is not None else None,
                "avg_speed_kph": round(float(speed_val), 1)
                if isinstance(speed_val, (int, float))
                else None,
            }
        )

    top3_in = live.get("top3") or []
    top3 = []
    if isinstance(top3_in, list):
        for code in top3_in[:3]:
            top3.append({"code": code, "name": _DRIVER_CODE_MAPPING.get(code, code)})

    return {
        "session_id": live.get("session_id"),
        "race": live.get("race"),
        "current_lap": current_lap,
        "total_laps": total_laps,
        "leader": leader,
        "leader_full_name": _DRIVER_CODE_MAPPING.get(leader, leader),
        "top3_drivers": top3,
        "top5_performance": top5,
        "leader_gap_to_p2_s": live.get("leader_gap_to_p2_s"),
    }


def _gap_trend(laps: list[dict], code: str, lookback: int = 3) -> float | None:
    """Returns gap change over last N laps. Negative = closing."""
    if not laps:
        return None
    vals = [
        lap["gaps"].get(code)
        for lap in laps[-lookback:]
        if isinstance((lap.get("gaps") or {}).get(code), (int, float))
    ]
    if len(vals) < 2:
        return None
    return round(float(vals[-1]) - float(vals[0]), 3)


def _load_laps_for_session(session_id: str | None) -> list[dict]:
    if not session_id:
        return []
    try:
        path = settings.processed_dir / session_id / "laps.json"
        if not path.exists():
            return []
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _build_llm_prompt(
    *,
    retrieved_context: str,
    messages: list[ChatMessage],
    category: str | None,
    live_context: dict[str, Any] | None,
    prediction_context: dict[str, Any] | None = None,
) -> str:
    live = json.dumps(live_context or {}, ensure_ascii=True)
    convo = []
    for m in messages[-10:]:
        convo.append(f"{m.role.upper()}: {m.content}")
    convo_text = "\n".join(convo)

    latest_user_query = ""
    for m in reversed(messages):
        if m.role == "user":
            latest_user_query = m.content
            break

    q_norm = _normalize_prompt(latest_user_query)
    wants_lap_summary = any(
        k in q_norm
        for k in (
            "lap summary", "summarize lap", "summary of lap",
            "give me a lap summary", "race summary",
        )
    )

    if (category or "").strip().lower() == "strategy":
        # ── Strategy path: use physics-based prediction output if available ──
        if prediction_context and prediction_context.get("snapshot"):
            pred_json = json.dumps(prediction_context, ensure_ascii=True, indent=2)
            return (
                "You are an F1 strategy engineer. A physics-based race simulation engine has already run "
                "the relevant scenario and produced the COMPUTED NUMBERS below.\n"
                "Your job is to interpret these numbers and explain what they mean in race-engineering terms.\n\n"
                "RULES:\n"
                "- Use the computed numbers as your primary source. Do NOT invent numbers.\n"
                "- Always use full driver names (e.g. 'Charles Leclerc', not 'LEC').\n"
                "- Be specific: cite gap values, tyre ages, lap counts, and time deltas from the data.\n"
                "- End your response with a clear RECOMMENDATION and KEY RISK.\n"
                "- If the error field is non-null, explain what data was missing.\n\n"
                "FIELD GLOSSARY:\n"
                "- net_time_cost_s: negative = pit is beneficial overall; positive = net loss\n"
                "- gap_trend_3lap_s: negative = closing on the car ahead\n"
                "- laps_past_cliff: laps beyond the tyre's performance cliff\n"
                "- net_pit_benefit_s (SC scenarios): positive = driver should pit under SC\n\n"
                f"SIMULATION OUTPUT:\n{pred_json}\n\n"
                f"CONVERSATION:\n{convo_text}\n\n"
                "Now give your race-engineer analysis."
            )
        else:
            # Fallback: compact strategy state when prediction engine has no data
            strategy_state = json.dumps(
                _compact_strategy_state(live_context), ensure_ascii=True
            )
            return (
                "You are an F1 strategy engineer assistant.\n"
                "Generate scenario-based, conditional what-if predictions from the provided live strategy state.\n"
                "Use concise, professional race-engineering language.\n"
                "Always use full driver names (e.g. 'Max Verstappen' not 'VER').\n"
                "If uncertainty is high, state assumptions explicitly.\n"
                "Terminology: gap_trend_3lap_s = gap change over last 3 laps (negative = closing).\n"
                "End with: RECOMMENDATION and KEY RISK.\n\n"
                f"Strategy state:\n{strategy_state}\n\n"
                f"Conversation:\n{convo_text}\n\n"
                "Now produce a what-if strategy answer."
            )

    summary_hint = (
        "If the user asks for a lap summary, provide a clear race-engineering summary covering: "
        "current race state, top order snapshot, notable events, and one-line implication for next laps.\n\n"
        if wants_lap_summary
        else ""
    )

    # ── Biographical / global knowledge queries ───────────────────────────────
    # For questions about a driver's personal facts (age, bio, career stats),
    # live telemetry is noise — strip it so the model focuses on its own knowledge.
    if _is_global_knowledge_query(latest_user_query):
        return (
            "You are AI Pit Crew, a Formula 1 expert assistant.\n"
            "Answer the user's question using your own verified knowledge of F1 history, "
            "drivers, teams, and statistics.\n"
            "Do NOT make up statistics. If you are unsure, say so clearly.\n"
            "Always use full driver names (e.g. 'Max Verstappen' not 'VER').\n"
            "Cite relevant retrieved context as [1], [2] if available.\n\n"
            f"Retrieved context:\n{retrieved_context or 'No context found.'}\n\n"
            f"Conversation:\n{convo_text}\n\n"
            "Provide a concise, factual answer."
        )

    return (
        "You are AI Pit Crew, a Formula 1 race analyst.\n"
        "Answer using the retrieved context, live context, and your own verified knowledge of F1 history.\n"
        "For historical questions (race winners, pole positions, championships), use your own knowledge first.\n"
        "Always use full driver names (e.g. 'Lewis Hamilton' not 'HAM').\n"
        "If data is missing, say so clearly. Cite relevant chunks as [1], [2], etc.\n\n"
        f"{summary_hint}"
        f"Live context:\n{live}\n\n"
        f"Retrieved context:\n{retrieved_context or 'No context found.'}\n\n"
        f"Conversation:\n{convo_text}\n\n"
        "Provide a concise, factual answer."
    )


def _strategy_local_fallback(
    *,
    query: str,
    live_context: dict[str, Any] | None,
) -> str:
    """
    Fallback when all LLM providers are unavailable.
    Uses the prediction engine for data-grounded output, not keyword templates.
    """
    # Try to import prediction engine — it may not be installed yet
    try:
        from src.backend.services.prediction_engine import run_prediction
        live = live_context or {}
        session_id = live.get("session_id") or ""
        pred = run_prediction(query, session_id, live)

        if pred.get("error") or not pred.get("snapshot"):
            raise RuntimeError(pred.get("error", "No data"))

        snap_data = pred["snapshot"]
        scenario = pred["scenario"] or {}
        scenario_type = pred["scenario_type"]
        nums = scenario.get("key_numbers", {})

        lines = [
            f"[Local prediction — {scenario_type.replace('_', ' ').upper()}]",
            f"{snap_data['race_name']} · Lap {snap_data['current_lap']}/{snap_data['total_laps']} "
            f"· {snap_data['laps_remaining']} to go · Pit loss: {snap_data['pit_loss_s']}s",
            "",
        ]

        if scenario_type == "safety_car":
            lines += [
                f"SC type: {nums.get('sc_type')} · Compression: {nums.get('gap_compression_total_s')}s total",
                f"Pit loss under SC: {nums.get('pit_loss_under_sc_s')}s (normal: {nums.get('pit_loss_normal_s')}s)",
            ]
            should_pit = nums.get("drivers_should_pit") or []
            lines.append(
                f"Should pit: {', '.join(should_pit)}" if should_pit
                else "No driver benefits from pitting under these conditions."
            )
            for pc in (nums.get("position_changes") or [])[:4]:
                lines.append(
                    f"  {pc['driver']}: P{pc['from']} → P{pc['to']} "
                    f"({'+'if pc['delta']>0 else ''}{pc['delta']})"
                )

        elif scenario_type == "pit_stop":
            lines += [
                f"Driver: {nums.get('driver')} P{nums.get('current_position')} · "
                f"{nums.get('current_tyre')} age {nums.get('current_tyre_age_laps')} laps",
                f"Onto {nums.get('new_compound')} · Pit loss: {nums.get('pit_loss_s')}s · "
                f"Tyre benefit: {nums.get('tyre_time_benefit_s')}s · Net: {nums.get('net_time_cost_s')}s",
                f"Rejoin: P{nums.get('rejoin_position')}",
                f"Verdict: {nums.get('verdict')}",
            ]
            if nums.get("undercut_threat"):
                ut = nums["undercut_threat"]
                lines.append(
                    f"Undercut risk from {ut['from_driver']}: {ut['threat_level']} "
                    f"(gap {ut.get('current_gap_behind_s')}s)"
                )

        elif scenario_type == "tyre_degradation":
            for w in (nums.get("cliff_warnings") or []):
                lines.append(f"  WARNING: {w}")
            for name, proj in list((nums.get("driver_projections") or {}).items())[:4]:
                times = proj.get("projected_lap_times_next_5") or []
                lines.append(
                    f"{name} P{proj['position']} · {proj['tyre']} age {proj['tyre_age_now']} · "
                    f"status: {proj['status']}"
                )
                if times:
                    lines.append(f"  Next 5: {' / '.join(str(t) for t in times[:5])}")

        elif scenario_type == "overtake_window":
            lines += [
                f"{nums.get('attacker')} P{nums.get('attacker_position')} vs "
                f"{nums.get('defender')} P{nums.get('defender_position')}",
                f"Gap: {nums.get('current_gap_s')}s · Pace delta: {nums.get('pace_delta_s_per_lap')}s/lap",
                f"Overtake probability: {int((nums.get('overtake_probability') or 0)*100)}% — "
                f"{nums.get('verdict')}",
            ]

        assumptions = scenario.get("assumptions") or []
        if assumptions:
            lines += ["", "Assumptions: " + "; ".join(assumptions[:3])]

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("Prediction engine fallback failed: %s", exc)

    # Hard fallback when prediction engine itself is unavailable
    live = live_context or {}
    lap = int(live.get("current_lap") or 0)
    total_laps = int(live.get("total_laps") or 0)
    leader = _DRIVER_CODE_MAPPING.get(str(live.get("leader") or ""), str(live.get("leader") or "--"))
    laps_left = max(0, total_laps - lap)
    return (
        f"Strategy analysis (offline) — lap {lap}/{total_laps or '--'}, {laps_left} to go.\n"
        f"Leader: {leader}.\n"
        "Detailed simulation unavailable — LLM and prediction engine both offline.\n"
        "Recommendation: monitor gap trend and tyre age before committing to a stop."
    )


# ── Provider: Ollama (local) ─────────────────────────────────────────────────

async def _ollama_complete(prompt: str) -> str:
    """
    Call the local Ollama server.
    Uses the OpenAI-compatible /api/chat endpoint so the same
    message format works as the cloud providers.
    Raises RuntimeError if Ollama is unreachable or returns an error.
    """
    global _OLLAMA_LAST_FAILURE

    # Skip quickly if Ollama recently failed
    if time.time() - _OLLAMA_LAST_FAILURE < _OLLAMA_SKIP_WINDOW:
        raise RuntimeError(
            f"Ollama skipped (failed {int(time.time() - _OLLAMA_LAST_FAILURE)}s ago)"
        )

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.15,
            "num_predict": 1024,    # max tokens to generate
        },
    }

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_s) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = (
                (data.get("message") or {}).get("content", "")
                or data.get("response", "")
            ).strip()
            if not content:
                raise RuntimeError("Ollama returned empty content")
            # Successful call — reset failure timestamp
            _OLLAMA_LAST_FAILURE = 0.0
            return content
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        _OLLAMA_LAST_FAILURE = time.time()
        raise RuntimeError(f"Ollama unreachable: {exc}") from exc
    except Exception as exc:
        _OLLAMA_LAST_FAILURE = time.time()
        raise RuntimeError(f"Ollama error: {exc}") from exc


# ── Provider: Gemini direct API ──────────────────────────────────────────────

async def _gemini_complete(prompt: str) -> str:
    """Call Google Gemini API directly."""
    root_env = Path(__file__).resolve().parent.parent.parent / ".env"
    env_vals = dotenv_values(root_env) if root_env.exists() else {}
    gemini_key = (
        settings.gemini_api_key
        or os.environ.get("F1VIZ_GEMINI_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
        or str(env_vals.get("F1VIZ_GEMINI_API_KEY", "") or "")
    )
    if not gemini_key:
        raise RuntimeError("Gemini API key is not configured.")

    gemini_model = settings.gemini_model or "gemini-2.0-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{gemini_model}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.15},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, params={"key": gemini_key}, json=payload)
        if resp.status_code == 429:
            cooldown_seconds = await _register_llm_rate_limit(resp)
            raise RuntimeError(
                f"Gemini rate limit reached. Cooling down for {int(cooldown_seconds)}s."
            )
        resp.raise_for_status()
        data = resp.json()
        await _reset_llm_backoff()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (
            ((candidates[0] or {}).get("content") or {}).get("parts") or []
        )
        return "".join(
            str(p.get("text", "")) for p in parts if isinstance(p, dict)
        ).strip()


# ── Provider: OpenRouter ──────────────────────────────────────────────────────

async def _openrouter_complete(prompt: str) -> str:
    """Call OpenRouter API (cloud fallback of last resort)."""
    root_env = Path(__file__).resolve().parent.parent.parent / ".env"
    env_vals = dotenv_values(root_env) if root_env.exists() else {}
    api_key = (
        settings.openrouter_api_key
        or settings.openrouter_api_key_unprefixed
        or os.environ.get("F1VIZ_OPENROUTER_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
        or str(env_vals.get("F1VIZ_OPENROUTER_API_KEY", "") or "")
    )
    if not api_key:
        raise RuntimeError("OpenRouter API key is not configured.")

    model = settings.chat_model or "google/gemini-2.0-flash-exp:free"
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Christen24/F1-viz",
        "X-Title": "F1-Viz Platform",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 429:
                    cooldown_seconds = await _register_llm_rate_limit(resp)
                    raise RuntimeError(
                        f"OpenRouter rate limit reached. Cooling down for {int(cooldown_seconds)}s."
                    )
                resp.raise_for_status()
                data = resp.json()
                await _reset_llm_backoff()
                choices = data.get("choices", [])
                if choices:
                    return str(
                        choices[0].get("message", {}).get("content", "")
                    ).strip()
                return ""
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < 2:
                    await asyncio.sleep(min(4, 2 ** attempt))
                    continue
                raise
            except RuntimeError:
                raise
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(min(4, 2 ** attempt))
                    continue
                raise
    return ""


# ── Main provider orchestrator ───────────────────────────────────────────────

async def _generate_llm_response(
    prompt: str,
    *,
    prefer_gemini: bool = False,
    skip_ollama: bool = False,
) -> tuple[str, str]:
    """
    Try providers in order: Ollama → OpenRouter → Gemini.
    Returns (answer_text, provider_name).

    prefer_gemini=True  → try Gemini before OpenRouter (but still after Ollama).
    skip_ollama=True    → bypass Ollama entirely (used for history/global queries
                          where local model knowledge is insufficient).
    """
    # Check API-provider cooldown (does not apply to Ollama)
    cooldown_until, _ = await asyncio.to_thread(_get_limit_state)
    api_on_cooldown = cooldown_until > time.time()

    # ── 1. Ollama (local, primary) ────────────────────────────────────────
    if settings.ollama_enabled and not skip_ollama:
        try:
            answer = await _ollama_complete(prompt)
            logger.info("LLM response from Ollama (%s)", settings.ollama_model)
            return answer, "ollama"
        except RuntimeError as exc:
            logger.info("Ollama unavailable, trying cloud fallback: %s", exc)

    # ── 2. Cloud Fallback (OpenRouter or Gemini) ───────────────────────────
    providers = ["openrouter", "gemini"]
    if prefer_gemini:
        providers = ["gemini", "openrouter"]

    if not api_on_cooldown:
        for p in providers:
            try:
                if p == "gemini":
                    answer = await _gemini_complete(prompt)
                    logger.info("LLM response from Gemini")
                    return answer, "gemini"
                else:
                    answer = await _openrouter_complete(prompt)
                    logger.info("LLM response from OpenRouter")
                    return answer, "openrouter"
            except RuntimeError as exc:
                if "rate limit" in str(exc).lower() or "cooldown" in str(exc).lower():
                    # If this is the last provider in the loop, re-raise to trigger cooldown error below
                    if p == providers[-1]:
                        raise
                    logger.warning("%s on cooldown/rate-limited. Trying fallback.", p)
                else:
                    logger.warning("%s failed: %s. Trying fallback.", p, exc)

    remaining = int(cooldown_until - time.time())
    raise RuntimeError(f"All providers on cooldown ({remaining}s remaining).")


@router.post("/stream")
async def stream_chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    user_prompt = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), None
    )
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    async def event_stream() -> AsyncIterator[str]:
        try:
            is_strategy_mode = (req.category or "").strip().lower() == "strategy"
            global_only_query = (not is_strategy_mode) and _is_global_knowledge_query(
                user_prompt
            )
            direct_historical_answer: dict[str, Any] | None = None

            if global_only_query and _HISTORICAL_WINNER_RE.search(
                _normalize_prompt(user_prompt)
            ):
                try:
                    direct = answer_from_json(
                        query=user_prompt,
                        session_id=req.session_id,
                        live_context=req.live_context,
                    )
                    direct_sources = direct.get("sources") or []
                    if any(
                        str(s.get("category", "")).lower() == "historical_result"
                        for s in direct_sources
                    ):
                        direct_historical_answer = direct
                except Exception:
                    direct_historical_answer = None

            if direct_historical_answer is not None:
                answer = str(direct_historical_answer.get("answer") or "").strip()
                sources = direct_historical_answer.get("sources") or []
                yield _sse_event({"type": "sources", "sources": sources})
                yield _sse_event({"type": "delta", "text": answer})
                yield _sse_event({"type": "done"})
                return
            elif global_only_query and _HISTORICAL_WINNER_RE.search(
                _normalize_prompt(user_prompt)
            ):
                try:
                    wiki = await retrieve_wikipedia_context(user_prompt)
                    wiki_context = str(wiki.get("context") or "").strip()
                    if wiki_context:
                        lines = [
                            ln.strip() for ln in wiki_context.splitlines() if ln.strip()
                        ]
                        answer = lines[1] if len(lines) > 1 else lines[0]
                        yield _sse_event({"type": "sources", "sources": wiki.get("sources", [])})
                        yield _sse_event({"type": "delta", "text": answer})
                        yield _sse_event({"type": "done"})
                        return
                except Exception:
                    pass

            _fastf1_hint = ""
            if global_only_query and _HISTORICAL_WINNER_RE.search(
                _normalize_prompt(user_prompt)
            ):
                direct = answer_from_json(
                    query=user_prompt,
                    session_id=req.session_id,
                    live_context=req.live_context,
                )
                direct_sources = direct.get("sources") or []
                if any(
                    str(s.get("category", "")).lower() == "historical_result"
                    for s in direct_sources
                ):
                    _fastf1_hint = (
                        f"[FastF1 cached data — may be stale, verify before using]: "
                        f"{direct.get('answer', '')}"
                    )

            ckey = _cache_key(
                session_id=req.session_id,
                user_prompt=user_prompt,
                category=req.category,
                live_context=req.live_context,
            )
            cached = _cache_get(ckey)
            if cached:
                yield _sse_event({"type": "sources", "sources": []})
                yield _sse_event({"type": "delta", "text": cached})
                yield _sse_event({"type": "done"})
                return

            llm_requested = bool(req.allow_llm)
            direct_strategy_mode = is_strategy_mode and llm_requested

            if direct_strategy_mode:
                rag: dict[str, Any] = {"context": "", "sources": []}
            elif is_strategy_mode:
                rag = retrieve_local_context(
                    query=user_prompt,
                    session_id=req.session_id,
                    top_k=req.top_k,
                )
            else:
                rag = await _retrieve_pitcrew_context(
                    query=user_prompt,
                    session_id=req.session_id,
                    top_k=req.top_k,
                    season=req.season,
                    event_name=req.event_name,
                    global_only=global_only_query,
                )
            yield _sse_event({"type": "sources", "sources": rag.get("sources", [])})

            rag_context = rag.get("context", "")
            if _fastf1_hint:
                rag_context = (
                    f"{_fastf1_hint}\n\n{rag_context}" if rag_context else _fastf1_hint
                )

            # ── Prediction engine (strategy mode only) ────────────────────
            prediction_context: dict[str, Any] | None = None
            if is_strategy_mode and req.session_id:
                try:
                    from src.backend.services.prediction_engine import run_prediction
                    prediction_context = run_prediction(
                        query=user_prompt,
                        session_id=req.session_id,
                        live_context=req.live_context or {},
                    )
                    if prediction_context.get("error"):
                        logger.info(
                            "Prediction engine: %s", prediction_context["error"]
                        )
                        prediction_context = None
                except Exception as pred_exc:
                    logger.warning("Prediction engine failed: %s", pred_exc)
                    prediction_context = None

            prompt = _build_llm_prompt(
                retrieved_context=rag_context,
                messages=req.messages,
                category=req.category,
                live_context=req.live_context,
                prediction_context=prediction_context,
            )

            query_wants_llm = (
                True
                if (is_strategy_mode and llm_requested)
                else (
                    _should_use_llm_strategy(user_prompt)
                    if is_strategy_mode
                    else _should_use_llm(user_prompt)
                )
            )

            # Budget check only applies to API providers, not Ollama
            budget_ok = True
            budget_reason: str | None = None
            if llm_requested and query_wants_llm:
                # If Ollama is enabled and healthy, skip budget gate
                ollama_healthy = (
                    settings.ollama_enabled
                    and settings.ollama_skip_rate_limit_tracking
                    and (time.time() - _OLLAMA_LAST_FAILURE >= _OLLAMA_SKIP_WINDOW)
                )
                if not ollama_healthy:
                    budget_ok, budget_reason = _llm_budget_check(req.session_id)

            cooldown_until, streak = await asyncio.to_thread(_get_limit_state)
            if (
                llm_requested
                and query_wants_llm
                and budget_ok
                and cooldown_until > time.time()
                and not settings.ollama_enabled
            ):
                remaining = int(max(1, round(cooldown_until - time.time())))
                budget_ok = False
                budget_reason = f"provider_cooldown_{remaining}s"

            if llm_requested and query_wants_llm and is_strategy_mode and budget_ok:
                interval_ok, interval_reason = _strategy_llm_interval_ok(req.session_id)
                if not interval_ok:
                    budget_ok = False
                    budget_reason = interval_reason

            use_llm = (
                settings.chat_allow_llm
                and (is_strategy_mode or (not settings.chat_force_local_rag))
                and llm_requested
                and query_wants_llm
                and budget_ok
            )

            if use_llm:
                try:
                    # Global knowledge queries skip Ollama (weak on F1 history)
                    skip_ollama = global_only_query
                    answer, provider = await _generate_llm_response(
                        prompt,
                        prefer_gemini=is_strategy_mode,
                        skip_ollama=skip_ollama,
                    )
                    if not answer.strip():
                        raise RuntimeError("Empty response from LLM")

                    # Only count against rate-limit budget for API providers
                    if provider != "ollama":
                        _record_llm_call(req.session_id)

                    if is_strategy_mode:
                        _LLM_LAST_STRATEGY_CALL_BY_SESSION[
                            req.session_id or "__global__"
                        ] = time.time()

                except Exception as exc:
                    logger.warning(
                        "LLM generation unavailable (%s), using local fallback", exc
                    )
                    if is_strategy_mode:
                        answer = _strategy_local_fallback(
                            query=user_prompt,
                            live_context=req.live_context,
                        )
                    else:
                        answer = answer_from_local_rag(
                            query=user_prompt,
                            retrieval=rag,
                            live_context=None if global_only_query else req.live_context,
                        )
            else:
                logger.debug("Using local RAG / prediction fallback")
                if is_strategy_mode:
                    answer = _strategy_local_fallback(
                        query=user_prompt,
                        live_context=req.live_context,
                    )
                else:
                    answer = answer_from_local_rag(
                        query=user_prompt,
                        retrieval=rag,
                        live_context=None if global_only_query else req.live_context,
                    )

            if not answer.strip():
                if global_only_query:
                    if _HISTORICAL_WINNER_RE.search(_normalize_prompt(user_prompt)):
                        answer = (
                            "I couldn't verify that winner from authoritative sources. "
                            "Please try the full event name, e.g. "
                            "'who won the 2024 São Paulo Grand Prix'."
                        )
                    else:
                        try:
                            answer_text, _ = await _generate_llm_response(
                                f"The user asked: '{user_prompt}'. "
                                "Answer based on your knowledge of Formula 1. "
                                "Be concise and factual.",
                                skip_ollama=True,  # historical — skip local model
                            )
                            answer = answer_text
                        except Exception:
                            answer = (
                                "I don't have enough specific data to answer that accurately. "
                                "Please try rephrasing or asking about a different event."
                            )
                else:
                    local = answer_from_json(
                        query=user_prompt,
                        session_id=req.session_id,
                        live_context=req.live_context,
                    )
                    answer = local.get("answer", "")

            if answer.strip():
                _cache_set(ckey, answer)

            yield _sse_event({"type": "delta", "text": answer})
            yield _sse_event({"type": "done"})

        except RuntimeError as e:
            if "cooldown" in str(e).lower() or "limit reached" in str(e).lower():
                yield _sse_event({"type": "error", "message": str(e)})
                return
            logger.error("RuntimeError in event_stream: %s", e)
            yield _sse_event({"type": "error", "message": str(e)})
            return
        except Exception as exc:
            logger.exception("Chat stream failed")
            yield _sse_event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
