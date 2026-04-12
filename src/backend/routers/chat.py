"""
Chat router with SSE streaming for RAG responses.
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
from typing import Any, AsyncIterator, Literal

import httpx
from dotenv import dotenv_values
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.backend.config import settings
from src.backend.services.local_rag import retrieve_local_context, answer_from_local_rag
from src.backend.services.json_chat import answer_from_json
from src.backend.services.rag import f1_knowledge
from src.backend.services.wiki_fallback import retrieve_wikipedia_context

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
_LLM_COOLDOWN_UNTIL = 0.0
_LLM_RATE_LIMIT_STREAK = 0
_LLM_COOLDOWN_BASE_SECONDS = 8.0
_LLM_COOLDOWN_MAX_SECONDS = 120.0
_ANSWER_CACHE: dict[str, tuple[float, str]] = {}
_ANSWER_CACHE_TTL_SECONDS = float(max(1, settings.chat_answer_cache_ttl_seconds))
_ANSWER_CACHE_VERSION = "v2"
_LLM_CALLS_WINDOW = 60.0
_LLM_CALLS_GLOBAL: deque[float] = deque()
_LLM_CALLS_BY_SESSION: dict[str, deque[float]] = defaultdict(deque)
_GLOBAL_RAG_DISABLED_REASON: str | None = None
_HISTORICAL_WINNER_RE = re.compile(r"\bwho\s+won\b.*\b20\d{2}\b", flags=re.IGNORECASE)

_DRIVER_CODE_MAPPING = {
    # 2024 / 2025 Grid
    "VER": "Max Verstappen",
    "PER": "Sergio Perez",
    "HAM": "Lewis Hamilton",
    "RUS": "George Russell",
    "LEC": "Charles Leclerc",
    "SAI": "Carlos Sainz",
    "NOR": "Lando Norris",
    "PIA": "Oscar Piastri",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",
    "OCO": "Esteban Ocon",
    "ALB": "Alexander Albon",
    "SAR": "Logan Sargeant",
    "TSU": "Yuki Tsunoda",
    "RIC": "Daniel Ricciardo",
    "LAW": "Liam Lawson",
    "HUL": "Nico Hulkenberg",
    "MAG": "Kevin Magnussen",
    "BOT": "Valtteri Bottas",
    "ZHO": "Guanyu Zhou",
    "BEA": "Oliver Bearman",
    "COL": "Franco Colapinto",
    "ANT": "Andrea Kimi Antonelli",
    "BOR": "Gabriel Bortoleto",
    "HAD": "Isack Hadjar",
    # Historical / Recent
    "VET": "Sebastian Vettel",
    "RAI": "Kimi Räikkönen",
    "MSC": "Michael Schumacher",
    "ROS": "Nico Rosberg",
    "BUT": "Jenson Button",
    "MAS": "Felipe Massa",
    "WEB": "Mark Webber",
    "GRO": "Romain Grosjean",
    "KVY": "Daniil Kvyat",
}


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
    if per_session and len(per_session) >= max(1, settings.chat_llm_max_calls_per_session_per_minute):
        return False, "session_llm_budget_reached"

    return True, None


def _record_llm_call(session_id: str | None) -> None:
    now = time.time()
    sid = session_id or "__global__"
    _LLM_CALLS_GLOBAL.append(now)
    _LLM_CALLS_BY_SESSION[sid].append(now)


def _extract_retry_after_seconds(resp: httpx.Response) -> float | None:
    raw = (resp.headers.get("Retry-After") or resp.headers.get("retry-after") or "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            try:
                dt = parsedate_to_datetime(raw)
                delta = (dt.timestamp() - time.time())
                if delta > 0:
                    return delta
            except Exception:
                pass

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    details = ((payload.get("error") or {}).get("details") or [])
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


def _register_llm_rate_limit(resp: httpx.Response) -> float:
    global _LLM_COOLDOWN_UNTIL
    global _LLM_RATE_LIMIT_STREAK

    retry_after = _extract_retry_after_seconds(resp)
    exp_backoff = min(
        _LLM_COOLDOWN_MAX_SECONDS,
        _LLM_COOLDOWN_BASE_SECONDS * (2 ** min(_LLM_RATE_LIMIT_STREAK, 5)),
    )
    cooldown_seconds = max(retry_after or 0.0, exp_backoff)
    _LLM_COOLDOWN_UNTIL = time.time() + cooldown_seconds
    _LLM_RATE_LIMIT_STREAK = min(_LLM_RATE_LIMIT_STREAK + 1, 12)
    return cooldown_seconds


def _reset_llm_backoff() -> None:
    global _LLM_COOLDOWN_UNTIL
    global _LLM_RATE_LIMIT_STREAK
    _LLM_COOLDOWN_UNTIL = 0.0
    _LLM_RATE_LIMIT_STREAK = 0


def _should_use_llm(query: str) -> bool:
    """
    Keep easy race-stat intents on local RAG to reduce provider calls.
    Use LLM only for comparative/strategic/explanatory asks.
    """
    q = _normalize_prompt(query)
    lap_summary_tokens = [
        "lap summary",
        "summarize lap",
        "summary of lap",
        "give me a lap summary",
        "race summary",
    ]
    simple_tokens = [
        "who is leading",
        "leader",
        "summary",
        "position",
        "speed",
        "where is",
        "how is",
        "top 3",
        "top 5",
        "pit stop",
    ]
    complex_tokens = [
        "why",
        "explain",
        "compare",
        "strategy",
        "predict",
        "insight",
        "analyze",
        "analysis",
        "what if",
    ]
    # Historical winner intents should stay deterministic to avoid hallucinations.
    if _HISTORICAL_WINNER_RE.search(q):
        return False
    # Prefer GenAI for richer summary explanations when requested.
    if any(k in q for k in lap_summary_tokens):
        return True
    if _is_global_knowledge_query(query):
        return True
    if any(k in q for k in complex_tokens):
        return True
    if any(k in q for k in simple_tokens):
        return False
    # default conservative behavior for free-tier usage
    return False


def _should_use_llm_strategy(query: str) -> bool:
    q = _normalize_prompt(query)
    if any(
        k in q
        for k in (
            "what if",
            "best strategy",
            "optimize",
            "simulate",
            "probability",
            "tradeoff",
            "compare",
            "expected",
            "should we",
            "should i",
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

        for source in (payload.get("sources") or []):
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

    # Local (active race/session) retrieval
    local_payload = {"context": "", "sources": []}
    if not global_only:
        local_payload = retrieve_local_context(
            query=query,
            session_id=session_id,
            top_k=max(2, min(top_k, 6)),
        )

    # Global historical knowledge retrieval (Wikipedia/docs/race reports)
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
            logger.info("Global RAG disabled for this backend run: %s", _GLOBAL_RAG_DISABLED_REASON)

    # If global KB is empty/missing for this query, attempt live Wikipedia fallback.
    if global_only and not str(global_payload.get("context") or "").strip():
        try:
            wiki_payload = await retrieve_wikipedia_context(query)
            if str(wiki_payload.get("context") or "").strip():
                global_payload = wiki_payload
        except Exception as exc:
            logger.debug("Wikipedia live fallback failed: %s", exc)

    return _merge_rag_payloads(local_payload, global_payload)


def _is_global_knowledge_query(query: str) -> bool:
    q = _normalize_prompt(query)
    if _HISTORICAL_WINNER_RE.search(q):
        return True
    global_tokens = (
        "born",
        "birth",
        "nationality",
        "biography",
        "bio",
        "where is",
        "where was",
        "which team",
        "team principal",
        "headquarters",
        "founded",
        "champion",
        "world champion",
        "constructor champion",
    )
    if any(tok in q for tok in global_tokens):
        return True
    if re.search(r"\b20\d{2}\b", q) and any(tok in q for tok in ("who won", "winner", "podium")):
        return True
    return False


def _compact_strategy_state(live_context: dict[str, Any] | None) -> dict[str, Any]:
    live = live_context or {}
    leader = str(live.get("leader") or "")
    current_lap = int(live.get("current_lap") or 0)
    total_laps = int(live.get("total_laps") or 0)

    positions_raw = live.get("positions") or {}
    gaps_raw = live.get("gaps") or {}
    tyres_raw = live.get("tyres") or {}
    avg_speed_raw = live.get("avg_speed") or {}

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
                "tyre": str(tyre_val) if tyre_val is not None else None,
                "avg_speed_kph": round(float(speed_val), 1) if isinstance(speed_val, (int, float)) else None,
            }
        )

    # Enhance top3 with names
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


def _build_llm_prompt(
    *,
    retrieved_context: str,
    messages: list[ChatMessage],
    category: str | None,
    live_context: dict[str, Any] | None,
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
            "lap summary",
            "summarize lap",
            "summary of lap",
            "give me a lap summary",
            "race summary",
        )
    )
    if (category or "").strip().lower() == "strategy":
        strategy_state = json.dumps(_compact_strategy_state(live_context), ensure_ascii=True)
        return (
            "You are an F1 strategy engineer assistant.\n"
            "Generate scenario-based, conditional what-if predictions from the provided live strategy state.\n"
            "Use concise, professional race-engineering language.\n"
            "Always use full proper names for drivers (e.g., 'Max Verstappen' instead of 'VER') for clarity.\n"
            "If uncertainty is high, state assumptions explicitly.\n"
            "Ground your answer in the provided state only and avoid fabricating exact telemetry.\n"
            "End with: recommendation + key risk.\n\n"
            f"Strategy state:\n{strategy_state}\n\n"
            f"Conversation:\n{convo_text}\n\n"
            "Now produce a what-if strategy answer tailored to the user's question."
        )

    summary_hint = (
        "If the user asks for a lap summary, provide a clear race-engineering summary with:\n"
        "1) current race state (lap progress, leader),\n"
        "2) top order snapshot,\n"
        "3) notable lap events/pit impact,\n"
        "4) one-line implication for next laps.\n"
        "Avoid generic filler and keep it concise.\n\n"
        if wants_lap_summary
        else ""
    )

    return (
        "You are AI Pit Crew, a Formula 1 race analyst.\n"
        "Answer using the retrieved context, live context, AND your own verified knowledge of F1 history.\n"
        "Retrieved context may include both active-race data and global historical F1 knowledge.\n"
        "IMPORTANT: If retrieved context contains '[FastF1 cached data — may be stale]', "
        "cross-check it against your own knowledge. If your knowledge conflicts with FastF1 data, "
        "prefer your own verified knowledge and note the discrepancy.\n"
        "For historical questions (race winners, pole positions, championships), use your own knowledge "
        "as the primary source and only supplement with retrieved context.\n"
        "Always use full proper names for drivers (e.g., 'Lewis Hamilton' instead of 'HAM') in your responses.\n"
        "If data is missing, say that clearly.\n"
        "Cite relevant chunks as [1], [2], etc.\n\n"
        f"{summary_hint}"
        f"Live context:\n{live}\n\n"
        f"Retrieved context:\n{retrieved_context or 'No context found.'}\n\n"
        f"Conversation:\n{convo_text}\n\n"
        "Now provide a concise, factual answer."
    )


def _strategy_local_fallback(
    *,
    query: str,
    live_context: dict[str, Any] | None,
) -> str:
    q = (query or "").lower()
    live = live_context or {}
    lap = int(live.get("current_lap") or 0)
    total_laps = int(live.get("total_laps") or 0)
    leader = str(live.get("leader") or "--")
    top3 = live.get("top3") or []
    top3_txt = ", ".join(str(x) for x in top3) if isinstance(top3, list) and top3 else "N/A"
    laps_left = max(0, total_laps - lap) if total_laps > 0 and lap > 0 else 0

    header = f"Strategy fallback mode (local) - lap {lap}/{total_laps if total_laps else '--'}."

    if "safety car" in q or "sc" in q:
        return (
            f"{header}\n"
            f"Scenario: Safety Car now.\n"
            f"Recommendation: prepare immediate pit-window evaluation for leaders; reduced pit-loss can justify stop timing.\n"
            f"Key risk: restart volatility and position shuffle in top order ({top3_txt})."
        )

    if any(k in q for k in ("crash", "dnf", "retire", "retirement", "out of race")):
        return (
            f"{header}\n"
            f"Scenario: potential retirement/crash event involving lead group (leader: {leader}).\n"
            f"Recommendation: immediately pivot to track-position defense for current P2/P3 and reassess pit windows under likely neutralization.\n"
            f"Key risk: incident timing uncertainty can invalidate pre-planned stop sequence and create sudden overcut/undercut swings."
        )

    if any(k in q for k in ("red flag", "vsc", "virtual safety car")):
        return (
            f"{header}\n"
            "Scenario: race control intervention (VSC/Red Flag).\n"
            "Recommendation: preserve optionality by delaying irreversible tyre commitments until restart/resume conditions are clear.\n"
            "Key risk: wrong restart tyre can cost multiple positions in first green-flag laps."
        )

    if "pit" in q:
        return (
            f"{header}\n"
            f"Scenario: pit decision for front-runners (leader: {leader}).\n"
            f"Recommendation: pit only if rejoin traffic is manageable and out-lap temperature window is favorable.\n"
            f"Key risk: track-position loss versus undercut gain uncertainty."
        )

    if any(k in q for k in ("tyre", "tire", "soft", "medium", "hard")):
        stint_hint = (
            "soft-leaning finish can be viable"
            if laps_left and laps_left <= 10
            else "balanced medium strategy is safer for consistency"
        )
        return (
            f"{header}\n"
            f"Scenario: tyre strategy with ~{laps_left} laps remaining.\n"
            f"Recommendation: {stint_hint}, prioritizing clean air over raw compound pace.\n"
            f"Key risk: degradation cliff if stint length exceeds target."
        )

    if any(k in q for k in ("rain", "wet", "inter")):
        return (
            f"{header}\n"
            "Scenario: weather shift contingency.\n"
            "Recommendation: keep a short call window for crossover tyres and avoid overcommitting to slick stint length.\n"
            "Key risk: delayed stop can cause major position loss in rapid grip transitions."
        )

    if any(k in q for k in ("damage", "front wing", "puncture", "engine", "power unit", "penalty")):
        return (
            f"{header}\n"
            "Scenario: performance-limiting issue (damage/penalty/reliability).\n"
            "Recommendation: shift to risk-minimizing strategy, prioritize clean air and fewer on-track fights until pace delta is known.\n"
            "Key risk: running normal attack strategy with reduced performance accelerates net position loss."
        )

    return (
        f"{header}\n"
        f"Scenario interpreted from query: \"{query.strip()}\".\n"
        f"Leader: {leader}; Top order: {top3_txt}; laps left: {laps_left}.\n"
        "Recommendation: optimize for flexible pit timing and clean-air track position until trigger certainty increases.\n"
        "Key risk: ambiguous triggers reduce confidence; define event timing/driver for higher-precision prediction."
    )


async def _generate_llm_response(prompt: str) -> str:
    """Generate LLM response via OpenRouter (OpenAI-compatible)."""
    # Rate limit check handles backoff internally
    now = time.time()
    last = _LLM_STATE["last_limit_time"]
    if now - last < LLM_BACKOFF_SECONDS:
        remaining = int(LLM_BACKOFF_SECONDS - (now - last))
        raise RuntimeError(f"OpenRouter temporarily on cooldown ({remaining}s remaining).")

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

    model = (
        settings.chat_model
        or os.environ.get("F1VIZ_CHAT_MODEL", "")
        or "google/gemini-2.0-flash-exp:free"
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Christen24/F1-viz",
        "X-Title": "F1-Viz Platform",
        "Content-Type": "application/json"
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
                    cooldown_seconds = _register_llm_rate_limit(resp)
                    raise RuntimeError(f"OpenRouter rate limit reached. Cooling down for {int(cooldown_seconds)}s.")
                
                resp.raise_for_status()
                data = resp.json()
                _reset_llm_backoff()
                
                choices = data.get("choices", [])
                if choices:
                    return str(choices[0].get("message", {}).get("content", "")).strip()
                return ""
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < 2:
                    await asyncio.sleep(min(4, 2 ** attempt))
                    continue
                raise
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(min(4, 2 ** attempt))
                    continue
                raise

    return ""


@router.post("/stream")
async def stream_chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    user_prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    async def event_stream() -> AsyncIterator[str]:
        try:
            is_strategy_mode = (req.category or "").strip().lower() == "strategy"
            global_only_query = (not is_strategy_mode) and _is_global_knowledge_query(user_prompt)
            direct_historical_answer: dict[str, Any] | None = None

            if global_only_query and _HISTORICAL_WINNER_RE.search(_normalize_prompt(user_prompt)):
                try:
                    direct = answer_from_json(
                        query=user_prompt,
                        session_id=req.session_id,
                        live_context=req.live_context,
                    )
                    direct_sources = direct.get("sources") or []
                    if any(str(s.get("category", "")).lower() == "historical_result" for s in direct_sources):
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
            elif global_only_query and _HISTORICAL_WINNER_RE.search(_normalize_prompt(user_prompt)):
                # Deterministic historical fallback via Wikipedia summary parser.
                try:
                    wiki = await retrieve_wikipedia_context(user_prompt)
                    wiki_context = str(wiki.get("context") or "").strip()
                    if wiki_context:
                        lines = [ln.strip() for ln in wiki_context.splitlines() if ln.strip()]
                        answer = lines[1] if len(lines) > 1 else lines[0]
                        yield _sse_event({"type": "sources", "sources": wiki.get("sources", [])})
                        yield _sse_event({"type": "delta", "text": answer})
                        yield _sse_event({"type": "done"})
                        return
                except Exception:
                    pass

            # Gather FastF1 hint as supplementary context (not authoritative)
            # The LLM will validate/override this using its own knowledge.
            _fastf1_hint = ""
            if global_only_query and _HISTORICAL_WINNER_RE.search(_normalize_prompt(user_prompt)):
                direct = answer_from_json(
                    query=user_prompt,
                    session_id=req.session_id,
                    live_context=req.live_context,
                )
                direct_sources = direct.get("sources") or []
                if any(str(s.get("category", "")).lower() == "historical_result" for s in direct_sources):
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
                rag = {"context": "", "sources": []}
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

            # Inject FastF1 supplementary hint if available
            rag_context = rag.get("context", "")
            if _fastf1_hint:
                rag_context = f"{_fastf1_hint}\n\n{rag_context}" if rag_context else _fastf1_hint

            prompt = _build_llm_prompt(
                retrieved_context=rag_context,
                messages=req.messages,
                category=req.category,
                live_context=req.live_context,
            )
            query_wants_llm = (
                True
                if (is_strategy_mode and llm_requested)
                else (_should_use_llm_strategy(user_prompt) if is_strategy_mode else _should_use_llm(user_prompt))
            )
            budget_ok = True
            budget_reason: str | None = None
            if llm_requested and query_wants_llm:
                budget_ok, budget_reason = _llm_budget_check(req.session_id)
            use_llm = (
                settings.chat_allow_llm
                and (is_strategy_mode or (not settings.chat_force_local_rag))
                and llm_requested
                and query_wants_llm
                and budget_ok
            )
            if llm_requested and query_wants_llm and not use_llm:
                logger.info(
                    "Skipping LLM (allow_llm=%s, request_allow_llm=%s, force_local=%s, budget_ok=%s, reason=%s)",
                    settings.chat_allow_llm,
                    llm_requested,
                    settings.chat_force_local_rag,
                    budget_ok,
                    budget_reason,
                )
            if use_llm:
                try:
                    answer = await _generate_llm_response(prompt)
                    if not answer.strip():
                        raise RuntimeError("Gemini returned empty content")
                    _record_llm_call(req.session_id)
                except Exception as exc:
                    logger.warning("LLM generation unavailable, falling back to local RAG answer: %s", exc)
                    if is_strategy_mode:
                        answer = (
                            "Prediction Model requires GenAI for scenario simulation right now, "
                            f"but the provider is unavailable ({exc}). "
                            "Please retry shortly."
                        )
                    else:
                        answer = answer_from_local_rag(
                            query=user_prompt,
                            retrieval=rag,
                            live_context=None if global_only_query else req.live_context,
                        )
            else:
                logger.debug("Using local RAG mode for query")
                if is_strategy_mode:
                    reason = (
                        "LLM budget reached" if not budget_ok else "LLM disabled by server configuration"
                    )
                    answer = (
                        "Prediction Model is configured for GenAI responses. "
                        f"Generation is currently unavailable ({reason}). Please retry shortly."
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
                            "I couldn't verify that winner from authoritative sources right now. "
                            "Please try the full event name (for example: 'who won the 2024 São Paulo Grand Prix')."
                        )
                    else:
                    # Final attempt: try LLM without restrictive RAG constraint if it was skipped or empty
                        try:
                            answer = await _generate_llm_response(
                                f"The user asked: '{user_prompt}'. I couldn't find specific documents in my RAG system. "
                                "Please answer based on your internal historical knowledge of Formula 1. "
                                "Be concise and factual."
                            )
                        except Exception:
                            answer = "I don't have enough specific data in my current database to answer that accurately. Please try rephrasing or asking about a different event."
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
