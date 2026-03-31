"""
Chat router with SSE streaming for RAG responses.
"""
from __future__ import annotations

import json
import logging
import os
import asyncio
import hashlib
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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
_LLM_COOLDOWN_UNTIL = 0.0
_LLM_COOLDOWN_SECONDS = 300.0
_ANSWER_CACHE: dict[str, tuple[float, str]] = {}
_ANSWER_CACHE_TTL_SECONDS = 20.0


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


def _should_use_llm(query: str) -> bool:
    """
    Keep easy race-stat intents on local RAG to reduce provider calls.
    Use LLM only for comparative/strategic/explanatory asks.
    """
    q = _normalize_prompt(query)
    simple_tokens = [
        "who is leading",
        "leader",
        "lap summary",
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
    if any(k in q for k in complex_tokens):
        return True
    if any(k in q for k in simple_tokens):
        return False
    # default conservative behavior for free-tier usage
    return False


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
    live_context: dict[str, Any] | None = None


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


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
    if (category or "").strip().lower() == "strategy":
        return (
            "You are an F1 strategy engineer assistant.\n"
            "Generate scenario-based, conditional what-if predictions from the provided race context.\n"
            "Use concise, practical race-engineering language.\n"
            "If uncertainty is high, state assumptions explicitly.\n"
            "Ground your answer in retrieved/live context only and avoid fabricating exact telemetry.\n"
            "End with: recommendation + key risk.\n\n"
            f"Live context:\n{live}\n\n"
            f"Retrieved context:\n{retrieved_context or 'No context found.'}\n\n"
            f"Conversation:\n{convo_text}\n\n"
            "Now produce a what-if strategy answer tailored to the user's question."
        )

    return (
        "You are AI Pit Crew, a Formula 1 race analyst.\n"
        "Answer strictly from the retrieved context and live context.\n"
        "If data is missing, say that clearly.\n"
        "Cite relevant chunks as [1], [2], etc.\n\n"
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

    return (
        f"{header}\n"
        f"Leader: {leader}; Top order: {top3_txt}.\n"
        "Recommendation: ask a specific what-if (Safety Car, pit timing, tyre switch, weather) for targeted prediction.\n"
        "Key risk: broad scenario questions reduce confidence without a defined trigger."
    )


async def _gemini_generate(prompt: str) -> str:
    global _LLM_COOLDOWN_UNTIL
    if time.time() < _LLM_COOLDOWN_UNTIL:
        remaining = int(_LLM_COOLDOWN_UNTIL - time.time())
        raise RuntimeError(f"Gemini temporarily on cooldown after rate-limit ({remaining}s remaining).")

    root_env = Path(__file__).resolve().parent.parent.parent / ".env"
    env_vals = dotenv_values(root_env) if root_env.exists() else {}
    gemini_key = (
        settings.gemini_api_key
        or settings.gemini_api_key_unprefixed
        or os.environ.get("F1VIZ_GEMINI_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
        or str(env_vals.get("F1VIZ_GEMINI_API_KEY", "") or "")
        or str(env_vals.get("GEMINI_API_KEY", "") or "")
    )
    if not gemini_key or "PASTE_GEMINI_API_KEY_HERE" in gemini_key:
        raise RuntimeError("Gemini API key is not configured.")
    gemini_model = (
        settings.gemini_model_unprefixed
        or settings.gemini_model
        or os.environ.get("F1VIZ_GEMINI_MODEL", "")
        or os.environ.get("GEMINI_MODEL", "")
        or str(env_vals.get("F1VIZ_GEMINI_MODEL", "") or "")
        or "gemini-2.0-flash"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {"temperature": 0.15},
    }
    headers = {"x-goog-api-key": gemini_key}

    data: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(2):
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                _LLM_COOLDOWN_UNTIL = time.time() + _LLM_COOLDOWN_SECONDS
                raise RuntimeError("Gemini rate limit/quota reached (HTTP 429).")
            if resp.status_code >= 500:
                if attempt == 1:
                    resp.raise_for_status()
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            break

    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(str(p.get("text", "")) for p in parts if p.get("text"))


@router.post("/stream")
async def stream_chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    user_prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    async def event_stream() -> AsyncIterator[str]:
        try:
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

            rag = retrieve_local_context(
                query=user_prompt,
                session_id=req.session_id,
                top_k=req.top_k,
            )
            yield _sse_event({"type": "sources", "sources": rag.get("sources", [])})

            prompt = _build_llm_prompt(
                retrieved_context=rag.get("context", ""),
                messages=req.messages,
                category=req.category,
                live_context=req.live_context,
            )
            is_strategy_mode = (req.category or "").strip().lower() == "strategy"
            use_llm = (not settings.chat_force_local_rag) and (is_strategy_mode or _should_use_llm(user_prompt))
            if use_llm:
                try:
                    answer = await _gemini_generate(prompt)
                    if not answer.strip():
                        raise RuntimeError("Gemini returned empty content")
                except Exception as exc:
                    logger.warning("LLM generation unavailable, falling back to local RAG answer: %s", exc)
                    if is_strategy_mode:
                        answer = _strategy_local_fallback(
                            query=user_prompt,
                            live_context=req.live_context,
                        )
                    else:
                        answer = answer_from_local_rag(
                            query=user_prompt,
                            retrieval=rag,
                            live_context=req.live_context,
                        )
            else:
                logger.debug("Using local RAG mode for query")
                if is_strategy_mode:
                    answer = _strategy_local_fallback(
                        query=user_prompt,
                        live_context=req.live_context,
                    )
                else:
                    answer = answer_from_local_rag(
                        query=user_prompt,
                        retrieval=rag,
                        live_context=req.live_context,
                    )

            if not answer.strip():
                local = answer_from_json(
                    query=user_prompt,
                    session_id=req.session_id,
                    live_context=req.live_context,
                )
                answer = local.get("answer", "")

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
