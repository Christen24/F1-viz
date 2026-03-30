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
    live_context: dict[str, Any] | None,
) -> str:
    live = live_context or {}
    base = "|".join(
        [
            str(session_id or ""),
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
    live_context: dict[str, Any] | None,
) -> str:
    live = json.dumps(live_context or {}, ensure_ascii=True)
    convo = []
    for m in messages[-10:]:
        convo.append(f"{m.role.upper()}: {m.content}")
    convo_text = "\n".join(convo)
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
                live_context=req.live_context,
            )
            try:
                if settings.chat_force_local_rag or not _should_use_llm(user_prompt):
                    raise RuntimeError("Forced local RAG mode")
                answer = await _gemini_generate(prompt)
                if not answer.strip():
                    raise RuntimeError("Gemini returned empty content")
            except Exception:
                logger.exception("LLM generation failed, falling back to local RAG answer")
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
