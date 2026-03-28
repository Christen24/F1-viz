"""
Chat router with SSE streaming for RAG responses.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import openai as openai_legacy
from pydantic import BaseModel, Field

from src.backend.config import settings
from src.backend.services.rag import f1_knowledge

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


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


def _build_system_prompt(live_context: dict[str, Any] | None) -> str:
    live_ctx = json.dumps(live_context or {}, ensure_ascii=True)
    return (
        "You are AI Pit Crew, an expert Formula 1 race assistant.\n"
        "Use retrieved context first. If context is missing, say you are unsure.\n"
        "Be concise and factual. Include source references as [1], [2] when relevant.\n"
        "Never fabricate race results.\n"
        f"Live dashboard context: {live_ctx}\n"
    )


def _build_fallback_answer(
    user_prompt: str,
    rag: dict[str, Any],
    live_context: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("The LLM provider is currently unavailable, so I switched to local RAG fallback mode.")

    live = live_context or {}
    leader = live.get("leader")
    lap = live.get("current_lap")
    if leader or lap:
        pieces = []
        if leader:
            pieces.append(f"leader: {leader}")
        if lap:
            pieces.append(f"lap: {lap}")
        lines.append(f"Live context -> {', '.join(pieces)}.")

    sources = rag.get("sources", []) or []
    if sources:
        lines.append("Most relevant knowledge chunks:")
        for s in sources[:3]:
            title = s.get("title") or "Untitled"
            category = s.get("category") or "general"
            src = s.get("source") or "-"
            lines.append(f"- [{s.get('rank', '?')}] {title} ({category})")
            lines.append(f"  source: {src}")
    else:
        lines.append("No indexed knowledge chunks were found for this query.")

    lines.append(f"Query: {user_prompt}")
    lines.append("Once provider access is restored, responses will stream with full model reasoning again.")
    return "\n".join(lines)


async def _gemini_complete(messages: list[dict[str, str]]) -> str:
    gemini_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise RuntimeError("Gemini API key is not configured")

    prompt_lines = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        prompt_lines.append(f"{role}: {content}")
    prompt = "\n\n".join(prompt_lines)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={gemini_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(p.get("text", "")) for p in parts if p.get("text"))
    return text


async def _stream_or_complete(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    # Prefer Gemini when configured
    if settings.gemini_api_key or os.environ.get("GEMINI_API_KEY"):
        text = await _gemini_complete(messages)
        if text:
            yield text
        return

    # Fallback to OpenAI
    if AsyncOpenAI is not None:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        stream = await client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.2,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
        return

    def _legacy_complete() -> str:
        openai_legacy.api_key = settings.openai_api_key
        response = openai_legacy.ChatCompletion.create(
            model=settings.chat_model,
            messages=messages,
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"] or ""  # type: ignore[index]

    text = await asyncio.to_thread(_legacy_complete)
    if text:
        yield text


@router.post("/stream")
async def stream_chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    has_gemini = bool(settings.gemini_api_key or os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(settings.openai_api_key)
    if not has_gemini and not has_openai:
        raise HTTPException(status_code=500, detail="No LLM API key configured (Gemini/OpenAI)")

    user_prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    async def event_stream() -> AsyncIterator[str]:
        try:
            try:
                rag = await f1_knowledge(
                    user_prompt,
                    top_k=req.top_k,
                    season=req.season,
                    category=req.category,
                    event_name=req.event_name,
                )
            except Exception as rag_exc:
                logger.exception("RAG retrieval failed; continuing without DB context")
                rag = {"context": "", "sources": [], "rag_error": str(rag_exc)}

            yield _sse_event({"type": "sources", "sources": rag["sources"]})

            messages = [
                {"role": "system", "content": _build_system_prompt(req.live_context)},
                {
                    "role": "system",
                    "content": (
                        "Retrieved F1 knowledge context:\n"
                        f"{rag['context'] or 'No matching context retrieved.'}"
                    ),
                },
            ] + [{"role": m.role, "content": m.content} for m in req.messages]

            try:
                async for delta in _stream_or_complete(messages):
                    yield _sse_event({"type": "delta", "text": delta})
            except Exception:
                logger.exception("LLM response failed, serving local fallback answer")
                fallback = _build_fallback_answer(
                    user_prompt=user_prompt,
                    rag=rag,
                    live_context=req.live_context,
                )
                yield _sse_event({"type": "delta", "text": fallback})

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
