"""
Chat router with SSE streaming for RAG responses.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.backend.services.json_chat import answer_from_json

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


@router.post("/stream")
async def stream_chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    user_prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    async def event_stream() -> AsyncIterator[str]:
        try:
            local = answer_from_json(
                query=user_prompt,
                session_id=req.session_id,
                live_context=req.live_context,
            )
            yield _sse_event({"type": "sources", "sources": local.get("sources", [])})
            yield _sse_event({"type": "delta", "text": local.get("answer", "")})
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
