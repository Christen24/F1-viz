"""
RAG retrieval utilities backed by PostgreSQL + pgvector.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None  # type: ignore

from src.backend.config import settings

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "who", "what", "when", "where", "why", "how", "did", "does", "do",
    "tell", "about", "please", "f1", "formula", "one",
}


@dataclass
class KnowledgeHit:
    id: int
    source: str
    category: str
    title: str | None
    content: str
    season: int | None
    event_name: str | None
    score: float


def _normalized_db_url() -> str:
    # asyncpg expects postgresql://, not postgresql+asyncpg://
    return settings.database_url.replace("+asyncpg", "")


async def retrieve_knowledge(
    query: str,
    *,
    top_k: int = 6,
    season: int | None = None,
    category: str | None = None,
    event_name: str | None = None,
) -> list[KnowledgeHit]:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL / F1VIZ_DATABASE_URL is not configured")
    if asyncpg is None:
        raise RuntimeError("asyncpg is not installed. Run: python -m pip install asyncpg==0.30.0")

    conn = await asyncpg.connect(_normalized_db_url())
    try:
        # Gemini-only runtime path: no OpenAI embeddings.
        return await _retrieve_knowledge_keyword(
            conn,
            query=query,
            top_k=top_k,
            season=season,
            category=category,
            event_name=event_name,
        )
    finally:
        await conn.close()


async def _retrieve_knowledge_keyword(
    conn,
    *,
    query: str,
    top_k: int,
    season: int | None,
    category: str | None,
    event_name: str | None,
) -> list[KnowledgeHit]:
    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9]+", (query or "").lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    ][:8]
    clauses: list[str] = []
    args: list[Any] = []
    idx = 1
    score_sql = "0.1::float"

    if tokens:
        token_clauses: list[str] = []
        score_parts: list[str] = []
        for tok in tokens:
            cond = f"(content ILIKE ${idx}::text OR COALESCE(title, '') ILIKE ${idx}::text)"
            token_clauses.append(cond)
            score_parts.append(f"(CASE WHEN {cond} THEN 1 ELSE 0 END)")
            args.append(f"%{tok}%")
            idx += 1
        clauses.append("(" + " OR ".join(token_clauses) + ")")
        score_sql = f"(({ ' + '.join(score_parts) })::float / {max(1, len(score_parts))}::float)"
    else:
        clauses.append(f"(content ILIKE ${idx}::text OR COALESCE(title, '') ILIKE ${idx}::text)")
        args.append(f"%{query}%")
        idx += 1

    if season is not None:
        clauses.append(f"season = ${idx}")
        args.append(season)
        idx += 1
    if category:
        clauses.append(f"category = ${idx}")
        args.append(category)
        idx += 1
    if event_name:
        clauses.append(f"event_name ILIKE ${idx}")
        args.append(f"%{event_name}%")
        idx += 1

    where_sql = f"WHERE {' AND '.join(clauses)}"
    limit_param = f"${idx}"
    args.append(max(1, min(top_k, 20)))

    rows = await conn.fetch(
        f"""
        SELECT
            id, source, category, title, content, season, event_name,
            {score_sql} AS score
        FROM f1_knowledge
        {where_sql}
        ORDER BY score DESC, scraped_at DESC
        LIMIT {limit_param}::int
        """,
        *args,
    )

    return [
        KnowledgeHit(
            id=int(r["id"]),
            source=str(r["source"]),
            category=str(r["category"]),
            title=r["title"],
            content=str(r["content"]),
            season=r["season"],
            event_name=r["event_name"],
            score=float(r["score"] or 0.0),
        )
        for r in rows
    ]


async def f1_knowledge(
    query: str,
    *,
    top_k: int = 6,
    season: int | None = None,
    category: str | None = None,
    event_name: str | None = None,
) -> dict[str, Any]:
    hits = await retrieve_knowledge(
        query,
        top_k=top_k,
        season=season,
        category=category,
        event_name=event_name,
    )

    context_blocks = []
    sources: list[dict[str, Any]] = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(
            (
                f"[{i}] {hit.title or 'Untitled'}\n"
                f"category={hit.category} score={hit.score:.3f}\n"
                f"source={hit.source}\n"
                f"{hit.content}"
            )
        )
        sources.append(
            {
                "id": hit.id,
                "rank": i,
                "title": hit.title,
                "source": hit.source,
                "category": hit.category,
                "season": hit.season,
                "event_name": hit.event_name,
                "score": round(hit.score, 4),
            }
        )

    return {
        "context": "\n\n---\n\n".join(context_blocks),
        "sources": sources,
    }
