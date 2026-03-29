"""
RAG retrieval utilities backed by PostgreSQL + pgvector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None  # type: ignore

from src.backend.config import settings


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
    clauses = ["(content ILIKE $1::text OR COALESCE(title, '') ILIKE $1::text)"]
    args: list[Any] = [f"%{query}%"]
    idx = 2

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
            0.5::float AS score
        FROM f1_knowledge
        {where_sql}
        ORDER BY scraped_at DESC
        LIMIT {limit_param}::int
        """,
        *args,
    )

    # If strict keyword search yields nothing, return the latest chunks to keep chat functional.
    if not rows:
        rows = await conn.fetch(
            """
            SELECT
                id, source, category, title, content, season, event_name,
                0.2::float AS score
            FROM f1_knowledge
            ORDER BY scraped_at DESC
            LIMIT $1::int
            """,
            max(1, min(top_k, 20)),
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
