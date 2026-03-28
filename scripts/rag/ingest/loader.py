"""
loader.py — Upsert embedded chunks into the f1_knowledge pgvector table.

Idempotent: matches on (source, content_hash) so re-running the pipeline
after a scrape update will replace changed chunks without creating duplicates.
Old chunks from a re-scraped source are deleted before new ones are inserted.
"""

import asyncio
import hashlib
import logging
import os

import asyncpg
from dotenv import load_dotenv

from chunker import Chunk

def _load_env() -> None:
    here = os.path.dirname(__file__)
    # Single-source env: project root .env
    root_env = os.path.abspath(os.path.join(here, "../../../.env"))
    load_dotenv(dotenv_path=root_env)


_load_env()

log = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def _connect() -> asyncpg.Connection:
    db_url = os.environ.get("DATABASE_URL", os.environ.get("F1VIZ_DATABASE_URL", "")).replace("+asyncpg", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL (or F1VIZ_DATABASE_URL) not set")
    return await asyncpg.connect(db_url)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def load_chunks(
    embedded: list[tuple[Chunk, list[float]]],
    source_url: str,
) -> int:
    """
    Delete all existing chunks for source_url, then insert the new ones.

    Args:
        embedded: list of (Chunk, embedding_vector) from the embedder
        source_url: the URL being ingested (used to delete stale chunks)

    Returns:
        Number of rows inserted.
    """
    if not embedded:
        log.warning("No chunks to load for %s", source_url)
        return 0

    conn = await _connect()

    try:
        async with conn.transaction():
            # Remove stale chunks from a previous scrape of this URL
            deleted = await conn.execute(
                "DELETE FROM f1_knowledge WHERE source = $1", source_url
            )
            if deleted != "DELETE 0":
                log.info("Removed stale chunks: %s from %s", deleted, source_url)

            # Insert new chunks
            rows = [
                (
                    chunk.source,
                    chunk.category,
                    chunk.title,
                    chunk.content,
                    _hash(chunk.content),
                    str(vector),        # pgvector accepts '[0.1, 0.2, ...]' string format
                    chunk.token_count,
                    chunk.season,
                    chunk.event_name,
                )
                for chunk, vector in embedded
            ]

            await conn.executemany(
                """
                INSERT INTO f1_knowledge
                    (source, category, title, content, content_hash,
                     embedding, token_count, season, event_name)
                VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9)
                ON CONFLICT (source, content_hash) DO UPDATE SET
                    category   = EXCLUDED.category,
                    title      = EXCLUDED.title,
                    embedding  = EXCLUDED.embedding,
                    scraped_at = NOW()
                """,
                rows,
            )

            log.info("Loaded %d chunks for %s", len(rows), source_url)
            return len(rows)
    finally:
        await conn.close()


async def load_all(
    embedded_batches: list[tuple[str, list[tuple[Chunk, list[float]]]]],
) -> int:
    """
    Load multiple sources in sequence.

    Args:
        embedded_batches: list of (source_url, embedded_chunks) tuples

    Returns:
        Total number of rows inserted.
    """
    total = 0
    for source_url, embedded in embedded_batches:
        total += await load_chunks(embedded, source_url)
    return total
