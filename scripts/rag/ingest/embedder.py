"""
embedder.py — Generate embeddings for text chunks using OpenAI.

Batches chunks into groups of 100 to maximise throughput while staying
within the API's token-per-request limits. Retries on transient errors
with exponential backoff.
"""

import asyncio
import logging
import os
from typing import Sequence

import openai as openai_legacy
from dotenv import load_dotenv

from chunker import Chunk

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

def _load_env() -> None:
    here = os.path.dirname(__file__)
    # Single-source env: project root .env
    root_env = os.path.abspath(os.path.join(here, "../../../.env"))
    load_dotenv(dotenv_path=root_env)


_load_env()

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    os.environ.get("F1VIZ_EMBEDDING_MODEL", "text-embedding-3-small"),
)
BATCH_SIZE = 100
MAX_RETRIES = 5

# Cost per 1M tokens in USD — update if OpenAI pricing changes.
_COST_PER_1M_TOKENS: dict[str, float] = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
    "text-embedding-ada-002": 0.100,
}

def _client():
    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("F1VIZ_OPENAI_API_KEY"))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (or F1VIZ_OPENAI_API_KEY) not set")
    if AsyncOpenAI is not None:
        return AsyncOpenAI(api_key=api_key)
    openai_legacy.api_key = api_key
    return None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """
    Embed a list of Chunk objects.

    Returns:
        List of (chunk, embedding_vector) tuples in the same order as input.
    """
    if not chunks:
        return []

    client = _client()
    results: list[tuple[Chunk, list[float]]] = []
    total_tokens = 0
    num_batches = (len(chunks) - 1) // BATCH_SIZE + 1

    # Process in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.content for c in batch]
        batch_num = i // BATCH_SIZE + 1
        log.info("Embedding batch %d/%d (%d chunks)", batch_num, num_batches, len(batch))

        vectors, usage = await _embed_with_retry(client, texts)
        total_tokens += usage
        log.info("Batch %d/%d — %d tokens used", batch_num, num_batches, usage)
        results.extend(zip(batch, vectors))

    cost_per_1m = _COST_PER_1M_TOKENS.get(EMBEDDING_MODEL, 0.0)
    estimated_cost = total_tokens / 1_000_000 * cost_per_1m
    log.info(
        "Embedding complete — model: %s, total tokens: %d, estimated cost: $%.6f",
        EMBEDDING_MODEL, total_tokens, estimated_cost,
    )

    return results


async def _embed_with_retry(client, texts: list[str]) -> tuple[list[list[float]], int]:
    """Call the embeddings API with exponential backoff on failure.

    Returns a tuple of (embedding vectors, total tokens used).
    """
    for attempt in range(MAX_RETRIES):
        try:
            if AsyncOpenAI is not None and client is not None:
                response = await client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                )
                vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                return vectors, response.usage.total_tokens

            # Legacy openai package fallback (non-async)
            def _legacy_embed():
                return openai_legacy.Embedding.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                )

            response = await asyncio.to_thread(_legacy_embed)
            data = sorted(response["data"], key=lambda x: x["index"])
            vectors = [item["embedding"] for item in data]
            usage = response.get("usage", {}).get("total_tokens", 0)
            return vectors, int(usage)
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            log.warning("Embedding attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
            await asyncio.sleep(wait)

    raise RuntimeError("Embedding failed after all retries")
