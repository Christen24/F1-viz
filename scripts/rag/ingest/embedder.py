"""
embedder.py — Generate embeddings for text chunks.

Primary provider: Gemini embeddings API.
Fallback: deterministic zero vectors (for keyword-only retrieval mode).
"""

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv

from chunker import Chunk

def _load_env() -> None:
    here = os.path.dirname(__file__)
    # Single-source env: project root .env
    root_env = os.path.abspath(os.path.join(here, "../../../.env"))
    load_dotenv(dotenv_path=root_env)


_load_env()

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("F1VIZ_EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = 32
MAX_RETRIES = 5
VECTOR_DIM = 1536

def _openrouter_api_key() -> str:
    return os.environ.get("F1VIZ_OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")


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

    openrouter_key = _openrouter_api_key()
    if not openrouter_key:
        # Keyword retrieval path in this project does not use vector similarity.
        vectors = [[0.0] * VECTOR_DIM for _ in chunks]
        log.warning(
            "OpenRouter API key not set. Embedding fallback active: generated %d zero vectors (dim=%d).",
            len(vectors),
            VECTOR_DIM,
        )
        return list(zip(chunks, vectors))

    results: list[tuple[Chunk, list[float]]] = []
    num_batches = (len(chunks) - 1) // BATCH_SIZE + 1

    # Process in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.content for c in batch]
        batch_num = i // BATCH_SIZE + 1
        log.info("Embedding batch %d/%d (%d chunks)", batch_num, num_batches, len(batch))

        vectors = await _embed_with_retry(openrouter_key, texts)
        log.info("Batch %d/%d embedded via OpenRouter", batch_num, num_batches)
        results.extend(zip(batch, vectors))

    log.info("Embedding complete — provider: OpenRouter, model: %s", EMBEDDING_MODEL)

    return results


def _coerce_dim(values: list[float], dim: int = VECTOR_DIM) -> list[float]:
    if len(values) == dim:
        return values
    if len(values) > dim:
        return values[:dim]
    return values + ([0.0] * (dim - len(values)))


async def _openrouter_embed_text(client: httpx.AsyncClient, api_key: str, text: str) -> list[float]:
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Christen24/F1-viz",
        "X-Title": "F1-Viz Platform",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }
    
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    # OpenRouter/OpenAI format: {"data": [{"embedding": [...]}]}
    values = data.get("data", [{}])[0].get("embedding", [])
    if values:
        return _coerce_dim([float(v) for v in values])
    
    raise RuntimeError(f"OpenRouter embedding returned no data for model {EMBEDDING_MODEL}")


async def _embed_with_retry(api_key: str, texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                vectors = []
                for text in texts:
                    vec = await _openrouter_embed_text(client, api_key, text)
                    vectors.append(vec)
            return vectors
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            log.warning("Embedding attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
            await asyncio.sleep(wait)

    raise RuntimeError("Embedding failed after all retries")
