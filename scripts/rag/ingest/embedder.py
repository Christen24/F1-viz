"""
embedder.py — Generate embeddings for text chunks.

Provider: Ollama (nomic-embed-text)
"""

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv

from chunker import Chunk


def _load_env() -> None:
    here = os.path.dirname(__file__)
    root_env = os.path.abspath(os.path.join(here, "../../../.env"))
    load_dotenv(dotenv_path=root_env)


_load_env()

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("F1VIZ_EMBEDDING_MODEL", "nomic-embed-text")

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434/api/embed"
)

BATCH_SIZE = 32
MAX_RETRIES = 5
VECTOR_DIM = 768


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """
    Embed a list of Chunk objects.

    Returns:
        List of (chunk, embedding_vector) tuples.
    """
    if not chunks:
        return []

    results: list[tuple[Chunk, list[float]]] = []

    num_batches = (len(chunks) - 1) // BATCH_SIZE + 1

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        texts = [c.content for c in batch]

        batch_num = i // BATCH_SIZE + 1

        log.info(
            "Embedding batch %d/%d (%d chunks)",
            batch_num,
            num_batches,
            len(batch),
        )

        vectors = await _embed_with_retry(texts)

        results.extend(zip(batch, vectors))

    log.info(
        "Embedding complete — provider: Ollama, model: %s",
        EMBEDDING_MODEL,
    )

    return results


async def _ollama_embed_text(
    client: httpx.AsyncClient,
    text: str,
) -> list[float]:

    resp = await client.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
    )

    resp.raise_for_status()

    data = resp.json()

    embeddings = data.get("embeddings")

    if not embeddings:
        raise RuntimeError("No embeddings returned from Ollama")

    vector = embeddings[0]

    if len(vector) != VECTOR_DIM:
        log.warning(
            "Embedding dimension %d differs from expected %d",
            len(vector),
            VECTOR_DIM,
        )

    return [float(v) for v in vector]


async def _embed_with_retry(
    texts: list[str],
) -> list[list[float]]:

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:

                vectors = []

                for text in texts:
                    vec = await _ollama_embed_text(client, text)
                    vectors.append(vec)

                return vectors

        except Exception as exc:

            if attempt == MAX_RETRIES - 1:
                raise

            wait = 2 ** attempt

            log.warning(
                "Embedding attempt %d failed: %s — retrying in %ds",
                attempt + 1,
                exc,
                wait,
            )

            await asyncio.sleep(wait)

    raise RuntimeError("Embedding failed after all retries")