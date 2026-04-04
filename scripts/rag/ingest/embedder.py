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

EMBEDDING_MODEL = os.environ.get("F1VIZ_GEMINI_EMBED_MODEL", "gemini-embedding-001")
BATCH_SIZE = 32
MAX_RETRIES = 5
VECTOR_DIM = 1536
_MODEL_FALLBACKS = ("gemini-embedding-001", "text-embedding-004")


def _normalize_model_name(model: str) -> str:
    m = (model or "").strip()
    if m.startswith("models/"):
        m = m[len("models/"):]
    return m or "gemini-embedding-001"


def _embed_url(model: str) -> str:
    model_name = _normalize_model_name(model)
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent"


def _gemini_api_key() -> str:
    return os.environ.get("F1VIZ_GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


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

    gemini_key = _gemini_api_key()
    if not gemini_key:
        # Keyword retrieval path in this project does not use vector similarity.
        vectors = [[0.0] * VECTOR_DIM for _ in chunks]
        log.warning(
            "Gemini API key not set. Embedding fallback active: generated %d zero vectors (dim=%d).",
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

        vectors = await _embed_with_retry(gemini_key, texts)
        log.info("Batch %d/%d embedded via Gemini", batch_num, num_batches)
        results.extend(zip(batch, vectors))

    log.info("Embedding complete — provider: Gemini, model: %s", EMBEDDING_MODEL)

    return results


def _coerce_dim(values: list[float], dim: int = VECTOR_DIM) -> list[float]:
    if len(values) == dim:
        return values
    if len(values) > dim:
        return values[:dim]
    return values + ([0.0] * (dim - len(values)))


async def _gemini_embed_text(client: httpx.AsyncClient, api_key: str, text: str) -> list[float]:
    model = _normalize_model_name(EMBEDDING_MODEL)
    tried: list[str] = []
    candidate_models = [model] + [m for m in _MODEL_FALLBACKS if m != model]

    last_exc: Exception | None = None
    for candidate in candidate_models:
        tried.append(candidate)
        payload = {
            "model": f"models/{candidate}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": VECTOR_DIM,
        }
        headers = {"x-goog-api-key": api_key}
        try:
            resp = await client.post(_embed_url(candidate), json=payload, headers=headers)
            if resp.status_code == 404:
                # Try next model fallback when model is unavailable/deprecated.
                continue
            resp.raise_for_status()
            data = resp.json()
            values = (
                data.get("embedding", {}).get("values")
                or data.get("embedding", {}).get("value")
                or []
            )
            if values:
                return _coerce_dim([float(v) for v in values])
        except Exception as exc:  # retry logic handled by caller
            last_exc = exc
            break

    if last_exc is not None:
        raise last_exc

    raise RuntimeError(
        f"Gemini embedding model unavailable for candidates: {', '.join(tried)}. "
        "Set F1VIZ_GEMINI_EMBED_MODEL=gemini-embedding-001 in .env."
    )


async def _embed_with_retry(api_key: str, texts: list[str]) -> list[list[float]]:
    model = _normalize_model_name(EMBEDDING_MODEL)
    payload = {
        "model": f"models/{model}",
        "outputDimensionality": VECTOR_DIM,
    }
    # Keep payload var so model resolution is visible in logs if exceptions bubble.
    _ = payload
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                vectors = []
                for text in texts:
                    vec = await _gemini_embed_text(client, api_key, text)
                    vectors.append(vec)
            return vectors
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            log.warning("Embedding attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
            await asyncio.sleep(wait)

    raise RuntimeError("Embedding failed after all retries")
