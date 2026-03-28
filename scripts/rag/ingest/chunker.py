"""
chunker.py — Split scraped text into overlapping chunks for embedding.

Uses tiktoken to count tokens accurately (same tokeniser as the embedding
model). Target: ~400 tokens per chunk with 50-token overlap, splitting
on paragraph and sentence boundaries where possible.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

import tiktoken

from scraper import ScrapedDocument

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50

_encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)


@dataclass
class Chunk:
    content: str
    token_count: int
    source: str
    category: str
    title: str | None
    season: int | None
    event_name: str | None


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def token_count(text: str) -> int:
    return len(_encoder.encode(text))


def token_split(text: str, max_tokens: int) -> tuple[str, str]:
    """Split text at max_tokens boundary, return (head, remainder)."""
    tokens = _encoder.encode(text)
    head_tokens = tokens[:max_tokens]
    tail_tokens = tokens[max_tokens:]
    return _encoder.decode(head_tokens), _encoder.decode(tail_tokens)


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_document(doc: ScrapedDocument) -> list[Chunk]:
    """
    Split a ScrapedDocument into overlapping token-bounded chunks.

    Strategy:
    1. Split text into paragraphs (double newline boundaries).
    2. Accumulate paragraphs into a chunk until CHUNK_TOKENS is reached.
    3. When a chunk is full, emit it and start the next chunk with the
       last OVERLAP_TOKENS of the previous chunk as a prefix.
    4. If a single paragraph exceeds CHUNK_TOKENS, split it at the
       sentence level, then at word level as a last resort.
    """
    if not doc.text.strip():
        log.warning("Empty document: %s", doc.url)
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", doc.text) if p.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    overlap_prefix = ""

    for para in paragraphs:
        para_tokens = token_count(para)

        # Paragraph too large — split recursively by sentence
        if para_tokens > CHUNK_TOKENS:
            sentences = _split_sentences(para)
            for sentence in sentences:
                sent_tokens = token_count(sentence)
                if current_tokens + sent_tokens > CHUNK_TOKENS:
                    if current:
                        chunks.append(_make_chunk(overlap_prefix, current, doc))
                        overlap_prefix = _compute_overlap(current)
                        current = []
                        current_tokens = 0
                current.append(sentence)
                current_tokens += sent_tokens
        else:
            if current_tokens + para_tokens > CHUNK_TOKENS:
                if current:
                    chunks.append(_make_chunk(overlap_prefix, current, doc))
                    overlap_prefix = _compute_overlap(current)
                    current = []
                    current_tokens = 0
            current.append(para)
            current_tokens += para_tokens

    # Emit any remaining content
    if current:
        chunks.append(_make_chunk(overlap_prefix, current, doc))

    log.debug("Chunked '%s' → %d chunks", doc.title, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(overlap_prefix: str, paragraphs: list[str], doc: ScrapedDocument) -> Chunk:
    body = "\n\n".join(paragraphs)
    content = f"{overlap_prefix}\n\n{body}".strip() if overlap_prefix else body
    return Chunk(
        content=content,
        token_count=token_count(content),
        source=doc.url,
        category=doc.category,
        title=doc.title,
        season=doc.season,
        event_name=doc.event_name,
    )


def _compute_overlap(paragraphs: list[str]) -> str:
    """Return the last OVERLAP_TOKENS tokens of the current chunk as a string."""
    full_text = "\n\n".join(paragraphs)
    tokens = _encoder.encode(full_text)
    overlap_tokens = tokens[-OVERLAP_TOKENS:]
    return _encoder.decode(overlap_tokens)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]
