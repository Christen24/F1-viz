"""
run_ingest.py — CLI orchestrator for the RAG ingestion pipeline.

Scrape → Chunk → Embed → Load

Usage:
    # Ingest all sources
    python scripts/rag/ingest/run_ingest.py

    # Ingest a specific category
    python scripts/rag/ingest/run_ingest.py --category driver
    python scripts/rag/ingest/run_ingest.py --category regulation

    # Ingest a single URL
    python scripts/rag/ingest/run_ingest.py --url https://en.wikipedia.org/wiki/Lewis_Hamilton

    # Ingest race reports for a specific event (added inline, not in sources.json)
    python scripts/rag/ingest/run_ingest.py \\
        --url https://www.autosport.com/f1/news/... \\
        --category race_report \\
        --title "2025 Bahrain GP Race Report" \\
        --season 2025 \\
        --event "Bahrain Grand Prix"

    # Dry run — scrape and chunk only, no DB writes
    python scripts/rag/ingest/run_ingest.py --category driver --dry-run
"""

import asyncio
import argparse
import json
import logging
import os
import sys

# Add parent directory to path so we can import from scripts/
sys.path.insert(0, os.path.dirname(__file__))

from scraper import scrape, ScrapedDocument
from chunker import chunk_document
from embedder import embed_chunks
from loader import load_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

SOURCES_FILE = os.path.join(os.path.dirname(__file__), "sources.json")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_source(source: dict, dry_run: bool = False) -> dict:
    """
    Run the full pipeline for a single source dict.

    Returns a status dict with counts.
    """
    url = source["url"]
    log.info("--- Processing: %s", source.get("title", url))

    # 1. Scrape
    doc: ScrapedDocument | None = await scrape(source)
    if doc is None or not doc.text.strip():
        return {"url": url, "status": "scrape_failed", "chunks": 0}

    # 2. Chunk
    chunks = chunk_document(doc)
    if not chunks:
        return {"url": url, "status": "no_chunks", "chunks": 0}

    log.info("  %d chunks generated (avg %d tokens)",
             len(chunks), sum(c.token_count for c in chunks) // len(chunks))

    if dry_run:
        log.info("  [dry-run] Skipping embed + load")
        return {"url": url, "status": "dry_run", "chunks": len(chunks)}

    # 3. Embed
    embedded = await embed_chunks(chunks)

    # 4. Load
    loaded = await load_chunks(embedded, url)

    return {"url": url, "status": "ok", "chunks": loaded}


async def run_pipeline(sources: list[dict], dry_run: bool = False) -> None:
    """Run the pipeline for a list of sources sequentially."""
    total_chunks = 0
    failed = []

    for source in sources:
        result = await run_source(source, dry_run=dry_run)
        if result["status"] == "ok":
            total_chunks += result["chunks"]
        elif result["status"] not in ("dry_run",):
            failed.append(result["url"])

    log.info("=== Ingest complete: %d chunks loaded, %d sources failed ===",
             total_chunks, len(failed))
    if failed:
        log.warning("Failed sources:\n  %s", "\n  ".join(failed))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_sources_file() -> list[dict]:
    with open(SOURCES_FILE) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the F1 RAG ingestion pipeline")
    parser.add_argument("--category", type=str,
                        choices=["driver", "team", "circuit", "regulation", "race_report"],
                        help="Filter sources by category")
    parser.add_argument("--url", type=str, help="Ingest a single URL")
    parser.add_argument("--title", type=str, help="Title for single URL mode")
    parser.add_argument("--season", type=int, help="Season for single URL mode")
    parser.add_argument("--event", type=str, help="Event name for single URL mode")
    parser.add_argument("--refresh", type=str,
                        choices=["quarterly", "monthly", "weekly", "race_weekend"],
                        help="Filter sources by refresh cadence")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scrape and chunk only — no database writes")
    args = parser.parse_args()

    # Single URL mode
    if args.url:
        category = args.category or "race_report"
        source = {
            "url": args.url,
            "title": args.title or args.url,
            "category": category,
        }
        if args.season:
            source["season"] = args.season
        if args.event:
            source["event_name"] = args.event

        asyncio.run(run_pipeline([source], dry_run=args.dry_run))
        return

    # Load from sources.json
    sources = load_sources_file()

    if args.category:
        sources = [s for s in sources if s.get("category") == args.category]
        log.info("Filtered to %d sources with category='%s'", len(sources), args.category)

    if args.refresh:
        sources = [s for s in sources if s.get("refresh") == args.refresh]
        log.info("Filtered to %d sources with refresh='%s'", len(sources), args.refresh)

    if not sources:
        log.error("No sources matched the given filters.")
        sys.exit(1)

    log.info("Starting ingest for %d sources (dry_run=%s)", len(sources), args.dry_run)
    asyncio.run(run_pipeline(sources, dry_run=args.dry_run))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
