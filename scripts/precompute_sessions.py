"""
Precompute race data before runtime usage.

Examples:
  python scripts/precompute_sessions.py --year 2024 --gp "Bahrain Grand Prix" --session R
  python scripts/precompute_sessions.py --year 2024 --gp "Japanese Grand Prix" --session R --force
"""
from __future__ import annotations

import argparse
import json

from src.backend.config import settings
from src.backend.services.exporter import process_and_export_session
from src.backend.services.fetcher import get_session_id
from src.backend.services.local_rag import warm_local_rag_cache


def _has_complete_export(session_id: str) -> bool:
    output_dir = settings.processed_dir / session_id
    metadata_path = output_dir / "metadata.json"
    track_path = output_dir / "track.json"
    chunks_dir = output_dir / "chunks"
    if not metadata_path.exists() or not track_path.exists() or not chunks_dir.exists():
        return False
    if not any(chunks_dir.glob("chunk_*.json.gz")):
        return False
    try:
        with open(track_path, encoding="utf-8") as f:
            payload = json.load(f)
        return isinstance(payload, list)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute and cache a race session.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--gp", type=str, required=True)
    parser.add_argument("--session", type=str, default="R")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-warm-chat", action="store_true")
    args = parser.parse_args()

    session_id = get_session_id(args.year, args.gp, args.session)
    if _has_complete_export(session_id) and not args.force:
        print(f"[cached] {session_id} already precomputed")
    else:
        print(f"[build] precomputing {session_id} ...")
        process_and_export_session(args.year, args.gp, args.session)
        print(f"[ok] {session_id} precomputed")

    if not args.no_warm_chat:
        warmed = warm_local_rag_cache(session_id)
        print(f"[chat-cache] warmed={warmed} session={session_id}")


if __name__ == "__main__":
    main()
