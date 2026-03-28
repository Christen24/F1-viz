"""
Initialize local PostgreSQL schema for RAG without requiring psql.

Usage:
    python scripts/rag/setup_local_db.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv


def _split_sql(sql: str) -> list[str]:
    chunks = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        current.append(line)
    merged = "\n".join(current)
    for stmt in merged.split(";"):
        s = stmt.strip()
        if s:
            chunks.append(s)
    return chunks


async def main() -> None:
    root_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env)

    db_url = os.environ.get("F1VIZ_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Set F1VIZ_DATABASE_URL in root .env before running setup.")

    schema_path = Path(__file__).resolve().parent / "db" / "schema.sql"
    indexes_path = Path(__file__).resolve().parent / "db" / "indexes.sql"

    conn = await asyncpg.connect(db_url.replace("+asyncpg", ""))
    try:
        for sql_path in (schema_path, indexes_path):
            sql = sql_path.read_text(encoding="utf-8")
            for stmt in _split_sql(sql):
                try:
                    await conn.execute(stmt)
                except Exception as exc:
                    msg = str(exc).lower()
                    if "extension" in msg and "vector" in msg:
                        raise RuntimeError(
                            "pgvector extension is missing in your local PostgreSQL.\n"
                            "Install pgvector for your PostgreSQL version, then run this script again."
                        ) from exc
                    raise
        print("Local RAG DB initialized successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
