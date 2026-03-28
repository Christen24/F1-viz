-- =============================================================================
-- Formula Chat — PostgreSQL Schema
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- RAG knowledge base
-- =============================================================================

CREATE TABLE IF NOT EXISTS f1_knowledge (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,
    category        TEXT NOT NULL,      -- driver | team | circuit | regulation | race_report
    title           TEXT,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,      -- SHA-256 of content for idempotent upsert
    embedding       VECTOR(1536) NOT NULL,
    token_count     INT,
    scraped_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season          INT,
    event_name      TEXT,
    UNIQUE (source, content_hash)
);
