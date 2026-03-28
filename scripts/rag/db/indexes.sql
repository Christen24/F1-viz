-- =============================================================================
-- Formula Chat — Index Definitions
-- Run after schema.sql and after data import for best performance
-- =============================================================================

-- f1_knowledge — HNSW vector index for fast approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_knowledge_embedding
    ON f1_knowledge
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_knowledge_category   ON f1_knowledge(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_season     ON f1_knowledge(season);
CREATE INDEX IF NOT EXISTS idx_knowledge_source     ON f1_knowledge(source);
