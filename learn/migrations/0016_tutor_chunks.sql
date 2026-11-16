-- Migration 0016: AI tutor retrieval index
-- Chunk text for course-grounded tutor answers. Vectors live in Vectorize
-- (one namespace per activity); this table keeps the encrypted text and a
-- content hash so unchanged chunks are not re-embedded.

CREATE TABLE IF NOT EXISTS tutor_chunks (
    id             TEXT PRIMARY KEY,
    activity_id    TEXT NOT NULL,
    source_type    TEXT NOT NULL,
    source_id      TEXT NOT NULL,
    source_title   TEXT,
    chunk_index    INTEGER NOT NULL DEFAULT 0,
    text_encrypted TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT,
    FOREIGN KEY (activity_id) REFERENCES activities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tutor_chunks_activity ON tutor_chunks(activity_id);
