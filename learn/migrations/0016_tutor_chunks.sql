CREATE TABLE IF NOT EXISTS tutor_chunks (
    id             TEXT PRIMARY KEY,
    activity_id    TEXT NOT NULL,
    source_type    TEXT NOT NULL,
    source_id      TEXT NOT NULL,
    source_title   TEXT,
    locator        TEXT,
    chunk_index    INTEGER NOT NULL DEFAULT 0,
    text_encrypted TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT,
    FOREIGN KEY (activity_id) REFERENCES activities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tutor_chunks_activity ON tutor_chunks(activity_id);
