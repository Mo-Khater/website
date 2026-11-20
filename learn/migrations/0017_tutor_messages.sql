CREATE TABLE IF NOT EXISTS tutor_messages (
    id                TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL,
    activity_id       TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    role              TEXT NOT NULL,
    content_encrypted TEXT NOT NULL,
    citations_json    TEXT,
    grounded          INTEGER,
    outcome           TEXT,
    withheld_encrypted TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (activity_id) REFERENCES activities(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id)     REFERENCES users(id)      ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tutor_messages_conversation ON tutor_messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tutor_messages_activity     ON tutor_messages(activity_id, created_at);
