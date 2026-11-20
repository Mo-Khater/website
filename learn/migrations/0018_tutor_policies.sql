CREATE TABLE IF NOT EXISTS tutor_policies (
    activity_id TEXT PRIMARY KEY,
    policy_json TEXT NOT NULL,
    updated_by  TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (activity_id) REFERENCES activities(id) ON DELETE CASCADE
);
