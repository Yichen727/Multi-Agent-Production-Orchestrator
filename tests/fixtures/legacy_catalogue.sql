
        -- One row per project, carrying only what the system actually knows: the id, a
        -- name, and how many clips the last ingest catalogued. `clip_count` is maintained
        -- by replace_project_shots — the ONLY writer of `shots` — inside the same
        -- transaction as the insert, so it cannot drift out of sync with the rows it counts.
        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT NOT NULL,
            clip_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS shots (
            shot_id INTEGER PRIMARY KEY,
            project_id INTEGER,
            file_path TEXT NOT NULL,
            shot_type TEXT,
            duration_seconds REAL,
            keywords TEXT,
            width INTEGER,
            height INTEGER,
            orientation TEXT,
            fps REAL,
            codec TEXT,
            has_audio INTEGER,
            scene_count INTEGER,
            description TEXT,
            people_count INTEGER,
            mood TEXT,
            embedding TEXT,
            source_mtime REAL,
            source_size INTEGER,
            audio_channels INTEGER,
            audio_sample_rate INTEGER,
            audio_bit_depth INTEGER,
            UNIQUE (project_id, file_path),
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        );

        CREATE INDEX IF NOT EXISTS idx_shots_project ON shots(project_id);
        CREATE INDEX IF NOT EXISTS idx_shots_orientation ON shots(orientation);
        CREATE INDEX IF NOT EXISTS idx_shots_shot_type ON shots(shot_type);
        CREATE INDEX IF NOT EXISTS idx_shots_duration ON shots(duration_seconds);

        -- Temporal event-based ingestion (Tier 2): one shots row (a physical file) has
        -- MANY clip_events rows — the ordered "what happens" segments inside it, each with
        -- a real start/end (from FFmpeg scene boundaries) and its own semantic embedding.
        -- ON DELETE CASCADE means replace_project_shots' DELETE of a project's shots also
        -- clears its events (foreign_keys=ON would otherwise reject the delete); ingest
        -- re-inserts the events (reused + freshly analysed) right after.
        CREATE TABLE IF NOT EXISTS clip_events (
            event_id INTEGER PRIMARY KEY,
            project_id INTEGER,
            shot_id INTEGER,
            file_path TEXT NOT NULL,
            event_order INTEGER,
            start_seconds REAL,
            end_seconds REAL,
            duration_seconds REAL,
            action TEXT,
            state_change TEXT,
            subjects TEXT,
            keywords TEXT,
            embedding TEXT,
            FOREIGN KEY (shot_id) REFERENCES shots(shot_id) ON DELETE CASCADE,
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_project ON clip_events(project_id);
        CREATE INDEX IF NOT EXISTS idx_events_shot ON clip_events(shot_id);
        CREATE INDEX IF NOT EXISTS idx_events_file ON clip_events(file_path);
