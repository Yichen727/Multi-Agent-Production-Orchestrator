"""SQLite schema v1. No application, model or configuration imports."""

SCHEMA_VERSION = 1
MEDIA_COLUMNS = (
    "file_path", "shot_type", "duration_seconds", "keywords", "width", "height",
    "orientation", "fps", "codec", "has_audio", "scene_count", "description",
    "people_count", "mood", "embedding", "source_mtime", "source_size",
    "audio_channels", "audio_sample_rate", "audio_bit_depth",
)
EVENT_COLUMNS = ("file_path", "event_order", "start_seconds", "end_seconds", "duration_seconds",
                 "action", "state_change", "subjects", "keywords", "embedding")


def schema_statements(prefix=""):
    clips, sets, events, steps = (prefix + name for name in
                                 ("clips", "event_sets", "clip_events", "analysis_steps"))
    return (
        """CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY, project_name TEXT NOT NULL,
            clip_count INTEGER NOT NULL DEFAULT 0)""",
        f"""CREATE TABLE {clips} (
            clip_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES projects(project_id),
            file_path TEXT NOT NULL, canonical_path TEXT NOT NULL,
            source_revision INTEGER NOT NULL DEFAULT 1 CHECK(source_revision>0),
            source_mtime REAL, source_mtime_ns INTEGER, source_size INTEGER, content_hash TEXT,
            online INTEGER NOT NULL DEFAULT 1 CHECK(online IN (0,1)),
            active_event_set_id INTEGER REFERENCES {sets}(event_set_id),
            shot_type TEXT, duration_seconds REAL, keywords TEXT, width INTEGER, height INTEGER,
            orientation TEXT, fps REAL, codec TEXT, has_audio INTEGER, scene_count INTEGER,
            description TEXT, people_count INTEGER, mood TEXT, embedding TEXT,
            audio_channels INTEGER, audio_sample_rate INTEGER, audio_bit_depth INTEGER,
            UNIQUE(project_id, canonical_path), UNIQUE(project_id, clip_id))""",
        f"""CREATE TABLE {sets} (
            event_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL, clip_id INTEGER NOT NULL,
            source_revision INTEGER NOT NULL CHECK(source_revision>0), source_duration REAL,
            segmentation_version TEXT NOT NULL, window_signature TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('complete','partial')),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(clip_id,source_revision,window_signature),
            UNIQUE(event_set_id,project_id,clip_id),
            FOREIGN KEY(project_id,clip_id) REFERENCES {clips}(project_id,clip_id))""",
        f"""CREATE TABLE {events} (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL, clip_id INTEGER NOT NULL, event_set_id INTEGER NOT NULL,
            file_path TEXT NOT NULL, event_order INTEGER NOT NULL CHECK(event_order>0),
            start_seconds REAL NOT NULL CHECK(start_seconds>=0),
            end_seconds REAL NOT NULL CHECK(end_seconds>start_seconds), duration_seconds REAL,
            start_tick INTEGER NOT NULL, end_tick INTEGER NOT NULL CHECK(end_tick>start_tick),
            action TEXT, state_change TEXT, subjects TEXT, keywords TEXT, embedding TEXT,
            UNIQUE(event_set_id,event_order), UNIQUE(event_set_id,start_tick,end_tick),
            FOREIGN KEY(event_set_id,project_id,clip_id)
                REFERENCES {sets}(event_set_id,project_id,clip_id))""",
        f"""CREATE TABLE {steps} (
            step_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL, clip_id INTEGER NOT NULL, source_revision INTEGER NOT NULL,
            step TEXT NOT NULL, config_version TEXT NOT NULL, cache_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK(status IN ('complete','partial','failed','missing')),
            details_json TEXT, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(clip_id,source_revision,step,config_version),
            FOREIGN KEY(project_id,clip_id) REFERENCES {clips}(project_id,clip_id))""",
    )


INDEX_STATEMENTS = (
    "CREATE INDEX idx_clips_project ON clips(project_id)",
    "CREATE INDEX idx_clips_orientation ON clips(orientation)",
    "CREATE INDEX idx_clips_shot_type ON clips(shot_type)",
    "CREATE INDEX idx_clips_duration ON clips(duration_seconds)",
    "CREATE INDEX idx_events_project ON clip_events(project_id)",
    "CREATE INDEX idx_events_clip ON clip_events(clip_id)",
    "CREATE INDEX idx_events_file ON clip_events(file_path)",
    "CREATE INDEX idx_steps_clip ON analysis_steps(clip_id,source_revision)",
    """CREATE TRIGGER validate_active_event_set BEFORE UPDATE OF active_event_set_id,source_revision ON clips
       WHEN NEW.active_event_set_id IS NOT NULL AND NOT EXISTS (
         SELECT 1 FROM event_sets e WHERE e.event_set_id=NEW.active_event_set_id
         AND e.clip_id=NEW.clip_id AND e.project_id=NEW.project_id AND e.source_revision=NEW.source_revision)
       BEGIN SELECT RAISE(ABORT,'Active events do not belong to this clip/source version'); END""",
)

# Exposed for fresh-schema fixtures; migrations execute statements individually in a transaction.
SCHEMA_DDL = ";\n".join((*schema_statements(), *INDEX_STATEMENTS)) + ";\n"
