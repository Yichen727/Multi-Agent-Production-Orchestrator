"""Database service — SQLAlchemy setup with production schema and demo data.

The catalogue is a SQLite database. By default it is **persisted to a file**
(``settings.METADATA_DB_PATH``) so real ingested rows survive app restarts — this is
what makes the Ingest Agent's incremental reuse work across sessions (unchanged files
are not re-probed / re-tagged on a later run). The demo catalogue is seeded ONLY when
the database is empty, so a real ingest is never clobbered by the demo data on the next
launch. Tests point ``METADATA_DB_PATH`` at ``:memory:`` for isolation.
"""

import sqlite3
from pathlib import Path
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.pool import StaticPool
from langchain_community.utilities.sql_database import SQLDatabase
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("database_service")


# Schema DDL — always applied (idempotent via IF NOT EXISTS), so an existing file just
# gains any missing tables.
_SCHEMA_DDL = """
        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT NOT NULL,
            client_name TEXT,
            created_date TEXT,
            frame_rate REAL,
            resolution TEXT
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
            camera_motion TEXT,
            lighting TEXT,
            mood TEXT,
            subject_position TEXT,
            embedding TEXT,
            source_mtime REAL,
            source_size INTEGER,
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        );

        CREATE TABLE IF NOT EXISTS quality_assessments (
            assessment_id INTEGER PRIMARY KEY,
            shot_id INTEGER,
            exposure_score REAL,
            white_balance_score REAL,
            contrast_score REAL,
            colour_consistency_score REAL,
            overall_score REAL,
            recommendations TEXT,
            FOREIGN KEY (shot_id) REFERENCES shots(shot_id)
        );

        CREATE TABLE IF NOT EXISTS transitions (
            transition_id INTEGER PRIMARY KEY,
            from_shot_id INTEGER,
            to_shot_id INTEGER,
            transition_type TEXT,
            duration_frames INTEGER,
            confidence_score REAL,
            FOREIGN KEY (from_shot_id) REFERENCES shots(shot_id),
            FOREIGN KEY (to_shot_id) REFERENCES shots(shot_id)
        );

        CREATE TABLE IF NOT EXISTS user_preferences (
            preference_id INTEGER PRIMARY KEY,
            user_id TEXT,
            key TEXT,
            value TEXT,
            updated_at TEXT
        );
"""


# Demo catalogue — seeded ONLY when the shots table is empty (fresh database), so a
# real ingest is never overwritten on the next launch.
_DEMO_SEED = """
        INSERT INTO projects VALUES
            (1, 'Demo Project - Short Film', 'Atomized Studios', '2025-06-01', 24.0, '3840x2160');

        INSERT INTO shots
            (shot_id, project_id, file_path, shot_type,
             duration_seconds, keywords,
             width, height, orientation, fps, codec, has_audio, scene_count,
             description, people_count, camera_motion, lighting, mood, subject_position,
             embedding, source_mtime, source_size)
        VALUES
            (1, 1, '/footage/scene01/take01.mov', 'wide_shot', 12.5, 'exterior,daylight,establishing', 3840, 2160, 'landscape', 24.0, 'prores', 1, 1, 'Exterior establishing wide of the location in daylight.', 0, 'static', 'natural', 'calm', 'center', NULL, NULL, NULL),
            (2, 1, '/footage/scene01/take02.mov', 'wide_shot', 13.1, 'exterior,daylight', 3840, 2160, 'landscape', 24.0, 'prores', 1, 1, 'Exterior daylight wide, second take.', 0, 'static', 'natural', 'calm', 'center', NULL, NULL, NULL),
            (3, 1, '/footage/scene01/take03.mov', 'close_up', 8.2, 'dialogue,interior', 3840, 2160, 'landscape', 24.0, 'prores', 1, 1, 'Interior close-up of a subject delivering dialogue.', 1, 'static', 'studio', 'tense', 'center', NULL, NULL, NULL),
            (4, 1, '/footage/scene02/take01.mov', 'establishing', 20.0, 'aerial,cityscape', 3840, 2160, 'landscape', 24.0, 'prores', 0, 3, 'Aerial establishing shot over a cityscape.', 0, 'pan', 'natural', 'cinematic', 'moving', NULL, NULL, NULL),
            (5, 1, '/footage/scene02/take02.mov', 'medium_shot', 15.3, 'dialogue,two-shot', 3840, 2160, 'landscape', 24.0, 'prores', 1, 2, 'Interior medium two-shot during dialogue.', 2, 'handheld', 'studio', 'calm', 'center', NULL, NULL, NULL);

        INSERT INTO user_preferences VALUES
            (1, 'editor_01', 'preferred_color_space', 'Rec.709', '2025-06-01'),
            (2, 'editor_01', 'preferred_output_format', 'ProRes 422 HQ', '2025-06-01');
"""


def _create_engine(db_target: str):
    """Open a SQLite database, ensure the schema, and seed demo data if empty.

    Args:
        db_target: A filesystem path for a persistent catalogue, or ':memory:' for an
            ephemeral one (used by tests).

    A single shared connection is kept alive via StaticPool — correct for the app's
    single-process model and required for an in-memory database to persist for the
    life of the process.
    """
    connection = sqlite3.connect(db_target, check_same_thread=False)
    connection.executescript(_SCHEMA_DDL)

    # Seed the demo catalogue only when the database is brand new (no shots). This is
    # what lets real ingested rows persist across restarts without the demo clobbering
    # them on the next launch.
    existing = connection.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    if existing == 0:
        connection.executescript(_DEMO_SEED)
        logger.info("Empty catalogue — seeded demo data.")

    connection.commit()

    engine = create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    return engine


def get_database():
    """Build the metadata database instance.

    Persists to ``settings.METADATA_DB_PATH`` (a file) by default so real ingested
    rows survive restarts; set it to ':memory:' for an ephemeral database. Swap to
    PostgreSQL for true production by changing DATABASE_URL in .env.

    Returns the SQLAlchemy engine and a LangChain ``SQLDatabase`` over it.
    """
    target = str(settings.METADATA_DB_PATH)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)

    engine = _create_engine(target)
    database = SQLDatabase(engine)
    where = "in-memory" if target == ":memory:" else target
    logger.info(f"Metadata database ready ({where}).")
    return engine, database


# Singleton instances. ``db`` is the read-oriented LangChain wrapper agents query
# with db.run("SELECT ..."); ``_engine`` is used for the write path (real ingestion).
_engine, db = get_database()


# Columns accepted by replace_project_shots, in the order the catalogue expects.
_SHOT_COLUMNS = (
    "project_id", "file_path", "shot_type",
    "duration_seconds",
    "keywords", "width", "height", "orientation",
    "fps", "codec", "has_audio", "scene_count", "description", "people_count",
    "camera_motion", "lighting", "mood", "subject_position", "embedding",
    "source_mtime", "source_size",
)


def replace_project_shots(project_id: int, rows: list[dict]) -> int:
    """Replace the catalogue for a project with freshly ingested rows.

    Deletes the project's existing shots, then inserts ``rows`` (each a dict whose
    keys are a subset of ``_SHOT_COLUMNS``; missing keys are stored as NULL). This
    is how REAL footage metadata, probed from the user's files, enters the database
    — replacing the demo seed once the user runs ingest. Attributes that were not
    measured should simply be omitted rather than guessed.

    Args:
        project_id: Project whose catalogue is being (re)built.
        rows: List of shot metadata dicts.

    Returns:
        The number of rows inserted.
    """
    from sqlalchemy import text as _sql

    placeholders = ", ".join(f":{c}" for c in _SHOT_COLUMNS)
    columns = ", ".join(_SHOT_COLUMNS)
    insert_sql = _sql(f"INSERT INTO shots ({columns}) VALUES ({placeholders})")

    with _engine.begin() as conn:
        conn.execute(_sql("DELETE FROM shots WHERE project_id = :pid"), {"pid": project_id})
        for row in rows:
            params = {c: row.get(c) for c in _SHOT_COLUMNS}
            params["project_id"] = project_id
            conn.execute(insert_sql, params)

    return len(rows)


def get_catalogued_paths(project_id: int) -> set[str]:
    """Return the set of file paths already catalogued for a project.

    Used by the Ingest Agent's "detect new footage" step to tell which files on
    disk are new versus already ingested.
    """
    from sqlalchemy import text as _sql

    with _engine.begin() as conn:
        result = conn.execute(
            _sql("SELECT file_path FROM shots WHERE project_id = :pid"),
            {"pid": project_id},
        )
        return {row[0] for row in result}


def get_catalogued_shots(project_id: int) -> dict:
    """Return existing catalogue rows for a project, keyed by file path.

    Each value is a dict of every column (including the source_mtime / source_size
    fingerprint). The Ingest Agent uses this to REUSE prior analysis — skipping the
    expensive ffprobe/scene-detection/GPT-4o Vision work for files whose path is
    already catalogued and whose content is unchanged.
    """
    from sqlalchemy import text as _sql

    with _engine.begin() as conn:
        result = conn.execute(
            _sql("SELECT * FROM shots WHERE project_id = :pid"),
            {"pid": project_id},
        )
        return {row._mapping["file_path"]: dict(row._mapping) for row in result}