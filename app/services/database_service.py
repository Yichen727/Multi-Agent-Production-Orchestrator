"""Database service — SQLite catalogue and persistence helpers."""

import json
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine, event, text as sql_text
from sqlalchemy.pool import StaticPool
from langchain_community.utilities.sql_database import SQLDatabase
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("database_service")


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_DDL = """
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
"""

_ADDED_SHOT_COLUMNS = (
    ("audio_channels", "INTEGER"),
    ("audio_sample_rate", "INTEGER"),
    ("audio_bit_depth", "INTEGER"),
)


def _migrate_shot_columns(connection) -> None:
    """Add missing columns to an existing shots table."""
    existing = {row[1] for row in connection.execute("PRAGMA table_info(shots)").fetchall()}
    for name, decl in _ADDED_SHOT_COLUMNS:
        if name not in existing:
            connection.execute(f"ALTER TABLE shots ADD COLUMN {name} {decl}")
    connection.commit()


def _migrate_project_columns(connection) -> None:
    """Add and backfill clip_count on existing projects."""
    existing = {row[1] for row in connection.execute("PRAGMA table_info(projects)").fetchall()}
    if "clip_count" not in existing:
        connection.execute("ALTER TABLE projects ADD COLUMN clip_count INTEGER NOT NULL DEFAULT 0")
        connection.execute(
            "UPDATE projects SET clip_count = ("
            "    SELECT COUNT(*) FROM shots WHERE shots.project_id = projects.project_id)"
        )
        connection.commit()


def _apply_pragmas(connection, *, in_memory: bool) -> None:
    """Configure SQLite connection settings."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if not in_memory:
        connection.execute("PRAGMA journal_mode = WAL")


def _create_engine(db_target: str):
    """Create the SQLite engine and initialise the schema.

    File databases use SQLAlchemy's normal connection pool for thread safety.
    In-memory databases use StaticPool so all operations share one connection.
    """
    in_memory = db_target == ":memory:"

    connection = sqlite3.connect(db_target, check_same_thread=False)
    _apply_pragmas(connection, in_memory=in_memory)
    connection.executescript(_SCHEMA_DDL)
    _migrate_shot_columns(connection)
    _migrate_project_columns(connection)
    connection.commit()

    if in_memory:
        return create_engine(
            "sqlite://",
            creator=lambda: connection,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

    connection.close()

    engine = create_engine(
        f"sqlite+pysqlite:///{Path(db_target).resolve().as_posix()}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _record): 
        _apply_pragmas(dbapi_connection, in_memory=False)

    return engine


def get_database():
    """Create the metadata database and LangChain SQL wrapper."""
    target = str(settings.METADATA_DB_PATH)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)

    engine = _create_engine(target)
    database = SQLDatabase(engine)
    where = "in-memory" if target == ":memory:" else target
    logger.info(f"Metadata database ready ({where}).")
    return engine, database

_engine, db = get_database()

# ── Shot operations ───────────────────────────────────────────────────────────
_SHOT_COLUMNS = (
    "project_id", "file_path", "shot_type",
    "duration_seconds",
    "keywords", "width", "height", "orientation",
    "fps", "codec", "has_audio", "scene_count", "description", "people_count",
    "mood", "embedding",
    "source_mtime", "source_size",
    "audio_channels", "audio_sample_rate", "audio_bit_depth",
)


def replace_project_shots(project_id: int, rows: list[dict]) -> int:
    """Replace all catalogue shots for a project."""
    from sqlalchemy import text as _sql

    placeholders = ", ".join(f":{c}" for c in _SHOT_COLUMNS)
    columns = ", ".join(_SHOT_COLUMNS)
    insert_sql = _sql(f"INSERT INTO shots ({columns}) VALUES ({placeholders})")

    with _engine.begin() as conn:
        conn.execute(
            _sql("INSERT OR IGNORE INTO projects (project_id, project_name) "
                 "VALUES (:pid, :name)"),
            {"pid": project_id, "name": f"Project {project_id}"},
        )
        conn.execute(_sql("DELETE FROM shots WHERE project_id = :pid"), {"pid": project_id})
        for row in rows:
            params = {c: row.get(c) for c in _SHOT_COLUMNS}
            params["project_id"] = project_id
            conn.execute(insert_sql, params)
        
        conn.execute(
            _sql("UPDATE projects SET clip_count = :n WHERE project_id = :pid"),
            {"n": len(rows), "pid": project_id},
        )

    return len(rows)


def get_project_info(project_id: int) -> dict | None:
    """Return project metadata, or None if it has not been ingested."""
    from sqlalchemy import text as _sql

    with _engine.begin() as conn:
        row = conn.execute(
            _sql("SELECT project_id, project_name, clip_count "
                 "FROM projects WHERE project_id = :pid"),
            {"pid": project_id},
        ).first()
        return dict(row._mapping) if row else None


def get_catalogued_paths(project_id: int) -> set[str]:
    """Return file paths already catalogued for a project."""
    from sqlalchemy import text as _sql

    with _engine.begin() as conn:
        result = conn.execute(
            _sql("SELECT file_path FROM shots WHERE project_id = :pid"),
            {"pid": project_id},
        )
        return {row[0] for row in result}


def get_catalogued_shots(project_id: int) -> dict:
    """Return existing catalogue rows keyed by file path."""
    from sqlalchemy import text as _sql

    with _engine.begin() as conn:
        result = conn.execute(
            _sql("SELECT * FROM shots WHERE project_id = :pid"),
            {"pid": project_id},
        )
        return {row._mapping["file_path"]: dict(row._mapping) for row in result}


# ── Temporal events ───────────────────────────────────────────────────────────
_EVENT_COLUMNS = (
    "project_id", "shot_id", "file_path", "event_order",
    "start_seconds", "end_seconds", "duration_seconds",
    "action", "state_change", "subjects", "keywords", "embedding",
)


def replace_project_events(project_id: int, events: list[dict]) -> int:
    """Replace all temporal events for a project."""
    from sqlalchemy import text as _sql

    placeholders = ", ".join(f":{c}" for c in _EVENT_COLUMNS)
    columns = ", ".join(_EVENT_COLUMNS)
    insert_sql = _sql(f"INSERT INTO clip_events ({columns}) VALUES ({placeholders})")

    with _engine.begin() as conn:
        shot_ids = {
            row._mapping["file_path"]: row._mapping["shot_id"]
            for row in conn.execute(
                _sql("SELECT shot_id, file_path FROM shots WHERE project_id = :pid"),
                {"pid": project_id},
            )
        }
        
        conn.execute(_sql("DELETE FROM clip_events WHERE project_id = :pid"),
                     {"pid": project_id})
        inserted = 0
        for ev in events:
            fp = ev.get("file_path")
            sid = shot_ids.get(fp)
            if sid is None:
                continue  
            params = {c: ev.get(c) for c in _EVENT_COLUMNS}
            params["project_id"] = project_id
            params["shot_id"] = sid
            subj = params.get("subjects")
            if isinstance(subj, (list, tuple)):
                params["subjects"] = json.dumps(list(subj))
            conn.execute(insert_sql, params)
            inserted += 1

    return inserted


def get_events_by_ids(project_id: int, event_ids: list[int]) -> dict:
    """Return events by ID, scoped to a project."""
    from sqlalchemy import text as _sql

    if not event_ids:
        return {}
    ids = [int(i) for i in event_ids]
    placeholders = ", ".join(f":id{i}" for i in range(len(ids)))
    params = {f"id{i}": v for i, v in enumerate(ids)}
    params["pid"] = project_id
    with _engine.begin() as conn:
        result = conn.execute(
            _sql(f"SELECT e.*, s.duration_seconds AS source_duration "
                 f"FROM clip_events e JOIN shots s ON s.shot_id = e.shot_id "
                 f"WHERE e.project_id = :pid AND e.event_id IN ({placeholders})"),
            params,
        )
        return {row._mapping["event_id"]: dict(row._mapping) for row in result}


def get_catalogued_events(project_id: int) -> dict:
    """Return existing events grouped by parent file path."""
    from sqlalchemy import text as _sql

    grouped: dict[str, list[dict]] = {}
    with _engine.begin() as conn:
        result = conn.execute(
            _sql("SELECT * FROM clip_events WHERE project_id = :pid "
                 "ORDER BY file_path, event_order"),
            {"pid": project_id},
        )
        for row in result:
            d = dict(row._mapping)
            grouped.setdefault(d["file_path"], []).append(d)
    return grouped