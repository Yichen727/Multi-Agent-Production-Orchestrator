"""Database service — SQLAlchemy setup with production schema and demo data.

The catalogue is a SQLite database. By default it is **persisted to a file**
(``settings.METADATA_DB_PATH``) so real ingested rows survive app restarts — this is
what makes the Ingest Agent's incremental reuse work across sessions (unchanged files
are not re-probed / re-tagged on a later run). The demo catalogue is seeded ONLY when
the database is empty, so a real ingest is never clobbered by the demo data on the next
launch. Tests point ``METADATA_DB_PATH`` at ``:memory:`` for isolation.
"""

import json
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine, event, text as sql_text
from sqlalchemy.pool import StaticPool
from langchain_community.utilities.sql_database import SQLDatabase
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("database_service")


# Schema DDL — always applied (idempotent via IF NOT EXISTS), so an existing file just
# gains any missing tables and indexes.
#
# NOTE (audit H-09): the ``UNIQUE (project_id, file_path)`` constraint below stops the
# same clip being catalogued twice in one project. Because ``CREATE TABLE IF NOT EXISTS``
# does NOT alter an already-existing table, this constraint only takes effect on a FRESH
# database (and every ``:memory:`` test run); a proper migration for pre-existing on-disk
# catalogues is tracked separately. The ``CREATE INDEX IF NOT EXISTS`` statements, by
# contrast, DO apply retroactively to an existing file.
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


# Tables from earlier designs that no agent or pipeline stage reads or writes any more:
# quality_assessments and transitions belonged to the removed Quality Agent, and
# user_preferences was never wired to anything. Dropped idempotently so a pre-existing
# on-disk catalogue is simplified too, not just a fresh one.
_DROP_UNUSED_TABLES = """
        DROP TABLE IF EXISTS quality_assessments;
        DROP TABLE IF EXISTS transitions;
        DROP TABLE IF EXISTS user_preferences;
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
             description, people_count, mood,
             embedding, source_mtime, source_size)
        VALUES
            (1, 1, '/footage/scene01/take01.mov', 'wide_shot', 12.5, 'exterior,daylight,establishing', 3840, 2160, 'landscape', 24.0, 'prores', 1, 1, 'Exterior establishing wide of the location in daylight.', 0, 'calm', NULL, NULL, NULL),
            (2, 1, '/footage/scene01/take02.mov', 'wide_shot', 13.1, 'exterior,daylight', 3840, 2160, 'landscape', 24.0, 'prores', 1, 1, 'Exterior daylight wide, second take.', 0, 'calm', NULL, NULL, NULL),
            (3, 1, '/footage/scene01/take03.mov', 'close_up', 8.2, 'dialogue,interior', 3840, 2160, 'landscape', 24.0, 'prores', 1, 1, 'Interior close-up of a subject delivering dialogue.', 1, 'tense', NULL, NULL, NULL),
            (4, 1, '/footage/scene02/take01.mov', 'establishing', 20.0, 'aerial,cityscape', 3840, 2160, 'landscape', 24.0, 'prores', 0, 3, 'Aerial establishing shot over a cityscape.', 0, 'cinematic', NULL, NULL, NULL),
            (5, 1, '/footage/scene02/take02.mov', 'medium_shot', 15.3, 'dialogue,two-shot', 3840, 2160, 'landscape', 24.0, 'prores', 1, 2, 'Interior medium two-shot during dialogue.', 2, 'calm', NULL, NULL, NULL);
"""


# Columns added to `shots` after the original schema shipped. `CREATE TABLE IF NOT
# EXISTS` will not alter a pre-existing on-disk table, so these are applied as an
# idempotent ALTER TABLE ADD COLUMN migration — additive, nullable, safe on every launch.
_ADDED_SHOT_COLUMNS = (
    ("audio_channels", "INTEGER"),
    ("audio_sample_rate", "INTEGER"),
    ("audio_bit_depth", "INTEGER"),
)

# Semantic columns removed from `shots`: nothing downstream read them (only `mood` is
# still used, in the embedded semantic text and the candidate preview), so they were pure
# schema weight. A fresh DB simply never creates them; a pre-existing on-disk table has
# them dropped here so `SELECT *` tools stop surfacing dead fields to the LLM.
_REMOVED_SHOT_COLUMNS = ("camera_motion", "lighting", "subject_position")


def _migrate_shot_columns(connection) -> None:
    """Reconcile the `shots` columns on both fresh and pre-existing databases.

    Adds the newer nullable columns (``CREATE TABLE IF NOT EXISTS`` will not alter an
    existing table) and drops the removed semantic ones. ``DROP COLUMN`` needs SQLite
    3.35+; on an older library the drop is skipped and the column just stays inert
    (nothing writes or reads it), so the catalogue still works.
    """
    existing = {row[1] for row in connection.execute("PRAGMA table_info(shots)").fetchall()}
    for name, decl in _ADDED_SHOT_COLUMNS:
        if name not in existing:
            connection.execute(f"ALTER TABLE shots ADD COLUMN {name} {decl}")
    for name in _REMOVED_SHOT_COLUMNS:
        if name in existing:
            try:
                connection.execute(f"ALTER TABLE shots DROP COLUMN {name}")
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not drop the unused shots.{name} column ({e}); "
                               "leaving it in place — nothing reads or writes it.")
    connection.commit()


def _apply_pragmas(connection, *, in_memory: bool) -> None:
    """Set the connection PRAGMAs (audit H-08) on a raw sqlite3 connection.

    Applied to EVERY connection (bootstrap and pooled), while no transaction is open —
    ``PRAGMA journal_mode`` / ``foreign_keys`` cannot change inside one, and both are
    per-connection settings, so a pooled connection that skipped them would silently
    lose FK enforcement (and with it the ``clip_events`` ON DELETE CASCADE).
      - ``foreign_keys=ON`` — actually enforce the declared FK relationships.
      - ``busy_timeout`` — wait rather than fail immediately under brief write contention.
      - ``journal_mode=WAL`` — concurrent readers + one writer (file databases only; WAL
        is not applicable to ``:memory:`` and is skipped there).
    """
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if not in_memory:
        connection.execute("PRAGMA journal_mode = WAL")


def _create_engine(db_target: str):
    """Open a SQLite database, ensure the schema, and seed demo data if empty.

    Args:
        db_target: A filesystem path for a persistent catalogue, or ':memory:' for an
            ephemeral one (used by tests).

    THREAD SAFETY (why a file catalogue is NOT StaticPool): LangGraph's ``ToolNode``
    executes an assistant turn's parallel tool calls in a THREAD POOL, so two catalogue
    queries can run at the same instant (e.g. Selection calling ``get_candidate_details``
    and ``get_clip_events`` together on a large curated set). A single shared DBAPI
    connection cannot serve them concurrently: their cursor traffic interleaves and a row
    ends up read against another statement's column metadata, surfacing as
    ``IndexError: tuple index out of range`` deep inside SQLAlchemy. So a FILE database is
    opened through SQLAlchemy's normal pool — every checkout gets its OWN sqlite3
    connection (WAL already allows concurrent readers alongside one writer), and the pool
    guarantees a connection is only ever used by one thread at a time.

    ``:memory:`` keeps the single-shared-connection StaticPool: an in-memory database
    exists only for the life of the connection that created it, so a per-checkout pool
    would hand out empty databases. That path is for the (single-threaded) tests.
    """
    in_memory = db_target == ":memory:"

    # Bootstrap connection: PRAGMAs, then schema + first-run seed.
    connection = sqlite3.connect(db_target, check_same_thread=False)
    _apply_pragmas(connection, in_memory=in_memory)
    connection.executescript(_SCHEMA_DDL)
    connection.executescript(_DROP_UNUSED_TABLES)
    _migrate_shot_columns(connection)

    # Seed the demo catalogue only when the database is brand new (no shots). This is
    # what lets real ingested rows persist across restarts without the demo clobbering
    # them on the next launch.
    existing = connection.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    if existing == 0:
        connection.executescript(_DEMO_SEED)
        logger.info("Empty catalogue — seeded demo data.")

    connection.commit()

    if in_memory:
        return create_engine(
            "sqlite://",
            creator=lambda: connection,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

    # File catalogue: hand the bootstrap connection back and let the pool open one
    # connection per checkout (see THREAD SAFETY above). check_same_thread=False because
    # a pooled connection may be reused by a different thread on a later checkout.
    connection.close()
    engine = create_engine(
        f"sqlite+pysqlite:///{Path(db_target).resolve().as_posix()}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _record):  # every NEW pooled connection
        _apply_pragmas(dbapi_connection, in_memory=False)

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
    "mood", "embedding",
    "source_mtime", "source_size",
    "audio_channels", "audio_sample_rate", "audio_bit_depth",
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
        # Ensure a parent projects row exists so the shots.project_id FK (now enforced,
        # audit H-08) is satisfied. Materialising the project on first ingest also avoids
        # orphan shots. INSERT OR IGNORE leaves an existing project (e.g. the demo
        # project 1) untouched.
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
    expensive ffprobe/scene-detection/GPT-5.4 Vision work for files whose path is
    already catalogued and whose content is unchanged.
    """
    from sqlalchemy import text as _sql

    with _engine.begin() as conn:
        result = conn.execute(
            _sql("SELECT * FROM shots WHERE project_id = :pid"),
            {"pid": project_id},
        )
        return {row._mapping["file_path"]: dict(row._mapping) for row in result}


# ── Temporal events (Tier 2: event-based ingestion) ────────────────────────────

# Columns accepted by replace_project_events, in the order the table expects. shot_id is
# resolved from the (project_id, file_path) of the freshly written shots, so callers pass
# file_path and need not know the autoincremented id.
_EVENT_COLUMNS = (
    "project_id", "shot_id", "file_path", "event_order",
    "start_seconds", "end_seconds", "duration_seconds",
    "action", "state_change", "subjects", "keywords", "embedding",
)


def replace_project_events(project_id: int, events: list[dict]) -> int:
    """Replace the temporal events for a project with a freshly built set.

    Each event dict carries a ``file_path`` (its parent clip); the parent ``shot_id`` is
    looked up from the shots table for this project at write time, so events stay linked
    even though ``replace_project_shots`` reassigns shot ids on every rebuild. Call this
    AFTER ``replace_project_shots`` (which cascade-deletes the old events). Events whose
    file has no catalogued shot are skipped rather than orphaned.

    Args:
        project_id: Project whose events are being (re)built.
        events: Ordered event dicts (keys a subset of ``_EVENT_COLUMNS`` minus the
            resolved ``shot_id``); ``subjects`` may be a list (stored as JSON text).

    Returns:
        The number of event rows inserted.
    """
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
        # Cascade may already have cleared these when shots were replaced; make the
        # rebuild idempotent regardless of call order.
        conn.execute(_sql("DELETE FROM clip_events WHERE project_id = :pid"),
                     {"pid": project_id})
        inserted = 0
        for ev in events:
            fp = ev.get("file_path")
            sid = shot_ids.get(fp)
            if sid is None:
                continue  # no parent shot for this file — never orphan an event
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
    """Return events (by id) for a project, keyed by event_id, with source duration.

    Scoped to ``project_id`` so an event id can never resolve across projects. Each value
    carries the event fields plus the parent clip's ``source_duration`` (for in/out
    range validation). Ids not found simply do not appear in the result.
    """
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
    """Return existing events for a project, grouped by parent file path.

    Each value is the ordered list of that clip's event dicts. The Ingest Agent uses
    this to REUSE events for unchanged clips (skipping the per-segment GPT-5.4 calls),
    mirroring how ``get_catalogued_shots`` lets it reuse clip-level analysis.
    """
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