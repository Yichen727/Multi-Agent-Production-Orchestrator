"""Versioned SQLite catalogue: stable IDs and atomic per-file publication."""

import json
import math
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text as sql_text
from sqlalchemy.pool import StaticPool
from langchain_community.utilities.sql_database import SQLDatabase
from app.config import settings
from app.services.catalogue_schema import MEDIA_COLUMNS, EVENT_COLUMNS, SCHEMA_DDL
from app.services.catalogue_migrations import initialise, step_key, window_signature
from app.services.media_identity import canonical_path, same_stat, fingerprint
from app.utils.logger import get_logger

logger = get_logger("database_service")
_SCHEMA_DDL = SCHEMA_DDL
_locks, _locks_guard = {}, threading.Lock()


def project_ingest_lock(project_id):
    """Serialize imports in the current UI process; worker leases belong to P2."""
    with _locks_guard:
        return _locks.setdefault(int(project_id), threading.RLock())


def _apply_pragmas(connection, *, in_memory):
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    if not in_memory:
        connection.execute("PRAGMA journal_mode=WAL")


def _create_engine(db_target, *, allow_migration=False):
    memory = db_target == ":memory:"
    connection = sqlite3.connect(db_target, check_same_thread=False)
    try:
        _apply_pragmas(connection, in_memory=memory)
        initialise(connection, allow_migration=allow_migration,
                   database_path=None if memory else db_target)
    except BaseException:
        connection.close()
        raise
    if memory:
        return create_engine("sqlite://", creator=lambda: connection, poolclass=StaticPool)
    connection.close()
    engine = create_engine(f"sqlite+pysqlite:///{Path(db_target).resolve().as_posix()}",
                           connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def pragmas(conn, _record):
        _apply_pragmas(conn, in_memory=False)
    return engine


def get_database():
    target = str(settings.METADATA_DB_PATH)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    engine = _create_engine(target)
    logger.info("Metadata database ready (%s).", target)
    return engine, SQLDatabase(engine)


_engine, db = get_database()


@contextmanager
def _write():
    with _engine.connect() as conn:
        conn.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def _insert(conn, table, values):
    columns = list(values)
    conn.execute(sql_text(f"INSERT INTO {table} ({','.join(columns)}) "
                          f"VALUES ({','.join(':'+c for c in columns)})"), values)
    return conn.execute(sql_text("SELECT last_insert_rowid()")).scalar_one()


def _legacy_clip(row):
    """Read-boundary alias, removed when P1b migrates the public model/state."""
    item = dict(row)
    item["shot_id"] = item["clip_id"]
    return item


def _changed(current, incoming):
    if incoming.get("content_hash") and current.get("content_hash"):
        return incoming["content_hash"] != current["content_hash"]
    return any(incoming.get(k) is not None and current.get(k) is not None
               and incoming[k] != current[k]
               for k in ("source_size", "source_mtime_ns", "source_mtime"))


def _step(conn, clip, name, status, config="v1", details=None):
    values = {"project_id": clip["project_id"], "clip_id": clip["clip_id"],
              "source_revision": clip["source_revision"], "step": name, "config_version": config,
              "cache_key": step_key(clip["clip_id"], clip["source_revision"], name, config),
              "status": status, "details_json": json.dumps(details) if details is not None else None}
    conn.execute(sql_text("""INSERT INTO analysis_steps
        (project_id,clip_id,source_revision,step,config_version,cache_key,status,details_json)
        VALUES (:project_id,:clip_id,:source_revision,:step,:config_version,:cache_key,:status,:details_json)
        ON CONFLICT(cache_key) DO UPDATE SET status=excluded.status,details_json=excluded.details_json,
            updated_at=CURRENT_TIMESTAMP"""), values)


def _publish_events(conn, clip, events, segmentation_version, complete):
    prepared = []
    for i, raw in enumerate(events, 1):
        start, end = float(raw["start_seconds"]), float(raw["end_seconds"])
        if not all(math.isfinite(n) for n in (start, end)) or not 0 <= start < end:
            raise ValueError("Event has an invalid source range")
        if clip.get("duration_seconds") and end > clip["duration_seconds"] + .005:
            raise ValueError("Event exceeds measured source duration")
        if raw.get("file_path") and canonical_path(raw["file_path"]) != clip["canonical_path"]:
            raise ValueError("Event does not belong to its parent clip")
        data = {key: raw.get(key) for key in EVENT_COLUMNS}
        data.update(file_path=clip["file_path"], event_order=int(raw.get("event_order") or i),
                    start_seconds=start, end_seconds=end,
                    duration_seconds=raw.get("duration_seconds", round(end - start, 3)),
                    start_tick=round(start * 1e6), end_tick=round(end * 1e6))
        if isinstance(data["subjects"], (list, tuple)):
            data["subjects"] = json.dumps(data["subjects"])
        prepared.append(data)
    signature = window_signature(prepared)
    existing = conn.execute(sql_text("""SELECT * FROM event_sets WHERE clip_id=:cid
        AND source_revision=:rev AND window_signature=:sig ORDER BY event_set_id LIMIT 1"""),
        {"cid": clip["clip_id"], "rev": clip["source_revision"], "sig": signature}).mappings().first()
    if existing:
        sid = existing["event_set_id"]
    else:
        sid = _insert(conn, "event_sets", {
            "project_id": clip["project_id"], "clip_id": clip["clip_id"],
            "source_revision": clip["source_revision"], "source_duration": clip.get("duration_seconds"),
            "segmentation_version": segmentation_version, "window_signature": signature,
            "status": "complete" if complete else "partial"})
    for data in prepared:
        row = conn.execute(sql_text("SELECT * FROM clip_events WHERE event_set_id=:sid AND event_order=:ord"),
                           {"sid": sid, "ord": data["event_order"]}).mappings().first()
        if row:
            updates = {k: v for k, v in data.items() if v is not None and
                       (v != "" or k not in ("action", "state_change", "keywords"))}
            if any(k in updates and updates[k] != row[k] for k in ("action", "state_change", "keywords")):
                updates["embedding"] = data.get("embedding")
            conn.execute(sql_text("UPDATE clip_events SET " + ",".join(f"{k}=:{k}" for k in updates)
                                  + " WHERE event_id=:eid"), {**updates, "eid": row["event_id"]})
        else:
            _insert(conn, "clip_events", {**data, "project_id": clip["project_id"],
                    "clip_id": clip["clip_id"], "event_set_id": sid})
    conn.execute(sql_text("UPDATE event_sets SET status=:status WHERE event_set_id=:sid"),
                 {"status": "complete" if complete else "partial", "sid": sid})
    conn.execute(sql_text("UPDATE clips SET active_event_set_id=:sid WHERE clip_id=:cid"),
                 {"sid": sid, "cid": clip["clip_id"]})
    return sid


def publish_clip(project_id, row, *, events=None, steps=None, segmentation_version="windows-v1",
                 events_complete=True, expected_revision=None):
    """Publish one file, events and statuses in one transaction, preserving existing IDs.

    events=None leaves events untouched. New source versions keep historical events,
    but only the active set is returned by catalogue queries.
    """
    pid, path = int(project_id), canonical_path(row["file_path"])
    with _write() as conn:
        conn.execute(sql_text("INSERT INTO projects(project_id,project_name) VALUES (:pid,:name) "
                              "ON CONFLICT(project_id) DO NOTHING"), {"pid": pid, "name": f"Project {pid}"})
        current = conn.execute(sql_text("SELECT * FROM clips WHERE project_id=:pid AND canonical_path=:path"),
                               {"pid": pid, "path": path}).mappings().first()
        if expected_revision is not None and (current["source_revision"] if current else 0) != expected_revision:
            raise ValueError("Catalogue source changed during analysis; result was not published")
        changed = bool(current and _changed(current, row))
        data = {k: row[k] for k in (*MEDIA_COLUMNS, "source_mtime_ns", "content_hash", "online") if k in row}
        data.update(file_path=path, canonical_path=path, project_id=pid)
        if current:
            cid = current["clip_id"]
            if changed:
                for k in ("shot_type", "keywords", "description", "people_count", "mood", "embedding"):
                    data.setdefault(k, None)
                data.update(source_revision=current["source_revision"] + 1, active_event_set_id=None)
            conn.execute(sql_text("UPDATE clips SET " + ",".join(f"{k}=:{k}" for k in data)
                                  + " WHERE clip_id=:cid"), {**data, "cid": cid})
            if current["file_path"] != path:
                conn.execute(sql_text("UPDATE clip_events SET file_path=:path WHERE clip_id=:cid"),
                             {"path": path, "cid": cid})
        else:
            cid = _insert(conn, "clips", data)
        clip = dict(conn.execute(sql_text("SELECT * FROM clips WHERE clip_id=:cid"), {"cid": cid}).mappings().one())
        if events is not None:
            _publish_events(conn, clip, events, segmentation_version, events_complete)
        for name, result in (steps or {}).items():
            result = {"status": result} if isinstance(result, str) else result
            _step(conn, clip, name, result["status"], result.get("config_version", "v1"), result.get("details"))
        if steps is None:
            for name, present in (("probe", bool(clip.get("duration_seconds"))),
                                  ("clip_tags", bool(clip.get("description") or clip.get("keywords"))),
                                  ("clip_embedding", bool(clip.get("embedding")))):
                _step(conn, clip, name, "complete" if present else "missing", "imported")
        if events is not None and "events" not in (steps or {}):
            _step(conn, clip, "events", "complete" if events_complete else "partial", segmentation_version)
        conn.execute(sql_text("UPDATE projects SET clip_count=(SELECT COUNT(*) FROM clips WHERE project_id=:pid) "
                              "WHERE project_id=:pid"), {"pid": pid})
        result = conn.execute(sql_text("SELECT * FROM clips WHERE clip_id=:cid"), {"cid": cid}).mappings().one()
        return _legacy_clip(result)


def upsert_project_clips(project_id, rows):
    for row in rows:
        publish_clip(project_id, row)
    return len(rows)


# Temporary function aliases for P1b callers; there are no legacy physical tables.
replace_project_shots = upsert_project_clips


def get_project_info(project_id):
    with _engine.connect() as conn:
        row = conn.execute(sql_text("SELECT * FROM projects WHERE project_id=:pid"), {"pid": project_id}).mappings().first()
        return dict(row) if row else None


def get_catalogued_clips(project_id):
    with _engine.connect() as conn:
        return {r["file_path"]: _legacy_clip(r) for r in conn.execute(
            sql_text("SELECT * FROM clips WHERE project_id=:pid ORDER BY clip_id"), {"pid": project_id}).mappings()}


get_catalogued_shots = get_catalogued_clips


def get_catalogued_paths(project_id):
    return set(get_catalogued_clips(project_id))


def get_analysis_steps(clip_id, source_revision):
    with _engine.connect() as conn:
        rows = conn.execute(sql_text("SELECT * FROM analysis_steps WHERE clip_id=:cid AND source_revision=:rev ORDER BY step_id"),
                            {"cid": clip_id, "rev": source_revision}).mappings()
        result = {}
        for row in rows:
            result[row["step"]] = {**row, "details": json.loads(row["details_json"]) if row["details_json"] else None}
        return result


def get_catalogued_events(project_id):
    with _engine.connect() as conn:
        rows = conn.execute(sql_text("""SELECT e.*, c.source_revision, c.duration_seconds AS source_duration
            FROM clip_events e JOIN clips c ON c.clip_id=e.clip_id AND c.project_id=e.project_id
            WHERE e.project_id=:pid AND e.event_set_id=c.active_event_set_id
            ORDER BY e.file_path,e.event_order"""), {"pid": project_id}).mappings()
        grouped = {}
        for row in rows:
            grouped.setdefault(row["file_path"], []).append(_legacy_clip(row))
        return grouped


def get_events_by_ids(project_id, event_ids):
    wanted = {int(i) for i in event_ids}
    return {e["event_id"]: e for group in get_catalogued_events(project_id).values() for e in group
            if e["event_id"] in wanted}


def replace_project_events(project_id, events):
    """Compatibility: update supplied files only, retaining identical-window IDs."""
    clips = {canonical_path(p): r for p, r in get_catalogued_clips(project_id).items()}
    grouped = {}
    for item in events:
        path = canonical_path(item["file_path"])
        if path not in clips:
            raise ValueError("Event parent is not in this project")
        grouped.setdefault(path, []).append(item)
    for path, items in grouped.items():
        publish_clip(project_id, clips[path], events=items)
    return len(events)


def mark_offline(project_id, paths):
    with _write() as conn:
        for path in paths:
            conn.execute(sql_text("UPDATE clips SET online=0 WHERE project_id=:pid AND canonical_path=:path"),
                         {"pid": project_id, "path": canonical_path(path)})


def validate_segment_source(segment, row):
    """Reject stale source/event references before any delivery file is written."""
    if segment.get("file_path") and canonical_path(segment["file_path"]) != row["canonical_path"]:
        raise ValueError("Plan ID/path mismatch; regenerate or explicitly relink the plan")
    if segment.get("source_revision") is None:
        raise ValueError("Plan has no source revision; regenerate it before export")
    if segment["source_revision"] != row["source_revision"]:
        raise ValueError("Plan refers to an older source revision")
    if segment.get("event_id") is not None:
        event = get_events_by_ids(row["project_id"], [segment["event_id"]]).get(int(segment["event_id"]))
        if not event or event["clip_id"] != row["clip_id"]:
            raise ValueError("Plan event is no longer active for this clip")
        if segment.get("event_set_id") not in (None, event["event_set_id"]):
            raise ValueError("Plan event set has changed")
    if not same_stat(row, fingerprint(row["file_path"])):
        raise ValueError("Media changed on disk or has no verified fingerprint; update the catalogue first")
