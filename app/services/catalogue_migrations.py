"""Transactional, explicitly deployed legacy catalogue migration; never calls models."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.services.catalogue_schema import (
    SCHEMA_VERSION, MEDIA_COLUMNS, EVENT_COLUMNS, schema_statements, INDEX_STATEMENTS,
)
from app.services.media_identity import canonical_path, fingerprint


class MigrationError(ValueError):
    pass


def window_signature(events) -> str:
    windows = [(int(e["event_order"]), round(float(e["start_seconds"]) * 1_000_000),
                round(float(e["end_seconds"]) * 1_000_000)) for e in events]
    return hashlib.sha256(json.dumps(sorted(windows), separators=(",", ":")).encode()).hexdigest()


def step_key(clip_id, revision, step, config_version) -> str:
    return hashlib.sha256(json.dumps([clip_id, revision, step, config_version],
                                    separators=(",", ":")).encode()).hexdigest()


def tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def schema_version(conn) -> int:
    if "schema_migrations" not in tables(conn):
        return 0
    return conn.execute("SELECT COALESCE(MAX(version),0) FROM schema_migrations").fetchone()[0]


def _legacy_rows(conn):
    clips = [dict(r) for r in conn.execute("SELECT * FROM shots ORDER BY shot_id")]
    events = [dict(r) for r in conn.execute("SELECT * FROM clip_events ORDER BY event_id")]
    by_id = {c["shot_id"]: c for c in clips}
    known_projects = {r[0] for r in conn.execute("SELECT project_id FROM projects")}
    paths, windows, orders = {}, set(), set()
    for clip in clips:
        unknown = set(clip) - {*MEDIA_COLUMNS, "shot_id", "project_id"}
        if unknown:
            raise MigrationError(f"Unsupported legacy clip columns: {sorted(unknown)}")
        if clip["project_id"] not in known_projects:
            raise MigrationError(f"Orphan project reference on clip {clip['shot_id']}")
        key = (clip["project_id"], canonical_path(clip["file_path"]))
        if key in paths:
            raise MigrationError(f"Canonical path conflict between IDs {paths[key]} and {clip['shot_id']}")
        paths[key] = clip["shot_id"]
        clip["canonical_path"] = key[1]
    for event in events:
        unknown = set(event) - {*EVENT_COLUMNS, "event_id", "shot_id", "project_id"}
        if unknown:
            raise MigrationError(f"Unsupported legacy event columns: {sorted(unknown)}")
        parent = by_id.get(event["shot_id"])
        if parent is None or parent["project_id"] != event["project_id"]:
            raise MigrationError(f"Orphan/cross-project event {event['event_id']}")
        if canonical_path(event["file_path"]) != parent["canonical_path"]:
            raise MigrationError(f"Event path differs from parent: {event['event_id']}")
        start, end = event.get("start_seconds"), event.get("end_seconds")
        if start is None or end is None or not all(math.isfinite(n) for n in (start, end)) or not 0 <= start < end:
            raise MigrationError(f"Invalid window on event {event['event_id']}")
        if parent.get("duration_seconds") and end > parent["duration_seconds"] + .005:
            raise MigrationError(f"Event exceeds source duration: {event['event_id']}")
        window = (event["shot_id"], round(start * 1e6), round(end * 1e6))
        order = (event["shot_id"], event.get("event_order"))
        if not event.get("event_order") or event["event_order"] <= 0 or window in windows or order in orders:
            raise MigrationError(f"Duplicate/invalid event window/order: {event['event_id']}")
        windows.add(window)
        orders.add(order)
    return clips, events


def _insert(conn, table, data):
    columns = list(data)
    conn.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                 [data[c] for c in columns])


def _backup(conn, database: Path) -> Path:
    folder = database.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    name = database.stem + ".pre-v1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = folder / (name + "-" + uuid.uuid4().hex[:8] + ".sqlite3")
    with closing(sqlite3.connect(target)) as backup:
        conn.backup(backup)
        backup.execute("PRAGMA journal_mode=DELETE")
        if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise MigrationError("Pre-migration backup failed integrity check")
    return target


def initialise(conn, *, allow_migration=False, database_path=None, failure_hook=None):
    """Create fresh v1, or explicitly migrate legacy v0 with backup and rollback."""
    conn.row_factory = sqlite3.Row
    current = schema_version(conn)
    if current > SCHEMA_VERSION:
        raise MigrationError("Catalogue schema is newer than this application")
    if current == SCHEMA_VERSION:
        return {"version": current, "migrated": False, "backup_path": None}
    existing = tables(conn)
    legacy = "shots" in existing
    if "clips" in existing or (existing - {"sqlite_sequence"} and not legacy):
        raise MigrationError("Unversioned/unsupported catalogue; inspect it before migration")
    if legacy and not allow_migration:
        raise MigrationError("Legacy catalogue requires: python -m scripts.migrate_catalogue <database-path>")
    # Preflight before any DDL. Repeat under the write lock to cover concurrent writers.
    if legacy:
        _legacy_rows(conn)
    backup = _backup(conn, Path(database_path)) if legacy and database_path else None
    conn.execute("BEGIN IMMEDIATE")
    try:
        if schema_version(conn) == SCHEMA_VERSION:
            conn.rollback()
            return {"version": SCHEMA_VERSION, "migrated": False, "backup_path": str(backup) if backup else None}
        clips, events = _legacy_rows(conn) if legacy else ([], [])
        prefix = "_p1a_" if legacy else ""
        for statement in schema_statements(prefix):
            conn.execute(statement)
        if legacy:
            columns = {r[1] for r in conn.execute("PRAGMA table_info(projects)")}
            if "clip_count" not in columns:
                conn.execute("ALTER TABLE projects ADD COLUMN clip_count INTEGER NOT NULL DEFAULT 0")
            grouped = {}
            for event in events:
                grouped.setdefault(event["shot_id"], []).append(event)
            for clip in clips:
                cid = clip["shot_id"]
                data = {k: clip.get(k) for k in MEDIA_COLUMNS}
                data.update(clip_id=cid, project_id=clip["project_id"], canonical_path=clip["canonical_path"])
                try:
                    stat = fingerprint(clip["file_path"])
                    if stat["source_size"] == clip.get("source_size") and stat["source_mtime"] == clip.get("source_mtime"):
                        data["source_mtime_ns"] = stat["source_mtime_ns"]
                except OSError:
                    data["online"] = 0
                _insert(conn, prefix + "clips", data)
                source_events = grouped.get(cid, [])
                if source_events:
                    _insert(conn, prefix + "event_sets", {
                        "project_id": clip["project_id"], "clip_id": cid, "source_revision": 1,
                        "source_duration": clip.get("duration_seconds"), "segmentation_version": "legacy",
                        "window_signature": window_signature(source_events), "status": "complete"})
                    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    for event in source_events:
                        data = {k: event.get(k) for k in EVENT_COLUMNS}
                        data.update(event_id=event["event_id"], clip_id=cid, project_id=clip["project_id"],
                                    event_set_id=sid, start_tick=round(event["start_seconds"] * 1e6),
                                    end_tick=round(event["end_seconds"] * 1e6))
                        _insert(conn, prefix + "clip_events", data)
                    conn.execute(f"UPDATE {prefix}clips SET active_event_set_id=? WHERE clip_id=?", (sid, cid))
                states = {"probe": bool(clip.get("duration_seconds")),
                          "clip_tags": bool(clip.get("description") or clip.get("keywords")),
                          "clip_embedding": bool(clip.get("embedding")), "events": bool(source_events)}
                for step, complete in states.items():
                    _insert(conn, prefix + "analysis_steps", {
                        "project_id": clip["project_id"], "clip_id": cid, "source_revision": 1,
                        "step": step, "config_version": "legacy", "cache_key": step_key(cid, 1, step, "legacy"),
                        "status": "complete" if complete else "missing"})
            if failure_hook:
                failure_hook()  # Deliberate failure injection for rollback regression tests.
            conn.execute("DROP TABLE clip_events")
            conn.execute("DROP TABLE shots")
            for name in ("clips", "event_sets", "clip_events", "analysis_steps"):
                conn.execute(f"ALTER TABLE {prefix}{name} RENAME TO {name}")
        for statement in INDEX_STATEMENTS:
            conn.execute(statement)
        conn.execute("UPDATE projects SET clip_count=(SELECT COUNT(*) FROM clips c WHERE c.project_id=projects.project_id)")
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (SCHEMA_VERSION,))
        if conn.execute("PRAGMA foreign_key_check").fetchone():
            raise MigrationError("Foreign-key validation failed")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise MigrationError("Catalogue integrity check failed")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return {"version": SCHEMA_VERSION, "migrated": legacy,
            "clips": len(clips), "events": len(events), "backup_path": str(backup) if backup else None}
