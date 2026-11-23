import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from app.services.catalogue_migrations import initialise, MigrationError

LEGACY_DDL = (Path(__file__).parent / "fixtures/legacy_catalogue.sql").read_text(encoding="utf-8")


@pytest.fixture
def legacy(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(LEGACY_DDL)
    conn.execute("INSERT INTO projects VALUES (1,'Example',2)")
    for cid in (17, 42):
        media = tmp_path / f"{cid}.mov"
        media.write_bytes(b"unchanged placeholder")
        stat = media.stat()
        conn.execute("""INSERT INTO shots(shot_id,project_id,file_path,duration_seconds,
            description,keywords,embedding,source_size,source_mtime)
            VALUES (?,1,?,10,'observed action','football','[0.1,0.2]',?,?)""",
            (cid, str(media), stat.st_size, stat.st_mtime))
        conn.execute("""INSERT INTO clip_events(event_id,project_id,shot_id,file_path,event_order,
            start_seconds,end_seconds,duration_seconds,action,subjects,keywords,embedding)
            VALUES (?,1,?,?,1,2,5,3,'a player runs','["player"]','running','[0.3,0.4]')""",
            (cid + 100, cid, str(media)))
    conn.commit()
    yield conn, path
    conn.close()


def test_migration_retains_ids_all_media_fields_and_event_content(legacy):
    conn, path = legacy
    old_clips = [dict(r) for r in conn.execute("SELECT * FROM shots ORDER BY shot_id")]
    old_events = [dict(r) for r in conn.execute("SELECT * FROM clip_events ORDER BY event_id")]
    report = initialise(conn, allow_migration=True, database_path=path)
    assert Path(report["backup_path"]).is_file()
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='shots'").fetchone() is None
    for old in old_clips:
        new = dict(conn.execute("SELECT * FROM clips WHERE clip_id=?", (old["shot_id"],)).fetchone())
        for key, value in old.items():
            assert new["clip_id" if key == "shot_id" else key] == value
        assert new["source_revision"] == 1
        assert new["source_mtime_ns"] is not None
    for old in old_events:
        new = dict(conn.execute("SELECT * FROM clip_events WHERE event_id=?", (old["event_id"],)).fetchone())
        for key, value in old.items():
            assert new["clip_id" if key == "shot_id" else key] == value
    assert conn.execute("SELECT COUNT(*) FROM analysis_steps WHERE config_version='legacy'").fetchone()[0] == 8
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    before = conn.total_changes
    assert initialise(conn, allow_migration=True, database_path=path)["migrated"] is False
    assert conn.total_changes == before


def test_migration_rollback_keeps_legacy_schema_and_results(legacy):
    conn, path = legacy
    def fail():
        raise RuntimeError("injected publication failure")
    with pytest.raises(RuntimeError, match="injected"):
        initialise(conn, allow_migration=True, database_path=path, failure_hook=fail)
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM clip_events").fetchone()[0] == 2
    assert not conn.execute("SELECT name FROM sqlite_master WHERE name LIKE '_p1a_%' OR name='schema_migrations'").fetchall()
    backups = list((path.parent / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as restored:
        assert restored.execute("SELECT embedding FROM shots WHERE shot_id=17").fetchone()[0] == "[0.1,0.2]"
    assert initialise(conn, allow_migration=True)["migrated"]


def test_legacy_database_requires_explicit_deployment(legacy):
    conn, _ = legacy
    with pytest.raises(MigrationError, match="migrate_catalogue"):
        initialise(conn)
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 2


def test_canonical_path_conflicts_are_reported_without_modifying_rows(legacy):
    conn, path = legacy
    first = conn.execute("SELECT file_path FROM shots WHERE shot_id=17").fetchone()[0]
    alias = str(Path(first).parent / "sub" / ".." / Path(first).name)
    conn.execute("UPDATE shots SET file_path=? WHERE shot_id=42", (alias,))
    conn.commit()
    with pytest.raises(MigrationError, match="Canonical path conflict"):
        initialise(conn, allow_migration=True, database_path=path)
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 2
    assert not (path.parent / "backups").exists()


def test_new_schema_enforces_event_uniqueness_and_parent_project(legacy):
    conn, _ = legacy
    initialise(conn, allow_migration=True)
    conn.execute("INSERT INTO projects VALUES (2,'Other',0)")
    event = dict(conn.execute("SELECT * FROM clip_events LIMIT 1").fetchone())
    event.pop("event_id")
    names = list(event)
    sql = f"INSERT INTO clip_events({','.join(names)}) VALUES ({','.join('?' for _ in names)})"
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, [event[n] for n in names])
    event["project_id"], event["event_order"] = 2, 9
    event["start_tick"], event["end_tick"] = 6_000_000, 8_000_000
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, [event[n] for n in names])
