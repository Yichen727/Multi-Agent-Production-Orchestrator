import sqlite3
import zipfile
import json

import pytest

from scripts import baseline
from pathlib import Path

_SCHEMA_DDL = (Path(__file__).parent / "fixtures/legacy_catalogue.sql").read_text(encoding="utf-8")


def test_backup_includes_committed_wal_and_excludes_secrets_media(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("OPENAI_API_KEY=must-not-copy")
    (root / "main.py").write_text("print('hello')")
    (root / "video.mov").write_bytes(b"must-not-copy")
    exports = root / "app/data/output/exports"
    exports.mkdir(parents=True)
    (exports / "example.json").write_text("{}")
    monkeypatch.setattr(baseline, "ROOT", root)
    monkeypatch.setattr(baseline, "dependency_versions", lambda: {"pytest": "9.1.1"})
    monkeypatch.setattr(baseline, "command_output", lambda args: "offline test")
    db_path = root / "catalogue.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA_DDL)
    conn.execute("INSERT INTO projects VALUES (1, 'test', 1)")
    conn.execute("INSERT INTO shots (project_id,file_path,duration_seconds) VALUES (1,'a.mov',10)")
    conn.commit()  # Connection remains open: committed row lives in WAL.
    try:
        backup = baseline.capture(db_path, tmp_path / "backups")
        assert baseline.verify(backup)["status"] == "ok"
        with sqlite3.connect(backup / "catalogue.sqlite3") as restored:
            assert restored.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 1
            assert restored.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        restored.close()
        assert baseline.verify(backup)["status"] == "ok"
        assert not any(name.endswith(("-wal", "-shm")) for name in
                       json.loads((backup / "manifest.json").read_text())["files"])
        with zipfile.ZipFile(backup / "source.zip") as archive:
            assert "main.py" in archive.namelist()
            assert ".env" not in archive.namelist()
            assert "video.mov" not in archive.namelist()
        (backup / "exports/example.json").write_text("tampered")
        with pytest.raises(ValueError, match="checksum"):
            baseline.verify(backup)
    finally:
        conn.close()


def test_legacy_manifest_ignores_transient_sidecars_but_rejects_new_wal(tmp_path):
    database = tmp_path / "catalogue.sqlite3"
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE example(id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    manifest = {"files": {
        database.name: {"bytes": database.stat().st_size, "sha256": baseline.sha256(database)},
        "catalogue.sqlite3-wal": {"bytes": 0, "sha256": "legacy-empty-wal"},
        "catalogue.sqlite3-shm": {"bytes": 32768, "sha256": "transient"},
    }}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert baseline.verify(tmp_path)["verified_files"] == 1
    (tmp_path / "catalogue.sqlite3-wal").write_bytes(b"unexpected pending write")
    with pytest.raises(ValueError, match="nonempty WAL"):
        baseline.verify(tmp_path)
