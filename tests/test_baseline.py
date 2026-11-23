import json
import sqlite3
import zipfile

import pytest

from scripts import baseline
from app.services.database_service import _SCHEMA_DDL


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
        with zipfile.ZipFile(backup / "source.zip") as archive:
            assert "main.py" in archive.namelist()
            assert ".env" not in archive.namelist()
            assert "video.mov" not in archive.namelist()
        (backup / "exports/example.json").write_text("tampered")
        with pytest.raises(ValueError, match="checksum"):
            baseline.verify(backup)
    finally:
        conn.close()
