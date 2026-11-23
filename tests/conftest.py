"""Offline tests: no real dotenv, catalogue, LangSmith, credentials or network."""

import os
import socket
import uuid
from pathlib import Path

import pytest

# Set BEFORE importing any app module: config.py otherwise overrides the environment.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["OPENAI_API_KEY"] = "offline-test-not-a-real-key"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["METADATA_DB_PATH"] = ":memory:"
os.environ["METRICS_ENABLED"] = "false"


def pytest_configure(config):
    # Keep temp files in the workspace; the Windows user temp tree may belong to
    # another user. A fresh directory also avoids pytest deleting prior run data.
    if not config.option.basetemp:
        config.option.basetemp = str(Path(__file__).resolve().parents[1] / ".local" /
                                     "pytest" / uuid.uuid4().hex)


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in the P0 regression suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    from app.config import settings
    from app.services import database_service as database
    from sqlalchemy import text

    engine = database._create_engine(":memory:")
    # Modules import _engine by value: point every consumer at the same isolated DB.
    import app.services.catalogue_resolver as resolver
    import app.services.retrieval_service as retrieval
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(resolver, "_engine", engine)
    monkeypatch.setattr(retrieval, "_engine", engine)
    monkeypatch.setattr(settings, "PROCESSED_OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(settings, "METRICS_OUTPUT_DIR", tmp_path / "metrics")
    monkeypatch.setattr(settings, "METRICS_ENABLED", False)
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: None)
    # Ingest's read tools use SQLDatabase; make its wrapper use the isolated engine too.
    from langchain_community.utilities.sql_database import SQLDatabase
    import app.agents.ingest_agent as ingest
    monkeypatch.setattr(ingest, "db", SQLDatabase(engine))
    yield
    engine.dispose()


@pytest.fixture
def seed_catalogue(tmp_path):
    from app.services.database_service import replace_project_shots, get_catalogued_shots

    def seed(project_id=1, names=("arrival.mov", "goal.mov")):
        paths = []
        for name in names:
            path = tmp_path / str(project_id) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test media placeholder; never decoded")
            paths.append(str(path))
        rows = [{"file_path": path, "duration_seconds": 10, "fps": 25,
                 "width": 1920, "height": 1080, "orientation": "landscape",
                 "shot_type": "wide_shot", "has_audio": 0,
                 "keywords": "football,stadium" if i else "arrival,people",
                 "description": "A player kicks the football" if i else "People arrive",
                 "source_mtime": 123, "source_size": 10}
                for i, path in enumerate(paths)]
        replace_project_shots(project_id, rows)
        return list(get_catalogued_shots(project_id).values())
    return seed
