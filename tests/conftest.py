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
        parent = Path(__file__).resolve().parents[1] / ".local" / "pytest"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid.uuid4().hex)


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in the P0 regression suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    from app.config import settings
    from app.services import database_service as database
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
    yield
    engine.dispose()


@pytest.fixture
def seed_catalogue(tmp_path):
    from app.services.database_service import upsert_project_clips, get_catalogued_clips
    from app.services.media_identity import fingerprint

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
                 **fingerprint(path)}
                for i, path in enumerate(paths)]
        upsert_project_clips(project_id, rows)
        return list(get_catalogued_clips(project_id).values())
    return seed


@pytest.fixture
def media_analysis(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from app.models.schemas import VisionTags, EventTags
    from app.services import ingest_service as ingest

    folder = tmp_path / "footage"
    folder.mkdir()
    media = folder / "same.mov"
    media.write_bytes(b"initial video")
    probe = Mock(return_value={"ok": True, "duration_seconds": 10, "width": 1920, "height": 1080,
        "orientation": "landscape", "fps": 25, "codec": "h264", "has_audio": False})
    clip_tags = Mock(return_value=VisionTags(scene_description="A player runs", keywords=["running"]))
    event_tags = Mock(return_value=EventTags(action="A player runs", keywords=["running"]))
    monkeypatch.setattr(ingest, "check_ffprobe_installed", lambda: True)
    monkeypatch.setattr(ingest, "probe_video_metadata", probe)
    monkeypatch.setattr(ingest, "detect_scene_cuts", Mock(return_value={
        "ran": True, "complete": True, "cut_times": [], "scene_count": 1}))
    monkeypatch.setattr(ingest, "extract_scene_representative_frames_b64", lambda *a, **k: (["mock-frame"], [5]))
    monkeypatch.setattr(ingest, "extract_frames_in_window_b64", Mock(return_value=["frame1", "frame2"]))
    monkeypatch.setattr(ingest, "build_event_windows", lambda *a, **k: [(0, 5), (5, 10)])
    monkeypatch.setattr(ingest, "analyze_frames", clip_tags)
    monkeypatch.setattr(ingest, "analyze_event", event_tags)
    monkeypatch.setattr(ingest, "embed_texts", Mock(side_effect=lambda texts: [[.1, .2] for _ in texts]))
    return folder, media, probe, clip_tags, event_tags
