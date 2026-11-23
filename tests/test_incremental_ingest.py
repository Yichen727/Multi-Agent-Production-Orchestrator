import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import text

from app.agents import ingest_agent as ingest
from app.models.schemas import VisionTags, EventTags
from app.services import database_service as db


@pytest.fixture
def media_analysis(tmp_path, monkeypatch):
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


def run(folder, **kwargs):
    ingest.ingest_footage.func(directory=str(folder), state={"project_id": "1"}, **kwargs)
    return ingest.get_last_ingest_result()


def identities():
    with db._engine.connect() as conn:
        return {table: list(conn.execute(text(f"SELECT {column} FROM {table} ORDER BY {column}")).scalars())
                for table, column in (("clips", "clip_id"), ("clip_events", "event_id"), ("event_sets", "event_set_id"))}


def test_three_unchanged_imports_keep_both_id_sets_and_skip_all_analysis(media_analysis, monkeypatch):
    folder, media, probe, clip_tags, event_tags = media_analysis
    assert run(folder).status == "success"
    original = identities()
    with db._engine.connect() as conn:
        original_steps = conn.execute(text("SELECT COUNT(*) FROM analysis_steps")).scalar_one()
    for _ in range(3):
        assert run(folder).reused_count == 1
        assert identities() == original
    with db._engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM analysis_steps")).scalar_one() == original_steps
    assert probe.call_count == clip_tags.call_count == 1
    assert event_tags.call_count == 2
    assert ingest.detect_scene_cuts.call_count == 1
    # An alternate spelling of the same root must not create another clip.
    alias = folder / ".." / folder.name
    assert run(alias).reused_count == 1
    assert identities() == original


def test_timestamp_only_change_reuses_hash_and_preserves_ids(media_analysis):
    folder, media, probe, clip_tags, event_tags = media_analysis
    run(folder)
    original = identities()
    before = media.stat()
    os.utime(media, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
    assert run(folder).reused_count == 1
    assert identities() == original
    assert probe.call_count == clip_tags.call_count == 1
    assert event_tags.call_count == 2


def test_partial_event_retry_only_analyzes_failed_window(media_analysis):
    folder, media, probe, clip_tags, event_tags = media_analysis
    event_tags.side_effect = [EventTags(action="Runs", keywords=["running"]), None,
                              EventTags(action="Stops", keywords=["stopping"])]
    assert run(folder).status == "partial_success"
    original = identities()
    assert run(folder).status == "success"
    assert identities() == original
    assert probe.call_count == clip_tags.call_count == 1
    assert event_tags.call_count == 3
    assert ingest.extract_frames_in_window_b64.call_count == 3


def test_missing_embeddings_do_not_trigger_visual_reanalysis(media_analysis):
    folder, media, probe, clip_tags, event_tags = media_analysis
    ingest.embed_texts.side_effect = lambda texts: None
    run(folder)
    original = identities()
    ingest.embed_texts.side_effect = lambda texts: [[.3, .4] for _ in texts]
    assert run(folder).status == "success"
    assert identities() == original
    assert probe.call_count == clip_tags.call_count == 1
    assert event_tags.call_count == 2


def test_source_change_retains_clip_id_but_invalidates_old_plan(media_analysis):
    folder, media, probe, clip_tags, event_tags = media_analysis
    run(folder)
    old = db.get_catalogued_shots(1)[str(media)]
    event = db.get_catalogued_events(1)[str(media)][0]
    media.write_bytes(b"replacement video with different contents")
    assert run(folder).status == "success"
    new = db.get_catalogued_shots(1)[str(media)]
    assert new["clip_id"] == old["clip_id"]
    assert new["source_revision"] == old["source_revision"] + 1
    assert event["event_id"] not in db.get_events_by_ids(1, [event["event_id"]])
    with pytest.raises(ValueError, match="older source"):
        db.validate_segment_source({"file_path": str(media), "source_revision": old["source_revision"]}, new)
    assert len(identities()["clips"]) == 1


def test_strict_verification_catches_same_size_same_mtime_replacement(media_analysis):
    folder, media, probe, clip_tags, event_tags = media_analysis
    run(folder)
    old = db.get_catalogued_shots(1)[str(media)]
    before = media.stat()
    media.write_bytes(b"x" * before.st_size)
    os.utime(media, ns=(before.st_atime_ns, before.st_mtime_ns))
    clip_tags.return_value = VisionTags(scene_description="A different observed scene", keywords=["changed"])
    run(folder, verify_content=True)
    new = db.get_catalogued_shots(1)[str(media)]
    assert new["clip_id"] == old["clip_id"]
    assert new["source_revision"] == 2
    assert new["description"] == "A different observed scene"
    assert probe.call_count == clip_tags.call_count == 2
    assert event_tags.call_count == 4


def test_force_retagging_same_windows_does_not_allocate_new_ids(media_analysis):
    folder, media, *_ = media_analysis
    run(folder)
    original = identities()
    run(folder, force_reanalyze=True)
    assert identities() == original


def test_append_and_scope_sync_keep_other_folders_and_missing_history(media_analysis, tmp_path):
    folder, media, *_ = media_analysis
    run(folder)
    other = tmp_path / "other"
    other.mkdir()
    (other / "b.mov").write_bytes(b"another video")
    run(other)
    original = identities()
    (folder / "remaining.mov").write_bytes(b"remaining video")
    media.unlink()
    run(folder, synchronize=True)
    rows = db.get_catalogued_shots(1)
    assert rows[str(media)]["online"] == 0
    assert rows[str(other / "b.mov")]["online"] == 1
    assert set(original["clips"]).issubset(identities()["clips"])


def test_per_file_publication_rolls_back_metadata_events_and_steps(media_analysis, monkeypatch):
    folder, media, *_ = media_analysis
    run(folder)
    original = identities()
    row = db.get_catalogued_shots(1)[str(media)]
    def fail(*args, **kwargs):
        raise RuntimeError("injected step write failure")
    monkeypatch.setattr(db, "_step", fail)
    with pytest.raises(RuntimeError):
        db.publish_clip(1, {**row, "description": "must roll back"}, events=[
            {"file_path": str(media), "event_order": 1, "start_seconds": 1, "end_seconds": 3}])
    assert identities() == original
    assert db.get_catalogued_shots(1)[str(media)]["description"] == row["description"]


def test_probe_failure_preserves_previous_observed_results(media_analysis):
    folder, media, probe, *_ = media_analysis
    run(folder)
    original = identities()
    probe.return_value = {"ok": False}
    assert run(folder, force_reanalyze=True).status == "failure"
    assert identities() == original
    assert db.get_catalogued_shots(1)[str(media)]["description"] == "A player runs"
    assert db.get_catalogued_shots(1)[str(media)]["online"] == 1


def test_concurrent_same_project_imports_share_finished_analysis(media_analysis):
    folder, media, probe, clip_tags, event_tags = media_analysis
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: run(folder), range(2)))
    assert len(identities()["clips"]) == 1
    assert len(identities()["clip_events"]) == 2
    assert probe.call_count == clip_tags.call_count == 1
    assert event_tags.call_count == 2


def test_scan_error_does_not_mark_unobserved_clips_offline(media_analysis, monkeypatch):
    folder, media, *_ = media_analysis
    run(folder)
    original = identities()
    def interrupted(*args, **kwargs):
        kwargs["onerror"](PermissionError("injected inaccessible subdirectory"))
        return iter(())
    monkeypatch.setattr(ingest.os, "walk", interrupted)
    assert run(folder, synchronize=True).status == "failure"
    assert identities() == original
    assert db.get_catalogued_shots(1)[str(media)]["online"] == 1


def test_delivery_checks_source_revision_and_disk_changes(media_analysis, monkeypatch):
    from app.agents import delivery_agent as delivery
    from app.services.timeline_service import build_clip_timeline

    folder, media, *_ = media_analysis
    run(folder)
    row = db.get_catalogued_shots(1)[str(media)]
    plan = build_clip_timeline([row])
    monkeypatch.setattr(delivery, "count_audio_streams", lambda path: 0)
    delivery.compile_plan(plan, project_id=1)
    result = delivery.get_last_delivery_result()
    assert result.status == "success"
    assert Path(result.xml_path).is_file()
    media.write_bytes(b"changed after the plan was built")
    delivery.compile_plan(plan, project_id=1)
    assert delivery.get_last_delivery_result().status == "failure"
    assert "changed on disk" in delivery.get_last_delivery_result().message


def test_unversioned_plan_is_refused_instead_of_assuming_current_source(media_analysis):
    folder, media, *_ = media_analysis
    run(folder)
    row = db.get_catalogued_shots(1)[str(media)]
    with pytest.raises(ValueError, match="no source revision"):
        db.validate_segment_source({"file_path": str(media)}, row)


def test_force_event_failure_keeps_existing_ids_and_observed_actions(media_analysis):
    folder, media, probe, clip_tags, event_tags = media_analysis
    run(folder)
    original = identities()
    event_tags.return_value = None
    assert run(folder, force_reanalyze=True).status == "partial_success"
    assert identities() == original
    assert all(e["action"] == "A player runs" for e in db.get_catalogued_events(1)[str(media)])
