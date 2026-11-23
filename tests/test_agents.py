import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agents import ingest_agent as ingest, selection_agent as selection, delivery_agent as delivery
from app.services import database_service as db
from app.services.retrieval_service import hybrid_search


def state(**kwargs):
    return {"project_id": "1", "editing_mode": "clip_assembly", "selected_candidates": [], **kwargs}


def test_ingest_missing_directory_records_failure(tmp_path):
    ingest.reset_last_ingest_result()
    ingest.ingest_footage.func(directory=str(tmp_path / "missing"), state=state())
    assert ingest.get_last_ingest_result().status == "failure"


def test_ingest_unchanged_file_skips_probe_and_model_work(tmp_path, monkeypatch):
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "a.mov").write_bytes(b"placeholder")
    probe = Mock(return_value={"ok": True, "orientation": "landscape", "duration_seconds": 10,
                               "width": 1920, "height": 1080, "fps": 25, "codec": "h264", "has_audio": False})
    monkeypatch.setattr(ingest, "check_ffprobe_installed", lambda: True)
    monkeypatch.setattr(ingest, "probe_video_metadata", probe)
    monkeypatch.setattr(ingest, "detect_scene_cuts", lambda *a, **k: {"cut_times": [], "scene_count": 1,
                                                                   "ran": True, "complete": True})
    unexpected_model = Mock(side_effect=AssertionError("Unexpected paid analysis"))
    monkeypatch.setattr(ingest, "analyze_frames", unexpected_model)
    monkeypatch.setattr(ingest, "embed_texts", unexpected_model)
    for _ in range(2):
        ingest.ingest_footage.func(directory=str(folder), analyze_content=False, state=state())
    assert probe.call_count == 1
    assert ingest.get_last_ingest_result().reused_count == 1
    unexpected_model.assert_not_called()


def test_planner_refuses_wrong_mode(seed_catalogue):
    row = seed_catalogue()[0]
    text = selection.plan_clip_assembly.func(str(row["shot_id"]), state=state(editing_mode="moment_assembly"))
    assert "does not apply" in text


def test_delivery_refuses_invalid_or_unresolved_plan():
    delivery.compile_plan({"segments": [{"valid": False, "file_path": "bad.mov"}]}, project_id=1)
    assert delivery.get_last_delivery_result().status == "failure"
    delivery.compile_plan({"segments": [{"shot_id": 999}]}, project_id=1)
    assert delivery.get_last_delivery_result().status == "failure"


def test_plan_extraction_trusts_tool_not_model_prose():
    from app.orchestrator.production_orchestrator import _extract_plan
    real = {"segments": [{"file_path": "real.mov"}]}
    messages = [ToolMessage(content="```json\n" + json.dumps(real) + "\n```",
                            tool_call_id="1", name="plan_clip_assembly"),
                AIMessage(content='```json\n{"segments": [{"file_path": "invented.mov"}]}\n```')]
    assert _extract_plan(messages) == real


def test_event_aware_search_returns_parent_clip(seed_catalogue):
    rows = seed_catalogue()
    db.replace_project_events(1, [{"file_path": rows[0]["file_path"], "event_order": 1,
                                  "start_seconds": 1, "end_seconds": 4,
                                  "action": "People celebrate the victory", "keywords": "celebration"}])
    found = hybrid_search(1, keywords="celebration", core_keywords="celebration")
    assert found[0]["file_path"] == rows[0]["file_path"]
    assert found[0]["matched_event"]["start_seconds"] == 1


@pytest.mark.known_issue
@pytest.mark.xfail(raises=AssertionError, reason="P1b: candidate whitelist currently exists only in the prompt")
def test_planner_refuses_same_project_unselected_clip(seed_catalogue):
    rows = seed_catalogue()
    text = selection.plan_clip_assembly.func(str(rows[1]["shot_id"]),
        state=state(selected_candidates=[rows[0]["file_path"]]))
    assert "```json" not in text


@pytest.mark.known_issue
@pytest.mark.xfail(raises=AssertionError, reason="P1a: exactly 200 files is incorrectly marked as truncated")
def test_ingest_cap_allows_exactly_200_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.os, "walk", lambda *a, **k: [(str(tmp_path), [], [f"{i}.mov" for i in range(200)])])
    files, truncated = ingest._find_video_files(tmp_path)
    assert len(files) == 200
    assert not truncated
