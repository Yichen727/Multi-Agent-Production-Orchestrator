import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from app.orchestrator import production_orchestrator as orch
from app.services import database_service as db, delivery_service as delivery, ffmpeg_service
from app.services.timeline_service import build_clip_timeline
from app.services.workspace_state import reconcile_workspace, set_candidates, save_edit_plan


def app():
    result = AppTest.from_file("app/ui/streamlit_app.py", default_timeout=25).run()
    assert not result.exception
    return result


def page(ui, name):
    ui.radio(key="workspace_page").set_value(name).run()
    assert not ui.exception
    return ui


def button(ui, label):
    return next(b for b in ui.button if b.label == label)


def test_pages_render_only_active_controls_and_never_call_models(seed_catalogue, monkeypatch):
    seed_catalogue()
    calls = {}
    for name in ("run_ingest", "run_search", "run_selection", "run_delivery"):
        calls[name] = Mock(side_effect=AssertionError("Page navigation must not perform work"))
        monkeypatch.setattr(orch, name, calls[name])
    frames = Mock(side_effect=AssertionError("No automatic media decoding"))
    monkeypatch.setattr(ffmpeg_service, "extract_sample_frames_b64", frames)
    ui = app()
    assert "library_filter" in [w.key for w in ui.text_input]
    assert not any(b.key in ("plan_generate", "export_compile") for b in ui.button)
    for destination in ("Find Footage", "Edit Plan", "Export", "Media Library"):
        page(ui, destination)
    for call in calls.values():
        call.assert_not_called()
    frames.assert_not_called()
    assert not any("Agent" in str(h.value) for h in ui.header)


def test_candidates_survive_widget_cleanup_pages_filters_and_bulk_actions(seed_catalogue):
    rows = seed_catalogue(1, tuple(f"clip-{i:02d}.mov" for i in range(25)))
    ui = app()
    assert len([c for c in ui.checkbox if c.key and c.key.startswith("candidate:")]) == 20
    ui.checkbox(key=f"candidate:library:{rows[0]['clip_id']}").check().run()
    assert ui.session_state["candidate_clip_ids"] == [rows[0]["clip_id"]]
    page(ui, "Export")
    page(ui, "Media Library")
    assert ui.checkbox(key=f"candidate:library:{rows[0]['clip_id']}").value
    ui.number_input(key="library_page").set_value(2).run()
    button(ui, "Select this page").click().run()
    assert set(ui.session_state["candidate_clip_ids"]) == {rows[0]["clip_id"], *[r["clip_id"] for r in rows[20:]]}
    ui.text_input(key="library_filter").set_value("clip-24").run()
    assert len([c for c in ui.checkbox if c.key and c.key.startswith("candidate:")]) == 1
    button(ui, "Clear this page").click().run()
    assert rows[24]["clip_id"] not in ui.session_state["candidate_clip_ids"]
    assert rows[0]["clip_id"] in ui.session_state["candidate_clip_ids"]
    ui.button(key="clear_candidates").click().run()
    assert ui.session_state["candidate_clip_ids"] == []


def test_search_and_library_share_ids_without_repeating_search(seed_catalogue, monkeypatch):
    rows = seed_catalogue()
    search = Mock(return_value=[{**r, "relevance": .9, "suggestion": "suggested"} for r in rows])
    monkeypatch.setattr(orch, "run_search", search)
    ui = page(app(), "Find Footage")
    ui.text_input(key="find_query").set_value("football")
    button(ui, "Find footage").click().run()
    assert not ui.exception
    ui.checkbox(key=f"candidate:find:{rows[1]['clip_id']}").check().run()
    page(ui, "Media Library")
    assert ui.checkbox(key=f"candidate:library:{rows[1]['clip_id']}").value
    ui.checkbox(key=f"candidate:library:{rows[1]['clip_id']}").uncheck().run()
    page(ui, "Find Footage")
    assert not ui.checkbox(key=f"candidate:find:{rows[1]['clip_id']}").value
    assert ui.text_input(key="find_query").value == "football"
    search.assert_called_once_with("football", 1, "editor_01")


def test_sidebar_repeat_import_plan_versions_and_explicit_export(media_analysis, monkeypatch):
    folder, media, probe, clip_tags, event_tags = media_analysis
    real_ingest, real_export = orch.run_ingest, orch.run_delivery
    imports, exports = Mock(wraps=real_ingest), Mock(wraps=real_export)
    monkeypatch.setattr(orch, "run_ingest", imports)
    monkeypatch.setattr(orch, "run_delivery", exports)
    monkeypatch.setattr(delivery, "count_audio_streams", lambda _: 0)

    def selection(intent, paths, pid, user, **spec):
        rows = db.get_catalogued_clips(pid)
        plan = build_clip_timeline([rows[p] for p in paths], aspect_ratio=spec["aspect_ratio"])
        return "Open with the movement.", plan
    generate = Mock(side_effect=selection)
    monkeypatch.setattr(orch, "run_selection", generate)
    ui = app()
    ui.text_input(key="footage_folder").set_value(str(folder)).run()
    ui.button(key="import_update").click().run()
    assert not ui.exception
    original = db.get_catalogued_clips(1)[str(media)]["clip_id"]
    event_ids = set(db.get_events_by_ids(1, [1, 2]))
    ui.button(key="import_update").click().run()
    assert not ui.exception
    assert ui.session_state["ingest_result"]["reused"] == 1
    assert db.get_catalogued_clips(1)[str(media)]["clip_id"] == original
    assert set(db.get_events_by_ids(1, [1, 2])) == event_ids
    assert probe.call_count == clip_tags.call_count == 1 and event_tags.call_count == 2
    ui.checkbox(key=f"candidate:library:{original}").check().run()
    page(ui, "Edit Plan")
    ui.text_area(key="plan_intent").set_value("Movement-led opening").run()
    ui.text_input(key="plan_name").set_value("Opening").run()
    ui.button(key="plan_generate").click().run()
    assert not ui.exception
    page(ui, "Media Library")
    page(ui, "Edit Plan")
    assert ui.text_area(key="plan_intent").value == "Movement-led opening"
    ui.button(key="plan_generate").click().run()
    assert len(ui.session_state["saved_edit_plans"]) == 2
    page(ui, "Export")
    assert len(ui.selectbox(key="export_plan").options) == 2
    ui.button(key="export_compile").click().run()
    assert not ui.exception
    output = ui.session_state["export_result"]
    assert output.status == "success"
    assert Path(output.xml_path).is_file()
    assert json.loads(Path(output.json_path).read_text())["clips"][0]["clip_id"] == original
    page(ui, "Edit Plan")
    page(ui, "Export")
    exports.assert_called_once()
    assert imports.call_count == 2 and generate.call_count == 2
    assert generate.call_args.kwargs["target_seconds"] is None


def test_migrate_p1b_state_and_restore_project_editor_workspaces(seed_catalogue):
    one = seed_catalogue()
    two = seed_catalogue(2, ("other.mov",))
    plan = build_clip_timeline(one[:1])
    state = {"workspace_key": (1, "alice"), "bin_clips": one,
             "clip_selected:" + one[0]["file_path"]: True,
             "last_timeline_plan": plan, "selection_output": "existing plan",
             "intent": "Original intent", "editing_mode_label": "Moment Assembly"}
    reconcile_workspace(state, 1, "alice", one)
    assert state["candidate_clip_ids"] == [one[0]["clip_id"]]
    assert state["ui_values"]["plan_intent"] == "Original intent"
    assert state["ui_values"]["plan_mode"] == "Selected Moments"
    assert len(state["saved_edit_plans"]) == 1
    for name in ("bin_clips", "last_timeline_plan", "selection_output", "intent"):
        assert name not in state
    reconcile_workspace(state, 2, "alice", two)
    assert state["candidate_clip_ids"] == [] and state["saved_edit_plans"] == []
    set_candidates(state, [two[0]["clip_id"]], True)
    reconcile_workspace(state, 1, "alice", one)
    assert state["candidate_clip_ids"] == [one[0]["clip_id"]]
    assert state["edit_report"] == "existing plan"
    reconcile_workspace(state, 1, "bob", one)
    assert state["candidate_clip_ids"] == [] and state["edit_plan"] is None
    reconcile_workspace(state, 1, "alice", one)
    assert state["candidate_clip_ids"] == [one[0]["clip_id"]]


def test_renamed_path_with_same_id_retains_candidates_and_offline_drops_them(seed_catalogue):
    row = seed_catalogue()[0]
    state = {}
    reconcile_workspace(state, 1, "alice", [row])
    set_candidates(state, [row["clip_id"]], True)
    renamed = {**row, "file_path": "new-display-path.mov"}
    reconcile_workspace(state, 1, "alice", [renamed])
    assert state["candidate_clip_ids"] == [row["clip_id"]]
    reconcile_workspace(state, 1, "alice", [{**renamed, "online": 0}])
    assert state["candidate_clip_ids"] == []


def test_library_query_avoids_vectors_and_project_creation_is_explicit(seed_catalogue):
    rows = seed_catalogue()
    assert all("embedding" not in c for c in db.get_library_clips(1))
    before = copy.deepcopy(db.list_projects())
    with pytest.raises(ValueError):
        db.create_project(" ")
    assert db.list_projects() == before
    pid = db.create_project("Campaign")
    assert db.get_project_info(pid)["project_name"] == "Campaign"
    assert db.get_library_clips(pid) == []


def test_browser_sessions_and_project_folders_are_isolated(seed_catalogue):
    rows = seed_catalogue()
    seed_catalogue(2, ("other.mov",))
    first, second = app(), app()
    first.checkbox(key=f"candidate:library:{rows[0]['clip_id']}").check().run()
    second.run()
    assert second.session_state["candidate_clip_ids"] == []
    first.text_input(key="footage_folder").set_value("project-one-media").run()
    first.checkbox(key="import_force").check().run()
    first.selectbox(key="project_picker").set_value(2).run()
    assert first.text_input(key="footage_folder").value != "project-one-media"
    assert not first.checkbox(key="import_force").value
    first.text_input(key="footage_folder").set_value("project-two-media").run()
    first.selectbox(key="project_picker").set_value(1).run()
    assert first.text_input(key="footage_folder").value == "project-one-media"
    assert first.checkbox(key="import_force").value
    assert first.session_state["candidate_clip_ids"] == [rows[0]["clip_id"]]
    first.text_input(key="workspace_editor_id").set_value("bob").run()
    assert first.session_state["candidate_clip_ids"] == []
    first.text_input(key="workspace_editor_id").set_value("editor_01").run()
    assert first.session_state["candidate_clip_ids"] == [rows[0]["clip_id"]]
    assert not first.exception


def test_named_project_creation_in_sidebar_is_an_explicit_action(seed_catalogue):
    seed_catalogue()
    ui = app()
    ui.text_input(key="new_project_name").set_value("Summer campaign")
    button(ui, "Create project").click().run()
    assert not ui.exception
    created = db.list_projects()[-1]
    assert created["project_name"] == "Summer campaign"
    assert ui.selectbox(key="project_picker").value == created["project_id"]
    assert ui.session_state["clip_metadata"] == []


def test_hidden_plan_is_not_revalidated_on_library_page(seed_catalogue, monkeypatch):
    from app.services import workspace_state
    rows = seed_catalogue()
    ui = app()
    plan = build_clip_timeline(rows[:1])
    ui.session_state["edit_plan"] = plan
    ui.session_state["saved_edit_plans"] = [{"id": "saved", "name": "Saved", "plan": plan, "report": ""}]
    check = Mock(side_effect=AssertionError("Inactive page must not read plan sources"))
    monkeypatch.setattr(workspace_state, "validate_edit_plan", check)
    ui.run()
    page(ui, "Find Footage")
    page(ui, "Media Library")
    assert not ui.exception
    check.assert_not_called()
