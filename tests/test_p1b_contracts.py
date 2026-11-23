import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.agents import selection_agent as selection
from app.models.schemas import EditPlan, SearchCandidate, SearchQuery
from app.orchestrator import production_orchestrator as orch
from app.services import database_service as db, search_service as search
from app.services import delivery_service as delivery
from app.services.plan_contract import read_edit_plan, validate_edit_plan
from app.services.timeline_service import build_clip_timeline, build_moment_timeline
from app.services.workspace_state import reconcile_workspace, SELECTION_PREFIX


def state(rows, **extra):
    return {"project_id": "1", "editing_mode": "clip_assembly", "target_seconds": None,
            "aspect_ratio": "16:9", "selected_candidates": [r["file_path"] for r in rows],
            "selected_clips": rows, **extra}


def extract(text):
    return json.loads(text.split("```json\n", 1)[1].split("```", 1)[0])


def test_canonical_contract_roundtrip_keeps_provenance(seed_catalogue):
    row = seed_catalogue()[0]
    candidate = SearchCandidate.model_validate(row).model_dump()
    assert candidate["clip_id"] == row["clip_id"]
    assert candidate["source_revision"] == row["source_revision"]
    assert "shot_id" not in candidate
    plan = build_clip_timeline([row])
    assert EditPlan.model_validate(plan).model_dump()["segments"][0]["clip_id"] == row["clip_id"]
    assert validate_edit_plan(plan, 1)["schema_version"] == 1


def test_legacy_plan_read_is_lossless_and_does_not_invent_a_revision(seed_catalogue, tmp_path):
    row = seed_catalogue()[0]
    old = {"segments": [{"shot_id": row["clip_id"], "file_path": row["file_path"],
                          "in_point": 0, "out_point": 10}]}
    path = tmp_path / "historical.json"
    original = json.dumps(old).encode()
    path.write_bytes(original)
    converted = read_edit_plan(path.read_text(), 1)
    assert converted["segments"][0]["clip_id"] == row["clip_id"]
    assert "shot_id" not in converted["segments"][0]
    assert "source_revision" not in converted["segments"][0]
    assert "shot_id" in old["segments"][0]
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="no source revision"):
        validate_edit_plan(converted, 1)


def test_verified_legacy_reference_can_be_read_without_changing_its_id(seed_catalogue):
    row = seed_catalogue()[0]
    plan = build_clip_timeline([row])
    plan.pop("schema_version")
    seg = plan["segments"][0]
    seg["shot_id"] = seg.pop("clip_id")
    assert validate_edit_plan(plan, 1)["segments"][0]["clip_id"] == row["clip_id"]


def test_legacy_id_drift_is_refused_even_when_path_still_resolves(seed_catalogue):
    rows = seed_catalogue()
    plan = {"segments": [{"shot_id": rows[1]["clip_id"], "file_path": rows[0]["file_path"]}]}
    with pytest.raises(ValueError, match="ID/path mismatch"):
        read_edit_plan(plan, 1)


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"project_id": 2},
])
def test_plan_version_and_project_scope_are_checked(seed_catalogue, change):
    plan = build_clip_timeline(seed_catalogue())
    plan.update(change)
    with pytest.raises(ValueError):
        read_edit_plan(plan, 1)


def test_new_plan_refuses_legacy_field(seed_catalogue):
    plan = build_clip_timeline(seed_catalogue())
    plan["segments"][0]["shot_id"] = plan["segments"][0]["clip_id"]
    with pytest.raises(ValueError, match="must use clip_id"):
        read_edit_plan(plan, 1)


@pytest.mark.parametrize("start,end", [(2, 8), (0, 11), (float("nan"), 10), (False, 10)])
def test_export_rejects_invalid_or_trimmed_whole_clip_before_writing(seed_catalogue, monkeypatch, start, end):
    plan = build_clip_timeline(seed_catalogue())
    plan["segments"][0].update(in_point=start, out_point=end)
    forbidden = Mock(side_effect=AssertionError("must not compile"))
    monkeypatch.setattr(delivery, "compile_project", forbidden)
    assert delivery.compile_plan(plan, project_id=1).status == "failure"
    forbidden.assert_not_called()


def test_search_parses_once_and_calls_direct_hybrid_search(seed_catalogue, monkeypatch):
    rows = seed_catalogue()
    db.upsert_project_clips(2, [{**rows[0], "file_path": rows[0]["file_path"]}])
    parser = Mock()
    parser.invoke.return_value = SearchQuery(core_keywords="football", keywords="football,soccer")
    llm = Mock()
    llm.with_structured_output.return_value = parser
    monkeypatch.setattr(search, "llm_fast", llm)
    result = orch.run_search("football", 1, "editor")
    assert result[0]["file_path"] == rows[1]["file_path"]
    assert all(c["project_id"] == 1 and "shot_id" not in c for c in result)
    llm.with_structured_output.assert_called_once_with(SearchQuery)
    parser.invoke.assert_called_once()


def test_failed_query_parse_falls_back_without_second_model_call(seed_catalogue, monkeypatch):
    seed_catalogue()
    parser = Mock()
    parser.invoke.side_effect = ValueError("bad structured output")
    llm = Mock()
    llm.with_structured_output.return_value = parser
    monkeypatch.setattr(search, "llm_fast", llm)
    assert orch.run_search("football", 1)
    parser.invoke.assert_called_once()


def test_blank_search_has_no_query_model_call(seed_catalogue, monkeypatch):
    seed_catalogue()
    llm = Mock(side_effect=AssertionError("no query model needed"))
    monkeypatch.setattr(search, "llm_fast", llm)
    assert len(orch.run_search("", 1)) == 2
    llm.with_structured_output.assert_not_called()


def test_structured_filters_match_direct_retrieval(seed_catalogue):
    from app.services.retrieval_service import hybrid_search
    seed_catalogue()
    query = SearchQuery(orientation="landscape", min_duration=5, max_duration=15)
    assert search.search_clips(query, 1) == hybrid_search(
        1, orientation="landscape", min_duration=5, max_duration=15, top_k=200)


@pytest.mark.parametrize("pid", [None, "", "wrong", 0, -1, True, 1.5])
def test_invalid_project_never_defaults_to_project_one(seed_catalogue, pid):
    seed_catalogue()
    with pytest.raises(ValueError):
        orch.run_search("", pid)
    with pytest.raises(ValueError):
        db.get_catalogued_clips(pid)
    with pytest.raises(ValueError):
        db.publish_clip(pid, {"file_path": "must-not-write.mov"})
    assert delivery.compile_plan({}, project_id=pid).status == "failure"


def test_candidate_details_and_events_do_not_expose_unselected_sources(seed_catalogue):
    rows = seed_catalogue()
    for fn in (selection.get_candidate_details, selection.get_clip_events):
        report = fn.func(str(rows[1]["clip_id"]), state=state(rows[:1]))
        assert rows[1]["file_path"] not in report
        assert rows[1]["description"] not in report


def test_moment_planner_enforces_candidate_whitelist_and_authoritative_target(seed_catalogue):
    rows = seed_catalogue()
    db.upsert_project_events(1, [{"file_path": r["file_path"], "event_order": 1,
        "start_seconds": 0, "end_seconds": 10, "action": "Runs"} for r in rows])
    events = [group[0] for group in db.get_catalogued_events(1).values()]
    by_clip = {e["clip_id"]: e for e in events}
    rows = list(db.get_catalogued_clips(1).values())
    selected = [by_clip[rows[0]["clip_id"]]]
    denied = selection.plan_moment_assembly.func(str(by_clip[rows[1]["clip_id"]]["event_id"]),
        state=state(rows[:1], editing_mode="moment_assembly"))
    assert "```json" not in denied
    for target in (None, 7):
        report = selection.plan_moment_assembly.func(str(selected[0]["event_id"]), target_seconds=1,
            state=state(rows[:1], editing_mode="moment_assembly", target_seconds=target))
        plan = extract(report)
        assert plan["target_seconds"] == target
        assert plan["aspect_ratio"] == "16:9"
        assert plan["project_id"] == 1


def test_source_revision_change_during_selection_is_refused(seed_catalogue):
    rows = seed_catalogue()
    frozen = copy.deepcopy(rows)
    with db._write() as conn:
        conn.exec_driver_sql("UPDATE clips SET source_revision=source_revision+1 WHERE clip_id=?",
                             (rows[0]["clip_id"],))
    text = selection.plan_clip_assembly.func(str(rows[0]["clip_id"]), state=state(frozen))
    assert "changed during this run" in text
    assert "```json" not in text


def test_excluded_material_does_not_accept_model_metadata_or_unselected_refs(seed_catalogue):
    rows = seed_catalogue()
    raw = json.dumps([{"ref": str(rows[0]["clip_id"]), "file_path": "invented.mov",
                       "start_seconds": 999, "label": "invented label"},
                      {"ref": str(rows[1]["clip_id"]), "file_path": rows[1]["file_path"]}])
    plan = extract(selection.plan_clip_assembly.func(str(rows[0]["clip_id"]),
        excluded_json=raw, state=state(rows[:1])))
    assert len(plan["excluded"]) == 1
    assert plan["excluded"][0]["file_path"] == rows[0]["file_path"]
    assert plan["excluded"][0]["start_seconds"] == 0
    assert "invented" not in json.dumps(plan)


def test_moment_export_checks_actual_event_bounds(seed_catalogue):
    row = seed_catalogue()[0]
    db.upsert_project_events(1, [{"file_path": row["file_path"], "event_order": 1,
        "start_seconds": 2, "end_seconds": 8, "action": "Runs"}])
    event = db.get_catalogued_events(1)[row["file_path"]][0]
    plan = build_moment_timeline([event])
    plan["segments"][0].update(in_point=0, out_point=10)
    assert delivery.compile_plan(plan, project_id=1).status == "failure"


def test_existing_catalogue_and_project_switch_restore_correct_workspace(seed_catalogue):
    rows = seed_catalogue()
    other = seed_catalogue(2, ("other.mov",))
    session = {}
    reconcile_workspace(session, 1, "alice", rows)
    assert bool(session["clip_metadata"]) and session["clip_metadata"] == rows
    session.update(edit_report="old explanation", export_result="old output",
                   search_results=rows, edit_plan=build_clip_timeline(rows))
    session["candidate_clip_ids"] = [rows[0]["clip_id"]]
    reconcile_workspace(session, 2, "alice", other)
    assert session["clip_metadata"] == other and bool(session["clip_metadata"])
    assert session["edit_plan"] is None and session["export_result"] is None
    assert session["search_results"] == []
    assert rows[0]["clip_id"] not in session["candidate_clip_ids"]
    reconcile_workspace(session, 3, "alice", [])
    assert not bool(session["clip_metadata"])


def test_workspace_refresh_preserves_valid_plan_but_discards_changed_sources(seed_catalogue):
    rows = seed_catalogue()
    session = {}
    reconcile_workspace(session, 1, "alice", rows)
    plan = build_clip_timeline(rows[:1])
    session["edit_plan"] = plan
    extra = seed_catalogue(1, ("append.mov",))
    all_rows = list(db.get_catalogued_clips(1).values())
    reconcile_workspace(session, 1, "alice", all_rows)
    assert session["edit_plan"] == validate_edit_plan(plan, 1)
    Path(rows[0]["file_path"]).write_bytes(b"changed source after planning")
    reconcile_workspace(session, 1, "alice", all_rows)
    assert session["edit_plan"] is None
    assert session["export_result"] is None


def test_concurrent_service_results_have_unique_run_ids_and_no_registry(tmp_path):
    def run(index):
        text, result = orch.run_ingest(str(tmp_path / str(index)), index + 1, "editor")
        assert text == result.message
        assert result.project_id == index + 1
        return result
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, range(3)))
    assert len({r.run_id for r in results}) == 3
    assert all(r.status == "failure" for r in results)


def test_streamlit_restores_existing_project_and_switches_without_ingestion(seed_catalogue, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.services import ffmpeg_service
    rows = seed_catalogue()
    other = seed_catalogue(2, ("other.mov",))
    monkeypatch.setattr(ffmpeg_service, "extract_sample_frames_b64", lambda *a, **k: [])
    app = AppTest.from_file("app/ui/streamlit_app.py", default_timeout=20).run()
    assert not app.exception
    assert app.session_state["clip_metadata"]
    assert [c["clip_id"] for c in app.session_state["clip_metadata"]] == [c["clip_id"] for c in rows]
    app.selectbox(key="project_picker").set_value(2).run()
    assert not app.exception
    assert [c["clip_id"] for c in app.session_state["clip_metadata"]] == [c["clip_id"] for c in other]
    assert app.selectbox(key="project_picker").format_func(2) == "Project 2"


def test_latest_refused_planning_call_does_not_resurrect_an_earlier_plan():
    from langchain_core.messages import ToolMessage
    valid = {"segments": [{"file_path": "earlier.mov"}]}
    messages = [
        ToolMessage(content="```json\n" + json.dumps(valid) + "\n```",
                    name="plan_clip_assembly", tool_call_id="old"),
        ToolMessage(content="Cannot plan: clip outside curated candidates",
                    name="plan_clip_assembly", tool_call_id="new")]
    assert orch._extract_plan(messages) is None


def test_legacy_export_reader_normalises_clips_without_retaining_aliases(seed_catalogue):
    row = seed_catalogue()[0]
    old = {"sequence": {"name": "Old export"}, "clips": [
        {"shot_id": row["clip_id"], "file_path": row["file_path"],
         "in_seconds": 2, "out_seconds": 8, "used_seconds": 6}]}
    loaded = read_edit_plan(old, 1)
    assert loaded["mode"] == "legacy_timeline"
    assert loaded["segments"][0]["in_point"] == 2
    assert loaded["segments"][0]["out_point"] == 8
    assert "shot_id" not in json.dumps(loaded)
    with pytest.raises(ValueError, match="no source revision"):
        validate_edit_plan(loaded, 1)


def test_editor_switch_clears_prior_outputs_and_selections(seed_catalogue):
    rows = seed_catalogue()
    session = {}
    reconcile_workspace(session, 1, "alice", rows)
    session["edit_plan"] = build_clip_timeline(rows)
    session["candidate_clip_ids"] = [rows[0]["clip_id"]]
    reconcile_workspace(session, 1, "bob", rows)
    assert bool(session["clip_metadata"]) and session["edit_plan"] is None
    assert rows[0]["clip_id"] not in session["candidate_clip_ids"]


@pytest.mark.parametrize("override", [
    {"aspect_ratio": "nonsense"}, {"target_seconds": float("nan")},
    {"project_id": "not-a-project"}, {"editing_mode": "invented-mode"},
])
def test_injected_invalid_specification_is_refused(seed_catalogue, override):
    rows = seed_catalogue()
    report = selection.plan_clip_assembly.func(str(rows[0]["clip_id"]), state=state(rows, **override))
    assert "```json" not in report
