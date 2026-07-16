"""Basic tests for MAPO agents (pipeline: Ingest → Search → Selection → Delivery)."""

import re
import pytest
from pathlib import Path
from app.models.state import ProductionState
from app.services.database_service import db


def test_database_initialized():
    """Verify the demo database has expected tables and data."""
    result = db.run("SELECT COUNT(*) FROM shots;")
    assert "5" in result


def test_production_state_fields():
    """Verify ProductionState has all required fields."""
    required = [
        "project_id", "messages", "loaded_preferences",
        "ingested_files", "shot_metadata", "search_results",
        "selected_shots", "recommendations", "remaining_steps",
    ]
    annotations = ProductionState.__annotations__
    for field in required:
        assert field in annotations, f"Missing field: {field}"


def test_production_state_has_no_quality_fields():
    """The Quality Agent was removed — its state fields must be gone too."""
    annotations = ProductionState.__annotations__
    assert "grade_assessments" not in annotations
    assert "rejected_shots" not in annotations


def test_shot_query():
    """Test a basic shot query."""
    result = db.run("SELECT shot_type FROM shots WHERE shot_id = 3;")
    assert "close_up" in result


def test_shots_table_has_orientation():
    """Real ingestion stores orientation; the demo seed is landscape."""
    result = db.run("SELECT orientation FROM shots WHERE shot_id = 1;")
    assert "landscape" in result


def test_replace_project_shots_round_trip():
    """replace_project_shots writes real rows and is queryable by orientation."""
    from app.services.database_service import replace_project_shots, db as _db
    try:
        replace_project_shots(99, [
            {"file_path": "/tmp/vid_a.mp4", "orientation": "portrait",
             "width": 1080, "height": 1920, "duration_seconds": 5.0},
            {"file_path": "/tmp/vid_b.mp4", "orientation": "landscape",
             "width": 1920, "height": 1080, "duration_seconds": 8.0},
        ])
        portrait = _db.run("SELECT file_path FROM shots WHERE project_id = 99 AND orientation = 'portrait';")
        assert "vid_a.mp4" in portrait
        assert "vid_b.mp4" not in portrait
    finally:
        _db.run("DELETE FROM shots WHERE project_id = 99;")


# ── Agent composition (skipped where heavy deps are absent) ───────────────────


def test_ingest_agent_owns_ingest_and_catalogue_tools():
    """Ingest Agent owns scan + detect-new + real ingestion + cataloguing + export."""
    pytest.importorskip("langgraph")
    from app.agents.ingest_agent import ingest_tools
    names = {t.name for t in ingest_tools}
    assert {"scan_footage_directory", "detect_new_footage", "ingest_footage",
            "get_shots_by_type", "export_metadata_json"} <= names


def test_vision_tags_schema():
    """The vision tagging model exists with expected fields, each documented.

    The field is named scene_description (not 'description') and every field has an
    explicit description — this is what prevents the model echoing the schema
    docstring into the field, the bug that polluted the catalogue. The richer semantic
    dimensions (camera_motion/lighting/mood/subject_position) are present too.
    """
    from app.models.schemas import VisionTags
    for field in ("scene_description", "shot_type", "objects", "keywords", "people_count",
                  "camera_motion", "lighting", "mood", "subject_position"):
        assert field in VisionTags.model_fields
    assert "description" not in VisionTags.model_fields  # renamed to avoid the echo bug
    for name, info in VisionTags.model_fields.items():
        assert info.description, f"{name} must carry an explicit Field(description=...)"


def test_search_agent_uses_unified_search_tool():
    """Search Agent exposes ONE unified hybrid tool (people is now a parameter)."""
    pytest.importorskip("langgraph")
    from app.agents.search_agent import search_tools
    names = {t.name for t in search_tools}
    assert "search_catalogue" in names


def test_shots_table_has_vision_and_technical_columns():
    """Catalogue stores the richer technical + vision metadata."""
    result = db.run("SELECT fps, codec, scene_count, description, people_count "
                    "FROM shots WHERE shot_id = 4;")
    assert "cityscape" in result.lower() or "aerial" in result.lower()


def test_search_agent_has_ground_truth_listing():
    """Search Agent keeps a ground-truth 'list all' alongside the unified tool."""
    pytest.importorskip("langgraph")
    from app.agents.search_agent import search_tools
    names = {t.name for t in search_tools}
    assert {"search_catalogue", "list_all_shots"} <= names


def test_selection_agent_owns_timeline_tools():
    """Selection Agent owns curation/planning/delivery tools — not rankers.

    It became an edit-timeline orchestrator, so the old score-ranking tools are gone; it
    now also owns plan_timeline (TIMED / FULL-CLIP segment planning). Quality scoring was
    dropped entirely (no quality_score column), so the old quality gate is gone too.
    """
    pytest.importorskip("langgraph")
    from app.agents.selection_agent import selection_tools
    names = {t.name for t in selection_tools}
    assert {"get_candidate_details", "plan_timeline",
            "generate_delivery_summary"} <= names
    assert "rank_shots_by_quality" not in names
    assert "compare_takes" not in names
    assert "filter_low_quality" not in names


def test_hybrid_search_returns_candidates_on_demo():
    """hybrid_search runs over the demo seed (lexical fallback, no embeddings)."""
    from app.services.retrieval_service import hybrid_search, group_size
    # Pure filter: landscape clips (all 5 demo rows are landscape).
    landscape = hybrid_search(1, orientation="landscape")
    assert len(landscape) == 5
    assert all(c["orientation"] == "landscape" for c in landscape)
    assert all("suggestion" in c and "group_size" in c for c in landscape)
    # Lexical recall on a keyword present in the demo descriptions/keywords.
    aerial = hybrid_search(1, keywords="aerial,cityscape")
    assert any("cityscape" in (c.get("description") or "").lower()
               or "aerial" in (c.get("keywords") or "").lower() for c in aerial)
    # group_size labels
    assert group_size(0) == "none"
    assert group_size(1) == "solo"
    assert group_size(3) == "group"
    assert group_size(None) == "unknown"


def test_shot_columns_include_semantic_fields():
    """The write contract carries the new semantic + embedding columns."""
    from app.services.database_service import _SHOT_COLUMNS
    for col in ("camera_motion", "lighting", "mood", "subject_position", "embedding"):
        assert col in _SHOT_COLUMNS


def test_removed_agent_modules_are_gone():
    """Legacy agents + the removed Quality/Metadata agents must not be importable.

    delivery_agent was RE-INTRODUCED as the 4th pipeline stage (Premiere export), so it
    now exists and must import cleanly — the truly removed modules stay gone.
    """
    import importlib
    for present in ("ingest_agent", "delivery_agent"):
        # These now EXIST — make sure they import cleanly.
        importlib.import_module(f"app.agents.{present}")
    for gone in ("metadata_agent", "quality_agent", "shottagger_agent",
                 "gradebot", "selectbot"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"app.agents.{gone}")


def test_delivery_agent_owns_export_tools():
    """Delivery Agent owns preview + Premiere-compile + structured-segment tools."""
    pytest.importorskip("langgraph")
    from app.agents.delivery_agent import delivery_tools
    names = {t.name for t in delivery_tools}
    assert {"preview_delivery_timeline", "compile_premiere_project",
            "compile_timeline_segments"} <= names
    # It is a compiler — it must NOT expose ranking / re-ordering tools.
    assert "rank_shots_by_quality" not in names


def test_delivery_agent_follows_three_part_contract():
    """Orchestrator imports assistant + tool_node + router per agent — they must exist."""
    pytest.importorskip("langgraph")
    from app.agents.delivery_agent import (
        delivery_assistant, delivery_tool_node, should_continue_delivery,
    )
    assert callable(delivery_assistant)
    assert callable(should_continue_delivery)


def test_production_state_has_delivery_output():
    """The Delivery stage records its compiled output on shared state (additively)."""
    assert "delivery_output" in ProductionState.__annotations__


# ── Premiere export compiler (pure, no langgraph / API needed) ────────────────


def _sample_clips():
    return [
        {"file_path": "/footage/scene01/wide.mov", "duration_seconds": 4.0, "fps": 24.0,
         "has_audio": True, "audio_streams": 1, "role": "establishing"},
        {"file_path": "/footage/scene02/goal.mov", "duration_seconds": 6.0, "fps": 24.0,
         "has_audio": True, "audio_streams": 2, "role": "climax"},
        {"file_path": "/footage/scene03/crowd.mov", "duration_seconds": 3.0, "fps": 24.0,
         "has_audio": False, "audio_streams": 0, "role": "reaction"},
    ]


def test_build_timeline_preserves_order_and_maps_time():
    """Timeline keeps the exact clip order and lays clips end-to-end (no gaps)."""
    from app.services.premiere_export_service import build_timeline
    tl = build_timeline(_sample_clips(), sequence_name="Test", fps=24.0)
    names = [c["name"] for c in tl["clips"]]
    assert names == ["wide.mov", "goal.mov", "crowd.mov"]  # order preserved
    # Sequential, contiguous placement in frames (24fps): 0-96, 96-240, 240-312.
    assert tl["clips"][0]["seq_start_frame"] == 0
    assert tl["clips"][0]["seq_end_frame"] == 96
    assert tl["clips"][1]["seq_start_frame"] == 96
    assert tl["clips"][1]["seq_end_frame"] == 240
    assert tl["clips"][2]["seq_start_frame"] == 240
    assert tl["sequence"]["total_frames"] == 312
    assert tl["sequence"]["clip_count"] == 3


def test_build_timeline_refuses_clip_without_file_path():
    """Never emit a media reference with no real path (anti-fabrication)."""
    from app.services.premiere_export_service import build_timeline
    with pytest.raises(ValueError):
        build_timeline([{"duration_seconds": 5.0}])


def test_to_fcp7_xml_is_valid_and_premiere_shaped():
    """Output parses as XML, is the xmeml v5 dialect, and has V1/A1/A2 track layout."""
    import xml.etree.ElementTree as ET
    from app.services.premiere_export_service import build_timeline, to_fcp7_xml
    tl = build_timeline(_sample_clips(), sequence_name="My Edit", fps=24.0)
    xml = to_fcp7_xml(tl)

    root = ET.fromstring(xml)  # raises if not well-formed
    assert root.tag == "xmeml"
    assert root.attrib.get("version") == "5"

    seq = root.find("sequence")
    assert seq is not None and seq.findtext("name") == "My Edit"

    # One video track carrying all three clips, in order.
    video_track = seq.find("./media/video/track")
    assert video_track is not None
    v_names = [ci.findtext("name") for ci in video_track.findall("clipitem")]
    assert v_names == ["wide.mov", "goal.mov", "crowd.mov"]

    # Audio: A1 present (2 clips have audio); A2 present (goal.mov has 2 streams).
    audio_tracks = seq.findall("./media/audio/track")
    assert len(audio_tracks) == 2  # A1 + A2
    a1_count = len(audio_tracks[0].findall("clipitem"))
    a2_count = len(audio_tracks[1].findall("clipitem"))
    assert a1_count == 2  # wide + goal have audio; crowd has none
    assert a2_count == 1  # only goal has a second audio stream

    # Media is linked by absolute file:// URLs (real references, never fabricated).
    pathurls = [p.text for p in seq.iter("pathurl")]
    assert pathurls and all(p.startswith("file://") for p in pathurls)


def test_to_fcp7_xml_a2_absent_when_no_secondary_audio():
    """A2 track only appears when a clip truly has a second audio stream."""
    import xml.etree.ElementTree as ET
    from app.services.premiere_export_service import build_timeline, to_fcp7_xml
    clips = [{"file_path": "/f/a.mov", "duration_seconds": 2.0, "fps": 24.0,
              "has_audio": True, "audio_streams": 1}]
    root = ET.fromstring(to_fcp7_xml(build_timeline(clips, fps=24.0)))
    assert len(root.findall("./sequence/media/audio/track")) == 1  # A1 only


# ── Timeline planning: duration parsing, allocation, TIMED / FULL modes ───────


def test_parse_target_duration_forms():
    """Detects seconds, minutes, combined and clock forms; None when no unit present."""
    from app.services.timeline_service import parse_target_duration
    assert parse_target_duration("Fast-paced 15s promo") == 15.0
    assert parse_target_duration("make it 30 seconds") == 30.0
    assert parse_target_duration("1 min") == 60.0
    assert parse_target_duration("2 minutes") == 120.0
    assert parse_target_duration("1m30s") == 90.0
    assert parse_target_duration("1:30") == 90.0
    # No duration → FULL CLIP MODE. "promo" must NOT be read as minutes.
    assert parse_target_duration("cinematic travel montage") is None
    assert parse_target_duration("a nice promo") is None
    assert parse_target_duration("") is None


def test_allocate_durations_proportional_and_capped():
    """Screen time is split by importance and never exceeds a clip's own length."""
    from app.services.timeline_service import allocate_durations
    # Equal-length clips, weights 1:3 → 25% / 75% of a 20s target.
    alloc = allocate_durations([100.0, 100.0], [1.0, 3.0], 20.0)
    assert alloc == [5.0, 15.0]
    assert abs(sum(alloc) - 20.0) < 1e-6
    # A short clip is capped at its real length; the surplus redistributes to the other.
    alloc2 = allocate_durations([2.0, 100.0], [1.0, 1.0], 20.0)
    assert alloc2[0] == 2.0                       # capped at source
    assert abs(sum(alloc2) - 20.0) < 1e-6         # target still met
    assert alloc2[1] == 18.0


def test_plan_segments_timed_mode_trims():
    """TIMED MODE produces trimmed, time-coded segments summing ~ to the target."""
    from app.services.timeline_service import plan_segments
    clips = [
        {"file_path": "/f/a.mov", "shot_id": 1, "duration_seconds": 12.0},
        {"file_path": "/f/b.mov", "shot_id": 2, "duration_seconds": 12.0},
    ]
    plan = plan_segments(clips, target_seconds=10.0, weights=[1.0, 4.0])
    assert plan["mode"] == "timed"
    assert plan["target_seconds"] == 10.0
    assert [s["duration"] for s in plan["segments"]] == [2.0, 8.0]  # 1:4 split
    assert plan["segments"][0]["in_point"] == 0.0
    assert plan["segments"][1]["out_point"] == 8.0
    assert abs(plan["total_seconds"] - 10.0) < 1e-6


def test_plan_segments_full_mode_keeps_full_length():
    """FULL CLIP MODE keeps each clip's full duration, order preserved."""
    from app.services.timeline_service import plan_segments
    clips = [
        {"file_path": "/f/a.mov", "shot_id": 1, "duration_seconds": 4.0},
        {"file_path": "/f/b.mov", "shot_id": 2, "duration_seconds": 6.0},
    ]
    plan = plan_segments(clips, target_seconds=None)
    assert plan["mode"] == "full"
    assert plan["target_seconds"] is None
    assert [s["duration"] for s in plan["segments"]] == [4.0, 6.0]
    assert [s["order"] for s in plan["segments"]] == [1, 2]
    assert plan["total_seconds"] == 10.0


def test_compile_timeline_segments_honours_trims(tmp_path):
    """Delivery's segment tool trims to the plan's out_points and preserves order.

    Uses REAL catalogued media (audit H-12 makes compilation validate files on disk), so
    the clips are actual temp files in their own project; durations come from the catalogue.
    """
    pytest.importorskip("langgraph")
    import json as _json
    import xml.etree.ElementTree as ET
    from app.agents.delivery_agent import compile_timeline_segments
    from app.services.database_service import replace_project_shots, _engine
    from sqlalchemy import text

    a = tmp_path / "take01.mov"; a.write_bytes(b"x")
    b = tmp_path / "take03.mov"; b.write_bytes(b"x")
    replace_project_shots(555, [
        {"file_path": str(a), "duration_seconds": 12.5, "fps": 24.0,
         "orientation": "landscape", "has_audio": 1},
        {"file_path": str(b), "duration_seconds": 8.2, "fps": 24.0,
         "orientation": "landscape", "has_audio": 1},
    ])
    try:
        # 12.5s and 8.2s clips, trimmed to 4s / 6s in that order.
        plan = {"mode": "timed", "target_seconds": 10.0, "total_seconds": 10.0, "segments": [
            {"order": 1, "file_path": str(a), "in_point": 0.0, "out_point": 4.0, "label": "open"},
            {"order": 2, "file_path": str(b), "in_point": 0.0, "out_point": 6.0, "label": "peak"},
        ]}
        # project_id is INJECTED from graph state (audit C-03/C-04) — resolution is scoped.
        out = compile_timeline_segments.invoke({
            "segments_json": _json.dumps(plan),
            "sequence_name": "TrimTest", "state": {"project_id": 555},
        })
        assert "compiled" in out.lower(), out

        # Find the written XML and verify the trims landed (24fps → 4s=96, 6s=144).
        m = re.search(r"(\S+\.xml)", out)
        assert m, out
        root = ET.fromstring(Path(m.group(1)).read_text(encoding="utf-8"))
        clipitems = root.findall("./sequence/media/video/track/clipitem")
        names = [c.findtext("name") for c in clipitems]
        assert names == ["take01.mov", "take03.mov"]  # order preserved
        outs = [int(c.findtext("out")) for c in clipitems]
        assert outs == [96, 144]  # trimmed lengths, not full 12.5s / 8.2s
    finally:
        with _engine.begin() as c:
            c.execute(text("DELETE FROM shots WHERE project_id = 555"))


def test_compile_timeline_segments_refuses_unresolved():
    """Segment compile refuses when a segment's media is not catalogued (no fabrication)."""
    pytest.importorskip("langgraph")
    import json as _json
    from app.agents.delivery_agent import compile_timeline_segments
    plan = {"segments": [{"order": 1, "file_path": "/nope/ghost.mov",
                          "in_point": 0.0, "out_point": 3.0}]}
    out = compile_timeline_segments.invoke({"segments_json": _json.dumps(plan)})
    assert "refus" in out.lower() or "no catalogued" in out.lower()


def test_file_service_removed():
    """The non-destructive file service belonged to the Quality Agent and is gone."""
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.services.file_service")


# ── Batch 1: project isolation, stable resolution, range validation ───────────


def test_resolver_enforces_project_isolation():
    """C-03/C-04: an identifier never resolves across projects (shot_id or file name)."""
    from app.services.catalogue_resolver import resolve_one
    from app.services.database_service import replace_project_shots, _engine
    from sqlalchemy import text
    replace_project_shots(77, [{"file_path": "/p77/take01.mov",
                                "duration_seconds": 4.0, "orientation": "landscape"}])
    try:
        assert resolve_one(77, "/p77/take01.mov") is not None      # its own clip
        assert resolve_one(77, "/footage/scene01/take02.mov") is None   # project 1's file
        assert resolve_one(1, "/p77/take01.mov") is None            # project 1 can't see it
        assert resolve_one(77, "3") is None                         # shot_id 3 is project 1
    finally:
        with _engine.begin() as c:
            c.execute(text("DELETE FROM shots WHERE project_id = 77"))


def test_resolver_ambiguous_identifier_not_silent():
    """H-01: a file name matching >1 clip raises instead of silently taking the first."""
    from app.services.catalogue_resolver import resolve_one, AmbiguousIdentifier
    # The demo project 1 holds two take01.mov (scene01 + scene02).
    with pytest.raises(AmbiguousIdentifier):
        resolve_one(1, "take01.mov")
    # A shot_id, and a full path, are unambiguous.
    assert resolve_one(1, "1")["shot_id"] == 1
    assert resolve_one(1, "/footage/scene01/take01.mov")["shot_id"] == 1


def test_build_timeline_rejects_illegal_explicit_range():
    """C-05: an explicit illegal in/out fails loudly; it is NOT expanded to the full clip."""
    from app.services.premiere_export_service import build_timeline, InvalidSegmentRange
    with pytest.raises(InvalidSegmentRange):  # zero-length
        build_timeline([{"file_path": "/f/a.mov", "duration_seconds": 8.0,
                         "in_point": 3.0, "out_point": 3.0}])
    with pytest.raises(InvalidSegmentRange):  # out beyond source
        build_timeline([{"file_path": "/f/a.mov", "duration_seconds": 8.0,
                         "in_point": 0.0, "out_point": 99.0}])
    # No explicit trim → full clip is used (the only time the full source is implied).
    tl = build_timeline([{"file_path": "/f/a.mov", "duration_seconds": 8.0}])
    assert tl["clips"][0]["used_seconds"] == 8.0


def test_plan_segments_flags_zero_length_trim():
    """C-06: a trim leaving no middle is flagged invalid, not silently emitted/dropped."""
    from app.services.timeline_service import plan_segments
    plan = plan_segments([{"file_path": "/f/short.mov", "shot_id": 9,
                           "duration_seconds": 2.0}],
                         target_seconds=None, head_trim=1.5, tail_trim=1.5)
    assert plan["valid"] is False
    assert plan["segments"][0]["valid"] is False
    assert plan["validation_errors"] and plan["validation_errors"][0]["order"] == 1


def test_compile_timeline_segments_refuses_invalid_plan():
    """C-06: Delivery refuses to compile a plan that carries an invalid segment."""
    pytest.importorskip("langgraph")
    import json as _json
    from app.agents.delivery_agent import compile_timeline_segments
    plan = {"segments": [{"order": 1, "shot_id": 1,
                          "file_path": "/footage/scene01/take01.mov",
                          "in_point": 0.0, "out_point": 0.0, "valid": False,
                          "validation_error": "no middle remains"}]}
    out = compile_timeline_segments.invoke(
        {"segments_json": _json.dumps(plan), "state": {"project_id": 1}})
    assert "refus" in out.lower() and "invalid" in out.lower()


def test_ingest_missing_directory_is_structured_failure():
    """C-08: a hard failure records a structured failure result (UI won't unlock on it)."""
    pytest.importorskip("langgraph")
    import app.agents.ingest_agent as ia
    ia.reset_last_ingest_result()
    msg = ia.ingest_footage.func(directory="/no/such/dir/xyz", project_id=1)
    res = ia.get_last_ingest_result()
    assert res is not None and res.status == "failure" and res.indexed_count == 0
    assert "not found" in msg.lower()


def test_ingest_refuses_when_scan_truncated(tmp_path, monkeypatch):
    """C-07: >cap files → refuse and DO NOT delete the existing catalogue."""
    pytest.importorskip("langgraph")
    import app.agents.ingest_agent as ia
    from app.services.database_service import (
        replace_project_shots, get_catalogued_paths, _engine,
    )
    from sqlalchemy import text
    replace_project_shots(88, [{"file_path": "/pre/keep.mov",
                                "duration_seconds": 3.0, "orientation": "landscape"}])
    try:
        for i in range(3):
            (tmp_path / f"c{i}.mp4").write_bytes(b"x")
        monkeypatch.setattr(ia, "_MAX_INGEST_FILES", 2)        # force truncation
        monkeypatch.setattr(ia, "check_ffprobe_installed", lambda: True)
        ia.reset_last_ingest_result()
        msg = ia.ingest_footage.func(directory=str(tmp_path), project_id=88)
        res = ia.get_last_ingest_result()
        assert res.status == "failure" and res.truncated is True
        assert "refus" in msg.lower()
        # The destructive rewrite was skipped — the pre-existing row survived.
        assert get_catalogued_paths(88) == {"/pre/keep.mov"}
    finally:
        with _engine.begin() as c:
            c.execute(text("DELETE FROM shots WHERE project_id = 88"))


# ── Batch 2: architecture convergence (C-09, H-03, H-04, H-06, H-07) ──────────


def test_orchestrator_is_fixed_pipeline_no_supervisor():
    """H-06/C-09: the orchestrator drives four explicit stages — no supervisor, no gate."""
    pytest.importorskip("langgraph")
    from app.orchestrator import production_orchestrator as orch
    # The LLM supervisor and the dead LangGraph approval gate are gone.
    for absent in ("mapo_agent", "supervisor", "human_approval_gate",
                   "_needs_approval", "verify_project", "mapo_graph"):
        assert not hasattr(orch, absent), f"{absent} should be removed"
    # The four explicit stage functions exist.
    for stage in ("run_ingest", "run_search", "run_selection", "run_delivery"):
        assert callable(getattr(orch, stage))
    # The compiled sub-agents are still exposed.
    for agent in ("ingest_agent", "search_agent", "selection_agent", "delivery_agent"):
        assert hasattr(orch, agent)


def test_run_delivery_requires_structured_plan():
    """H-04: Delivery refuses without a structured plan — no media-pool-order fallback."""
    pytest.importorskip("langgraph")
    from app.orchestrator import production_orchestrator as orch
    for bad in (None, {}, {"segments": []}):
        with pytest.raises(ValueError):
            orch.run_delivery(bad, 1, "editor_01")


def test_extract_plan_reads_tool_output_not_model_prose():
    """H-03: the plan is read from the plan_timeline TOOL message, immune to model rewrites."""
    pytest.importorskip("langgraph")
    from langchain_core.messages import ToolMessage, AIMessage
    from app.orchestrator import production_orchestrator as orch
    tool_plan = '{"mode":"timed","segments":[{"order":1,"file_path":"/f/a.mov"}]}'
    decoy = '{"mode":"HACKED","segments":[{"order":99,"file_path":"/evil.mov"}]}'
    msgs = [
        ToolMessage(content="ok\n```json\n" + tool_plan + "\n```",
                    tool_call_id="x", name="plan_timeline"),
        AIMessage(content="My plan:\n```json\n" + decoy + "\n```"),  # model prose decoy
    ]
    plan = orch._extract_plan(msgs)
    assert plan is not None and plan["mode"] == "timed"
    assert plan["segments"][0]["order"] == 1  # tool output won, not the AI-message decoy


def test_ingest_reads_footage_dir_from_state_not_global(tmp_path, monkeypatch):
    """H-07: the footage directory comes from state; the global setting is never mutated."""
    pytest.importorskip("langgraph")
    import app.agents.ingest_agent as ia
    from app.config import settings
    monkeypatch.setattr(ia, "check_ffprobe_installed", lambda: True)
    before = settings.RAW_FOOTAGE_DIR
    ia.reset_last_ingest_result()
    # An empty existing dir passed via state → "no supported files", proving it was used.
    msg = ia.ingest_footage.func(
        directory=None, state={"footage_dir": str(tmp_path), "project_id": 1})
    assert str(tmp_path) in msg
    assert settings.RAW_FOOTAGE_DIR == before           # global untouched (H-07)
    assert ia.get_last_ingest_result().status == "failure"


def test_run_search_returns_candidates_via_orchestrator():
    """H-06: the UI's Search path and the Search Agent share one retrieval entry point."""
    pytest.importorskip("langgraph")
    from app.orchestrator import production_orchestrator as orch
    cands = orch.run_search("aerial cityscape", 1)
    assert isinstance(cands, list) and len(cands) >= 1
    assert all("file_path" in c and "suggestion" in c for c in cands)


# ── Batch 3: delivery reliability (H-10, H-11, H-12, H-13) ────────────────────


def test_timebase_ntsc_mapping():
    """H-11: fractional NTSC rates map to (integer timebase, ntsc=True); integers don't."""
    from app.services.premiere_export_service import _timebase_ntsc
    assert _timebase_ntsc(23.976) == (24, True)
    assert _timebase_ntsc(29.97) == (30, True)
    assert _timebase_ntsc(59.94) == (60, True)
    assert _timebase_ntsc(30000 / 1001) == (30, True)   # the exact rational
    assert _timebase_ntsc(25.0) == (25, False)
    assert _timebase_ntsc(24.0) == (24, False)
    assert _timebase_ntsc(30.0) == (30, False)


def test_build_timeline_ntsc_no_frame_drift():
    """H-11: a 29.97 timeline uses timebase 30 + ntsc, and frames are exact (no drift)."""
    from app.services.premiere_export_service import build_timeline
    tl = build_timeline([{"file_path": "/f/a.mov", "duration_seconds": 100.0, "fps": 29.97}],
                        fps=29.97)
    assert tl["sequence"]["timebase"] == 30 and tl["sequence"]["ntsc"] is True
    assert tl["clips"][0]["out_frame"] == 3000  # 100s * 30 — integer, no accumulation error


def test_source_file_uses_real_media_params():
    """H-10: the source <file> carries the clip's OWN dims/rate/audio, not the sequence's."""
    import xml.etree.ElementTree as ET
    from app.services.premiere_export_service import build_timeline, to_fcp7_xml
    clips = [{"file_path": "/f/portrait.mov", "duration_seconds": 5.0, "fps": 30.0,
              "width": 720, "height": 1280, "has_audio": True, "audio_streams": 1,
              "audio_channels": 6, "audio_sample_rate": 44100, "audio_bit_depth": 24}]
    # Sequence deliberately differs from the source (1920x1080@24 vs 720x1280@30).
    root = ET.fromstring(to_fcp7_xml(build_timeline(clips, fps=24.0, width=1920, height=1080)))
    # Sequence raster = sequence params.
    assert root.findtext("./sequence/media/video/format/samplecharacteristics/width") == "1920"
    # Source <file> = the real source params.
    fchar = root.find("./sequence/media/video/track/clipitem/file/media/video/samplecharacteristics")
    assert fchar.findtext("width") == "720" and fchar.findtext("height") == "1280"
    assert root.findtext("./sequence/media/video/track/clipitem/file/rate/timebase") == "30"
    fa = root.find("./sequence/media/video/track/clipitem/file/media/audio")
    assert fa.findtext("samplecharacteristics/samplerate") == "44100"
    assert fa.findtext("samplecharacteristics/depth") == "24"
    assert fa.findtext("channelcount") == "6"  # channels, NOT the audio-stream count


def test_compile_project_refuses_missing_media(tmp_path):
    """H-12: compilation validates media on disk and refuses when a file is missing."""
    from app.services.premiere_export_service import compile_project, MediaValidationError
    clips = [{"file_path": str(tmp_path / "ghost.mov"), "duration_seconds": 3.0, "fps": 24.0}]
    with pytest.raises(MediaValidationError):
        compile_project(clips, write=False)            # verify_media defaults True
    # Structure-only compiles can still opt out of the disk check.
    result = compile_project(clips, write=False, verify_media=False)
    assert result["xml"].startswith("<?xml")


def test_compile_project_versions_exports(tmp_path):
    """H-13: re-exporting the same sequence writes two trackable versions (no overwrite)."""
    from app.services.premiere_export_service import compile_project
    media = tmp_path / "clip.mov"; media.write_bytes(b"x")
    clips = [{"file_path": str(media), "duration_seconds": 4.0, "fps": 24.0,
              "has_audio": False, "audio_streams": 0}]
    r1 = compile_project(clips, sequence_name="VerTest", project_id=321, write=True)
    r2 = compile_project(clips, sequence_name="VerTest", project_id=321, write=True)
    assert r1["xml_path"] != r2["xml_path"]                    # distinct versions
    assert Path(r1["xml_path"]).exists() and Path(r2["xml_path"]).exists()
    # Clean up the two written exports.
    for r in (r1, r2):
        Path(r["xml_path"]).unlink(missing_ok=True)
        Path(r["json_path"]).unlink(missing_ok=True)


# ── Tier 2: temporal event-based ingestion (events, windows, moment search) ────


def test_clip_events_table_exists():
    """The event-based ingestion layer added a clip_events table."""
    tables = db.run("SELECT name FROM sqlite_master WHERE type='table';")
    assert "clip_events" in tables


def test_event_tags_schema():
    """EventTags is action-oriented, has no 'description' echo field, all documented."""
    from app.models.schemas import EventTags
    for field in ("action", "subjects", "state_change", "keywords"):
        assert field in EventTags.model_fields
    assert "description" not in EventTags.model_fields  # avoid the schema-echo bug
    for name, info in EventTags.model_fields.items():
        assert info.description, f"{name} must carry an explicit Field(description=...)"


def test_clip_events_round_trip_and_cascade():
    """Events write/read by parent file, and cascade-delete when shots are replaced."""
    from app.services.database_service import (
        replace_project_shots, replace_project_events, get_catalogued_events, _engine,
    )
    from sqlalchemy import text
    replace_project_shots(66, [
        {"file_path": "/ev/a.mov", "duration_seconds": 20.0, "orientation": "landscape"},
    ])
    try:
        n = replace_project_events(66, [
            {"file_path": "/ev/a.mov", "event_order": 1, "start_seconds": 0.0,
             "end_seconds": 5.0, "duration_seconds": 5.0, "action": "a person walks in",
             "subjects": ["person"], "keywords": "entrance,walk-in"},
            {"file_path": "/ev/a.mov", "event_order": 2, "start_seconds": 5.0,
             "end_seconds": 12.0, "duration_seconds": 7.0, "action": "sits and talks",
             "subjects": ["person"], "keywords": "sit,dialogue"},
        ])
        assert n == 2
        grouped = get_catalogued_events(66)
        assert len(grouped["/ev/a.mov"]) == 2
        assert grouped["/ev/a.mov"][0]["event_order"] == 1
        # Replacing the shots cascade-clears the events (FK ON DELETE CASCADE).
        replace_project_shots(66, [
            {"file_path": "/ev/a.mov", "duration_seconds": 20.0, "orientation": "landscape"}])
        assert get_catalogued_events(66) == {}
    finally:
        with _engine.begin() as c:
            c.execute(text("DELETE FROM shots WHERE project_id = 66"))


def test_replace_events_skips_orphans():
    """An event whose file has no catalogued shot is skipped, never orphaned."""
    from app.services.database_service import (
        replace_project_shots, replace_project_events, _engine,
    )
    from sqlalchemy import text
    replace_project_shots(68, [{"file_path": "/ev/real.mov", "duration_seconds": 4.0}])
    try:
        n = replace_project_events(68, [
            {"file_path": "/ev/real.mov", "event_order": 1, "start_seconds": 0.0,
             "end_seconds": 2.0, "duration_seconds": 2.0, "action": "x"},
            {"file_path": "/ev/ghost.mov", "event_order": 1, "start_seconds": 0.0,
             "end_seconds": 2.0, "duration_seconds": 2.0, "action": "orphan"},
        ])
        assert n == 1  # the orphan (no parent shot) was skipped
    finally:
        with _engine.begin() as c:
            c.execute(text("DELETE FROM shots WHERE project_id = 68"))


def test_get_events_by_ids_is_project_scoped():
    """An event id never resolves across projects (mirrors shot/file isolation)."""
    from app.services.database_service import (
        replace_project_shots, replace_project_events, get_events_by_ids,
        get_catalogued_events, _engine,
    )
    from sqlalchemy import text
    replace_project_shots(69, [{"file_path": "/ev/p69.mov", "duration_seconds": 10.0}])
    try:
        replace_project_events(69, [
            {"file_path": "/ev/p69.mov", "event_order": 1, "start_seconds": 0.0,
             "end_seconds": 3.0, "duration_seconds": 3.0, "action": "y"}])
        an_id = get_catalogued_events(69)["/ev/p69.mov"][0]["event_id"]
        got = get_events_by_ids(69, [an_id])          # its own project sees it
        assert got and got[an_id]["source_duration"] == 10.0
        assert get_events_by_ids(1, [an_id]) == {}    # project 1 cannot see it
    finally:
        with _engine.begin() as c:
            c.execute(text("DELETE FROM shots WHERE project_id = 69"))


def test_build_event_windows_segments_and_caps():
    """Scene cuts become windows; long scenes subdivide; the count is capped."""
    from app.services.ffmpeg_service import build_event_windows
    windows = build_event_windows(30.0, [5.0, 8.0], long_scene_seconds=8.0)
    assert windows[0] == (0.0, 5.0)           # first scene kept whole
    assert (5.0, 8.0) in windows              # short middle scene kept whole
    assert windows[-1][1] == 30.0             # coverage reaches clip end
    assert any(w[0] >= 8.0 for w in windows[2:])  # long final scene subdivided
    # Cap: one 100s scene subdivides to many windows, then down-samples to <= cap.
    capped = build_event_windows(100.0, [], max_events=4, long_scene_seconds=8.0)
    assert 1 <= len(capped) <= 4
    # Unknown duration → no windows (caller falls back to whole-clip analysis).
    assert build_event_windows(0.0, [5.0]) == []


def test_plan_event_segments_uses_event_bounds():
    """EVENTS MODE trims each segment to the event's own in/out, order preserved."""
    from app.services.timeline_service import plan_event_segments
    events = [
        {"file_path": "/f/a.mov", "shot_id": 1, "start_seconds": 10.0,
         "end_seconds": 14.0, "source_duration": 30.0, "action": "the goal"},
        {"file_path": "/f/b.mov", "shot_id": 2, "start_seconds": 0.0,
         "end_seconds": 3.0, "clip_duration": 8.0, "action": "crowd roar"},
    ]
    plan = plan_event_segments(events)
    assert plan["mode"] == "events"
    assert [s["order"] for s in plan["segments"]] == [1, 2]
    assert plan["segments"][0]["in_point"] == 10.0
    assert plan["segments"][0]["out_point"] == 14.0
    assert plan["segments"][0]["duration"] == 4.0
    assert plan["total_seconds"] == 7.0
    assert plan["valid"] is True


def test_plan_event_segments_flags_out_of_range():
    """An event out-point beyond the source length is flagged invalid, not clamped."""
    from app.services.timeline_service import plan_event_segments
    plan = plan_event_segments([{"file_path": "/f/a.mov", "start_seconds": 0.0,
                                 "end_seconds": 99.0, "source_duration": 10.0}])
    assert plan["valid"] is False and plan["segments"][0]["valid"] is False


def test_search_events_ranks_moments_lexically():
    """search_events returns ranked moments with timecodes + parent-clip context."""
    from app.services.database_service import (
        replace_project_shots, replace_project_events, _engine,
    )
    from app.services.retrieval_service import search_events
    from sqlalchemy import text
    replace_project_shots(67, [{"file_path": "/ev/match.mov", "duration_seconds": 30.0,
                                "orientation": "landscape", "shot_type": "wide_shot"}])
    try:
        replace_project_events(67, [
            {"file_path": "/ev/match.mov", "event_order": 1, "start_seconds": 0.0,
             "end_seconds": 5.0, "duration_seconds": 5.0,
             "action": "players walk onto the pitch", "keywords": "entrance,walk-on"},
            {"file_path": "/ev/match.mov", "event_order": 2, "start_seconds": 10.0,
             "end_seconds": 14.0, "duration_seconds": 4.0,
             "action": "striker scores a goal and celebrates",
             "keywords": "goal,celebration,score"},
        ])
        res = search_events(67, keywords="goal celebration")
        assert res, "expected at least one moment"
        top = res[0]
        assert "goal" in (top.get("keywords") or "")          # the scoring moment ranked first
        assert top["start_seconds"] == 10.0 and top["end_seconds"] == 14.0  # timecodes
        assert top["shot_type"] == "wide_shot"                # parent-clip context joined
        assert "suggestion" in top
    finally:
        with _engine.begin() as c:
            c.execute(text("DELETE FROM shots WHERE project_id = 67"))


def test_selection_agent_owns_event_tools():
    """Selection Agent gains event inspection + moment-precise timeline planning."""
    pytest.importorskip("langgraph")
    from app.agents.selection_agent import selection_tools
    names = {t.name for t in selection_tools}
    assert {"get_clip_events", "plan_moment_timeline"} <= names


def test_ingest_result_carries_event_count():
    """IngestResult records how many temporal events were indexed (additive field)."""
    from app.models.schemas import IngestResult
    assert "event_count" in IngestResult.model_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
