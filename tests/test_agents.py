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
    """Delivery's segment tool trims to the plan's out_points and preserves order."""
    pytest.importorskip("langgraph")
    import json as _json
    import xml.etree.ElementTree as ET
    from app.agents.delivery_agent import compile_timeline_segments

    # Demo shots 1 (12.5s) and 3 (8.2s), trimmed to 4s / 6s in that order.
    plan = {"mode": "timed", "target_seconds": 10.0, "total_seconds": 10.0, "segments": [
        {"order": 1, "shot_id": 1, "file_path": "/footage/scene01/take01.mov",
         "in_point": 0.0, "out_point": 4.0, "label": "open"},
        {"order": 2, "shot_id": 3, "file_path": "/footage/scene01/take03.mov",
         "in_point": 0.0, "out_point": 6.0, "label": "peak"},
    ]}
    out = compile_timeline_segments.invoke({
        "segments_json": _json.dumps(plan),
        "sequence_name": "TrimTest", "project_id": 777,
    })
    assert "compiled" in out.lower()

    # Find the written XML and verify the trims landed (24fps demo → 4s=96, 6s=144).
    m = re.search(r"(\S+premiere_777_\S+\.xml)", out)
    assert m, out
    root = ET.fromstring(Path(m.group(1)).read_text(encoding="utf-8"))
    clipitems = root.findall("./sequence/media/video/track/clipitem")
    names = [c.findtext("name") for c in clipitems]
    assert names == ["take01.mov", "take03.mov"]  # order preserved
    outs = [int(c.findtext("out")) for c in clipitems]
    assert outs == [96, 144]  # trimmed lengths, not full 12.5s / 8.2s


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
