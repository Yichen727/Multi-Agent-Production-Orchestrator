import pytest

from app.services.timeline_service import build_clip_timeline, build_moment_timeline
from app.services.premiere_export_service import (
    InvalidSegmentRange, MediaValidationError, _timebase_ntsc,
    build_timeline, compile_project, fit_within,
)
from xml.etree import ElementTree as ET


def test_clip_assembly_keeps_whole_clips_and_order():
    plan = build_clip_timeline([{"file_path": "b.mov", "duration_seconds": 7},
                                {"file_path": "a.mov", "duration_seconds": 3}])
    assert plan["mode"] == "clip_assembly"
    assert plan["target_seconds"] is None
    assert [(s["name"], s["in_point"], s["out_point"]) for s in plan["segments"]] == [
        ("b.mov", 0, 7), ("a.mov", 0, 3)]


def test_moment_compression_preserves_protected_beats_and_event_boundaries():
    events = [{"event_id": i + 1, "file_path": "a.mov", "start_seconds": 10 * i,
               "end_seconds": 10 * (i + 1), "source_duration": 30} for i in range(3)]
    plan = build_moment_timeline(events, target_seconds=24, importance=[1, 3, 5], protect=[False, False, True])
    assert plan["valid"]
    assert len(plan["segments"]) == 3
    assert plan["segments"][2]["duration"] == 10
    assert plan["segments"][0]["duration"] < plan["segments"][1]["duration"]
    for s in plan["segments"]:
        assert s["event_start"] <= s["in_point"] < s["out_point"] <= s["event_end"]


def test_impossible_target_reports_overrun_without_removing_events():
    event = {"file_path": "a.mov", "start_seconds": 0, "end_seconds": 10, "source_duration": 10}
    plan = build_moment_timeline([event], target_seconds=1, protect=[True])
    assert len(plan["segments"]) == 1
    assert plan["total_seconds"] == 10
    assert plan["compression"]["shortfall_seconds"] == 9


def test_invalid_event_blocks_plan():
    plan = build_moment_timeline([{"file_path": "a.mov", "start_seconds": 8, "end_seconds": 15,
                                  "source_duration": 10}])
    assert not plan["valid"]
    assert not plan["segments"][0]["valid"]


@pytest.mark.parametrize("start,end", [(-1, 3), (2, 2), (8, 11)])
def test_export_refuses_illegal_ranges(start, end):
    with pytest.raises(InvalidSegmentRange):
        build_timeline([{"file_path": "a.mov", "duration_seconds": 10,
                         "in_point": start, "out_point": end}])


def test_integer_fps_export_preserves_order_trims_and_audio_links(tmp_path):
    clips = [{"file_path": str(tmp_path / name), "duration_seconds": 10,
              "in_point": start, "out_point": end, "fps": 25,
              "width": 1920, "height": 1080, "has_audio": True, "audio_streams": 2}
             for name, start, end in (("b.mov", 2, 5), ("a.mov", 1, 3))]
    result = compile_project(clips, write=False, verify_media=False)
    timeline = result["timeline"]
    assert timeline["sequence"]["total_frames"] == 125
    assert [(c["in_frame"], c["out_frame"], c["seq_start_frame"], c["seq_end_frame"])
            for c in timeline["clips"]] == [(50, 125, 0, 75), (25, 75, 75, 125)]
    root = ET.fromstring(result["xml"])
    video = root.findall("./sequence/media/video/track/clipitem")
    assert [item.findtext("name") for item in video] == ["b.mov", "a.mov"]
    assert len(root.findall("./sequence/media/audio/track")) == 2
    assert len(video[0].findall("link")) == 3


def test_export_refuses_missing_media(tmp_path):
    with pytest.raises(MediaValidationError):
        compile_project([{"file_path": str(tmp_path / "missing.mov"), "duration_seconds": 10}], write=False)


def test_fit_preserves_aspect_and_keeps_full_image():
    fit = fit_within(1920, 1080, 1080, 1920)
    assert fit["mode"] == "letterbox"
    assert fit["fitted_width"] <= 1080 and fit["fitted_height"] <= 1920
    assert fit["scale"] == pytest.approx(1080 / 1920)


@pytest.mark.known_issue
@pytest.mark.xfail(raises=AssertionError, reason="P5: arbitrary fractional FPS is incorrectly marked as NTSC")
def test_non_ntsc_fractional_fps_is_not_misclassified():
    assert _timebase_ntsc(30.056356)[1] is False


@pytest.mark.known_issue
@pytest.mark.xfail(raises=AssertionError, reason="P5: seconds-to-frames uses nominal 30 instead of 30000/1001")
def test_ntsc_60_seconds_uses_actual_rate():
    timeline = build_timeline([{"file_path": "a.mov", "duration_seconds": 60, "fps": 30000 / 1001}])
    assert timeline["sequence"]["total_frames"] == round(60 * 30000 / 1001)
