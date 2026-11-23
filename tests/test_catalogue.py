import pytest

from app.services import database_service as db
from app.services.catalogue_resolver import AmbiguousIdentifier, resolve_one, resolve_ordered
from app.services.retrieval_service import hybrid_search


def test_catalogue_is_empty_before_ingest():
    assert db.get_catalogued_clips(1) == {}
    assert db.get_catalogued_events(1) == {}


def test_project_isolation_for_paths_ids_and_retrieval(seed_catalogue):
    first = seed_catalogue(1)
    other = seed_catalogue(2)
    assert resolve_one(1, other[0]["clip_id"]) is None
    assert resolve_one(1, other[0]["file_path"]) is None
    assert {r["file_path"] for r in hybrid_search(1)} == {r["file_path"] for r in first}
    assert db.get_project_info(1)["clip_count"] == 2


def test_ambiguous_basename_is_refused(seed_catalogue):
    rows = seed_catalogue(1, ("a/same.mov", "b/same.mov"))
    with pytest.raises(AmbiguousIdentifier):
        resolve_one(1, "same.mov")
    assert resolve_one(1, rows[1]["clip_id"])["file_path"] == rows[1]["file_path"]


def test_resolution_preserves_order_repeats_and_reports_missing(seed_catalogue):
    rows = seed_catalogue()
    tokens = [str(rows[1]["clip_id"]), str(rows[0]["clip_id"]), str(rows[1]["clip_id"]), "absent.mov"]
    found, problems = resolve_ordered(1, tokens)
    assert [r["clip_id"] for r in found] == [rows[1]["clip_id"], rows[0]["clip_id"], rows[1]["clip_id"]]
    assert problems[0][0] == "absent.mov"


def test_event_lookup_is_project_scoped(seed_catalogue):
    one = seed_catalogue(1)[0]
    two = seed_catalogue(2)[0]
    for pid, row in ((1, one), (2, two)):
        db.upsert_project_events(pid, [{"file_path": row["file_path"], "event_order": 1,
                                        "start_seconds": 2, "end_seconds": 5, "duration_seconds": 3}])
    event = db.get_catalogued_events(2)[two["file_path"]][0]
    assert db.get_events_by_ids(1, [event["event_id"]]) == {}
    assert db.get_events_by_ids(2, [event["event_id"]])[event["event_id"]]["source_duration"] == 10


def test_unchanged_catalogue_write_preserves_clip_and_event_ids(seed_catalogue):
    one = seed_catalogue(1)[0]
    event = {"file_path": one["file_path"], "event_order": 1, "start_seconds": 0, "end_seconds": 5}
    db.upsert_project_events(1, [event])
    old_event_id = db.get_catalogued_events(1)[one["file_path"]][0]["event_id"]
    two = seed_catalogue(2)[0]
    db.upsert_project_events(2, [{**event, "file_path": two["file_path"]}])
    db.upsert_project_clips(1, [one])
    db.upsert_project_events(1, [event])
    assert db.get_catalogued_clips(1)[one["file_path"]]["clip_id"] == one["clip_id"]
    assert db.get_catalogued_events(1)[one["file_path"]][0]["event_id"] == old_event_id
