"""Versioned plan reader and verified catalogue references.

Legacy names are accepted only here. Reading never rewrites the original file or
invents a source revision for a historical reference.
"""

from copy import deepcopy
import json
import math

from app.services.catalogue_resolver import resolve_one
from app.services.contracts import project_id_value
from app.services.database_service import validate_segment_source, get_events_by_ids
from app.services.media_identity import canonical_path
from app.services.premiere_export_service import normalise_aspect_label

PLAN_VERSION = 1


def read_edit_plan(value, project_id) -> dict:
    pid = project_id_value(project_id)
    data = json.loads(value) if isinstance(value, (str, bytes)) else deepcopy(value)
    if isinstance(data, list):
        data = {"segments": data}
    if not isinstance(data, dict):
        raise ValueError("Timeline plan must be a JSON object")
    version = data.get("schema_version", 0)
    if type(version) is not int or version not in (0, PLAN_VERSION):
        raise ValueError("Unsupported timeline schema_version")
    if data.get("project_id") is not None and project_id_value(data["project_id"]) != pid:
        raise ValueError("Plan belongs to another project")
    if "segments" not in data and "sequence" in data and "clips" in data:
        if (not isinstance(data["sequence"], dict) or not isinstance(data["clips"], list)
                or any(not isinstance(c, dict) for c in data["clips"])):
            raise ValueError("Invalid exported timeline structure")
        sequence = data["sequence"]
        data = {**data,
                "aspect_ratio": sequence.get("aspect_ratio"),
                "segments": [{**c, "in_point": c.get("in_seconds", 0),
                              "out_point": c.get("out_seconds"),
                              "duration": c.get("used_seconds"),
                              "label": c.get("role", "")} for c in data["clips"]]}
        # Historical exports may contain trims, so do not claim whole-clip semantics.
        has_events = any(c.get("event_id") is not None for c in data["clips"])
        data["mode"] = ("legacy_timeline" if version == 0 else
                        data.get("mode", "moment_assembly" if has_events else "clip_assembly"))
        data.pop("clips")
        data.pop("sequence")
    if not isinstance(data.get("segments"), list) or not data["segments"]:
        raise ValueError("No segments to compile. Provide the Selection plan JSON.")
    for seg in data["segments"]:
        if not isinstance(seg, dict):
            raise ValueError("Every segment must be a JSON object")
        if "shot_id" in seg:
            if version != 0:
                raise ValueError("Version 1 plans must use clip_id")
            old_id = seg.pop("shot_id")
            if seg.get("clip_id") is not None and seg["clip_id"] != old_id:
                raise ValueError("Conflicting legacy and canonical clip identifiers")
            seg["clip_id"] = old_id
        if not seg.get("file_path"):
            raise ValueError("Every segment must include its original file_path")
        # Reading catches an ID which now points to a different path, even if the
        # old path still exists under another ID. Never silently substitute it.
        row = resolve_one(pid, seg.get("clip_id") if seg.get("clip_id") is not None else seg["file_path"])
        if not row or canonical_path(seg["file_path"]) != row["canonical_path"]:
            raise ValueError("Plan ID/path mismatch or unresolved clip in this project")
        if seg.get("clip_id") is None:
            seg["clip_id"] = row["clip_id"]
        if seg.get("source_duration") is not None and not math.isclose(
                float(seg["source_duration"]), float(row["duration_seconds"] or 0), abs_tol=0.002):
            raise ValueError("Plan source duration differs from the catalogue")
    data["schema_version"] = PLAN_VERSION
    data["project_id"] = pid
    if version == 0:
        data["legacy_import"] = True
    return data


def validate_edit_plan(value, project_id) -> dict:
    """Validate every reference and trim before any probing or output writes."""
    plan = read_edit_plan(value, project_id)
    if plan.get("valid") is False:
        raise ValueError("The plan contains invalid segment(s)")
    mode = plan.get("mode", "clip_assembly")
    if mode not in ("clip_assembly", "moment_assembly", "legacy_timeline"):
        raise ValueError("Unknown editing mode")
    if mode == "legacy_timeline" and not plan.get("legacy_import"):
        raise ValueError("Legacy timelines must originate from the compatibility reader")
    plan["aspect_ratio"] = normalise_aspect_label(plan.get("aspect_ratio"))
    for index, seg in enumerate(plan["segments"], 1):
        if seg.get("valid") is False:
            raise ValueError(f"Invalid segment #{index}")
        row = resolve_one(plan["project_id"], seg["clip_id"])
        validate_segment_source(seg, row)
        start, end = seg.get("in_point"), seg.get("out_point")
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("Invalid trim range")
        start, end = float(start), float(end)
        duration = float(row["duration_seconds"] or 0)
        if not all(math.isfinite(v) for v in (start, end, duration)) or not 0 <= start < end <= duration + 0.002:
            raise ValueError(f"Invalid trim range for segment #{index}")
        if mode == "clip_assembly" and (start != 0 or abs(end - duration) > 0.002):
            raise ValueError("Clip Assembly must preserve the whole source clip")
        if mode == "moment_assembly":
            if seg.get("event_id") is None:
                raise ValueError("Moment Assembly requires an active event_id")
            event = get_events_by_ids(plan["project_id"], [seg["event_id"]])[int(seg["event_id"])]
            if start < event["start_seconds"] - 0.002 or end > event["end_seconds"] + 0.002:
                raise ValueError("Moment trim exceeds the authoritative event boundaries")
        seg.update(in_point=start, out_point=end)
    return plan
