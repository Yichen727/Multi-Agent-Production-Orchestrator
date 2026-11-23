"""Project/editor workspace state independent of transient Streamlit widgets."""

from copy import deepcopy
import uuid

from app.services.contracts import project_id_value
from app.services.plan_contract import validate_edit_plan

# Kept only for the explicit P1b path-key migration.
SELECTION_PREFIX = "clip_selected:"
_FIELDS = {
    "messages": [], "clip_metadata": [], "candidate_clip_ids": [], "search_results": [],
    "edit_report": "", "edit_plan": None, "export_report": "", "export_result": None, "exported_plan_id": None,
    "saved_edit_plans": [], "active_plan_id": None, "ingest_result": None,
    "ui_values": {}, "preview_clip_id": None,
}
_LEGACY_FIELDS = {
    "bin_clips": "clip_metadata", "selection_output": "edit_report",
    "last_timeline_plan": "edit_plan", "delivery_output_text": "export_report",
    "delivery_result": "export_result",
}


def _migrate_legacy(state, clips):
    """Convert the current P1b workspace once; never match by basename."""
    if state.get("workspace_schema_version") == 2:
        return
    for old, new in _LEGACY_FIELDS.items():
        if old in state and new not in state:
            state[new] = deepcopy(state[old])
        state.pop(old, None)
    paths = {c["file_path"]: c["clip_id"] for c in clips if c.get("online", 1)}
    ids = set(state.get("candidate_clip_ids", []))
    for key in list(state):
        if key.startswith(SELECTION_PREFIX):
            if state[key] and key[len(SELECTION_PREFIX):] in paths:
                ids.add(paths[key[len(SELECTION_PREFIX):]])
            del state[key]
    state["candidate_clip_ids"] = sorted(ids)
    old_inputs = {"search_query": "find_query", "intent": "plan_intent",
                  "aspect_ratio": "plan_aspect", "target_seconds": "plan_target",
                  "seq_name": "export_name"}
    values = state.setdefault("ui_values", {})
    for old, new in old_inputs.items():
        if old in state:
            values.setdefault(new, state.pop(old))
    old_mode = state.pop("editing_mode_label", "")
    if old_mode:
        values.setdefault("plan_mode", "Selected Moments" if "Moment" in old_mode else "Full Clips")
    for old in ("ingest_done", "last_timeline", "select_all_bin"):
        state.pop(old, None)
    state["workspace_schema_version"] = 2


def reconcile_workspace(state, project_id, user_id, clips, *, verify_plan=True):
    key = (project_id_value(project_id), str(user_id))
    previous = state.get("workspace_key")
    # Migration applies only to the workspace the old state actually belonged to.
    _migrate_legacy(state, clips if previous is None or previous == key else state.get("bin_clips", []))
    if previous != key:
        archive = state.setdefault("workspace_archive", {})
        if previous is not None:
            archive[previous] = {k: deepcopy(state.get(k, v)) for k, v in _FIELDS.items()}
        restored = archive.get(key, {k: deepcopy(state[k]) for k in _FIELDS if k in state} if previous is None else {})
        for name, default in _FIELDS.items():
            state[name] = deepcopy(restored.get(name, default))
        for name in list(state):
            if name.startswith(("candidate:", "library_", "find_", "plan_", "export_", "footage_", "import_")) and name not in _FIELDS:
                del state[name]
            elif name.startswith(SELECTION_PREFIX):
                del state[name]
        state["workspace_key"] = key
        state["workspace_schema_version"] = 2
    for name, default in _FIELDS.items():
        state.setdefault(name, deepcopy(default))
    online = {c["clip_id"]: c for c in clips if c.get("online", 1)}
    state["clip_metadata"] = list(online.values())
    state["candidate_clip_ids"] = sorted(set(state["candidate_clip_ids"]) & online.keys())
    state["search_results"] = [
        c for c in state["search_results"] if c.get("clip_id") in online
        and c.get("source_revision") == online[c["clip_id"]]["source_revision"]]
    if state["edit_plan"] and verify_plan:
        try:
            validate_edit_plan(state["edit_plan"], key[0])
        except (ValueError, TypeError, KeyError, OSError):
            state.update(edit_plan=None, edit_report="", export_report="", export_result=None, exported_plan_id=None)
    # Preserve a pre-P1c plan as a named session version.
    if state["edit_plan"] and not state["saved_edit_plans"]:
        save_edit_plan(state, state["edit_plan"], state["edit_report"], "Recovered plan")


def set_candidates(state, clip_ids, selected):
    allowed = {c["clip_id"] for c in state["clip_metadata"]}
    requested = set(clip_ids) & allowed
    current = set(state["candidate_clip_ids"])
    state["candidate_clip_ids"] = sorted((current | requested) if selected else (current - requested))


def save_edit_plan(state, plan, report, name):
    """Save a verified immutable session version; JSON download persists it outside the session."""
    verified = validate_edit_plan(plan, state["workspace_key"][0])
    record = {"id": uuid.uuid4().hex, "name": str(name or "Edit plan").strip(),
              "plan": deepcopy(verified), "report": str(report or "")}
    state["saved_edit_plans"].append(record)
    state.update(edit_plan=verified, edit_report=record["report"], active_plan_id=record["id"],
                 export_report="", export_result=None, exported_plan_id=None)
    return record
