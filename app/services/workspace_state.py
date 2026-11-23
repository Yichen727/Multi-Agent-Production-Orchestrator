"""Pure session-state reconciliation for project switches and catalogue updates."""

from app.services.contracts import project_id_value
from app.services.plan_contract import validate_edit_plan

SELECTION_PREFIX = "clip_selected:"
_OUTPUT_DEFAULTS = {
    "messages": [], "bin_clips": [], "search_results": [], "ingest_done": False,
    "selection_output": "", "last_timeline": "", "last_timeline_plan": None,
    "delivery_output_text": "", "delivery_result": None,
}


def reconcile_workspace(state, project_id, user_id, clips) -> None:
    """Existing catalogues are usable immediately; only valid plans survive refresh."""
    key = (project_id_value(project_id), str(user_id))
    if state.get("workspace_key") != key:
        for name in list(state):
            if name.startswith(SELECTION_PREFIX) or name in (
                    "select_all_bin", "search_query", "intent", "editing_mode_label",
                    "aspect_ratio", "target_seconds", "seq_name"):
                del state[name]
        for name, default in _OUTPUT_DEFAULTS.items():
            state[name] = list(default) if isinstance(default, list) else default
        state["workspace_key"] = key
    online = {c["file_path"]: c for c in clips if c.get("online", 1)}
    for name in list(state):
        if name.startswith(SELECTION_PREFIX) and name[len(SELECTION_PREFIX):] not in online:
            del state[name]
    state["bin_clips"] = list(online.values())
    state["ingest_done"] = bool(online)
    state["search_results"] = [
        c for c in state.get("search_results", [])
        if c["file_path"] in online and c.get("source_revision") == online[c["file_path"]]["source_revision"]]
    plan = state.get("last_timeline_plan")
    if plan:
        try:
            validate_edit_plan(plan, key[0])
        except (ValueError, TypeError, KeyError, OSError):
            for name in ("selection_output", "last_timeline", "delivery_output_text"):
                state[name] = ""
            state["last_timeline_plan"] = None
            state["delivery_result"] = None
