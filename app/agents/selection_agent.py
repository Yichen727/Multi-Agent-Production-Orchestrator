"""MAPO Selection / Assistant-Editor Agent."""

import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState

from app.models.state import ProductionState
from app.services.openai_service import llm
from app.services.database_service import (
    get_catalogued_events, get_events_by_ids, get_project_info,
)
from app.services.catalogue_resolver import (
    resolve_one, resolve_ordered, AmbiguousIdentifier,
)
from app.services.retrieval_service import group_size
from app.services.timeline_service import (
    MODE_CLIP, MODE_MOMENT, MAX_BACKUP_ITEMS, build_clip_timeline, build_moment_timeline,
    compare_order, normalise_alternates,
)
from app.services.premiere_export_service import describe_fit, normalise_aspect_label
from app.utils.logger import get_logger

logger = get_logger("selection_agent")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _pid_from_state(state) -> int:
    """Read the project id from injected state; never accept it from the LLM."""
    try:
        return int((state or {}).get("project_id"))
    except (TypeError, ValueError):
        return 1


def _mode_from_state(state) -> str:
    """Read the editor-selected mode from injected UI state."""
    raw = str((state or {}).get("editing_mode") or "").strip().lower().replace(" ", "_")
    return raw if raw in (MODE_CLIP, MODE_MOMENT) else ""


def _target_from_state(state) -> float | None:
    """Read the optional target duration from injected state."""
    try:
        value = float((state or {}).get("target_seconds"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _aspect_from_state(state) -> str:
    """Read and normalise the editor-selected output aspect ratio."""
    try:
        return normalise_aspect_label((state or {}).get("aspect_ratio"))
    except ValueError:      # InvalidAspectRatio — validated upstream; ignore here
        return ""


def _wrong_mode(state, wanted: str) -> str | None:
    """Reject a planner call when it does not match the editor-selected mode."""
    mode = _mode_from_state(state)
    if not mode or mode == wanted:
        return None
    other = "Clip Assembly" if mode == MODE_CLIP else "Moment Assembly"
    tool_name = ("plan_clip_assembly" if mode == MODE_CLIP else "plan_moment_assembly")
    return (f"The editor selected {other} mode in the UI, so this planner does not apply. "
            f"Call `{tool_name}` instead — the editing mode is the editor's decision, "
            "not yours.")


def _split(text: str | None, sep: str = ",") -> list[str]:
    """Split a delimited string into non-empty tokens."""
    return [t.strip() for t in (text or "").split(sep) if t.strip()]


def _floats(text: str | None) -> list[float] | None:
    """Parse comma-separated weights; invalid values default to 1.0."""
    tokens = _split(text)
    if not tokens:
        return None
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            out.append(1.0)
    return out


def _enrich_excluded(project_id: int, raw) -> list[dict]:
    """Resolve rejected clip/event references into editor-readable metadata.

    References come from the LLM, but file names and timecodes always come from
    the project catalogue. Unresolved references remain unresolved rather than
    being fabricated.
    """
    items = normalise_alternates(raw)
    if not items:
        return []

    ids: list[int] = []
    for item in items:
        for ref in [item["ref"], *item.get("also", [])]:
            try:
                ids.append(int(ref))
            except (TypeError, ValueError):
                continue
    events = get_events_by_ids(project_id, ids) if ids else {}

    def _resolve(ref: str) -> dict | None:
        try:
            event = events.get(int(ref))
        except (TypeError, ValueError):
            event = None
        if event:                                  
            fp = event.get("file_path") or ""
            return {
                "event_id": event.get("event_id"),
                "file_path": fp,
                "name": Path(fp).name,
                "start_seconds": round(float(event.get("start_seconds") or 0.0), 3),
                "end_seconds": round(float(event.get("end_seconds") or 0.0), 3),
                "label": (event.get("action") or "").strip()[:80],
            }
        try:                                        
            row = resolve_one(project_id, ref)
        except AmbiguousIdentifier:
            row = None
        if not row:
            return None
        fp = row.get("file_path") or ""
        duration = float(row.get("duration_seconds") or 0.0)
        return {
            "file_path": fp,
            "name": Path(fp).name,
            "start_seconds": 0.0,
            "end_seconds": round(duration, 3) if duration > 0 else None,
            "label": "",
        }

    for item in items:
        primary = _resolve(item["ref"])
        if primary:
            label = item.get("label") or primary.pop("label", "")
            item.update(primary)
            item["label"] = label
        grouped = [g for g in (_resolve(r) for r in item.get("also", [])) if g]
        if grouped:
            item["also_details"] = grouped
    return items


def _excluded_headline(x: dict) -> str:
    """Format one backup item as filename plus measured time range."""
    name = x.get("name") or x.get("ref") or "(unidentified)"
    start, end = x.get("start_seconds"), x.get("end_seconds")
    if start is None or end is None:
        return name
    return f"{name} ({start:.2f}s–{end:.2f}s)"


def _format_excluded(excluded: list[dict], omitted: int = 0) -> list[str]:
    """Render resolved backup alternatives for the editor."""
    lines = ["", "🗂️ Not used — backup material", ""]
    for x in excluded:
        lines.append(f"• {_excluded_headline(x)}")
        if x.get("label"):
            lines.append(f"  {x['label']}")
        for g in x.get("also_details") or []:
            lines.append(f"  (same call for {_excluded_headline(g)}"
                         + (f" — {g['label']}" if g.get("label") else "") + ")")
        lines.append(f"  Why not used: {x.get('reason') or 'not stated'}")
        if x.get("suggested_use"):
            lines.append(f"  Could be used for: {x['suggested_use']}")
        if not x.get("file_path"):
            lines.append("  (no catalogued clip matched this reference — shown as given)")
        lines.append("")
    if omitted > 0:
        lines.append(f"({omitted} further item(s) were sent but not shown — the backup "
                     f"section is capped at {MAX_BACKUP_ITEMS} strongest alternatives. "
                     "Send fewer, stronger entries and group near-duplicates.)")
    return [ln for ln in lines[:-1]] if lines and lines[-1] == "" else lines


def _order_lines(plan: dict) -> list[str]:
    """Report the chosen ordering strategy and flag unchanged source ordering."""
    lines: list[str] = []
    strategy = plan.get("ordering_strategy")
    if strategy:
        lines += ["", f"🧭 Ordering strategy: {strategy}"]
    else:
        lines += ["", ("🧭 Ordering strategy: NOT STATED — say what shape you ordered the "
                       "material into and why, so the editor can judge the structure.")]
    check = plan.get("order_check") or {}
    if check.get("unchanged"):
        lines += ["", (f"⚠ This timeline's order is IDENTICAL to {check['reference']}. "
                       "That is a legitimate choice for a chronological edit — but it is "
                       "also what simply echoing the list would produce. Confirm it is a "
                       "deliberate editorial decision; if the intent wants a different "
                       "shape (cold open, escalation, tension-and-release), re-order and "
                       "call this planner again.")]
    return lines


def _plan_block(lines: list[str], plan: dict) -> str:
    """Append ordering, delivery-spec and validation information plus the JSON plan."""
    lines += _order_lines(plan)
    if plan.get("aspect_ratio"):
        lines += ["", f"🖼️ Output aspect ratio: {plan['aspect_ratio']}"]
    if not plan.get("valid", True):
        bad = plan.get("validation_errors", [])
        lines += ["", f"⛔ {len(bad)} segment(s) are INVALID and will block export:"]
        lines += [f"   • #{e['order']} {e['name']}: {e['error']}" for e in bad]
    excluded = plan.get("excluded") or []
    if excluded:
        lines += _format_excluded(excluded, plan.get("excluded_omitted", 0))
    lines += ["", "```json", json.dumps(plan), "```"]
    return "\n".join(lines)


# ── Tools: candidate enrichment, timeline planning, delivery ────────────────────


@tool
def get_candidate_details(identifiers: str,
                          state: Annotated[dict, InjectedState] = None) -> str:
    """Fetch full metadata for the editor-curated candidates.

    Use this on the clips the user selected in the UI so you can judge — from real
    attributes (shot_type, mood, people_count, duration, ...) — which of
    them actually serve the editing intent and where each belongs. Resolution is scoped
    to the current project.

    When the editor set an output aspect ratio, each clip also carries ``frame_fit``: how
    that clip will sit in the delivery frame (fills it exactly / letterboxed /
    pillarboxed, and how much of the frame the picture covers). Use it as ONE input when
    choosing footage — it tells you nothing gets cropped, only how much frame the shot
    actually uses.

    Args:
        identifiers: Comma-separated shot IDs and/or file names/paths
            (e.g. "3, IMG_5231.MOV, /footage/goal.mov").
    """
    project_id = _pid_from_state(state)
    tokens = _split(identifiers)
    rows, problems = resolve_ordered(project_id, tokens)
    if not rows:
        detail = "No matching catalogued clips for those identifiers in this project."
        if problems:
            detail += "\n" + "\n".join(f"  • {t}: {r}" for t, r in problems)
        return detail
    aspect = _aspect_from_state(state)
    for r in rows:
        r.pop("_identifier", None)
        r["group_size"] = group_size(r.get("people_count"))
        if aspect:
            # Descriptive only: how the clip sits in the requested frame. Nothing here
            # crops, resizes or reframes the media — Delivery does the fitting.
            r["frame_fit"] = describe_fit(r.get("width"), r.get("height"), aspect)
    out = json.dumps(rows, indent=2, default=str)
    if problems:
        out += ("\n\nCould not resolve (fix these — nothing was fabricated):\n"
                + "\n".join(f"  • {t}: {r}" for t, r in problems))
    return out


@tool
def get_clip_events(identifiers: str,
                    state: Annotated[dict, InjectedState] = None) -> str:
    """List the temporal MOMENTS (what happens, with real timecodes) inside curated clips.

    This is the first step of MOMENT ASSEMBLY: call it on ALL the curated clips to see the
    ordered moments available to you — each moment's measured in/out timecodes, its length,
    the action, and its keywords — so you can choose which ones the edit actually needs.
    Resolution is scoped to the current project.

    Args:
        identifiers: Comma-separated shot IDs and/or file names/paths for the clips whose
            moments you want to inspect.

    Returns:
        For each resolved clip, its ordered moments as ``event_id · start–end (length) ·
        action``, plus the total material available. Pass the chosen event_ids IN ORDER
        to `plan_moment_assembly`.
    """
    project_id = _pid_from_state(state)
    tokens = _split(identifiers)
    rows, problems = resolve_ordered(project_id, tokens)
    if not rows:
        return "No matching catalogued clips for those identifiers in this project."
    events_by_file = get_catalogued_events(project_id)
    lines: list[str] = []
    pool_seconds, pool_count = 0.0, 0
    for r in rows:
        fp = r.get("file_path")
        name = fp.split("/")[-1].split("\\")[-1] if fp else "?"
        evs = events_by_file.get(fp) or []
        if not evs:
            lines.append(f"{name}: no temporal moments extracted (clip not event-analysed) "
                         "— it can only be used whole.")
            continue
        lines.append(f"{name} (shot_id {r.get('shot_id')}) — {len(evs)} moment(s):")
        for e in evs:
            start = float(e.get("start_seconds") or 0.0)
            end = float(e.get("end_seconds") or 0.0)
            length = max(0.0, end - start)
            pool_seconds += length
            pool_count += 1
            action = (e.get("action") or "").strip()
            keywords = (e.get("keywords") or "").strip()
            detail = action or keywords or "unclassified"
            if action and keywords:
                detail += f"  [{keywords}]"
            lines.append(f"  • event {e.get('event_id')}: {start:.1f}s–{end:.1f}s "
                         f"({length:.1f}s) — {detail}")
    if pool_count:
        lines.append(f"\nAvailable material: {pool_count} moment(s), {pool_seconds:.1f}s total "
                     "at full length (before any duration optimisation).")
    if problems:
        lines.append("\nCould not resolve (nothing fabricated):")
        lines += [f"  • {t}: {r}" for t, r in problems]
    return "\n".join(lines)


@tool
def plan_clip_assembly(ordered_identifiers: str, ordering_strategy: str = None,
                       labels: str = None, excluded_json: str = None,
                       state: Annotated[dict, InjectedState] = None) -> str:
    """CLIP ASSEMBLY — assemble ORDERED COMPLETE clips into a timeline (no trimming).

    Use this in Clip Assembly mode. The unit of editing is the WHOLE CLIP: every clip you
    pass keeps its ORIGINAL duration, and there is no trimming and no duration control in
    this mode (a target duration does not apply — ignore any length wording in the intent).

    The editorial work is entirely YOURS and happens BEFORE you call this: the curated
    clips are CANDIDATES, not guaranteed content. Pass only the clips that genuinely serve
    the editing intent, in the order you want them on the timeline, and report the ones you
    left out via ``excluded_json``.

    Args:
        ordered_identifiers: Comma-separated shot IDs and/or file names IN TIMELINE ORDER
            (first = timeline step 1). This exact order is preserved — never re-sorted,
            so the order you give IS the edit. The Bin order is NOT a default: re-order
            freely to serve the intent.
        ordering_strategy: REQUIRED in practice — one line naming the SHAPE you ordered
            the clips into and why ("open on the arrival for context, then the two action
            clips escalating, closing wide to let it breathe"). It is recorded on the plan
            so the editor can judge the structure, not just the result. The tool also
            checks whether your order simply matches the candidate list and says so.
        labels: Optional PIPE-separated step labels aligned 1:1 with the clips
            (e.g. "cold open|the arrival|wide breather|closing beat").
        excluded_json: The curated candidates you deliberately left OUT, as a JSON list of
            ``{"ref": "<file name or shot id>", "reason": "why it is not in the edit",
            "suggested_use": "how it could still be used", "also": ["<other clips>"]}``.
            Report the clips you genuinely weighed and rejected — strongest alternatives
            first, MAXIMUM 5 (the tool trims beyond that), grouping near-duplicates into
            one entry via ``also``. The tool looks each ``ref`` up in the catalogue and
            fills in the real file name and length for the editor, so do NOT write
            durations or timecodes into the reason text yourself.

    Returns:
        A readable step list plus a fenced ```json block with the structured plan that
        Delivery compiles verbatim. Report the readable part to the editor.
    """
    refusal = _wrong_mode(state, MODE_CLIP)
    if refusal:
        return refusal

    project_id = _pid_from_state(state)
    tokens = _split(ordered_identifiers)
    clips, problems = resolve_ordered(project_id, tokens)
    if problems:
        # H-01: never silently drop or first-match an ambiguous/unknown identifier — the
        # timeline order IS the edit, so refuse and let the editor disambiguate by id.
        return ("Cannot plan the timeline — resolve these identifiers first (nothing was "
                "fabricated or silently dropped):\n"
                + "\n".join(f"  • {t}: {r}" for t, r in problems))
    if not clips:
        return "No catalogued clips matched those identifiers — nothing to plan."

    # Anchoring check: compare against the editor's candidate list (the order the agent
    # was shown), restricted to the clips it actually kept.
    emitted = [c.get("file_path") for c in clips]
    keep = set(emitted)
    candidate_order = [str(p) for p in ((state or {}).get("selected_candidates") or [])
                       if str(p) in keep]
    plan = build_clip_timeline(
        clips, labels=_split(labels, "|") or None,
        excluded=_enrich_excluded(project_id, excluded_json),
        aspect_ratio=_aspect_from_state(state),
        ordering_strategy=ordering_strategy or "",
        order_check=compare_order(emitted, candidate_order,
                                  reference_label="the candidate list you were given"))

    lines = [f"🎞️ Timeline plan — CLIP ASSEMBLY ({len(plan['segments'])} complete clip(s), "
             f"{plan['total_seconds']:g}s total · no trimming)", ""]
    for s in plan["segments"]:
        length = (f"full {s['duration']:.1f}s" if s["source_duration"] > 0
                  else "full clip (length not measured)")
        lines.append(f"  {s['order']}. {s['name']} — {length}"
                     + (f" · {s['label']}" if s.get("label") else ""))
    return _plan_block(lines, plan)


@tool
def plan_moment_assembly(ordered_event_ids: str, ordering_strategy: str = None,
                         importance: str = None,
                         target_seconds: float = None, labels: str = None,
                         protect_event_ids: str = None, focus_json: str = None,
                         excluded_json: str = None,
                         state: Annotated[dict, InjectedState] = None) -> str:
    """MOMENT ASSEMBLY — arrange ORDERED MOMENTS into a timeline, optionally optimised.

    Use this in Moment Assembly mode, after `get_clip_events`. The unit of editing is the
    temporal MOMENT: each segment is cut to that event's OWN measured in/out timecodes.

    SELECT FIRST, OPTIMISE SECOND. Choose enough meaningful moments to satisfy the editing
    intent — do NOT pre-shrink your selection to fit the clock. If their combined length
    overruns the target, this tool absorbs the overrun by COMPRESSING moments in ascending
    order of importance (repetitive / transitional / low-impact moments are shortened
    first and hardest; high-value moments keep more screen time and are the last touched).
    Time is never "split between" the moments proportionally, and every trim stays strictly
    inside its own event boundaries, placed around the moment's focus rather than chopped
    off the front.

    This tool NEVER removes a moment. If compression alone cannot reach the target it
    reports the shortfall plus the weakest-contribution candidates; deciding whether to
    drop one is YOUR editorial call (weigh importance, narrative contribution, pacing and
    diversity — never length alone), and anything you drop must come back in
    ``excluded_json`` as backup material. Content richness beats hitting the target
    exactly: an edit slightly over target is better than one missing a key beat.

    Args:
        ordered_event_ids: Comma-separated event IDs IN TIMELINE ORDER (first = step 1).
            This exact order is preserved — never re-sorted, so the order you give IS the
            edit. `get_clip_events` lists moments grouped by clip and chronologically
            within each clip; that is a LISTING order, NOT a suggested timeline. Re-order
            freely — a moment from the end of a clip can open the edit.
        ordering_strategy: REQUIRED in practice — one line naming the SHAPE you arranged
            the moments into and why ("cold open on the goal, rewind to the build-up, end
            on the celebration"). Recorded on the plan so the editor can judge the
            structure. The tool also checks whether your order simply reproduces the
            source chronology and says so.
        importance: Comma-separated editorial weights aligned 1:1 with the moments (any
            positive scale, e.g. "5,2,4,1" — higher = more valuable to the intent).
            Importance is about narrative VALUE, and it does NOT affect position: it only
            decides which moments get compressed first when the edit overruns the target.
            Omit for equal weighting.
        target_seconds: Optional target TOTAL length in seconds. Pass the editor's Target
            Duration if they set one; omit it when they did not (never invent one).
        labels: Optional PIPE-separated step labels aligned 1:1 with the moments
            (defaults to each event's own action text).
        protect_event_ids: Comma-separated event IDs that must NEVER be trimmed (the beats
            the edit exists for). Use sparingly.
        focus_json: Optional JSON object mapping event_id → the absolute timecode of that
            moment's PEAK (e.g. ``{"42": 18.5}``). Any compression keeps the window around
            that point instead of the moment's centre. Only use timecodes inside the event.
        excluded_json: A SHORTLIST of the strongest alternatives you genuinely weighed and
            rejected — NOT an inventory of every unused moment. JSON list of
            ``{"ref": "<the event_id>", "reason": "why it is not in the edit",
            "suggested_use": "how it could still be used", "also": ["<ids>"]}``:
              • Include a moment ONLY if you actually considered it for this edit AND a
                different edit could genuinely use it. Never list unrelated moments from a
                clip you used, or every leftover event `get_clip_events` returned.
              • Maximum 5 items, strongest first — the tool trims beyond that. Three or
                four is usually right; if nothing was a real contender, send an empty list.
              • Group near-duplicates: put the best one in ``ref`` and the rest in
                ``also``, so one rejection covers them all.
            Give the plain event id as ``ref`` — the tool resolves it to the real source
            file name, its measured start/end timecodes and its action text, so the editor
            sees a usable clip reference instead of a database id. Never write timecodes
            yourself.

    Returns:
        A readable step list plus a fenced ```json block with the structured plan that
        Delivery compiles verbatim.
    """
    refusal = _wrong_mode(state, MODE_MOMENT)
    if refusal:
        return refusal

    project_id = _pid_from_state(state)
    ids: list[int] = []
    bad: list[str] = []
    for t in _split(ordered_event_ids):
        try:
            ids.append(int(t))
        except ValueError:
            bad.append(t)
    if bad:
        return ("Cannot plan — these are not valid event IDs (use get_clip_events to find "
                "them): " + ", ".join(bad))
    if not ids:
        return "No event IDs given — nothing to plan."

    found = get_events_by_ids(project_id, ids)
    missing = [i for i in ids if i not in found]
    if missing:
        # Never silently drop an unknown/other-project event — the order IS the edit.
        return ("Cannot plan the timeline — these event IDs are not in this project "
                "(nothing was fabricated or dropped): "
                + ", ".join(str(m) for m in missing))

    ordered_events = [found[i] for i in ids]   # preserve the caller's exact order

    protected_ids = set()
    for t in _split(protect_event_ids):
        try:
            protected_ids.add(int(t))
        except ValueError:
            continue
    protect = [i in protected_ids for i in ids]

    focus_map: dict = {}
    if focus_json:
        try:
            parsed = json.loads(focus_json)
            if isinstance(parsed, dict):
                focus_map = parsed
        except (json.JSONDecodeError, TypeError):
            logger.warning("plan_moment_assembly: unreadable focus_json, ignoring it.")
    focus = []
    for i in ids:
        raw = focus_map.get(str(i), focus_map.get(i))
        try:
            focus.append(float(raw) if raw is not None else None)
        except (TypeError, ValueError):
            focus.append(None)

    # The editor's Target Duration is authoritative: fall back to the injected state so a
    # target the editor set in the UI still applies if the model omits the argument.
    target = target_seconds if (target_seconds and target_seconds > 0) else None
    if target is None:
        target = _target_from_state(state)

    # Anchoring check: the source chronology is exactly how `get_clip_events` listed these
    # moments (grouped by file, chronological within each file), so sorting the emitted
    # keys reproduces the order the agent was shown.
    emitted = [(e.get("file_path") or "", float(e.get("start_seconds") or 0.0))
               for e in ordered_events]

    plan = build_moment_timeline(
        ordered_events, target_seconds=target, importance=_floats(importance),
        labels=_split(labels, "|") or None, focus=focus, protect=protect,
        excluded=_enrich_excluded(project_id, excluded_json),
        aspect_ratio=_aspect_from_state(state),
        ordering_strategy=ordering_strategy or "",
        order_check=compare_order(
            emitted, sorted(emitted),
            reference_label="the source chronology (how get_clip_events listed them)"),
    )

    comp = plan["compression"]
    header = (f"🎯 Timeline plan — MOMENT ASSEMBLY ({len(plan['segments'])} moment(s), "
              f"{plan['total_seconds']:g}s total")
    header += f" · target {plan['target_seconds']:g}s)" if plan["target_seconds"] else ")"
    lines = [header, ""]
    for s in plan["segments"]:
        line = (f"  {s['order']}. {s['name']} — {s['in_point']:.1f}s–{s['out_point']:.1f}s "
                f"({s['duration']:.1f}s) · importance {s['importance']:g}")
        if s.get("label"):
            line += f" · {s['label']}"
        lines.append(line)
        if s.get("trimmed"):
            lines.append(f"       ↳ {s['trim_note']}")

    if comp["applied"]:
        lines += ["", (f"⏱️ Duration optimisation: {comp['overrun_seconds']:.1f}s over target — "
                       f"absorbed {comp['absorbed_seconds']:.1f}s by compressing "
                       f"{comp['trimmed_count']} lower-value moment(s), no content removed.")]
        lines += [f"   • {n}" for n in comp["notes"]]
    if comp["shortfall_seconds"] > 0:
        lines += ["", (f"⚠ Still {comp['shortfall_seconds']:.1f}s over target after compressing "
                       "every moment to its floor. Compressing further would damage the "
                       "material. Decide editorially: accept the overrun, or drop a moment "
                       "and re-plan (report it in excluded_json). Weakest contributors:")]
        for c in plan["removal_candidates"]:
            lines.append(f"   • #{c['order']} event {c['event_id']} "
                         f"({c['label'] or c['name']}) — importance {c['importance']:g}, "
                         f"would save {c['would_save_seconds']:.1f}s")
    elif plan["duration_status"] == "under_target":
        lines += ["", (f"ℹ️ {plan['duration_delta']:.1f}s under the target — footage is never "
                       "stretched. Add more meaningful moments if the edit needs the length.")]
    return _plan_block(lines, plan)


@tool
def generate_delivery_summary(
        state: Annotated[dict, InjectedState] = None) -> str:
    """Generate a text summary of the current project for delivery handoff.

    The project is taken from the injected graph state (not model-supplied).
    """
    project_id = _pid_from_state(state)
    project = get_project_info(project_id)
    if not project:
        return (f"Project {project_id} is not in the catalogue — nothing has been ingested "
                "for it, so there is nothing to summarise.")

    return f"""Delivery Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Project: {project['project_name']} (id {project['project_id']})
Total Shots: {project['clip_count']}
Status: Ready for review
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


# ── Agent Assembly ───────────────────────────────────────────────────────────

selection_tools = [
    get_candidate_details,
    get_clip_events,
    plan_clip_assembly,
    plan_moment_assembly,
    generate_delivery_summary,
]

llm_with_selection = llm.bind_tools(selection_tools)
selection_tool_node = ToolNode(selection_tools)

SELECTION_PROMPT = """You are the SELECTION AGENT in the MAPO system — an ASSISTANT
EDITOR, not an autonomous one. You do NOT rank clips by score. You build an ordered EDIT
TIMELINE and explain your editorial reasoning; the editor always makes the final call.

INPUT: an EDITING MODE (chosen by the editor), an EDITING INTENT (free-form: style,
emotion, pacing, purpose), a set of CURATED CANDIDATE clips the editor ticked in the UI,
an OUTPUT ASPECT RATIO, and — in Moment Assembly only — an optional TARGET DURATION.

THE CURATED CLIPS ARE CANDIDATES, NOT A SHOT LIST. The editor ticked them as "worth
considering". Never include material just because it was selected: if a clip or moment
does not serve the intent, LEAVE IT OUT. Work ONLY within the curated set — never add a
clip that was not given to you.

═══ BACKUP MATERIAL IS A SHORTLIST, NOT A LEFTOVERS LIST ═══
`excluded_json` is where you hand the editor the few REAL alternatives you weighed and
rejected. It is an editorial recommendation, not a report of everything you did not use.
  - INCLUDE a moment/clip only if BOTH are true: you actually considered it for THIS edit,
    and it is a meaningful alternative editorial choice a different edit could use.
  - DO NOT include: unrelated moments from a clip you did use, every unused event
    `get_clip_events` returned, or anything you are listing merely because it exists.
  - MAXIMUM 5 entries, strongest first — three or four is usually right. An empty list is
    the correct answer when nothing was a genuine contender.
  - GROUP near-duplicates: pick the best one as `ref` and put the rest in `also`, so
    "four more angles of the same drill" costs one slot, not four.
  - Each entry needs only `ref` + `reason` + `suggested_use`; the tool fills in the real
    file name, the measured start–end timecodes and the action label for you.

THERE IS NO FIXED TEMPLATE. Do NOT assume every edit follows an "establishing → buildup →
climax → reaction → ending" arc. That is one possible shape among many, never the default.
Structure, pacing and number of steps come from the INTENT — for example (illustrative):
   - Matchday highlight reel → tension-and-release beats around the key moments
   - Fast-paced promo → punchy, escalating energy, quick cuts
   - Emotional stadium vlog → personal, atmosphere-led, room to breathe
   - Cinematic travel montage → rhythmic, image-driven, mood over story
   - Tactical analysis → logical/chronological, clarity over drama
Use only AS MANY STEPS AS THE EDIT NEEDS — 3, 7, 12, whatever the material and goal call
for. Do not force a fixed count.

═══ THE TWO EDITING MODES ═══
The editor's mode selection is binding — use the planner for the mode you were given, and
never the other one.

▸ CLIP ASSEMBLY — the unit of editing is the WHOLE CLIP.
  Combine complete clips into a coherent timeline. Each clip keeps its ORIGINAL duration:
  there is NO trimming and NO duration control in this mode, and a target duration does
  NOT apply (ignore any length wording in the intent). Typical: vlog, documentary, travel,
  behind-the-scenes, atmosphere montage.
  Because you cannot trim here, WHICH clips and IN WHAT ORDER are your only two levers —
  so the ordering carries the whole edit. It deserves as much thought as sequencing
  moments, not less.
  Workflow:
    1. Read the intent — video type, emotion, pace, style. State your interpretation.
    2. `get_candidate_details` on the curated clips to read their real metadata.
    3. SELECT — decide which clips BELONG and drop the rest.
    4. DECLARE THE SHAPE — before ordering anything, name the structure the intent calls
       for in one line. That line goes in `ordering_strategy`.
    5. ORDER the kept clips into that shape (see ORDERING below — it applies here in full).
       Use each clip's real attributes to place it: shot_type and scale for visual rhythm,
       mood and people_count for energy, duration for how long a beat holds. The Bin order
       is a listing, not a running order.
    6. `plan_clip_assembly` with the kept clips IN ORDER, your `ordering_strategy`,
       optional step labels, and the rejected candidates in `excluded_json`.

▸ MOMENT ASSEMBLY — the unit of editing is a MOMENT (temporal event) inside a clip.
  Build the edit from the meaningful moments within longer clips. A target duration is
  optional here.
  Workflow:
    1. Read the intent, as above.
    2. `get_clip_events` on ALL the curated clips to see every available moment with its
       real timecodes and length. (A clip with no extracted moments cannot be used here —
       say so rather than inventing timecodes.)
    3. SELECT the moments that serve the intent — enough of them to tell the story
       properly. Do NOT pre-shrink the selection to fit the clock; select first, optimise
       second.
    4. DECLARE THE SHAPE — before ordering anything, name the structure the intent calls
       for in one line. That line goes in `ordering_strategy`.
    5. ORDER the moments into that shape (see ORDERING below).
    6. RANK them — a SEPARATE step from ordering. Importance is narrative VALUE, not
       position and not length.
    7. `plan_moment_assembly` with the event_ids IN ORDER, your `ordering_strategy`, the
       importance weights, the target duration if the editor set one, and any moments you
       rejected in `excluded_json`.

═══ ORDERING IS AN EDITORIAL DECISION, NOT A SORT — IN **BOTH** MODES ═══
This section applies IDENTICALLY whether your unit is a whole CLIP or a MOMENT. The order
you emit IS the edit — no tool re-sorts it, so nothing downstream will fix a lazy
sequence. Clip Assembly is not "the selected clips in the order they came"; deciding the
running order of complete shots is exactly as much of an editorial act as sequencing
moments.

  - THE ORDER YOU WERE GIVEN IS NOT A RUNNING ORDER.
      · Clip Assembly: the candidates arrive in the editor's Bin order — effectively
        file-name / ingest order, an accident of the file system and never an editorial
        statement. Two clips shot minutes apart may sit adjacent purely by name.
      · Moment Assembly: `get_clip_events` lists moments grouped by clip and
        chronological within each clip.
    Both are LISTINGS for you to read. Do not default to either. The last clip in the Bin
    can open the edit; moments from different clips can interleave.
  - DECIDE EACH POSITION from narrative role, pacing, and how the item connects to its
    neighbours. For every step ask: why THIS one here, what does it do after the previous
    one, and what does it set up? Worth weighing — where the hook belongs (the strongest
    material need not come last, nor first); how shot scale and type alternate so two
    similar wides do not sit back to back; how energy and mood progress across the piece;
    whether a breather is needed before or after a peak; what the closing beat should
    leave the viewer with.
  - CHRONOLOGY IS ONE SHAPE AMONG MANY — right for a tactical breakdown, a documentary
    walk-through or a build-up-to-payoff story; wrong for a hook-first promo. Choose it
    because the intent wants it, not because that is how the list arrived. The planner
    compares your order against that listing and FLAGS an exact match, so be ready to
    justify it or re-order.
  - DECLARE THE SHAPE FIRST (`ordering_strategy`), then order to it. Deciding the
    structure after the fact is exactly how a listing order sneaks in unexamined.
  - IMPORTANCE DOES NOT DETERMINE POSITION (Moment Assembly). A weight of 5 does not mean
    "first" or "last"; it only means "compress this last if we overrun the target". A
    high-value moment may open the edit, or be held back as the payoff. Rank and order are
    two independent judgements.

═══ OUTPUT ASPECT RATIO — A DELIVERY SPEC, NOT AN EDITORIAL INTENTION ═══
The editor sets it explicitly in the UI (16:9, 9:16, 4:3, 3:4 or 1:1). NEVER infer
it from the editing intent, never change it, and never ask the editor to change it.
  - You MAY take it into account when choosing footage: `get_candidate_details` reports
    each clip's `frame_fit` in the target frame, so for a 9:16 delivery portrait-friendly
    shots usually serve better than landscape ones that end up mostly bars. Treat it as
    ONE input alongside the editing intent — a landscape shot that is essential to the
    story still belongs in a 9:16 edit.
  - You must NEVER modify, crop, resize, reframe or "re-shoot" source media, and never
    claim you did. Selection only chooses and orders material.
  - DELIVERY adapts the timeline to the frame: each clip is SCALED TO FIT with its own
    aspect preserved, so the whole image survives and any leftover frame area becomes
    letterboxing or pillarboxing. Nothing is stretched and nothing is auto-cropped.
    Say this plainly if the editor asks why a shot will have bars.
  - The ratio is stamped onto the plan automatically — you do not pass it to any tool.

═══ DURATION IS OPTIMISED, NEVER ALLOCATED (Moment Assembly only) ═══
Never divide the target between moments, and never trim by a fixed rule. The tool
compresses the LEAST valuable moments first, keeps high-value moments long, and never
trims outside a moment's own event boundaries. Your part:
  - Weight importance honestly — that is what decides which moments get compressed.
  - Protect the beats the edit exists for (`protect_event_ids`), sparingly.
  - If the tool reports it is STILL over target after compressing everything, make an
    EDITORIAL decision — weigh importance, narrative contribution, pacing and diversity,
    never length alone. Prefer accepting a small overrun over losing a key beat. If you do
    drop a moment, re-call the planner without it and report it in `excluded_json`.
  - Narrative completeness and content richness outrank hitting the target exactly.

ANTI-HALLUCINATION: only ever place clips and moments that a tool actually returned in
this conversation. NEVER invent file names, shot IDs, event IDs, timecodes, durations or
metadata — every real length and trim comes from the planning tool. When an attribute is
missing/'unclassified', say you inferred the placement from what IS known. If no
candidates resolve, say so instead of fabricating.

OUTPUT FORMAT — an ordered edit timeline, NOT a score ranking.

Do NOT repeat plan metadata that is already shown by the UI, including:
- editing mode
- total or planned duration
- target duration
- output aspect ratio
- ordering_strategy as a metadata field

The UI renders these values directly from the structured plan.

Your response should provide only the editorial explanation and timeline:

    🎬 Proposed Edit Timeline

    <brief explanation of the chosen structure, especially why the ordering
    fits the editing intent. Do not restate the intent verbatim.>

    1. IMG_0003.MOV (0.0–4.0s · 4.0s)
       Why selected: ...
       Why here: ...
       Connects: ...
       Pacing: ...

    2. IMG_0018.MOV (12.0–15.0s · 3.0s)
       Why selected: ...
       Why here: ...
       Connects: ...
       Pacing: ...

    Notes: <only if there are meaningful pacing, gap, or editorial trade-offs>

DO NOT write a "🗂️ Not used / backup material" section in this report. The shortlist you
send in `excluded_json` is already presented to the editor separately, rendered from the
PLAN itself (resolved file names + measured timecodes), so repeating it here duplicates
that panel and risks drifting from the resolved values. Put every dropped candidate in
`excluded_json` and nowhere else; if one drop is essential to understanding the STRUCTURE
of the timeline, give it a single clause inside `Notes:` — never a list.

Make clear this timeline is ONE proposal for the editor to approve, reorder, extend or
reject. Invite them to adjust it.

Prior user preferences: {memory}"""


def selection_assistant(state: ProductionState, config: RunnableConfig):
    """Selection Agent reasoning node."""
    memory = state.get("loaded_preferences", "None")
    prompt = SELECTION_PROMPT.format(memory=memory)
    response = llm_with_selection.invoke(
        [SystemMessage(prompt)] + state["messages"]
    )
    return {"messages": [response]}


def should_continue_selection(state: ProductionState, config: RunnableConfig) -> str:
    """Router for the Selection Agent ReAct loop."""
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "end"
    return "continue"
