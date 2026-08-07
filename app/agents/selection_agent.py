"""Selection / Assistant-Editor Agent — edit-timeline orchestration.

The editorial stage of the MAPO pipeline (Ingest → Search → Selection → Delivery). It
does not produce a quality-ranked LIST; it produces an ordered EDIT TIMELINE from the
clips the editor curated in the UI, which the Delivery Agent then compiles verbatim.

There are exactly TWO user-facing editing modes, chosen by the editor in the UI. They
differ only in the UNIT of editing:

    CLIP ASSEMBLY   — combine COMPLETE clips into a coherent timeline. Each clip keeps
                      its original duration; nothing is trimmed and a target duration
                      does not apply. The agent's job is purely editorial: drop the
                      candidates that do not serve the intent, decide the order, and
                      explain it. (vlog, documentary, travel, BTS, atmosphere montage)
    MOMENT ASSEMBLY — build the edit from meaningful MOMENTS inside longer clips. The
                      agent inspects each clip's temporal events, picks the relevant
                      ones, ranks their importance and arranges them. A target duration
                      is OPTIONAL here and is applied as an optimisation over already-
                      chosen moments — never as an allocation that splits time between
                      them.

Division of labour: the LLM does the EDITORIAL reasoning (what belongs, in what order,
what matters most, what to drop and why); ``timeline_service`` does the deterministic
arithmetic (boundaries, compression, validation). The agent is an assistant editor — it
proposes and explains; the editor always makes the final call (Human-in-the-Loop).
"""

import json
from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState
from sqlalchemy import text as _sql

from app.models.state import ProductionState
from app.services.openai_service import llm
from app.services.database_service import (
    db, get_catalogued_events, get_events_by_ids,
)
from app.services.catalogue_resolver import resolve_ordered
from app.services.retrieval_service import group_size
from app.services.timeline_service import (
    MODE_CLIP, MODE_MOMENT, build_clip_timeline, build_moment_timeline,
)
# Output-aspect helpers only — Selection READS the delivery spec to prefer footage that
# suits the frame; adapting the media to it is exclusively Delivery's job.
from app.services.premiere_export_service import describe_fit, normalise_aspect_label
from app.utils.logger import get_logger

logger = get_logger("selection_agent")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _pid_from_state(state) -> int:
    """Current project id from the injected graph state (defaults to 1).

    The MODEL never supplies this — it is injected by the ToolNode from
    ``ProductionState`` (see ``InjectedState`` on the tools), so a tool call can never
    target another project's catalogue (audit C-03). Identifier resolution is then
    scoped to this id by ``catalogue_resolver``.
    """
    try:
        return int((state or {}).get("project_id"))
    except (TypeError, ValueError):
        return 1


def _mode_from_state(state) -> str:
    """The EDITOR's chosen editing mode, injected from state (never model-supplied).

    The mode is a UI decision, so the model cannot switch it by calling the other
    planner: each planning tool checks this and refuses a mismatch. An empty/unknown
    value means "not set by the UI" and leaves both planners open (direct/programmatic
    invocation).
    """
    raw = str((state or {}).get("editing_mode") or "").strip().lower().replace(" ", "_")
    return raw if raw in (MODE_CLIP, MODE_MOMENT) else ""


def _target_from_state(state) -> float | None:
    """The editor's optional Target Duration (seconds), injected from state."""
    try:
        value = float((state or {}).get("target_seconds"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _aspect_from_state(state) -> str:
    """The editor's OUTPUT aspect ratio label, injected from state (``''`` if unset).

    An explicit user input and an output SPECIFICATION — never inferred from the editing
    prompt and never model-supplied, so the agent cannot change the delivery frame. The
    planners stamp it onto the plan so it reaches Delivery unchanged.
    """
    try:
        return normalise_aspect_label((state or {}).get("aspect_ratio"))
    except ValueError:      # InvalidAspectRatio — validated upstream; ignore here
        return ""


def _wrong_mode(state, wanted: str) -> str | None:
    """Refusal message when a planner is called in the other editing mode, else ``None``."""
    mode = _mode_from_state(state)
    if not mode or mode == wanted:
        return None
    other = "Clip Assembly" if mode == MODE_CLIP else "Moment Assembly"
    tool_name = ("plan_clip_assembly" if mode == MODE_CLIP else "plan_moment_assembly")
    return (f"The editor selected {other} mode in the UI, so this planner does not apply. "
            f"Call `{tool_name}` instead — the editing mode is the editor's decision, "
            "not yours.")


def _split(text: str | None, sep: str = ",") -> list[str]:
    return [t.strip() for t in (text or "").split(sep) if t.strip()]


def _floats(text: str | None) -> list[float] | None:
    """Parse a comma-separated weight list; unparseable entries fall back to 1.0."""
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


def _plan_block(lines: list[str], plan: dict) -> str:
    """Append the fenced ```json plan the orchestrator reads back (never model prose)."""
    if plan.get("aspect_ratio"):
        lines += ["", (f"🖼️ Output aspect ratio: {plan['aspect_ratio']} — Delivery scales "
                       "each clip to FIT this frame with its own aspect preserved "
                       "(letterbox/pillarbox as needed). No source media is cropped, "
                       "resized or reframed.")]
    if not plan.get("valid", True):
        bad = plan.get("validation_errors", [])
        lines += ["", f"⛔ {len(bad)} segment(s) are INVALID and will block export:"]
        lines += [f"   • #{e['order']} {e['name']}: {e['error']}" for e in bad]
    excluded = plan.get("excluded") or []
    if excluded:
        lines += ["", f"🗂️ Alternative / backup material ({len(excluded)} not in the edit):"]
        for x in excluded:
            note = x["reason"] or "no reason given"
            if x["suggested_use"]:
                note += f" · could be used for: {x['suggested_use']}"
            lines.append(f"   • {x['name'] or x['ref']}: {note}")
    lines += ["", "```json", json.dumps(plan), "```"]
    return "\n".join(lines)


# ── Tools: candidate enrichment, timeline planning, delivery ────────────────────


@tool
def get_candidate_details(identifiers: str,
                          state: Annotated[dict, InjectedState] = None) -> str:
    """Fetch full metadata for the editor-curated candidates.

    Use this on the clips the user selected in the UI so you can judge — from real
    attributes (shot_type, camera_motion, mood, people_count, duration, ...) — which of
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
def plan_clip_assembly(ordered_identifiers: str, labels: str = None,
                       excluded_json: str = None,
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
            (first = timeline step 1). This exact order is preserved — never re-sorted.
        labels: Optional PIPE-separated step labels aligned 1:1 with the clips
            (e.g. "cold open|the arrival|wide breather|closing beat").
        excluded_json: The candidates you deliberately left OUT, as a JSON list of
            ``{"ref": "<file name or shot id>", "reason": "why it is not in the edit",
            "suggested_use": "how it could still be used"}``. Never drop a candidate
            silently — everything you exclude must be reported here as backup material.

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

    plan = build_clip_timeline(clips, labels=_split(labels, "|") or None,
                               excluded=excluded_json,
                               aspect_ratio=_aspect_from_state(state))

    lines = [f"🎞️ Timeline plan — CLIP ASSEMBLY ({len(plan['segments'])} complete clip(s), "
             f"{plan['total_seconds']:g}s total · no trimming)", ""]
    for s in plan["segments"]:
        length = (f"full {s['duration']:.1f}s" if s["source_duration"] > 0
                  else "full clip (length not measured)")
        lines.append(f"  {s['order']}. {s['name']} — {length}"
                     + (f" · {s['label']}" if s.get("label") else ""))
    return _plan_block(lines, plan)


@tool
def plan_moment_assembly(ordered_event_ids: str, importance: str = None,
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
            This exact order is preserved — never re-sorted. Each id must belong to the
            current project.
        importance: Comma-separated editorial weights aligned 1:1 with the moments (any
            positive scale, e.g. "5,2,4,1" — higher = more valuable to the intent, so
            compressed later and less). Omit for equal weighting.
        target_seconds: Optional target TOTAL length in seconds. Pass the editor's Target
            Duration if they set one; omit it when they did not (never invent one).
        labels: Optional PIPE-separated step labels aligned 1:1 with the moments
            (defaults to each event's own action text).
        protect_event_ids: Comma-separated event IDs that must NEVER be trimmed (the beats
            the edit exists for). Use sparingly.
        focus_json: Optional JSON object mapping event_id → the absolute timecode of that
            moment's PEAK (e.g. ``{"42": 18.5}``). Any compression keeps the window around
            that point instead of the moment's centre. Only use timecodes inside the event.
        excluded_json: Moments you considered but left OUT, as a JSON list of
            ``{"ref": "<event id or clip name>", "reason": "why it is not in the edit",
            "suggested_use": "how it could still be used"}``.

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

    plan = build_moment_timeline(
        ordered_events, target_seconds=target, importance=_floats(importance),
        labels=_split(labels, "|") or None, focus=focus, protect=protect,
        excluded=excluded_json, aspect_ratio=_aspect_from_state(state),
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
    project = db.run(
        _sql("SELECT project_name, client_name, frame_rate, resolution "
             "FROM projects WHERE project_id = :pid"),
        parameters={"pid": project_id}, include_columns=True,
    )
    shot_count = db.run(
        _sql("SELECT COUNT(*) as total FROM shots WHERE project_id = :pid"),
        parameters={"pid": project_id}, include_columns=True,
    )

    return f"""Delivery Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Project: {project}
Total Shots: {shot_count}
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
does not serve the intent, LEAVE IT OUT and report it as alternative/backup material
(what it is, why it is not in the edit, and how it could still be used). Work ONLY within
the curated set — never add a clip that was not given to you.

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
  Workflow:
    1. Read the intent — video type, emotion, pace, style. State your interpretation.
    2. `get_candidate_details` on the curated clips to read their real metadata.
    3. Decide which clips BELONG (drop the irrelevant ones) and in what ORDER.
    4. `plan_clip_assembly` with the kept clips IN ORDER, optional step labels, and every
       dropped candidate in `excluded_json` with a reason and a suggested alternative use.

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
    4. RANK them: give each an importance weight reflecting narrative contribution, not
       length. Order them into the timeline the intent calls for.
    5. `plan_moment_assembly` with the event_ids IN ORDER, the importance weights, the
       target duration if the editor set one, and any moments you rejected in
       `excluded_json`.

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

OUTPUT FORMAT — an ordered edit timeline, NOT a score ranking. For EVERY step explain
(a) why this material is in the edit, (b) why it sits at THIS position and how it connects
to the surrounding segments, and (c) what it does for the pacing/rhythm:

    🎬 Proposed Edit Timeline — <one-line read of the intent + the structure you chose>
    Mode: <CLIP ASSEMBLY | MOMENT ASSEMBLY> · <target/length as the planner reported it>
    Output frame: <the aspect ratio, and — only if it influenced your picks — one line on
                   how, e.g. "9:16: favoured the portrait shots; the two landscape clips
                   are essential to the story and will be letterboxed">


    1. IMG_0003.MOV  (0.0–4.0s · 4.0s)
       Why selected: Crowd atmosphere is the energy the intent asks to open on.
       Why here: Sets the tempo immediately; nothing needs to precede it.
       Connects: — (first segment)
       Pacing: High-energy cold open, establishes the rhythm.
    2. IMG_0018.MOV  (12.0–15.0s · 3.0s)
       Why selected: Gives the viewer a subject to follow after the wide opener.
       Why here: Pulls focus inward from the crowd to the players.
       Connects: Cuts from wide atmosphere to a human subject.
       Pacing: Steadies briefly before the action ramps up.

    (…as many steps as the intent needs — no more, no fewer.)

    ---
    Not used (backup material): <each excluded candidate — why it is out, and where it
    could still work>
    Notes: <pacing, gaps, duration trade-offs, alternatives the editor should weigh>

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
