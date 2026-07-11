"""Selection / Editorial Assistant Agent — edit-timeline orchestration.

The editorial stage of the MAPO pipeline (Ingest → Search → Selection → Delivery). It
does not produce a quality-ranked LIST; it produces an ordered EDIT TIMELINE from the
clips the editor curated in the UI, which the Delivery Agent then compiles verbatim.

Division of labour (design intent):
    - The LLM parses the editing INTENT (video type, emotion, pace, style).
    - It then chooses a timeline STRUCTURE that fits THAT intent — there is no fixed
      template. A highlight reel, a cinematic vlog, a promo montage, a tactical
      breakdown and a travel video can each want a different shape, pacing, and number
      of steps. The agent lays the curated clips into ordered "Timeline Steps",
      explaining for each why it sits there, how it connects to the previous clip, and
      what it does for the pacing.
    - Quality is used ONLY to FILTER (drop clearly weak clips), never to order.

It is an assistant editor: it proposes a timeline and explains it; the editor always
makes the final decision (Human-in-the-Loop). Its input is the user-selected
candidates, not raw Search output.
"""

import json

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from sqlalchemy import text as _sql

from app.models.state import ProductionState
from app.services.openai_service import llm
from app.services.database_service import db, _engine
from app.services.retrieval_service import group_size
from app.services.timeline_service import parse_target_duration, plan_segments
from app.utils.logger import get_logger

logger = get_logger("selection_agent")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_identifiers(identifiers: str) -> list[dict]:
    """Fetch full metadata for candidates named by shot_id OR file path/name.

    ``identifiers`` is a comma-separated list; each entry is either a numeric shot_id
    or a file-path substring (the UI curates by path). Returns catalogue rows as
    dicts — never fabricated.
    """
    tokens = [t.strip() for t in (identifiers or "").split(",") if t.strip()]
    if not tokens:
        return []

    ids = [int(t) for t in tokens if t.isdigit()]
    paths = [t for t in tokens if not t.isdigit()]

    clauses = []
    params: dict = {}
    if ids:
        id_binds = ", ".join(f":id{i}" for i in range(len(ids)))
        clauses.append(f"shot_id IN ({id_binds})")
        params.update({f"id{i}": v for i, v in enumerate(ids)})
    for i, p in enumerate(paths):
        clauses.append(f"file_path LIKE :p{i}")
        params[f"p{i}"] = f"%{p}%"

    if not clauses:
        return []

    query = (
        "SELECT shot_id, file_path, shot_type, "
        "duration_seconds, orientation, fps, keywords, description, "
        "people_count, camera_motion, lighting, mood, subject_position "
        f"FROM shots WHERE {' OR '.join(clauses)} ORDER BY shot_id"
    )
    with _engine.begin() as conn:
        rows = conn.execute(_sql(query), params)
        return [dict(r._mapping) for r in rows]


def _resolve_in_order(identifiers: str) -> list[dict]:
    """Resolve a comma-separated identifier list PRESERVING the given order (and repeats).

    Unlike ``_resolve_identifiers`` (which sorts by shot_id), this keeps the exact
    order the caller supplied — needed for timeline planning, where the sequence IS the
    edit. Each token is a shot_id or a file-path substring; unresolved tokens are
    dropped (the tool reports the count) — never fabricated.
    """
    tokens = [t.strip() for t in (identifiers or "").split(",") if t.strip()]
    ordered: list[dict] = []
    for tok in tokens:
        where, param = ("shot_id = :v", {"v": int(tok)}) if tok.isdigit() \
            else ("file_path LIKE :v", {"v": f"%{tok}%"})
        query = (
            "SELECT shot_id, file_path, shot_type, duration_seconds, fps, "
            "camera_motion, mood, people_count FROM shots "
            f"WHERE {where} ORDER BY shot_id LIMIT 1"
        )
        with _engine.begin() as conn:
            row = conn.execute(_sql(query), param).fetchone()
        if row is not None:
            ordered.append(dict(row._mapping))
    return ordered


# ── Tools: candidate enrichment, timeline planning, delivery ────────────────────


@tool
def get_candidate_details(identifiers: str) -> str:
    """Fetch full metadata for the editor-curated candidates.

    Use this on the clips the user selected in the UI so you can place each in the
    timeline from its real attributes (shot_type, camera_motion, mood, people_count,
    duration, etc.).

    Args:
        identifiers: Comma-separated shot IDs and/or file names/paths
            (e.g. "3, IMG_5231.MOV, /footage/goal.mov").
    """
    rows = _resolve_identifiers(identifiers)
    if not rows:
        return "No matching catalogued clips for those identifiers."
    for r in rows:
        r["group_size"] = group_size(r.get("people_count"))
    return json.dumps(rows, indent=2, default=str)


@tool
def plan_timeline(ordered_identifiers: str, importance: str = None,
                  target_duration_text: str = None,
                  head_trim: float = 0.0, tail_trim: float = 0.0) -> str:
    """Turn an ORDERED clip list into a structured timeline plan (segments for Delivery).

    This is what makes the timeline concrete: it resolves each clip's real duration,
    decides the operating mode, and returns machine-readable SEGMENTS the Delivery Agent
    compiles verbatim. Call it once you have decided the clip ORDER and each clip's
    IMPORTANCE.

    MODE is chosen automatically:
      • TRIM MODE — you pass ``head_trim`` and/or ``tail_trim``. Drops that many seconds
        off the START / END of EVERY clip and keeps the FULL remaining middle; clips are
        assembled sequentially in order, untrimmed otherwise. Use this for a per-clip
        head/tail trim like "trim the first 2s and last 2s of each clip". Takes
        precedence over any target duration.
      • TIMED MODE — no head/tail trim, but a total duration is detected in
        ``target_duration_text`` (e.g. "15s", "1 min", "1:30"). The target length is
        split across the clips PROPORTIONALLY TO IMPORTANCE and each clip is TRIMMED to
        its share, giving a time-coded timeline whose total ≈ the target.
      • FULL CLIP MODE — neither. Clips keep their FULL length, concatenated in order.

    Args:
        ordered_identifiers: Comma-separated shot IDs and/or file names IN TIMELINE ORDER
            (first = timeline step 1). This exact order is preserved — never re-sorted.
        importance: Optional comma-separated importance weights aligned 1:1 with the
            clips (any positive scale, e.g. "3,5,2,4" or "1,1,3"). Higher = more screen
            time in TIMED MODE. Omit for equal weighting.
        target_duration_text: A TOTAL target length for the whole edit ("make it 30s",
            "1:30"). The tool auto-detects the duration. This is NOT for per-clip trims —
            a phrase like "trim the first/last 2 seconds of each clip" is a per-clip trim,
            so use head_trim/tail_trim for it and do NOT rely on this argument.
        head_trim: Seconds to drop from the START of EVERY clip (per-clip trim).
        tail_trim: Seconds to drop from the END of EVERY clip (per-clip trim).

    Returns:
        A readable step list plus a fenced ```json block containing the structured plan
        (mode, target_seconds, head_trim, tail_trim, total_seconds, and per-segment
        in/out points). Report the readable part to the editor; the JSON is consumed
        downstream by Delivery.
    """
    clips = _resolve_in_order(ordered_identifiers)
    if not clips:
        return "No catalogued clips matched those identifiers — nothing to plan."

    weights = None
    if importance:
        parsed = []
        for tok in importance.split(","):
            tok = tok.strip()
            try:
                parsed.append(float(tok))
            except ValueError:
                parsed.append(1.0)
        weights = parsed

    head_trim = max(0.0, float(head_trim or 0.0))
    tail_trim = max(0.0, float(tail_trim or 0.0))
    trimming = head_trim > 0 or tail_trim > 0

    # A per-clip trim is NOT a target length — don't let the intent text's trim wording
    # be mis-read as a target when the editor asked for head/tail trimming.
    target = None if trimming else parse_target_duration(target_duration_text)
    plan = plan_segments(clips, target, weights=weights,
                         head_trim=head_trim, tail_trim=tail_trim)

    mode = plan["mode"]
    if mode == "trim":
        header = (f"🎬 Timeline plan — TRIM MODE (drop first {head_trim:g}s + last "
                  f"{tail_trim:g}s of each clip; {plan['total_seconds']:g}s total)")
    elif mode == "timed":
        header = (f"🎬 Timeline plan — TIMED MODE (target {plan['target_seconds']:g}s, "
                  f"actual {plan['total_seconds']:g}s)")
    else:
        header = f"🎬 Timeline plan — FULL CLIP MODE ({plan['total_seconds']:g}s total, no trimming)"

    lines = [header, ""]
    no_middle = []
    for s in plan["segments"]:
        if mode == "trim":
            lines.append(
                f"  {s['order']}. {s['name']} — keep {s['in_point']:.1f}s–{s['out_point']:.1f}s "
                f"({s['duration']:.1f}s of {s['source_duration']:.1f}s)")
            if s["source_duration"] > 0 and s["duration"] <= 0:
                no_middle.append(s["name"])
        elif mode == "timed":
            lines.append(
                f"  {s['order']}. {s['name']} — {s['in_point']:.1f}s–{s['out_point']:.1f}s "
                f"({s['duration']:.1f}s of {s['source_duration']:.1f}s) · importance {s['importance']:g}")
        else:
            lines.append(f"  {s['order']}. {s['name']} — full {s['duration']:.1f}s")
    if no_middle:
        lines += ["", (f"⚠ {len(no_middle)} clip(s) are shorter than "
                       f"{head_trim + tail_trim:g}s and have no middle left after trimming: "
                       + ", ".join(no_middle))]
    lines += ["", "```json", json.dumps(plan), "```"]
    return "\n".join(lines)


@tool
def generate_delivery_summary(project_id: int = 1) -> str:
    """Generate a text summary of the project for delivery handoff.

    Args:
        project_id: Project identifier.
    """
    project = db.run(
        f"SELECT project_name, client_name, frame_rate, resolution FROM projects WHERE project_id = {project_id};",
        include_columns=True,
    )
    shot_count = db.run(
        f"SELECT COUNT(*) as total FROM shots WHERE project_id = {project_id};",
        include_columns=True,
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
    plan_timeline,
    generate_delivery_summary,
]

llm_with_selection = llm.bind_tools(selection_tools)
selection_tool_node = ToolNode(selection_tools)

SELECTION_PROMPT = """You are the SELECTION AGENT in the MAPO system — an assistant
editor, not an autonomous one. You do NOT rank clips by score. You build an ordered
EDIT TIMELINE from the clips the editor has already curated, and explain it. The editor
always makes the final decision.

INPUT: an editing intent PLUS a set of curated candidate clips (identified by file name
or shot id) that the user selected in the UI. Work ONLY with those clips.

THERE IS NO FIXED TEMPLATE. Do NOT assume every edit follows a traditional
"establishing → buildup → climax → reaction → ending" narrative arc. That is only ONE
possible structure among many — never the default. The timeline's structure, pacing,
and length are decided by the USER'S EDITING INTENT. Different intents want different
shapes, for example (illustrative, not exhaustive):
   - Matchday highlight reel → tension-and-release beats around the key moments
   - Fast-paced football / brand promo → punchy, escalating energy, quick cuts
   - Emotional stadium vlog → personal, atmosphere-led, room to breathe
   - Cinematic travel montage → rhythmic, image-driven, mood over story
   - Tactical analysis → logical/chronological, clarity over drama
   - Player introduction → build recognition, then reveal
Choose the ordering that best serves THIS intent.

WORKFLOW:
1. INTERPRET THE EDITING INTENT — infer video type, emotion, pace, and style. State
   your interpretation, and the timeline structure you'll use for it, so the editor can
   correct you.
2. FETCH DETAILS — call `get_candidate_details` on the curated clips to read their real
   metadata (shot_type, camera_motion, lighting, mood, people_count, duration, ...).
3. DECIDE ORDER + IMPORTANCE — order the clips into whatever sequence best delivers the
   intent, using each clip's real attributes. Use only AS MANY STEPS AS THE EDIT NEEDS —
   3, 7, 12, however many the curated set and goal call for. Do NOT force a fixed count
   and do NOT invent clips. Assign each clip an IMPORTANCE weight (any positive scale,
   higher = more significant to the intent) — the key moments should weigh more.
4. PLAN THE TIMELINE — call `plan_timeline` with the clips IN ORDER and their importance
   weights. It returns the structured timeline and runs in one of three modes, chosen
   automatically from the arguments you pass:
     • TRIM MODE — the intent asks to drop the first/last N seconds of EACH clip (a
       per-clip head/tail trim, keeping the middle). Pass `head_trim` and/or `tail_trim`
       (in seconds). Every clip keeps its FULL remaining middle and clips are assembled
       sequentially in order. DO NOT put that trim wording into `target_duration_text`.
     • TIMED MODE — the intent names a TOTAL length for the whole edit ("15s", "1 min",
       "1:30"): pass the intent text as `target_duration_text`; clips are TRIMMED and
       screen time allocated PROPORTIONALLY to importance, producing a time-coded timeline.
     • FULL CLIP MODE — no total length and no per-clip trim: clips keep their FULL
       duration, concatenated in order, untrimmed.
   CRITICAL — do NOT confuse a PER-CLIP trim ("trim the first/last 2s of each clip" →
   head_trim/tail_trim) with a TOTAL target length ("make the whole edit 30s" →
   target_duration_text). They are different arguments; never route a per-clip trim
   through target_duration_text.
   You do NOT compute the trims yourself — `plan_timeline` does. Report its result and,
   if it trimmed clips, tell the editor which moments were shortened and why.
   Optional free-form step labels ("cold open", "hero moment", "outro") may help the
   editor, but are never a required fixed set.

ANTI-HALLUCINATION: only ever place clips that a tool actually returned in this
conversation. NEVER invent file names, shot IDs, durations, or metadata — the real
lengths and trims come only from `plan_timeline`. When an attribute is
missing/'unclassified', say you inferred the placement from what IS known rather than
guessing the missing field. If no candidates resolve, say so instead of fabricating.

OUTPUT FORMAT — an ordered edit timeline, NOT a score ranking. For EVERY step explain
(a) why the clip sits at this position, (b) how it connects to the previous clip, and
(c) what it does for the overall pacing/rhythm:

    🎬 Proposed Edit Timeline — <your one-line read of the intent + the structure chosen>

    1. IMG_0003.MOV  (0.0–4.0s · 4.0s)     ← show the allotted time in TIMED MODE
       Why here: Opens on crowd atmosphere to set the energy immediately.
       Connects: —  (first clip)
       Pacing: A high-energy cold open; establishes tempo for what follows.
    2. IMG_0018.MOV  (4.0–7.0s · 3.0s)
       Why here: Players entering the pitch gives the viewer a subject to follow.
       Connects: Cuts from the wide crowd to the players — pulls focus inward.
       Pacing: Steadies the rhythm briefly before the action ramps up.
    3. IMG_0042.MOV  (7.0–15.0s · 8.0s)
       Why here: The attacking sequence is the momentum peak of the reel.
       Connects: Escalates directly off the entrance build-up.
       Pacing: Fastest section — drives the edit's energy (weighted most, so most time).

    (…as many steps as the intent needs — no more, no fewer. In FULL CLIP MODE just show
     each clip's full length instead of a trimmed in/out.)

    ---
    Mode: <TRIM (first Xs / last Ys of each clip), TIMED (target Xs), or FULL CLIP> — as
          reported by plan_timeline.
    Notes: <pacing / gaps / alternatives the editor should weigh>

Make clear this timeline is ONE proposal for the editor to approve, reorder, extend, or
reject — not the only correct order. Invite the editor to adjust it.

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
