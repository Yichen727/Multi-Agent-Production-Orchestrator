"""Delivery Agent — compile the edit timeline into a Premiere Pro project file.

The FINAL stage of the MAPO pipeline (Ingest → Search → Selection → **Delivery**).

It takes the ordered edit timeline the Selection Agent produced — laid out in whatever
structure fit the editor's intent (no fixed narrative arc) — and compiles it, in that
exact order, into a Premiere Pro–importable project: an **FCP7 XML** (``xmeml`` v5)
document that Adobe Premiere Pro imports natively (File ▸ Import), plus a neutral JSON
intermediate.

STRICT ROLE — this agent is a PROJECT COMPILER, not an editor:
    - It NEVER re-orders, ranks, or drops clips. The order it is given IS the timeline
      order (establishing → buildup → climax → reaction → ending, as laid out upstream).
    - It NEVER fabricates media. Every clip must resolve to a real catalogued file; each
      is referenced by its ABSOLUTE path. If any identifier does not resolve, it refuses
      rather than invent a filename.
    - Time is mapped from real durations: unless explicit in/out points are supplied, the
      full clip is used and clips are laid end-to-end (a straight assembly).
    - Track layout: V1 = video, A1 = original/ambient audio, A2 = secondary audio only
      when a file genuinely carries a second audio stream.

The heavy lifting (XML/JSON generation) lives in
``app.services.premiere_export_service``; the tools here just resolve the ordered clips
against the catalogue and hand them to the compiler.
"""

import json
from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState

from app.models.state import ProductionState
from app.services.openai_service import llm
from app.services.catalogue_resolver import (
    resolve_one, resolve_ordered, AmbiguousIdentifier,
)
from app.services.premiere_export_service import build_timeline, compile_project
from app.services.ffmpeg_service import count_audio_streams
from app.utils.logger import get_logger

logger = get_logger("delivery_agent")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _pid_from_state(state) -> int:
    """Current project id from the injected graph state (defaults to 1).

    The model never supplies this — the ToolNode injects it from ``ProductionState`` —
    so Delivery resolves media, and names its output file, ONLY within the current
    project (audit C-04). Cross-project or ambiguous identifiers cannot be compiled.
    """
    try:
        return int((state or {}).get("project_id"))
    except (TypeError, ValueError):
        return 1


def _to_compiler_clips(rows: list[dict], roles: list[str] | None,
                       probe_audio: bool) -> list[dict]:
    """Shape resolved catalogue rows into the compiler's clip dicts (real data only)."""
    clips = []
    for i, r in enumerate(rows):
        streams = None
        if probe_audio and r.get("file_path"):
            n = count_audio_streams(r["file_path"])
            # Trust the probe only when it finds something; otherwise fall back to the
            # catalogue's has_audio flag (never invent a second stream).
            streams = n if n > 0 else (1 if r.get("has_audio") else 0)
        clips.append({
            "file_path": r["file_path"],
            "shot_id": r.get("shot_id"),
            "duration_seconds": r.get("duration_seconds") or 0.0,
            "fps": r.get("fps") or None,
            "width": r.get("width") or None,
            "height": r.get("height") or None,
            "has_audio": bool(r.get("has_audio")),
            "audio_streams": streams,
            "audio_channels": r.get("audio_channels") or None,
            "audio_sample_rate": r.get("audio_sample_rate") or None,
            "audio_bit_depth": r.get("audio_bit_depth") or None,
            "role": (roles[i] if roles and i < len(roles) else ""),
        })
    return clips


def _parse_roles(roles: str | None, count: int) -> list[str] | None:
    if not roles:
        return None
    parsed = [r.strip() for r in roles.split(",")]
    return parsed[:count]


# ── Tools ──────────────────────────────────────────────────────────────────────


@tool
def preview_delivery_timeline(ordered_identifiers: str, roles: str = None,
                              sequence_name: str = "MAPO Edit",
                              state: Annotated[dict, InjectedState] = None) -> str:
    """Dry-run the timeline compile: resolve the ordered clips and show the layout.

    Use this BEFORE compiling to confirm every clip resolves and to see the computed
    in/out points, sequence timestamps, and track assignment — WITHOUT writing a file.
    Order is preserved exactly as given. Resolution is scoped to the current project.

    Args:
        ordered_identifiers: Comma-separated shot IDs and/or file names/paths IN
            TIMELINE ORDER (first = first clip on the timeline). This must be the exact
            order the Selection Agent laid out — do NOT re-sort.
        roles: Optional comma-separated free-form timeline-step labels aligned 1:1 with
            the clips (e.g. "cold open, hero moment, outro" — whatever the Selection
            Agent used, if anything). Purely descriptive; never reorders.
        sequence_name: Name for the sequence.

    Returns:
        A grounded, numbered preview of the timeline, or a clear error listing any
        identifiers that did not resolve (or that were ambiguous).
    """
    project_id = _pid_from_state(state)
    tokens = [t.strip() for t in (ordered_identifiers or "").split(",") if t.strip()]
    rows, problems = resolve_ordered(project_id, tokens)
    if problems:
        return ("Cannot build the timeline — resolve these identifiers first (no media "
                "will be fabricated):\n"
                + "\n".join(f"  • {t}: {r}" for t, r in problems))
    if not rows:
        return "No clips provided. Give the ordered clip list from the Selection Agent."

    role_list = _parse_roles(roles, len(rows))
    clips = _to_compiler_clips(rows, role_list, probe_audio=False)
    timeline = build_timeline(clips, sequence_name=sequence_name)

    seq = timeline["sequence"]
    lines = [
        f"🎬 Timeline preview — {seq['name']}",
        f"   {seq['clip_count']} clips · {seq['total_seconds']:.1f}s "
        f"· {seq['timebase']} fps · {seq['width']}x{seq['height']}",
        "",
    ]
    for c in timeline["clips"]:
        role = f"[{c['role']}] " if c["role"] else ""
        audio = ("A1+A2" if c["audio_streams"] >= 2
                 else "A1" if c["audio_streams"] == 1 else "no audio")
        lines.append(
            f"  {c['order']}. {role}{c['name']}  "
            f"@ {c['seq_start_seconds']:.1f}s–{c['seq_end_seconds']:.1f}s  "
            f"(in {c['in_seconds']:.1f}s / out {c['out_seconds']:.1f}s)  "
            f"· V1 + {audio}"
        )
    lines.append("")
    lines.append("Order is exactly as provided. Call compile_premiere_project to write the file.")
    return "\n".join(lines)


@tool
def compile_premiere_project(ordered_identifiers: str, roles: str = None,
                             sequence_name: str = "MAPO Edit",
                             state: Annotated[dict, InjectedState] = None) -> str:
    """Compile the ordered timeline into a Premiere-importable FCP7 XML (+ JSON).

    Resolves each identifier to its real catalogued clip, maps time sequentially from
    the measured durations, assigns tracks (V1 video, A1 original audio, A2 only when a
    clip has a genuine second audio stream), and writes an ``xmeml`` v5 XML that Adobe
    Premiere Pro imports natively — plus a JSON intermediate. Order is PRESERVED exactly;
    no clip is re-ranked or dropped. Media resolution and the output filename are both
    scoped to the current project.

    Args:
        ordered_identifiers: Comma-separated shot IDs and/or file names/paths IN
            TIMELINE ORDER (first = first clip). Use the exact order from the Selection
            Agent's timeline — never re-sort.
        roles: Optional comma-separated free-form timeline-step labels aligned 1:1 with
            the clips (recorded as clip comments; descriptive only).
        sequence_name: Name for the Premiere sequence.

    Returns:
        The paths to the written .xml (Premiere import) and .json files plus a summary,
        or a clear error naming any identifiers that did not resolve (or were ambiguous).
    """
    project_id = _pid_from_state(state)
    tokens = [t.strip() for t in (ordered_identifiers or "").split(",") if t.strip()]
    rows, problems = resolve_ordered(project_id, tokens)
    if problems:
        return ("Refusing to compile — resolve these identifiers first (every media "
                "reference must be a real catalogued clip):\n"
                + "\n".join(f"  • {t}: {r}" for t, r in problems))
    if not rows:
        return "No clips provided. Supply the ordered clip list from the Selection Agent."

    role_list = _parse_roles(roles, len(rows))
    clips = _to_compiler_clips(rows, role_list, probe_audio=True)

    try:
        result = compile_project(clips, sequence_name=sequence_name,
                                 project_id=project_id, write=True)
    except Exception as e:  # keep the ReAct loop alive with a grounded error
        logger.error(f"Compile failed: {e}")
        return f"Compile failed: {e}"

    seq = result["timeline"]["sequence"]
    order_summary = " → ".join(
        f"{c['order']}.{(c['role'] + ':') if c['role'] else ''}{c['name']}"
        for c in result["timeline"]["clips"]
    )
    return (
        "✅ Premiere project compiled (FCP7 XML — import via File ▸ Import in Premiere Pro).\n"
        f"  XML : {result['xml_path']}\n"
        f"  JSON: {result['json_path']}\n"
        f"  Sequence '{seq['name']}': {seq['clip_count']} clips · "
        f"{seq['total_seconds']:.1f}s · {seq['timebase']} fps · {seq['width']}x{seq['height']}\n"
        f"  Timeline order (preserved): {order_summary}"
    )


@tool
def compile_timeline_segments(segments_json: str, sequence_name: str = "MAPO Edit",
                              state: Annotated[dict, InjectedState] = None) -> str:
    """Compile STRUCTURED timeline segments (from the Selection plan) into Premiere FCP7 XML.

    This is the preferred delivery path: it consumes the Selection Agent's
    `plan_timeline` output directly, so trims (in/out points) and order are honoured
    exactly. Each segment already carries its file, in_point, out_point and optional
    label — this tool resolves the real media, applies those trims, and writes the XML
    (+ JSON). It NEVER re-orders, drops, or re-times a segment. Media resolution and the
    output filename are scoped to the current project.

    Args:
        segments_json: JSON — either the full plan object from `plan_timeline`
            (``{"mode":..., "segments":[...]}``) or a bare list of segment objects. Each
            segment needs ``file_path`` or ``shot_id``; optional ``in_point`` /
            ``out_point`` (seconds) and ``label``.
        sequence_name: Name for the Premiere sequence.

    Returns:
        The written .xml / .json paths + a summary, or a clear error naming any segment
        whose media did not resolve, was ambiguous, or was flagged invalid upstream.
    """
    project_id = _pid_from_state(state)
    try:
        data = json.loads(segments_json)
    except (json.JSONDecodeError, TypeError) as e:
        return f"Could not parse segments_json: {e}. Pass the plan_timeline JSON verbatim."
    segments = data.get("segments") if isinstance(data, dict) else data
    if not segments:
        return "No segments to compile. Provide the plan_timeline JSON."

    # C-06: refuse any segment the planner flagged invalid (real footage but nothing left
    # after trimming). Never silently emit or repair it.
    invalid = [s for s in segments if s.get("valid") is False]
    if invalid:
        listing = "\n".join(
            f"  • #{s.get('order', '?')} {s.get('name') or s.get('file_path')}: "
            f"{s.get('validation_error') or 'invalid segment range'}" for s in invalid)
        return ("Refusing to compile — the plan contains invalid segment(s); fix the trim "
                f"or drop the clip in Selection first:\n{listing}")

    clips, unresolved = [], []
    for i, seg in enumerate(segments, start=1):
        ident = seg.get("shot_id")
        ident = str(ident) if ident is not None else (seg.get("file_path") or "")
        try:
            row = resolve_one(project_id, ident)
        except AmbiguousIdentifier as e:
            unresolved.append(str(e))
            continue
        if row is None:
            unresolved.append(seg.get("file_path") or seg.get("shot_id") or f"segment #{i}")
            continue
        streams = count_audio_streams(row["file_path"])
        streams = streams if streams > 0 else (1 if row.get("has_audio") else 0)
        clips.append({
            "file_path": row["file_path"],
            "shot_id": row.get("shot_id"),
            "duration_seconds": row.get("duration_seconds") or 0.0,
            "fps": row.get("fps") or None,
            "width": row.get("width") or None,
            "height": row.get("height") or None,
            "has_audio": bool(row.get("has_audio")),
            "audio_streams": streams,
            "audio_channels": row.get("audio_channels") or None,
            "audio_sample_rate": row.get("audio_sample_rate") or None,
            "audio_bit_depth": row.get("audio_bit_depth") or None,
            "role": seg.get("label") or "",
            # Honour the plan's trims. out_point None → compiler uses the full clip.
            "in_point": seg.get("in_point"),
            "out_point": seg.get("out_point"),
        })

    if unresolved:
        return ("Refusing to compile — these segments matched no catalogued clip: "
                f"{unresolved}. Every media reference must be real.")

    try:
        result = compile_project(clips, sequence_name=sequence_name,
                                 project_id=project_id, write=True)
    except Exception as e:
        logger.error(f"Segment compile failed: {e}")
        return f"Compile failed: {e}"

    seq = result["timeline"]["sequence"]
    mode = data.get("mode", "full") if isinstance(data, dict) else "full"
    order_summary = " → ".join(
        f"{c['order']}.{(c['role'] + ':') if c['role'] else ''}{c['name']} "
        f"({c['used_seconds']:.1f}s)"
        for c in result["timeline"]["clips"]
    )
    return (
        f"✅ Premiere project compiled from {mode.upper()} timeline segments "
        "(FCP7 XML — import via File ▸ Import in Premiere Pro).\n"
        f"  XML : {result['xml_path']}\n"
        f"  JSON: {result['json_path']}\n"
        f"  Sequence '{seq['name']}': {seq['clip_count']} segments · "
        f"{seq['total_seconds']:.1f}s · {seq['timebase']} fps · {seq['width']}x{seq['height']}\n"
        f"  Segments (order + trim preserved): {order_summary}"
    )


# ── Agent Assembly ───────────────────────────────────────────────────────────

delivery_tools = [
    preview_delivery_timeline,
    compile_premiere_project,
    compile_timeline_segments,
]

llm_with_delivery = llm.bind_tools(delivery_tools)
delivery_tool_node = ToolNode(delivery_tools)

DELIVERY_PROMPT = """You are the DELIVERY AGENT in the MAPO system — the FINAL stage
(Ingest → Search → Selection → Delivery). You are a PROJECT COMPILER, not an editor.

INPUT: an ordered edit timeline from the Selection Agent — a sequence of clips (by file
name or shot id) laid out as ordered timeline steps. The structure is whatever fit the
editor's intent (it is NOT a fixed narrative arc), and each step may or may not carry a
free-form label. Your job is to turn that exact sequence into a Premiere Pro–importable
project file, in exactly the order given.

HARD RULES:
1. PRESERVE ORDER EXACTLY. The order the Selection Agent gives you IS the timeline order.
   NEVER re-rank, re-sort, or drop a clip. Pass the clips to the tools in that same order.
2. NEVER FABRICATE. Only ever compile clips that resolve to a real catalogued file. If
   an identifier does not resolve, the tool will tell you — report it and ask the editor
   to fix the list. Do NOT invent file names, durations, or timings.
3. You do NOT make creative choices (no trimming, no reordering, no quality judgement).
   Time mapping uses the full measured clip duration laid end-to-end unless explicit
   in/out points are provided.

WORKFLOW:
- PREFERRED — if the Selection Agent provided STRUCTURED timeline segments (a
  `plan_timeline` JSON object, with per-segment in/out points), call
  `compile_timeline_segments` with that JSON verbatim. This honours any trims (TIMED
  MODE) and the exact order. This is the right path whenever segments are available.
- FALLBACK — if you only have a plain ordered clip list (no segments/trims), call
  `preview_delivery_timeline` to confirm every clip resolves, then
  `compile_premiere_project` with the SAME ordered list.
- Either way: report the output paths back to the editor. If any clip/segment fails to
  resolve, STOP and report exactly which identifiers were bad — never substitute or
  invent a clip.

TRACK LAYOUT (handled by the compiler; explain it to the editor): V1 = main video,
A1 = original/ambient audio, A2 = secondary audio only when a clip truly has a second
audio stream.

OUTPUT: confirm the sequence was compiled, give the .xml path (for File ▸ Import in
Premiere Pro) and the .json path, and restate the preserved clip order.

Prior user preferences: {memory}"""


def delivery_assistant(state: ProductionState, config: RunnableConfig):
    """Delivery Agent reasoning node."""
    memory = state.get("loaded_preferences", "None")
    prompt = DELIVERY_PROMPT.format(memory=memory)
    response = llm_with_delivery.invoke(
        [SystemMessage(prompt)] + state["messages"]
    )
    return {"messages": [response]}


def should_continue_delivery(state: ProductionState, config: RunnableConfig) -> str:
    """Router for the Delivery Agent ReAct loop."""
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "end"
    return "continue"
