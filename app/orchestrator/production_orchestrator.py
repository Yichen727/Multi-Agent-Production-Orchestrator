"""Production Orchestrator — explicit four-stage MAPO pipeline."""

import json
import re
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, ToolMessage

from app.models.state import ProductionState

from app.agents.selection_agent import (
    selection_assistant, selection_tool_node, should_continue_selection,
)
from app.services.ingest_service import ingest_footage
from app.services.delivery_service import compile_plan
from app.services.search_service import search_clips
from app.services.database_service import get_catalogued_clips
from app.services.contracts import project_id_value, target_duration
from app.services.plan_contract import validate_edit_plan
from app.services.timeline_service import MODE_CLIP, MODE_MOMENT
from app.services.premiere_export_service import normalise_aspect_label
from app.utils.logger import get_logger
from app.services.metrics_service import measure_stage

logger = get_logger("orchestrator")

# ── Memory ───────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()
in_memory_store = InMemoryStore()


# ── Build Sub-Agent Graphs ───────────────────────────────────────────────────


def _build_agent_graph(name, assistant_fn, tool_node, should_continue_fn):
    """Build the Selection reasoning loop (assistant -> tools -> assistant)."""
    graph = StateGraph(ProductionState)
    graph.add_node(f"{name}_assistant", assistant_fn)
    graph.add_node(f"{name}_tools", tool_node)
    graph.add_edge(START, f"{name}_assistant")
    graph.add_conditional_edges(
        f"{name}_assistant",
        should_continue_fn,
        {"continue": f"{name}_tools", "end": END},
    )
    graph.add_edge(f"{name}_tools", f"{name}_assistant")
    return graph.compile(
        name=f"{name}_agent",
        checkpointer=checkpointer,
        store=in_memory_store,
    )


selection_agent = _build_agent_graph("selection", selection_assistant, selection_tool_node, should_continue_selection)

# Delivery is deterministic and does not require an agent graph.
logger.info("Selection graph ready; ingestion, search, and delivery use direct services.")


# ── Shared state / config helpers ─────────────────────────────────────────────

def _base_state(project_id, extra: dict | None = None) -> dict:
    """Create isolated initial state for one pipeline stage."""
    state: dict = {
        "project_id": str(project_id),
        "messages": [],
        "loaded_preferences": "",
        "footage_dir": "",
        "ingested_files": [], "clip_metadata": [], "search_results": [],
        "search_candidates": [], "selected_candidates": [], "selected_clips": [],
        "editing_mode": "", "target_seconds": None, "aspect_ratio": "",
        "recommendations": [], "edit_timeline": None, "delivery_output": None,
    }
    if extra:
        state.update(extra)
    return state


def _config(stage: str, user_id, project_id, tag: str | None = None) -> dict:
    """Create a fresh checkpoint thread for one stage invocation."""
    parts = [p for p in ("mapo", stage, str(user_id), str(project_id), tag,
                         uuid.uuid4().hex[:12]) if p]
    return {"configurable": {
        "thread_id": "-".join(parts),
        "user_id": user_id,
    }}


def _pid(project_id) -> int:
    return project_id_value(project_id)


# ── Structured output extraction ─────────────────────────────────────────────

_PLAN_TOOLS = ("plan_clip_assembly", "plan_moment_assembly")


def _extract_plan(messages) -> dict | None:
    """Extract the latest structured timeline plan from a planning tool output."""
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) not in _PLAN_TOOLS:
            continue
        content = m.content if isinstance(m.content, str) else ""
        for match in re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL):
            try:
                obj = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "segments" in obj:
                return obj
        # A refused latest planning call must not resurrect an earlier plan.
        return None
    return None


# ── Pipeline stages ─────────────────────────────────────────────────────────────


@measure_stage("ingest")
def run_ingest(directory: str, project_id, user_id, **options) -> tuple[str, object]:
    """Index footage directly; result belongs to this run."""
    result = ingest_footage(directory, _pid(project_id), **options)
    return result.message, result


@measure_stage("search")
def run_search(query: str, project_id, user_id="editor") -> list[dict]:
    """Parse once and retrieve directly; no result narration call."""
    return search_clips(query, _pid(project_id))


def _normalise_editing_mode(mode) -> str:
    """Convert a UI editing mode to its canonical identifier."""
    raw = str(mode or "").strip().lower().replace(" ", "_")
    if raw not in (MODE_CLIP, MODE_MOMENT):
        raise ValueError("Unknown editing mode")
    return raw


@measure_stage("selection")
def run_selection(intent: str, selected_paths: list[str], project_id, user_id,
                  editing_mode: str = MODE_CLIP,
                  target_seconds: float | None = None,
                  aspect_ratio: str = "") -> tuple[str, dict | None]:
    """③ Selection — create an ordered editorial timeline from curated clips."""
    mode = _normalise_editing_mode(editing_mode)
    aspect = normalise_aspect_label(aspect_ratio)   # raises on an unusable value
    project_id = _pid(project_id)
    target = target_duration(target_seconds) if mode == MODE_MOMENT else None
    from app.services.media_identity import canonical_path
    catalogue = {canonical_path(p): c for p, c in get_catalogued_clips(project_id).items()}
    selected_clips = []
    for path in selected_paths:
        row = catalogue.get(canonical_path(path))
        if not row or not row["online"]:
            raise ValueError("Every curated candidate must be an online clip in this project")
        selected_clips.append(row)
    if not selected_clips:
        raise ValueError("Select at least one curated candidate")

    clip_list = "\n".join(f"- {p}" for p in selected_paths)
    aspect_line = (
        f"OUTPUT ASPECT RATIO: {aspect} — this is my delivery SPEC, not an editorial "
        "instruction. You may prefer footage that suits this frame, but you must never "
        "crop, resize or reframe any media; Delivery scales each clip to FIT the frame "
        "with its aspect preserved (letterbox/pillarbox where needed).\n\n"
    ) if aspect else ""
    common = (
        f"My editing intent: {intent}\n\n"
        f"These are the CANDIDATE clips I ticked in the UI (work ONLY within this set, but "
        f"do NOT assume all of them belong in the edit):\n{clip_list}\n\n"
        + aspect_line
    )
    if mode == MODE_CLIP:
        message = common + (
            "EDITING MODE: CLIP ASSEMBLY — the unit of editing is the WHOLE CLIP. Every "
            "clip you keep stays at its original duration; there is no trimming and no "
            "target duration in this mode, so ignore any length wording in my intent.\n\n"
            "The list above is my Bin order (roughly file-name order) — it is NOT a "
            "running order and carries no editorial meaning. Re-order freely: with no "
            "trimming available, WHICH clips and IN WHAT ORDER are the only two levers, "
            "so the sequencing carries the whole edit.\n\n"
            "Call get_candidate_details on the candidates, decide from my intent which "
            "clips genuinely belong, then choose the SHAPE the edit should take and state "
            "it in one line. Call plan_clip_assembly with the kept clips IN THAT ORDER, "
            "that line as ordering_strategy, and every dropped candidate in excluded_json "
            "(with why it is out and how it could still be used). Structure, pacing and "
            "the number of steps come from my intent — do NOT assume a fixed narrative "
            "arc. For each step explain why the clip is in the edit, why it sits there, "
            "how it connects to its neighbours, and what it does for the pacing."
        )
    else:
        target_line = (f"TARGET DURATION: {target:g} seconds."
                       if target else "TARGET DURATION: N/A — I did not set one.")
        message = common + (
            "EDITING MODE: MOMENT ASSEMBLY — the unit of editing is a MOMENT (temporal "
            f"event) inside a clip.\n{target_line}\n\n"
            "Call get_clip_events on ALL the candidate clips to see the moments available "
            "with their real timecodes — that listing is chronological per clip and is "
            "NOT a running order. SELECT FIRST, OPTIMISE SECOND: choose enough meaningful "
            "moments to satisfy my intent before worrying about the clock. Then choose the "
            "SHAPE the edit should take and state it in one line, ORDER the moments into "
            "it, and RANK them separately by narrative contribution (importance decides "
            "what gets compressed, never where a moment sits). Call "
            "plan_moment_assembly with the event_ids IN ORDER, that line as "
            "ordering_strategy, the importance weights"
            + (", the target duration" if target else "")
            + ", and any moments you rejected in excluded_json. If the planner reports it "
            "is still over target after compressing, make an editorial decision rather "
            "than cutting by length — a small overrun beats losing a key beat. For each "
            "step explain why the moment is in the edit, why it sits there, how it "
            "connects to its neighbours, and what it does for the pacing."
        )

    state = _base_state(project_id, {
        "messages": [HumanMessage(content=message)],
        "selected_candidates": [c["file_path"] for c in selected_clips],
        "selected_clips": selected_clips,
        "editing_mode": mode,
        "target_seconds": target,
        "aspect_ratio": aspect,
    })
    result = selection_agent.invoke(state, config=_config("select", user_id, project_id))
    plan = _extract_plan(result["messages"])
    if plan is not None:
        plan = validate_edit_plan(plan, project_id)
        if plan["mode"] != mode or plan.get("aspect_ratio") != aspect or plan.get("target_seconds") != target:
            raise ValueError("Plan does not match the editor-selected specification")
        allowed_ids = {c["clip_id"] for c in selected_clips}
        if any(seg["clip_id"] not in allowed_ids for seg in plan["segments"]):
            raise ValueError("Plan contains an unselected clip")
    return result["messages"][-1].content, plan


@measure_stage("delivery")
def run_delivery(plan: dict, project_id, user_id,
                 sequence_name: str = "MAPO Edit"):
    """④ Delivery — compile the Selection plan into a Premiere project."""
    if not (plan and plan.get("segments")):
        raise ValueError(
            "Delivery requires a structured timeline plan (ordered segments) from "
            "Selection. Generate the edit timeline in ③ Selection first — there is no "
            "implicit media-pool-order fallback.")

    result = compile_plan(plan, sequence_name=sequence_name, project_id=_pid(project_id))
    return result.message, result


logger.info("MAPO pipeline orchestrator ready (Ingest -> Search -> Selection -> Delivery).")
