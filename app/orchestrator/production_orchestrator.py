"""Production Orchestrator — explicit four-stage MAPO pipeline."""

import hashlib
import json
import re
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, ToolMessage

from app.models.state import ProductionState

from app.agents.ingest_agent import (
    ingest_assistant, ingest_tool_node, should_continue_ingest,
    reset_last_ingest_result, get_last_ingest_result,
)
from app.agents.search_agent import (
    search_assistant, search_tool_node, should_continue_search,
)
from app.agents.selection_agent import (
    selection_assistant, selection_tool_node, should_continue_selection,
)
from app.agents.delivery_agent import (
    compile_plan, reset_last_delivery_result, get_last_delivery_result,
)
from app.services.retrieval_service import (hybrid_search, expand_query_terms,
                                            hoist_orientation)
from app.services.timeline_service import MODE_CLIP, MODE_MOMENT
from app.services.premiere_export_service import normalise_aspect_label
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("orchestrator")

# ── Memory ───────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()
in_memory_store = InMemoryStore()


# ── Build Sub-Agent Graphs ───────────────────────────────────────────────────


def _build_agent_graph(name, assistant_fn, tool_node, should_continue_fn):
    """Helper: build a ReAct sub-agent graph (assistant → tools → assistant)."""
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


ingest_agent = _build_agent_graph("ingest", ingest_assistant, ingest_tool_node, should_continue_ingest)
search_agent = _build_agent_graph("search", search_assistant, search_tool_node, should_continue_search)
selection_agent = _build_agent_graph("selection", selection_assistant, selection_tool_node, should_continue_selection)

# Delivery is deterministic and does not require an agent graph.
logger.info("3 sub-agent graphs compiled + deterministic delivery compiler "
            "(fixed-pipeline orchestration, no supervisor).")


# ── Shared state / config helpers ─────────────────────────────────────────────

def _base_state(project_id, extra: dict | None = None) -> dict:
    """Create isolated initial state for one pipeline stage."""
    state: dict = {
        "project_id": str(project_id),
        "messages": [],
        "loaded_preferences": "",
        "footage_dir": "",
        "ingested_files": [], "shot_metadata": [], "search_results": [],
        "search_candidates": [], "selected_candidates": [], "selected_shots": [],
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
    try:
        return int(project_id)
    except (TypeError, ValueError):
        return project_id


def _qhash(query: str) -> str:
    """Return a short digest used to identify a search invocation."""
    return hashlib.md5((query or "").encode("utf-8")).hexdigest()[:8]


# ── Structured output extraction ─────────────────────────────────────────────

_PLAN_TOOLS = (None, "plan_clip_assembly", "plan_moment_assembly")


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
    return None


_SEARCH_TOOLS = (None, "search_catalogue", "list_all_shots")


def _extract_candidates(messages) -> list[dict] | None:
    """Extract the latest structured candidate list from a search tool output."""
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) not in _SEARCH_TOOLS:
            continue
        content = m.content if isinstance(m.content, str) else ""
        for match in re.findall(r"```json\s*(\[.*?\])\s*```", content, re.DOTALL):
            try:
                obj = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list):
                return obj
    return None


# ── Pipeline stages ─────────────────────────────────────────────────────────────


def run_ingest(directory: str, project_id, user_id) -> tuple[str, object]:
    """① Ingest — analyse footage and build the project catalogue."""
    reset_last_ingest_result()
    msg = (f"Run ingest analysis on the footage directory '{directory}' for project "
           f"{project_id}. Call the ingest_footage tool now to build the catalogue.")
    state = _base_state(project_id, {
        "messages": [HumanMessage(content=msg)],
        "footage_dir": str(directory),
    })
    result = ingest_agent.invoke(state, config=_config("ingest", user_id, project_id))
    return result["messages"][-1].content, get_last_ingest_result()


def _search_direct(query: str, project_id) -> list[dict]:
    """Run deterministic hybrid retrieval for a search query."""
    residual, orientation = hoist_orientation(query, None)
    core, related = (expand_query_terms(residual)
                     if residual and residual.strip() else ("", ""))
    keywords = ", ".join(t for t in (core, related) if t)
    return hybrid_search(_pid(project_id), keywords=keywords or None,
                         core_keywords=core or None, orientation=orientation)


def run_search(query: str, project_id, user_id="editor") -> list[dict]:
    """② Search — retrieve candidate clips using the search agent or fallback."""
    if settings.OPENAI_API_KEY:
        try:
            config = _config("search", user_id, project_id, _qhash(query))
            state = _base_state(project_id, {"messages": [HumanMessage(content=query)]})
            result = search_agent.invoke(state, config=config)
            candidates = _extract_candidates(result["messages"])
            if candidates is not None:
                return candidates
            logger.warning("Search agent produced no structured result; using direct search.")
        except Exception as e:  # keep the UI responsive on any agent/LLM failure
            logger.warning(f"Search agent failed ({e}); using direct search.")
    return _search_direct(query, project_id)


def _normalise_editing_mode(mode) -> str:
    """Convert a UI editing mode to its canonical identifier."""
    raw = str(mode or "").strip().lower().replace(" ", "_")
    return MODE_MOMENT if raw == MODE_MOMENT else MODE_CLIP


def run_selection(intent: str, selected_paths: list[str], project_id, user_id,
                  editing_mode: str = MODE_CLIP,
                  target_seconds: float | None = None,
                  aspect_ratio: str = "") -> tuple[str, dict | None]:
    """③ Selection — create an ordered editorial timeline from curated clips."""
    mode = _normalise_editing_mode(editing_mode)
    aspect = normalise_aspect_label(aspect_ratio)   # raises on an unusable value
    target = None
    if mode == MODE_MOMENT:
        try:
            target = float(target_seconds) if target_seconds else None
        except (TypeError, ValueError):
            target = None
        if target is not None and target <= 0:
            target = None

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
        "selected_candidates": list(selected_paths),
        "editing_mode": mode,
        "target_seconds": target,
        "aspect_ratio": aspect,
    })
    result = selection_agent.invoke(state, config=_config("select", user_id, project_id))
    return result["messages"][-1].content, _extract_plan(result["messages"])


def run_delivery(plan: dict, project_id, user_id,
                 sequence_name: str = "MAPO Edit"):
    """④ Delivery — compile the Selection plan into a Premiere project."""
    if not (plan and plan.get("segments")):
        raise ValueError(
            "Delivery requires a structured timeline plan (ordered segments) from "
            "Selection. Generate the edit timeline in ③ Selection first — there is no "
            "implicit media-pool-order fallback.")

    reset_last_delivery_result()
    summary = compile_plan(plan, sequence_name=sequence_name, project_id=_pid(project_id))
    return summary, get_last_delivery_result()


logger.info("MAPO pipeline orchestrator ready (Ingest -> Search -> Selection -> Delivery).")
