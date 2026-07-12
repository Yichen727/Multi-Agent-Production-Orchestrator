"""Production Orchestrator — the explicit four-stage pipeline driver.

This module IS the orchestration layer (audit H-06). MAPO is a strict, user-driven
linear pipeline and the orchestrator drives it as a fixed state machine — there is no
LLM "supervisor" routing between agents. Each stage is an explicit function that invokes
exactly one specialised ReAct sub-agent (or, for Search, the retrieval service directly):

    ① run_ingest    → ingest_agent      (build the catalogue)
    ② run_search    → retrieval_service (hybrid recall — retrieval only)
    ③ run_selection → selection_agent   (intent-driven edit timeline)
    ④ run_delivery  → delivery_agent     (compile the timeline → Premiere FCP7 XML)

The Streamlit UI is a thin presentation layer that calls these functions in order; the
code path, the architecture diagram, and the thesis description are therefore the same
(no "documented supervisor, actually bypassed by the UI" discrepancy).

HUMAN-IN-THE-LOOP (audit C-09): there is ONE HITL mechanism — the editor's direct
control in the UI. The **curation checkboxes** decide which candidate clips participate,
and the explicit **Export** button is what triggers Delivery. There is no separate
LangGraph ``interrupt()`` approval gate (the old one was dead code — it fired on state
fields no agent wrote, and the UI never resumed it), so it has been removed rather than
left as a misleading no-op.
"""

import json
import re

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, ToolMessage

from app.models.state import ProductionState

# The four agents of the linear pipeline:
#   Ingest    = scan, classify, catalogue footage (build the knowledge base)
#   Search    = retrieval-only candidate search
#   Selection = intent-aware editorial orchestration (edit timeline)
#   Delivery  = compile the timeline into a Premiere-importable project (FCP7 XML)
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
    delivery_assistant, delivery_tool_node, should_continue_delivery,
)
from app.services.retrieval_service import hybrid_search, expand_query, hoist_orientation
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
delivery_agent = _build_agent_graph("delivery", delivery_assistant, delivery_tool_node, should_continue_delivery)

logger.info("All 4 sub-agent graphs compiled (fixed-pipeline orchestration, no supervisor).")


# ASCII-only log lines below: the Windows console codec (gbk) cannot encode emoji/arrows.


# ── Shared state / config helpers ─────────────────────────────────────────────


def _base_state(project_id, extra: dict | None = None) -> dict:
    """A fresh ProductionState for one stage invocation.

    project_id and footage_dir are carried ON state (not via a mutated global), so
    concurrent sessions stay isolated (audit H-07).
    """
    state: dict = {
        "project_id": str(project_id),
        "messages": [],
        "loaded_preferences": "",
        "footage_dir": "",
        "ingested_files": [], "shot_metadata": [], "search_results": [],
        "search_candidates": [], "selected_candidates": [], "selected_shots": [],
        "recommendations": [], "edit_timeline": None, "delivery_output": None,
    }
    if extra:
        state.update(extra)
    return state


def _config(stage: str, user_id, project_id) -> dict:
    """A per-stage runnable config with a stable thread id and the user id."""
    return {"configurable": {
        "thread_id": f"mapo-{stage}-{user_id}-{project_id}",
        "user_id": user_id,
    }}


def _pid(project_id) -> int:
    try:
        return int(project_id)
    except (TypeError, ValueError):
        return project_id


def _extract_plan(messages) -> dict | None:
    """Pull the structured timeline plan from the ``plan_timeline`` TOOL output.

    Reads the plan out of the plan_timeline ToolMessage — the tool's own output, which
    the model cannot rewrite — rather than scraping the assistant's prose. So the model
    reformatting its narration can never corrupt or hide the plan Delivery receives
    (audit H-03). Returns the most recent structured plan, or ``None`` if none was produced.
    """
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        if getattr(m, "name", None) not in (None, "plan_timeline"):
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


# ── The four pipeline stages (the ONE real orchestration path) ─────────────────


def run_ingest(directory: str, project_id, user_id) -> tuple[str, object]:
    """① Ingest — invoke the Ingest sub-agent to build the catalogue.

    The footage directory is passed on state (``footage_dir``), never by mutating a
    global setting (audit H-07). Returns ``(narration, IngestResult | None)``; the
    caller unlocks later stages ONLY on a real structured success (audit C-08).
    """
    reset_last_ingest_result()
    msg = (f"Run ingest analysis on the footage directory '{directory}' for project "
           f"{project_id}. Call the ingest_footage tool now to build the catalogue.")
    state = _base_state(project_id, {
        "messages": [HumanMessage(content=msg)],
        "footage_dir": str(directory),
    })
    result = ingest_agent.invoke(state, config=_config("ingest", user_id, project_id))
    return result["messages"][-1].content, get_last_ingest_result()


def run_search(query: str, project_id) -> list[dict]:
    """② Search — hybrid retrieval (retrieval only), the same path the Search agent uses.

    Runs the query through the shared query understanding (orientation hoist + synonym
    expansion) and ``hybrid_search`` so the UI's direct search and the Search Agent's
    tool behave identically. Returns candidate dicts; never ranks a "best" clip.
    """
    residual, orientation = hoist_orientation(query, None)
    expanded = expand_query(residual) if residual and residual.strip() else None
    return hybrid_search(_pid(project_id), keywords=expanded or None,
                         orientation=orientation)


def run_selection(intent: str, selected_paths: list[str],
                  project_id, user_id) -> tuple[str, dict | None]:
    """③ Selection — invoke the Selection sub-agent on the curated clips.

    Returns ``(narration, structured_plan | None)``. The structured plan is read from the
    plan_timeline tool output (audit H-03), never from the model's prose.
    """
    clip_list = "\n".join(f"- {p}" for p in selected_paths)
    message = (
        f"My editing intent: {intent}\n\n"
        f"The editor has selected these clips for the edit (work ONLY with these):\n{clip_list}\n\n"
        "Fetch their details, decide the order and each clip's importance, then call "
        "plan_timeline (passing my editing-intent text so it can detect any target "
        "duration). The timeline's structure, pacing and number of steps are driven by my "
        "intent — do NOT assume a fixed narrative arc. For each step explain why the clip "
        "sits there, how it connects to the previous clip, and what it does for the pacing."
    )
    state = _base_state(project_id, {
        "messages": [HumanMessage(content=message)],
        "selected_candidates": list(selected_paths),
    })
    result = selection_agent.invoke(state, config=_config("select", user_id, project_id))
    return result["messages"][-1].content, _extract_plan(result["messages"])


def run_delivery(plan: dict, project_id, user_id,
                 sequence_name: str = "MAPO Edit") -> str:
    """④ Delivery — compile the Selection timeline into a Premiere project.

    REQUIRES the structured plan from Selection (audit H-04): Delivery is driven ONLY by
    the explicit ordered segments the Selection Agent produced. There is NO fallback to
    the Bin/media-pool order — without a structured plan there is no defined edit order
    to compile, so this raises rather than invent one.
    """
    if not (plan and plan.get("segments")):
        raise ValueError(
            "Delivery requires a structured timeline plan (ordered segments) from "
            "Selection. Generate the edit timeline in ③ Selection first — there is no "
            "implicit media-pool-order fallback.")
    message = (
        f"Compile the timeline into a Premiere Pro project named '{sequence_name}'.\n\n"
        "The Selection Agent produced these STRUCTURED timeline segments. Call "
        "compile_timeline_segments with segments_json set to EXACTLY this JSON — do not "
        "re-order, add, drop, or re-time any segment:\n\n"
        f"{json.dumps(plan)}"
    )
    state = _base_state(project_id, {
        "messages": [HumanMessage(content=message)],
        "edit_timeline": plan,
    })
    result = delivery_agent.invoke(state, config=_config("deliver", user_id, project_id))
    return result["messages"][-1].content


logger.info("MAPO pipeline orchestrator ready (Ingest -> Search -> Selection -> Delivery).")
