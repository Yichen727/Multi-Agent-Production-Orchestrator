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

import hashlib
import json
import re
import uuid

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


def _config(stage: str, user_id, project_id, tag: str | None = None) -> dict:
    """A per-INVOCATION runnable config: a FRESH thread id plus the user id.

    Every stage call gets its own checkpointer thread. Each stage invocation is already
    self-contained — ``_base_state`` seeds ``messages`` with the one HumanMessage carrying
    everything the agent needs — so there is nothing to gain from replaying an earlier
    run's history, and two concrete things to lose:

      * A CORRUPTED history is replayed forever. When a tool call raises (the parallel
        ToolNode threads hitting the shared-SQLite race, for example), the checkpoint keeps
        the assistant message whose ``tool_calls`` never got their ToolMessages. Re-running
        the stage on that same thread re-sends that history and OpenAI rejects the whole
        request: "An assistant message with 'tool_calls' must be followed by tool messages
        responding to each 'tool_call_id'". The stage then stays broken until the process
        restarts, even though the original bug is gone.
      * History grows without bound. Re-running Selection with a new intent used to append
        to the previous run's transcript — stale timelines in context, and the token bill
        for them.

    ``tag`` is an optional human-readable discriminator (e.g. a query digest) kept in the
    thread id for log/debug traceability only; it never makes two invocations share a
    thread.
    """
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
    """A stable short digest of a query, used only to LABEL that search's agent thread.

    Every stage invocation already gets its own thread (see ``_config``), so this is a
    debugging aid — it makes a search's checkpoint thread identifiable in the logs — not
    the isolation mechanism."""
    return hashlib.md5((query or "").encode("utf-8")).hexdigest()[:8]


# Timeline-planning tools whose ToolMessage carries the structured plan Delivery consumes.
# Both the clip-level planner and the event/moment-precise planner emit the same fenced
# ```json plan shape, so the extractor accepts either (audit H-03).
_PLAN_TOOLS = (None, "plan_timeline", "plan_moment_timeline")


def _extract_plan(messages) -> dict | None:
    """Pull the structured timeline plan from a planning TOOL output.

    Reads the plan out of the plan_timeline / plan_moment_timeline ToolMessage — the
    tool's own output, which the model cannot rewrite — rather than scraping the
    assistant's prose. So the model reformatting its narration can never corrupt or hide
    the plan Delivery receives (audit H-03). Returns the most recent structured plan, or
    ``None`` if none was produced.
    """
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


# Search tools whose ToolMessage carries the structured candidate list the UI renders.
_SEARCH_TOOLS = (None, "search_catalogue", "list_all_shots")


def _extract_candidates(messages) -> list[dict] | None:
    """Pull the structured candidate list from the search TOOL output.

    Mirrors ``_extract_plan``: reads the fenced ```json array the ``search_catalogue`` /
    ``list_all_shots`` ToolMessage carries (the tool's OWN output, which the model cannot
    rewrite), so the model reformatting its narration can never corrupt or hide the
    candidates the UI renders. Returns the most recent list (possibly empty, when the
    search genuinely matched nothing), or ``None`` when the agent never ran a search tool —
    in which case ``run_search`` falls back to deterministic direct retrieval.
    """
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


def _search_direct(query: str, project_id) -> list[dict]:
    """Deterministic retrieval — query understanding (orientation hoist + synonym
    expansion) feeding ``hybrid_search`` directly, with no LLM agent in the loop.

    This is the fallback the Search stage uses when the Search Agent is unavailable
    (no API key), errors, or completes a turn without actually searching — so the UI
    always receives candidates.
    """
    residual, orientation = hoist_orientation(query, None)
    expanded = expand_query(residual) if residual and residual.strip() else None
    return hybrid_search(_pid(project_id), keywords=expanded or None,
                         orientation=orientation)


def run_search(query: str, project_id, user_id="editor") -> list[dict]:
    """② Search — invoke the Search sub-agent to retrieve candidate clips.

    The Search Agent translates the natural-language query into a structured
    ``search_catalogue`` call; the structured candidate list is read back from that tool's
    ToolMessage (via ``_extract_candidates``), NOT from the model's prose, so the UI
    receives the exact ``hybrid_search`` rows it renders. Retrieval stays CLIP-level: the
    returned unit is always a whole clip (choosing a moment within it is Selection's job).
    Recall is event-aware inside ``hybrid_search`` — a clip surfaces when a moment inside
    it matches, attached as a ``matched_event`` hint — but it never returns a moment and
    never ranks a "best" clip.

    Degrades gracefully (audit-consistent with the rest of the system): with no API key,
    an agent error, or an agent turn that never searched, it falls back to the
    deterministic ``_search_direct`` path so the UI always gets results.
    """
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


def run_selection(intent: str, selected_paths: list[str],
                  project_id, user_id) -> tuple[str, dict | None]:
    """③ Selection — invoke the Selection sub-agent on the curated clips.

    Returns ``(narration, structured_plan | None)``. The structured plan is read from the
    planning tool's output (audit H-03), never from the model's prose.

    The agent DECIDES for itself, from the editing intent, whether to build a whole-clip
    timeline (``plan_timeline`` — head/tail/timed/full) or a MOMENT-PRECISE one
    (``get_clip_events`` → ``plan_moment_timeline``, trimming to exact event boundaries).
    An intent that targets specific moments ("a 30s celebration reel") steers it to the
    event path; there is no separate UI switch.
    """
    clip_list = "\n".join(f"- {p}" for p in selected_paths)
    message = (
        f"My editing intent: {intent}\n\n"
        f"The editor has selected these clips for the edit (work ONLY with these):\n{clip_list}\n\n"
        "Fetch their details and decide the order and each clip's importance. Then plan "
        "the timeline: if my intent targets specific MOMENTS (a particular action/beat), "
        "inspect the clips' temporal events with get_clip_events and call "
        "plan_moment_timeline with the chosen event_ids in order (trims to the exact "
        "moment); otherwise call plan_timeline (passing my editing-intent text so it can "
        "detect any target duration). The timeline's structure, pacing and number of steps "
        "are driven by my intent — do NOT assume a fixed narrative arc. For each step "
        "explain why the clip sits there, how it connects to the previous clip, and what "
        "it does for the pacing."
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
