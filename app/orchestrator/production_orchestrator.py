"""Production Orchestrator — Supervisor + Human-in-the-Loop gate.

Assembles the four agents of the linear MAPO pipeline into a unified LangGraph:

    Ingest → Search → Selection → Delivery

- Project verification gate (HITL)
- Supervisor routing to the specialised agents (the editor drives the order via
  the UI; Search/Selection/Delivery stay locked until Ingest completes)
- Approval gate (HITL) before editorial recommendations are finalised
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt
from langgraph_supervisor import create_supervisor
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.models.state import ProductionState
from app.services.openai_service import llm

# The four agents of the linear pipeline:
#   Ingest    = scan, classify, catalogue footage (build the knowledge base)
#   Search    = retrieval-only candidate search
#   Selection = intent-aware editorial orchestration (edit timeline)
#   Delivery  = compile the timeline into a Premiere-importable project (FCP7 XML)
from app.agents.ingest_agent import (
    ingest_assistant, ingest_tool_node, should_continue_ingest,
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

from app.utils.logger import get_logger

logger = get_logger("orchestrator")

# ── Memory ───────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()
in_memory_store = InMemoryStore()


# ── Build Sub-Agent Graphs ───────────────────────────────────────────────────


def _build_agent_graph(name, assistant_fn, tool_node, should_continue_fn):
    """Helper: build a ReAct sub-agent graph."""
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

logger.info("All 4 sub-agent graphs compiled.")


# ── Supervisor ───────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are the MAPO ORCHESTRATOR SUPERVISOR, the central
coordination agent for the Multi-Agent Production Orchestrator system.

You route each production task to the appropriate specialised agent. You do NOT
perform tasks yourself — you coordinate the team. The system is an assistant editor,
never an autonomous one; the human stays in control.

THE PIPELINE IS A STRICT LINEAR FLOW:

    1. ingest_agent — Build the knowledge base. Scan the footage directory, extract
       metadata, classify shots, and store the catalogue (SQLite + JSON).
       Route for: "run ingest", "index footage", "scan / catalogue the footage".

    2. search_agent — RETRIEVAL ONLY. Finds candidate clips matching a natural-
       language query and returns them with metadata and an optional relevance
       score. Does NOT rank or recommend a "best" clip.
       Route for: "find / show me / which clips have ..." style queries.

    3. selection_agent — Intent-aware EDITORIAL orchestration. Interprets editing
       intent (style, emotion, pace, duration), assigns each curated clip a narrative
       role, and lays out an ordered edit timeline.
       Route for: "recommend / best / build the timeline / what should I use for ..."

    4. delivery_agent — PROJECT COMPILER. Takes the ordered edit timeline and compiles
       it into a Premiere Pro–importable project file (FCP7 XML + JSON). It preserves
       clip order exactly and makes NO creative decision.
       Route for: "export / deliver / render the timeline / make a Premiere project".

ROUTING RULES:
- Search, Selection and Delivery only make sense AFTER ingest has built the catalogue.
- For a creative request, prefer routing search_agent FIRST to gather candidates,
  THEN selection_agent to build the timeline from them — never let search_agent recommend.
- Route delivery_agent LAST, only once an ordered timeline exists; it just compiles.
- Pass full conversation context so agents build on each other's work.
- After the relevant agent completes, summarise the result concisely."""

supervisor = create_supervisor(
    agents=[ingest_agent, search_agent, selection_agent, delivery_agent],
    model=llm,
    prompt=SUPERVISOR_PROMPT,
    state_schema=ProductionState,
    output_mode="last_message",
).compile()

logger.info("Supervisor initialized with 4 sub-agents.")


# ── Human-in-the-Loop Gates ─────────────────────────────────────────────────


def verify_project(state: ProductionState, config: RunnableConfig):
    """HITL Gate 1: Verify project context before agents begin."""
    if not state.get("project_id"):
        interrupt(HumanMessage(
            content="Please provide the project ID or project name to begin."
        ))

    project_id = state.get("project_id", "1")
    return {
        "project_id": project_id,
        "messages": [SystemMessage(content=f"Project verified: ID {project_id}. Starting orchestration.")],
    }


def human_approval_gate(state: ProductionState, config: RunnableConfig):
    """HITL Gate 2: Pause for human review of editorial recommendations."""

    messages = state.get("messages", [])
    summary = (
        "\n".join(str(m.content) for m in messages[-3:])
        if messages
        else "No recommendations."
    )

    approval_request = HumanMessage(
        content=(
            "🔍 REVIEW REQUIRED\n\n"
            f"Recent recommendations:\n{summary}\n\n"
            "Please approve (type 'approved') or request changes."
        )
    )

    interrupt(approval_request)

    return {
        "messages": [approval_request]
    }


# ── Assemble Full Graph ──────────────────────────────────────────────────────

mapo_graph = StateGraph(ProductionState)

mapo_graph.add_node("verify_project", verify_project)
mapo_graph.add_node("supervisor", supervisor)
mapo_graph.add_node("human_approval_gate", human_approval_gate)

mapo_graph.add_edge(START, "verify_project")
mapo_graph.add_edge("verify_project", "supervisor")


# Pause for human review whenever the pipeline has produced an editorial decision
# the editor must own: a selection or a ranked recommendation.
def _needs_approval(state: ProductionState, config: RunnableConfig) -> str:
    if state.get("selected_shots") or state.get("recommendations"):
        return "approval"
    return "end"


mapo_graph.add_conditional_edges(
    "supervisor",
    _needs_approval,
    {"approval": "human_approval_gate", "end": END},
)
mapo_graph.add_edge("human_approval_gate", END)

# Compile
mapo_agent = mapo_graph.compile(
    name="MAPO_Full_System",
    checkpointer=checkpointer,
    store=in_memory_store,
)

logger.info("✅ Full MAPO system compiled successfully.")
