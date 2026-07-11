"""LangGraph shared state definition for the MAPO multi-agent system.

The system is a strict, user-driven linear pipeline:

    Ingest → Search → Selection

Ingest builds the catalogue; Search retrieves candidates; Selection ranks and
explains. There is no automated quality scoring — quality is the editor's call,
informed by the Selection Agent's reasoning.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps


class ProductionState(TypedDict):
    """Central state shared across all agents.

    Fields:
        project_id: Current production project identifier.
        messages: Conversation history (user, agent, tool messages).
        loaded_preferences: User/project preferences from long-term memory.
        ingested_files: List of file paths processed by the Ingest Agent.
        shot_metadata: Per-shot metadata catalogued by the Ingest Agent.
        search_results: Candidate clips retrieved by the Search Agent (retrieval only).
        search_candidates: Structured candidate dicts from hybrid_search, shown in the
            UI curation layer for the editor to check/uncheck.
        selected_candidates: File paths the editor curated in the UI — the explicit
            input to the Selection Agent (it works only on these).
        selected_shots: Clips the editor has chosen to keep (Human-in-the-Loop).
        recommendations: Intent-aware, ranked clip recommendations from the Selection
            Agent, each paired with an explanation of why it was recommended.
        edit_timeline: Ordered edit timeline produced by the Selection Agent. Its
            structure and length are driven by the editing intent (no fixed arc);
            steps may carry optional free-form labels.
        delivery_output: Paths + summary of the Premiere-importable project files
            (FCP7 XML + JSON intermediate) compiled by the Delivery Agent from the
            edit timeline. Compile-only — no creative decision is recorded here.
        remaining_steps: Managed — prevents infinite ReAct loops.

    New fields should be added additively to stay backward-compatible.
    """
    project_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_preferences: str
    ingested_files: list
    shot_metadata: list
    search_results: list
    search_candidates: list
    selected_candidates: list
    selected_shots: list
    recommendations: list
    edit_timeline: list
    delivery_output: list
    remaining_steps: RemainingSteps
