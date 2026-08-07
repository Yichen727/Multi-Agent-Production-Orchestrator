"""LangGraph shared state definition for the MAPO multi-agent system.

The system is a strict, user-driven linear pipeline with a UI Human-in-the-Loop
curation layer between retrieval and editing:

    Ingest → Search → Curation (UI HITL) → Selection → Delivery

- Ingest builds the searchable catalogue (the knowledge base).
- Search retrieves candidate clips (hybrid recall, retrieval only — never "best").
- Curation is the editor checking which candidates participate (the Streamlit Bin).
- Selection orchestrates the curated clips into an intent-driven edit timeline
  (no fixed narrative arc; there is NO automated quality scoring — quality is the
  editor's call, informed by the Selection Agent's reasoning).
- Delivery compiles that timeline into a Premiere-importable project file.

``ProductionState`` is the single object threaded through the whole graph and is the
TYPED data contract between stages (audit H-05): each stage reads the previous stage's
field and writes its own, rather than re-parsing natural-language ``messages``. The
Pydantic shapes live in :mod:`app.models.schemas`. New fields must be added additively
to stay backward-compatible.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps

from app.models.schemas import DeliveryResult, EditPlan, SearchCandidate


class ProductionState(TypedDict):
    """Central state shared across all agents (the inter-stage data contract).

    Fields:
        project_id: Current production project identifier.
        messages: Conversation history (user, agent, tool messages).
        loaded_preferences: User/project preferences from long-term memory.
        footage_dir: The footage directory for THIS run, carried on state instead of a
            mutated global setting, so concurrent sessions can't clobber each other's
            directory (audit H-07). The Ingest tools read it when no explicit directory
            argument is given.
        ingested_files: File paths processed by the Ingest stage.
        shot_metadata: Per-clip catalogue rows written by the Ingest stage.
        search_results: Candidate clips retrieved by the Search stage (retrieval only).
        search_candidates: Structured ``SearchCandidate`` items from ``hybrid_search``,
            shown in the UI curation layer for the editor to check/uncheck.
        selected_candidates: File paths the editor curated in the UI — the explicit
            input to the Selection stage (it works only on these).
        selected_shots: Clips the editor has chosen to keep (Human-in-the-Loop).
        editing_mode: The editor's Selection mode — ``'clip_assembly'`` (whole clips, no
            trimming, no duration control) or ``'moment_assembly'`` (moments inside clips,
            optional target duration). It is a UI decision, injected into the Selection
            tools from state so the model cannot switch modes on its own.
        target_seconds: The editor's optional Target Duration in seconds. Applies to
            MOMENT ASSEMBLY only (``None`` = N/A, the default); Clip Assembly ignores it.
        aspect_ratio: The editor's OUTPUT aspect ratio label ("16:9", "9:16", "1:1" or a
            "4:3" or "3:4"; ``""`` = none requested). An explicit user input
            and an output SPECIFICATION — never inferred from the editing prompt. Selection
            may let it influence WHICH footage it picks but never modifies media; the label
            travels on the plan so Delivery can scale each clip to fit that frame.
        recommendations: (Legacy) intent-aware clip notes from the Selection stage; the
            editor always makes the final decision — nothing here is auto-selected.
        edit_timeline: The Selection stage's structured ``EditPlan`` (mode + ordered
            segments). Structure and length are intent-driven (no fixed arc). This is the
            object the Delivery stage compiles verbatim.
        delivery_output: The Delivery stage's structured ``DeliveryResult`` (written FCP7
            XML/JSON artefact paths + sequence shape). Compile-only — no creative call.
        remaining_steps: Managed — prevents infinite ReAct loops.

    New fields should be added additively to stay backward-compatible.
    """
    project_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_preferences: str
    footage_dir: str
    ingested_files: list[str]
    shot_metadata: list[dict]
    search_results: list[SearchCandidate]
    search_candidates: list[SearchCandidate]
    selected_candidates: list[str]
    selected_shots: list[dict]
    editing_mode: str
    target_seconds: Optional[float]
    aspect_ratio: str
    recommendations: list[dict]
    edit_timeline: Optional[EditPlan]
    delivery_output: Optional[DeliveryResult]
    remaining_steps: RemainingSteps
