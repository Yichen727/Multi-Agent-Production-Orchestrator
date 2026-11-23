"""Shared LangGraph state for the MAPO pipeline."""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps

from app.models.schemas import DeliveryResult, EditPlan, SearchCandidate


class ProductionState(TypedDict):
    """Shared state and typed data contract between pipeline stages."""

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
