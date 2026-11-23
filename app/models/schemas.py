"""Pydantic schemas for structured agent I/O."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional


class ShotMetadata(BaseModel):
    """Metadata for a single shot extracted by the Ingest Agent."""
    shot_id: int
    file_path: str
    shot_type: str  # wide_shot, close_up, establishing, medium_shot, etc.
    duration_seconds: float
    objects_detected: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class VisionTags(BaseModel):
    """Visual metadata extracted from clip frames."""
    scene_description: str = Field(
        default="",
        description="One factual sentence describing what is happening in this shot.",
    )
    shot_type: str = Field(
        default="unknown",
        description="Closest of: close_up, medium_shot, wide_shot, establishing, "
                    "over_shoulder, pov, aerial, or 'unknown'.",
    )
    setting: str = Field(
        default="unknown",
        description="'interior', 'exterior', or 'unknown'.",
    )
    objects: list[str] = Field(
        default_factory=list,
        description="Salient objects or subjects visibly present in the frames.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Short lowercase searchable tags (subjects, actions, mood, "
                    "environment). Include common synonyms where natural.",
    )
    people_count: int = Field(
        default=0,
        description="Approximate number of distinct people visible (0 if none). "
                    "Do NOT identify who they are.",
    )
    mood: str = Field(
        default="unknown",
        description="Emotional tone conveyed by the imagery: one of 'calm', "
                    "'energetic', 'tense', 'cinematic', or 'unknown' if unclear.",
    )


class EventTags(BaseModel):
    """Action-oriented metadata for a temporal event."""
    action: str = Field(
        default="",
        description="One factual verb-led sentence describing what HAPPENS across the "
                    "frames in order (e.g. 'a person walks in from the left and sits "
                    "down'). Describe motion/action, not a static caption.",
    )
    subjects: list[str] = Field(
        default_factory=list,
        description="The entities that DO or UNDERGO the action (people, vehicles, "
                    "objects). Not every visible object — only the ones involved in what "
                    "happens.",
    )
    state_change: str = Field(
        default="",
        description="What is DIFFERENT between the first and last frame (position, "
                    "presence, framing, activity). Empty if nothing changes / a single "
                    "held moment.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Short lowercase searchable action/event tags an editor would use to "
                    "find this MOMENT (verbs and event nouns: 'kick-off', 'celebration', "
                    "'entrance', 'handshake'). No invented proper nouns.",
    )


class ClipEvent(BaseModel):
    """Persisted temporal event with source-clip boundaries."""
    model_config = ConfigDict(extra="ignore")

    event_id: Optional[int] = None
    shot_id: Optional[int] = None
    file_path: str
    event_order: int = 0
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    duration_seconds: float = 0.0
    action: str = ""
    state_change: str = ""
    subjects: list[str] = Field(default_factory=list)
    keywords: str = ""
    relevance: Optional[float] = None


class TransitionSuggestion(BaseModel):
    """Transition suggestion between two shots."""
    from_shot_id: int
    to_shot_id: int
    transition_type: str  # cut, dissolve, wipe, fade
    duration_frames: int
    confidence: float


class ProductionReport(BaseModel):
    """Final production report metadata."""
    project_name: str
    total_shots: int
    total_scenes: int
    best_takes: list[int]
    report_path: str


class EditingIntent(BaseModel):
    """Structured interpretation of the user's creative editing prompt."""
    video_type: str = ""   
    pace: str = ""         
    emotion: str = ""      
    style: str = ""        
    notes: str = ""


class ClipRecommendation(BaseModel):
    """Explainable clip recommendation from the Selection Agent."""
    shot_id: int
    file_path: str
    rank: int
    score: float = Field(ge=0, le=100)
    explanation: str  


class TimelineEntry(BaseModel):
    """One ordered clip in an edit timeline."""
    order: int 
    label: str = ""  
    file_path: str
    shot_id: Optional[int] = None
    rationale: str  


class EditTimeline(BaseModel):
    """The Selection Agent's output: an ordered edit structure."""
    intent: EditingIntent = Field(default_factory=EditingIntent)
    entries: list[TimelineEntry] = Field(default_factory=list)
    notes: str = ""


class SearchCandidate(BaseModel):
    """One candidate returned by hybrid retrieval."""
    model_config = ConfigDict(extra="ignore")

    shot_id: Optional[int] = None
    file_path: str
    shot_type: Optional[str] = None
    duration_seconds: Optional[float] = None
    orientation: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    keywords: Optional[str] = None
    description: Optional[str] = None
    people_count: Optional[int] = None
    mood: Optional[str] = None
    relevance: Optional[float] = None
    group_size: str = "unknown"
    suggestion: str = "neutral"


class TimelineSegment(BaseModel):
    """One ordered segment in the Selection timeline."""
    order: int
    shot_id: Optional[int] = None
    event_id: Optional[int] = None
    file_path: str
    name: str = ""
    label: str = ""
    importance: float = 1.0
    source_duration: float = 0.0
    event_start: Optional[float] = None
    event_end: Optional[float] = None
    event_duration: Optional[float] = None
    in_point: Optional[float] = 0.0
    out_point: Optional[float] = 0.0
    duration: float = 0.0
    trimmed: bool = False
    trim_note: str = ""
    protected: bool = False
    valid: bool = True
    validation_error: Optional[str] = None


class ExcludedMaterial(BaseModel):
    """Material considered but excluded from the final edit."""
    ref: str = ""
    name: str = ""
    reason: str = ""
    suggested_use: str = ""
    event_id: Optional[int] = None
    file_path: Optional[str] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    label: str = ""
    also: list[str] = Field(default_factory=list)
    also_details: list[dict] = Field(default_factory=list)


class EditPlan(BaseModel):
    """Structured timeline plan produced by the Selection stage."""
    mode: str = "clip_assembly"
    aspect_ratio: Optional[str] = None
    ordering_strategy: str = ""
    order_check: dict = Field(default_factory=dict)
    target_seconds: Optional[float] = None
    raw_seconds: float = 0.0
    total_seconds: float = 0.0
    duration_status: str = "unconstrained"
    duration_delta: float = 0.0
    valid: bool = True
    segments: list[TimelineSegment] = Field(default_factory=list)
    excluded: list[ExcludedMaterial] = Field(default_factory=list)
    excluded_omitted: int = 0
    explanation: str = ""


class IngestResult(BaseModel):
    """Structured result of an ingest run."""
    status: Literal["success", "partial_success", "failure"]
    project_id: int
    indexed_count: int = 0
    reused_count: int = 0
    unreadable_count: int = 0
    event_count: int = 0
    truncated: bool = False
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DeliveryResult(BaseModel):
    """Structured result of a Delivery export."""
    status: Literal["success", "failure"] = "success"
    xml_path: Optional[str] = None
    json_path: Optional[str] = None
    sequence_name: str = ""
    project_id: Optional[int] = None
    clip_count: int = 0
    total_frames: int = 0
    # Output frame and fit statistics.
    aspect_ratio: Optional[str] = None
    width: int = 0
    height: int = 0
    letterboxed_clips: int = 0
    pillarboxed_clips: int = 0
    message: str = ""
    warnings: list[str] = Field(default_factory=list)