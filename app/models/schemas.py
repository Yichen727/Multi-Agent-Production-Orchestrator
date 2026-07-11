"""Pydantic schemas for structured agent I/O."""

from pydantic import BaseModel, Field
from typing import Optional


class ShotMetadata(BaseModel):
    """Metadata for a single shot extracted by the Ingest Agent."""
    shot_id: int
    file_path: str
    shot_type: str  # wide_shot, close_up, establishing, medium_shot, etc.
    duration_seconds: float
    objects_detected: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class VisionTags(BaseModel):
    """Video frame tags."""
    # NOTE: every field carries an explicit Field(description=...). Without it, and
    # with a field literally named "description", structured-output models tend to
    # echo the SCHEMA's docstring into that field — which is exactly the bug that
    # polluted the catalogue. The field is named scene_description for the same reason.
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
    camera_motion: str = Field(
        default="unknown",
        description="Dominant camera movement across the frames: one of 'pan', "
                    "'tilt', 'zoom', 'static', 'handheld', or 'unknown' if unclear.",
    )
    lighting: str = Field(
        default="unknown",
        description="Overall lighting character: one of 'natural', 'low_light', "
                    "'backlit', 'studio', or 'unknown' if unclear.",
    )
    mood: str = Field(
        default="unknown",
        description="Emotional tone conveyed by the imagery: one of 'calm', "
                    "'energetic', 'tense', 'cinematic', or 'unknown' if unclear.",
    )
    subject_position: str = Field(
        default="unknown",
        description="Where the main subject sits in the frame: one of 'center', "
                    "'left', 'right', 'moving', or 'unknown' if unclear / no subject.",
    )


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
    """Structured interpretation of the user's creative editing prompt.

    Produced by the Selection/Editorial Assistant Agent before ranking, so that
    selection criteria are derived from intent rather than raw quality score alone.
    """
    video_type: str = ""   # e.g. trailer, documentary, social_reel, narrative
    pace: str = ""         # e.g. fast, medium, slow
    emotion: str = ""      # e.g. energetic, tense, calm, uplifting
    style: str = ""        # e.g. cinematic, handheld, observational
    notes: str = ""


class ClipRecommendation(BaseModel):
    """An explainable clip recommendation from the Selection Agent.

    The editor always makes the final decision (Human-in-the-Loop); the agent
    recommends and explains, it does not auto-select.
    """
    shot_id: int
    file_path: str
    rank: int
    score: float = Field(ge=0, le=100)
    explanation: str  # WHY this clip fits the user's editing intent


class TimelineEntry(BaseModel):
    """One clip placed as a step in the edit timeline by the Selection Agent.

    Ordering is INTENT-DRIVEN, not quality-ranked: the Selection Agent chooses whatever
    sequence best serves the user's editing intent (there is NO fixed narrative arc).
    Quality only gates whether a clip is included at all — it never decides the order.
    """
    order: int  # position in the timeline, 1-based
    label: str = ""  # optional, free-form step label (e.g. "cold open", "hero moment")
    file_path: str
    shot_id: Optional[int] = None
    rationale: str  # why this clip sits at this position + how it connects / paces


class EditTimeline(BaseModel):
    """The Selection Agent's output: an ordered edit structure, not a score list.

    Built ONLY from the clips the editor curated in the UI, ordered to fit the editing
    intent (free structure, any number of steps). The editor keeps final control
    (Human-in-the-Loop).
    """
    intent: EditingIntent = Field(default_factory=EditingIntent)
    entries: list[TimelineEntry] = Field(default_factory=list)
    notes: str = ""