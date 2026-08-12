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
    mood: str = Field(
        default="unknown",
        description="Emotional tone conveyed by the imagery: one of 'calm', "
                    "'energetic', 'tense', 'cinematic', or 'unknown' if unclear.",
    )


class EventTags(BaseModel):
    """Action-oriented tags for ONE temporal segment (event) of a clip.

    This is the "what HAPPENS" layer, distinct from :class:`VisionTags` (the "what is
    IN the frame" layer). The model is shown several frames spanning a KNOWN time window
    IN ORDER and asked to describe the ACTION and the CHANGE across them — verbs, not
    just nouns. The window's start/end seconds are supplied by ingest from real FFmpeg
    scene boundaries, so the model never invents timecodes; it only describes what
    happens inside a window it is told about.

    As with VisionTags: every field carries an explicit Field(description=...) and none is
    named 'description' (the schema-echo bug). Prefer empty/'unknown' over a guess.
    """
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
    """One persisted temporal event: a segment of a clip with a real start/end.

    Mirrors a ``clip_events`` row. ``start_seconds``/``end_seconds`` are against the
    SOURCE clip and come from FFmpeg scene boundaries (never model-guessed), so an event
    can be trimmed to VERBATIM by Selection/Delivery via the existing in/out machinery.
    Unknown extra keys are ignored so a partial DB row validates without fabrication.
    """
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


# ── Pipeline stage data contracts (Ingest → Search → Curation → Selection → Delivery) ──
#
# These mirror the plain dicts the services already produce/consume today
# (``retrieval_service.hybrid_search`` candidates and the ``timeline_service`` timeline
# plans) so they can become the TYPED contract that flows on ``ProductionState`` between
# stages (audit H-03 / H-05), replacing the current messages-only + fenced-JSON hand-off.
#
# Batch 0 introduces the definitions and re-types ``ProductionState`` against them.
# Populating/validating actual instances at the tool and UI boundaries (so Delivery no
# longer regex-scrapes a Markdown plan, and the UI no longer gates ingest on "did the
# agent return any text") is wired in the later batches. Every field is optional / has a
# default so a partial upstream dict validates without fabricating values.


class SearchCandidate(BaseModel):
    """One retrieval candidate as returned by ``retrieval_service.hybrid_search``.

    Mirrors that function's per-row output dict exactly: real catalogue metadata plus a
    calibrated ``relevance`` (``None`` when no free-text query was given — never a
    fabricated confidence), a coarse ``group_size`` label, and a UI ``suggestion`` marker
    ('suggested' | 'neutral' | 'low'). Unknown extra keys are ignored, not rejected.
    """
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
    """One ordered segment of an edit timeline (``timeline_service`` builders).

    ``order`` is 1-based and preserved end-to-end. ``in_point``/``out_point``/``duration``
    are seconds against the SOURCE clip (Delivery maps them to frames verbatim); both
    points are ``None`` only when the source length is unknown, in which case the
    compiler uses the whole clip rather than a fabricated range.

    The MOMENT ASSEMBLY fields (``event_id``/``event_start``/``event_end``/
    ``event_duration``/``trimmed``/``trim_note``/``protected``) record the original event
    boundary a segment came from and any duration optimisation applied inside it — a
    trim can never leave ``[event_start, event_end]``. They stay unset in CLIP ASSEMBLY,
    where nothing is ever trimmed.

    ``valid``/``validation_error`` carry segment-range validation (audit C-05/C-06): a
    segment defaults to valid, and the builder flips it rather than silently expanding an
    illegal range back to the full clip.
    """
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
    """Candidate material the Selection Agent deliberately left OUT of the edit.

    This is a CURATED SHORTLIST (capped at ``timeline_service.MAX_BACKUP_ITEMS``), not an
    inventory of everything unused: only material the agent genuinely weighed and rejected
    AND that is a meaningful alternative editorial choice. Unrelated leftovers from
    ``get_clip_events`` do not belong here. ``also``/``also_details`` group near-identical
    rejected moments under one entry so grouping costs a single slot; the plan's
    ``excluded_omitted`` counts anything trimmed past the cap.

    The agent supplies only ``ref`` (an event id, or a clip's file name) plus its
    reasoning; the Selection tool RESOLVES that against the catalogue to fill in
    ``file_path`` / ``name`` / ``start_seconds`` / ``end_seconds`` / ``label``, so the
    editor reads "warmup.mov 18.0s-26.0s — passing drill" rather than "Event 9". Those
    fields stay ``None`` when nothing matched — an unresolved reference is shown as given,
    never dressed up with an invented file or timecode.
    """
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
    """The Selection stage's structured timeline plan.

    ``mode`` is one of the two user-facing editing modes — ``'clip_assembly'`` (ordered
    complete clips, no trimming, no target duration) or ``'moment_assembly'`` (ordered
    temporal moments with an optional target duration). The mode changes only how the
    timeline is GENERATED; ``segments`` are ordered and compiled VERBATIM by the Delivery
    stage either way (order + in/out points preserved — no re-ordering, dropping, or
    re-trimming).

    ``raw_seconds`` is the length before any duration optimisation, ``total_seconds``
    after it, and ``duration_status`` is 'unconstrained' | 'on_target' | 'under_target' |
    'over_target'.

    ``aspect_ratio`` is the editor's OUTPUT specification ("16:9", "9:16", "4:3", "3:4" or
    "1:1"), carried through Selection untouched — Selection may let it influence which
    footage it picks but never crops or resizes media. Delivery reads it from here and
    scales each clip to FIT that frame with its own aspect preserved.

    ``ordering_strategy`` is the agent's own one-line statement of the SHAPE it arranged
    the material into (declared before it ordered anything), and ``order_check`` records
    whether the emitted order turned out identical to the order the material was listed in
    — ``{"checked", "unchanged", "reference", "items"}``. Ordering is the core editorial
    act of this stage, so it is captured explicitly rather than left implicit in the
    sequence of segments. An ``unchanged`` order is not an error (chronological suits
    plenty of edits); it is flagged so a deliberate choice can be told apart from simply
    echoing the candidate list.
    """
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
    """Structured outcome of an ingest run (audit C-07 / C-08 / L-03).

    ``status`` distinguishes a clean run from a partial one (some files unreadable) or a
    hard failure (missing directory, no ffprobe, empty directory, DB write failure), so
    the UI can unlock Search/Selection/Delivery on real success + ``indexed_count > 0``
    rather than on "the agent returned some text". ``truncated`` flags that the scan hit
    the file cap and the destructive catalogue rewrite was refused.
    """
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
    """Structured outcome of a Delivery compile (FCP7 XML + JSON intermediate).

    Records the written artefact paths and the sequence shape so the UI can offer
    downloads and surface a validation status without re-parsing agent prose (audit
    H-12 / M-14). ``status`` is 'failure' when pre-export media/range validation rejects
    the compile instead of emitting a "successful" XML pointing at invalid media.
    """
    status: Literal["success", "failure"] = "success"
    xml_path: Optional[str] = None
    json_path: Optional[str] = None
    sequence_name: str = ""
    project_id: Optional[int] = None
    clip_count: int = 0
    total_frames: int = 0
    # The delivery frame actually written, and the honest consequence of scaling to fit
    # it (nothing was cropped or stretched to avoid these).
    aspect_ratio: Optional[str] = None
    width: int = 0
    height: int = 0
    letterboxed_clips: int = 0
    pillarboxed_clips: int = 0
    message: str = ""
    warnings: list[str] = Field(default_factory=list)