"""Vision service — GPT-5.4 frame analysis for automatic shot tagging.

Turns sampled video frames into structured, searchable tags (shot type, objects,
keywords, a one-line description, and an approximate people count). This is the
"let the OpenAI model watch the footage and tag it" layer of ingestion.

Hard rule: the model tags only what it can actually see. It is instructed to leave
fields at their defaults rather than invent content, and it does NOT attempt to
identify *who* a person is — only that people are present and roughly how many.
"""

import time

from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.models.schemas import VisionTags, EventTags
from app.services.openai_service import get_llm
from app.utils.logger import get_logger

logger = get_logger("vision_service")


# ── Retry policy ──────────────────────────────────────────────────────────────
#
# A single transient failure used to cost a clip its ENTIRE semantic layer: no tags AND
# no embedding, which makes it invisible to vector search rather than merely under-tagged.
# That is the most expensive failure mode in ingest, so the call is retried.
_RETRY_BACKOFF_SECONDS = (1.0, 4.0)

# Errors that will fail identically on every retry — retrying a bad key on 200 clips just
# burns time. Matched against the exception text (provider exception types vary).
_PERMANENT_ERROR_MARKERS = (
    "invalid_api_key", "authentication", "unauthorized", "401",
    "permission", "model_not_found", "does not exist",
)


def _is_permanent(err: Exception) -> bool:
    """True when retrying ``err`` cannot possibly help (auth / unknown model)."""
    text = str(err).lower()
    return any(marker in text for marker in _PERMANENT_ERROR_MARKERS)


def _reduce_frames(frames_b64: list[str]) -> list[str]:
    """Halve a frame set, keeping temporal order plus the first and last frame.

    The final retry goes out with this reduced set: it also clears failures caused by
    too many images in one request (context limits / provider per-request caps), which a
    plain retry would hit again.
    """
    if len(frames_b64) <= 2:
        return frames_b64
    keep = sorted({*range(0, len(frames_b64), 2), len(frames_b64) - 1})
    return [frames_b64[i] for i in keep]


def _image_content(instruction: str, frames_b64: list[str]) -> list[dict]:
    """Build the multimodal message content: the instruction then the frames, in order."""
    content: list[dict] = [{"type": "text", "text": instruction}]
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


def _invoke_vision(schema, system_prompt: str, instruction: str,
                   frames_b64: list[str], label: str):
    """Invoke the vision model with retries, returning a validated ``schema`` or None.

    Retries transient failures (network, rate limit, structured-output parse) with
    backoff; the last attempt uses a reduced frame set. Returns ``None`` once every
    attempt is exhausted or the error is permanent, so the caller still leaves the
    fields unclassified rather than fabricating them.
    """
    attempts = max(1, settings.VISION_MAX_ATTEMPTS)
    try:
        vlm = get_llm(model=settings.VISION_MODEL).with_structured_output(schema)
    except Exception as e:
        # Client construction (bad model name / config) must still return None: callers
        # rely on this never raising, so a failure leaves fields unclassified.
        logger.error(f"{label} could not initialise the vision model: {e}")
        return None

    for attempt in range(1, attempts + 1):
        # Final attempt: fewer frames, in case the request size was the problem.
        frames = _reduce_frames(frames_b64) if attempt == attempts and attempts > 1 \
            else frames_b64
        try:
            return vlm.invoke([
                SystemMessage(system_prompt),
                HumanMessage(content=_image_content(instruction, frames)),
            ])
        except Exception as e:  # network / model / parsing — degrade gracefully
            if _is_permanent(e):
                logger.error(f"{label} failed permanently (not retrying): {e}")
                return None
            if attempt == attempts:
                logger.error(f"{label} failed after {attempts} attempt(s): {e}")
                return None
            delay = _RETRY_BACKOFF_SECONDS[min(attempt - 1, len(_RETRY_BACKOFF_SECONDS) - 1)]
            logger.warning(f"{label} attempt {attempt}/{attempts} failed ({e}); "
                           f"retrying in {delay}s.")
            time.sleep(delay)
    return None


_VISION_SYSTEM = """You are a video footage tagger for a post-production catalogue.
You are shown a few frames sampled from a SINGLE video clip. Describe ONLY what is
visibly present.

Rules:
- Be factual and concise. If you are unsure, leave a field empty / default — do NOT
  guess or embellish.
- shot_type: choose the closest of close_up, medium_shot, wide_shot, establishing,
  over_shoulder, pov, aerial, or 'unknown' if unclear.
- setting: 'interior', 'exterior', or 'unknown'.
- objects: the salient objects/subjects actually visible.
- keywords: short, lowercase, searchable tags an editor would use to find this clip
  (subjects, actions, mood, environment). No invented proper nouns.
- people_count: the approximate number of distinct people visible (0 if none). Do NOT
  attempt to identify who they are.
- camera_motion: the dominant camera movement — 'pan', 'tilt', 'zoom', 'static',
  'handheld', or 'unknown' if you cannot tell from a few frames.
- lighting: 'natural', 'low_light', 'backlit', 'studio', or 'unknown'.
- mood: the emotional tone of the imagery — 'calm', 'energetic', 'tense',
  'cinematic', or 'unknown'.
- subject_position: where the main subject sits — 'center', 'left', 'right',
  'moving', or 'unknown' if there is no clear subject.

For every one of these semantic fields, prefer 'unknown' over a guess. These tags feed
search and edit decisions, so a wrong label is worse than an honest 'unknown'.
"""

_VISION_INSTRUCTION = (
    "Tag this clip from the sampled frames below. Return only what you can see."
)


def _sampling_context(frames_b64: list[str], duration_seconds: float | None,
                      scene_count: int | None,
                      sampled_timestamps: list[float] | None,
                      sampling_strategy: str | None) -> str:
    """Build an optional context block describing how the frames were sampled.

    Gives the model grounding (clip length, how many shots, where each frame sits
    on the timeline) so it can better judge shot type / camera motion / whether
    several distinct scenes are present. It is guidance only — the model is still
    told to describe ONLY what is visibly present and never to invent content.
    Returns "" when no context is available.
    """
    lines: list[str] = []
    if duration_seconds and duration_seconds > 0:
        lines.append(f"- Clip duration: ~{duration_seconds:.1f}s")
    if scene_count and scene_count > 0:
        lines.append(f"- Detected shot/scene cuts: ~{scene_count} distinct shot(s); "
                     "frames may span several different scenes.")
    if sampled_timestamps:
        ts = ", ".join(f"{t:.1f}s" for t in sampled_timestamps)
        lines.append(f"- The {len(frames_b64)} frame(s) were sampled at: {ts}")
    if sampling_strategy:
        lines.append(f"- Sampling strategy: {sampling_strategy}")
    if not lines:
        return ""
    return (
        "\n\nContext about the sampled frames (guidance only — still describe ONLY "
        "what is visibly present, and prefer 'unknown' over a guess):\n"
        + "\n".join(lines)
    )


def analyze_frames(frames_b64: list[str], duration_seconds: float | None = None,
                   scene_count: int | None = None,
                   sampled_timestamps: list[float] | None = None,
                   sampling_strategy: str | None = None) -> VisionTags | None:
    """Analyse sampled frames with GPT-5.4 Vision and return structured tags.

    Args:
        frames_b64: Base64-encoded JPEG frames from one clip.
        duration_seconds: Clip duration (optional context for the model).
        scene_count: Detected number of shots (optional context).
        sampled_timestamps: Seconds each frame was taken at (optional context).
        sampling_strategy: How frames were chosen, e.g. 'scene-midpoint' or
            'adaptive-even' (optional context).

    Returns:
        A ``VisionTags`` instance, or ``None`` if vision analysis is unavailable
        (no API key, no frames) or every attempt failed — callers should then leave
        the shot's semantic fields unclassified rather than fabricate them.

    Transient failures are retried (see :func:`_invoke_vision`). The extra arguments
    are optional, so the legacy ``analyze_frames(frames_b64)`` call still works
    unchanged.
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY — skipping vision analysis.")
        return None
    if not frames_b64:
        return None

    instruction = _VISION_INSTRUCTION + _sampling_context(
        frames_b64, duration_seconds, scene_count, sampled_timestamps,
        sampling_strategy,
    )
    return _invoke_vision(VisionTags, _VISION_SYSTEM, instruction, frames_b64,
                          "Clip vision analysis")


# ── Event-level (temporal) analysis: "what HAPPENS", not "what is in frame" ─────

_EVENT_SYSTEM = """You are a video EVENT tagger for a post-production catalogue. You are
shown several frames sampled IN CHRONOLOGICAL ORDER from a SINGLE short time window of one
clip. Your job is to describe WHAT HAPPENS across these frames — the action and the change
— NOT to caption a single frame.

Rules:
- Read the frames as a sequence: first frame → last frame. Focus on MOTION and CHANGE
  (someone enters, sits, stands, hands over an object, the camera pushes in, a ball is
  kicked). Use VERBS.
- action: one factual sentence, verb-led, describing what happens across the window. If
  nothing changes (a single held moment), say what is being held.
- subjects: only the entities involved in the action, not every visible object.
- state_change: what is DIFFERENT between the first and last frame. Leave empty if nothing
  changes.
- keywords: short lowercase searchable ACTION/EVENT tags an editor would use to find this
  MOMENT (verbs and event nouns). No invented proper nouns.
- Describe ONLY what is visibly present. If unsure, leave a field empty — do NOT guess,
  do NOT invent events that are not shown, and do NOT identify WHO anyone is.
"""


def analyze_event(frames_b64: list[str], start_seconds: float | None = None,
                  end_seconds: float | None = None,
                  clip_duration: float | None = None) -> EventTags | None:
    """Analyse one temporal window's frames with GPT-5.4 and return action tags.

    Unlike :func:`analyze_frames` (one static summary for the whole clip), this is called
    per event window with SEVERAL ordered frames from that window, so the model can report
    the ACTION and the CHANGE within it. The window's start/end come from real FFmpeg
    boundaries and are passed only as grounding context — the model never invents timecodes.

    Args:
        frames_b64: Ordered base64 JPEG frames sampled across ONE window.
        start_seconds / end_seconds: The window's real time bounds (context only).
        clip_duration: The parent clip's duration (context only).

    Returns:
        An ``EventTags`` instance, or ``None`` when vision is unavailable (no key/frames)
        or every attempt failed — the caller then leaves the event's action fields
        unclassified rather than fabricating them. Transient failures are retried (see
        :func:`_invoke_vision`).
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY — skipping event analysis.")
        return None
    if not frames_b64:
        return None

    ctx = []
    if start_seconds is not None and end_seconds is not None:
        ctx.append(f"- This window spans ~{start_seconds:.1f}s–{end_seconds:.1f}s of the clip.")
    if clip_duration and clip_duration > 0:
        ctx.append(f"- The full clip is ~{clip_duration:.1f}s long.")
    ctx.append(f"- The {len(frames_b64)} frame(s) below are in chronological order.")
    instruction = ("Describe what HAPPENS across these frames, in order. Return only what "
                   "you can see.\n\nContext (guidance only):\n" + "\n".join(ctx))

    window = (f"{start_seconds:.1f}-{end_seconds:.1f}s"
              if start_seconds is not None and end_seconds is not None else "window")
    return _invoke_vision(EventTags, _EVENT_SYSTEM, instruction, frames_b64,
                          f"Event analysis ({window})")
