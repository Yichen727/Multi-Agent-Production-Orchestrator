"""Vision service — GPT-5.4 frame analysis for automatic shot tagging.

Turns sampled video frames into structured, searchable tags (shot type, objects,
keywords, a one-line description, and an approximate people count). This is the
"let the OpenAI model watch the footage and tag it" layer of ingestion.

Hard rule: the model tags only what it can actually see. It is instructed to leave
fields at their defaults rather than invent content, and it does NOT attempt to
identify *who* a person is — only that people are present and roughly how many.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.models.schemas import VisionTags
from app.services.openai_service import get_llm
from app.utils.logger import get_logger

logger = get_logger("vision_service")


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
        (no API key, no frames) or the call failed — callers should then leave the
        shot's semantic fields unclassified rather than fabricate them.

    The extra arguments are optional, so the legacy ``analyze_frames(frames_b64)``
    call still works unchanged.
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

    content = [{"type": "text", "text": instruction}]
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    try:
        vlm = get_llm(model=settings.VISION_MODEL).with_structured_output(VisionTags)
        return vlm.invoke([SystemMessage(_VISION_SYSTEM), HumanMessage(content=content)])
    except Exception as e:  # network / model / parsing — degrade gracefully
        logger.error(f"Vision analysis failed: {e}")
        return None
