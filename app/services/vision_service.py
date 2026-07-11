"""Vision service — GPT-4o frame analysis for automatic shot tagging.

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


def analyze_frames(frames_b64: list[str]) -> VisionTags | None:
    """Analyse sampled frames with GPT-4o Vision and return structured tags.

    Args:
        frames_b64: Base64-encoded JPEG frames from one clip.

    Returns:
        A ``VisionTags`` instance, or ``None`` if vision analysis is unavailable
        (no API key, no frames) or the call failed — callers should then leave the
        shot's semantic fields unclassified rather than fabricate them.
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY — skipping vision analysis.")
        return None
    if not frames_b64:
        return None

    content = [{"type": "text", "text": _VISION_INSTRUCTION}]
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
