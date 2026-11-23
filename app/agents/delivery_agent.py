"""Delivery — compile the edit timeline into a Premiere Pro project file."""

import json

from app.models.schemas import DeliveryResult
from app.services.catalogue_resolver import resolve_one, AmbiguousIdentifier
from app.services.premiere_export_service import compile_project, InvalidAspectRatio
from app.services.ffmpeg_service import count_audio_streams
from app.utils.logger import get_logger

logger = get_logger("delivery_agent")


# Store the result of the most recent delivery run for the UI/orchestrator.
_LAST_DELIVERY_RESULT: DeliveryResult | None = None


def _record_delivery_result(result: DeliveryResult) -> DeliveryResult:
    global _LAST_DELIVERY_RESULT
    _LAST_DELIVERY_RESULT = result
    return result


def reset_last_delivery_result() -> None:
    """Clear the previous delivery result."""
    global _LAST_DELIVERY_RESULT
    _LAST_DELIVERY_RESULT = None


def get_last_delivery_result() -> DeliveryResult | None:
    """Return the result of the most recent delivery run."""
    return _LAST_DELIVERY_RESULT


# ── Helpers ────────────────────────────────────────────────────────────────────

def _refuse(message: str, project_id: int | None = None) -> str:
    """Record a failed delivery and return the error message."""
    _record_delivery_result(DeliveryResult(status="failure", project_id=project_id,
                                           message=message))
    return message


def _record_success(result: dict, project_id: int, message: str) -> None:
    """Store structured information about a successful export."""
    seq = result["timeline"]["sequence"]
    _record_delivery_result(DeliveryResult(
        status="success",
        xml_path=result.get("xml_path"),
        json_path=result.get("json_path"),
        sequence_name=seq.get("name") or "",
        project_id=project_id,
        clip_count=seq.get("clip_count") or 0,
        total_frames=seq.get("total_frames") or 0,
        aspect_ratio=seq.get("aspect_ratio"),
        width=seq.get("width") or 0,
        height=seq.get("height") or 0,
        letterboxed_clips=seq.get("letterboxed_clips") or 0,
        pillarboxed_clips=seq.get("pillarboxed_clips") or 0,
        message=message,
    ))


def _as_pid(value) -> int:
    """Convert project ID to int, defaulting to 1."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _frame_summary(seq: dict) -> str:
    """Summarise the output frame and any fit adjustments."""
    parts = [f"{seq['width']}x{seq['height']}"]
    if seq.get("aspect_ratio"):
        parts.insert(0, f"frame {seq['aspect_ratio']}")
    boxed = []
    if seq.get("letterboxed_clips"):
        boxed.append(f"{seq['letterboxed_clips']} letterboxed")
    if seq.get("pillarboxed_clips"):
        boxed.append(f"{seq['pillarboxed_clips']} pillarboxed")
    line = " · ".join(parts)
    if boxed:
        line += (f" · scaled to fit: {', '.join(boxed)} "
                 "(full image kept — nothing cropped or stretched)")
    return line


# ── Compile ────────────────────────────────────────────────────────────────────

def compile_plan(plan, *, sequence_name: str = "MAPO Edit", project_id=1) -> str:
    """Compile a Selection plan into a Premiere Pro project."""
    project_id = _as_pid(project_id)

    # Accept either a dict/list or a JSON representation of the plan.
    if isinstance(plan, (str, bytes)):
        try:
            data = json.loads(plan)
        except (json.JSONDecodeError, TypeError) as e:
            return _refuse(f"Could not parse the timeline plan JSON: {e}", project_id)
    else:
        data = plan
    segments = data.get("segments") if isinstance(data, dict) else data
    if not segments or not isinstance(segments, list):
        return _refuse("No segments to compile. Provide the Selection plan JSON.", project_id)

    # Reject invalid segments rather than silently repairing them.
    invalid = [s for s in segments if s.get("valid") is False]
    if invalid:
        listing = "\n".join(
            f"  • #{s.get('order', '?')} {s.get('name') or s.get('file_path')}: "
            f"{s.get('validation_error') or 'invalid segment range'}" for s in invalid)
        return _refuse("Refusing to compile — the plan contains invalid segment(s); fix the "
                       f"trim or drop the clip in Selection first:\n{listing}", project_id)

    clips, unresolved = [], []
    for i, seg in enumerate(segments, start=1):
        ident = seg.get("shot_id")
        ident = str(ident) if ident is not None else (seg.get("file_path") or "")
        try:
            row = resolve_one(project_id, ident)
        except AmbiguousIdentifier as e:
            unresolved.append(str(e))
            continue
        if row is None:
            unresolved.append(seg.get("file_path") or seg.get("shot_id") or f"segment #{i}")
            continue
        streams = count_audio_streams(row["file_path"])
        streams = streams if streams > 0 else (1 if row.get("has_audio") else 0)
        clips.append({
            "file_path": row["file_path"],
            "shot_id": row.get("shot_id"),
            "duration_seconds": row.get("duration_seconds") or 0.0,
            "fps": row.get("fps") or None,
            "width": row.get("width") or None,
            "height": row.get("height") or None,
            "has_audio": bool(row.get("has_audio")),
            "audio_streams": streams,
            "audio_channels": row.get("audio_channels") or None,
            "audio_sample_rate": row.get("audio_sample_rate") or None,
            "audio_bit_depth": row.get("audio_bit_depth") or None,
            "role": seg.get("label") or "",
            "in_point": seg.get("in_point"),
            "out_point": seg.get("out_point"),
        })

    if unresolved:
        return _refuse("Refusing to compile — these segments matched no catalogued clip: "
                       f"{unresolved}. Every media reference must be real.", project_id)

    aspect_ratio = data.get("aspect_ratio") if isinstance(data, dict) else None

    try:
        result = compile_project(clips, sequence_name=sequence_name,
                                 project_id=project_id, aspect_ratio=aspect_ratio,
                                 write=True)
    except InvalidAspectRatio as e:
        return _refuse(f"Refusing to compile — the plan's output aspect ratio is unusable: "
                       f"{e} Fix it in ③ Selection and re-generate the timeline.", project_id)
    except Exception as e:
        logger.error(f"Segment compile failed: {e}")
        return _refuse(f"Compile failed: {e}", project_id)

    seq = result["timeline"]["sequence"]
    mode = (data.get("mode") if isinstance(data, dict) else None) or "clip_assembly"
    mode = mode.replace("_", " ")
    order_summary = " → ".join(
        f"{c['order']}.{(c['role'] + ':') if c['role'] else ''}{c['name']} "
        f"({c['used_seconds']:.1f}s)"
        for c in result["timeline"]["clips"]
    )
    summary = (
        f"✅ Premiere project compiled from {mode.upper()} timeline segments "
        "(FCP7 XML — import via File ▸ Import in Premiere Pro).\n"
        f"  XML : {result['xml_path']}\n"
        f"  JSON: {result['json_path']}\n"
        f"  Sequence '{seq['name']}': {seq['clip_count']} segments · "
        f"{seq['total_seconds']:.1f}s · {seq['timebase']} fps\n"
        f"  Output: {_frame_summary(seq)}\n"
        f"  Segments (order + trim preserved): {order_summary}"
    )
    _record_success(result, project_id,
                    f"{seq['clip_count']} segments compiled from the {mode} timeline.")
    return summary
