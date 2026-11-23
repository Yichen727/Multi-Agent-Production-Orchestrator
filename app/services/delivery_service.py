"""Delivery — compile the edit timeline into a Premiere Pro project file."""

import uuid
from app.services.metrics_service import current_run_id
from app.services.contracts import project_id_value
from app.services.plan_contract import validate_edit_plan

from app.models.schemas import DeliveryResult
from app.services.catalogue_resolver import resolve_one, AmbiguousIdentifier
from app.services.database_service import validate_segment_source
from app.services.premiere_export_service import compile_project, InvalidAspectRatio
from app.services.ffmpeg_service import count_audio_streams
from app.utils.logger import get_logger

logger = get_logger("delivery_service")


def _refuse(message: str, project_id: int | None = None) -> DeliveryResult:
    return DeliveryResult(status="failure", project_id=project_id, message=message,
                          run_id=current_run_id() or uuid.uuid4().hex)


def _record_success(result: dict, project_id: int, message: str) -> DeliveryResult:
    """Return structured information owned by this invocation."""
    seq = result["timeline"]["sequence"]
    return DeliveryResult(
        run_id=current_run_id() or uuid.uuid4().hex,
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
    )


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

def compile_plan(plan, *, sequence_name: str = "MAPO Edit", project_id=1) -> DeliveryResult:
    """Compile a Selection plan into a Premiere Pro project."""
    try:
        project_id = project_id_value(project_id)
    except ValueError as exc:
        return _refuse(str(exc))

    try:
        data = validate_edit_plan(plan, project_id)
    except (ValueError, TypeError, KeyError, OSError, AmbiguousIdentifier) as exc:
        return _refuse(f"Refusing stale/unverified timeline: {exc}", project_id)
    segments = data["segments"]

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
        ident = seg.get("clip_id")
        ident = str(ident) if ident is not None else (seg.get("file_path") or "")
        try:
            row = resolve_one(project_id, ident)
        except AmbiguousIdentifier as e:
            unresolved.append(str(e))
            continue
        if row is None:
            unresolved.append(seg.get("file_path") or seg.get("clip_id") or f"segment #{i}")
            continue
        try:
            validate_segment_source(seg, row)
        except (ValueError, OSError) as e:
            return _refuse(f"Refusing stale/unverified segment #{i}: {e}", project_id)
        streams = count_audio_streams(row["file_path"])
        streams = streams if streams > 0 else (1 if row.get("has_audio") else 0)
        clips.append({
            "file_path": row["file_path"],
            "clip_id": row["clip_id"],
            "source_revision": row["source_revision"],
            "event_id": seg.get("event_id"),
            "event_set_id": seg.get("event_set_id"),
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
    return _record_success(result, project_id, summary)
