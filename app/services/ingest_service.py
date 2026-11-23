"""Incremental footage ingestion; no reasoning agent or global result registry."""

import json
import os
import uuid
from pathlib import Path

from app.config import settings
from app.models.schemas import IngestResult
from app.services.openai_service import embed_texts
from app.services.database_service import (
    get_catalogued_clips, get_catalogued_events,
    publish_clip, get_analysis_steps, project_ingest_lock, mark_offline,
)
from app.services.contracts import project_id_value
from app.services.metrics_service import current_run_id
from app.services.media_identity import canonical_path, fingerprint, same_stat, content_hash
from app.services.ffmpeg_service import (
    generate_proxy,
    check_ffprobe_installed, probe_video_metadata, detect_scene_cuts,
    choose_frame_count,
    extract_scene_representative_frames_b64, extract_sample_frames_b64,
    build_event_windows, extract_frames_in_window_b64,
)
from app.services.vision_service import analyze_frames, analyze_event
from app.utils.logger import get_logger

logger = get_logger("ingest_service")


# Ingest and event extraction limits
_MAX_INGEST_FILES = 200
_MIN_EVENTS_PER_CLIP = 12        
_MAX_EVENTS_PER_CLIP = 40        
_SECONDS_PER_EVENT = 6.0       
_FRAMES_PER_EVENT = 3


def _event_budget(duration_seconds: float | None) -> int:
    """Calculate the event-window budget for a clip."""
    if not duration_seconds or duration_seconds <= 0:
        return _MIN_EVENTS_PER_CLIP
    want = int(round(float(duration_seconds) / _SECONDS_PER_EVENT))
    return max(_MIN_EVENTS_PER_CLIP, min(want, _MAX_EVENTS_PER_CLIP))


# Columns preserved when reusing cached events.
_EVENT_REUSE_COLUMNS = (
    "file_path", "event_order", "start_seconds", "end_seconds", "duration_seconds",
    "action", "state_change", "subjects", "keywords", "embedding",
)


def _event_semantic_text(action: str | None, state_change: str | None,
                         keywords: str) -> str:
    """Build the text embedded for moment-level search."""
    parts = []
    if action:
        parts.append(action)
    if state_change:
        parts.append(state_change)
    if keywords:
        parts.append(keywords.replace(",", " "))
    return ". ".join(parts).strip()


class EventAnalysis(list):
    """Measured windows, including explicit failed windows, with tagging status."""

    def __init__(self):
        super().__init__()
        self.attempt_failures = 0

    @property
    def complete(self):
        return not self.attempt_failures and all(e.get("action") or e.get("keywords") for e in self)


def _build_clip_events(path, duration, cut_times, *, existing_events=None, force=False):
    cached = {(round(e["start_seconds"] * 1e6), round(e["end_seconds"] * 1e6)): e
              for e in (existing_events or [])}
    windows = ([(e["start_seconds"], e["end_seconds"]) for e in existing_events]
               if existing_events and not force else
               build_event_windows(duration, cut_times, max_events=_event_budget(duration)))
    events = EventAnalysis()
    pending = []
    for order, (start, end) in enumerate(windows, 1):
        previous = cached.get((round(start * 1e6), round(end * 1e6)), {})
        item = {k: previous.get(k) for k in _EVENT_REUSE_COLUMNS}
        item.update(file_path=path, event_order=order, start_seconds=round(start, 3),
                    end_seconds=round(end, 3), duration_seconds=round(end - start, 3))
        if force or not (item.get("action") or item.get("keywords")):
            frames = extract_frames_in_window_b64(path, start, end, count=_FRAMES_PER_EVENT)
            tags = analyze_event(frames, start_seconds=start, end_seconds=end,
                                 clip_duration=duration) if frames else None
            if tags is not None and (tags.action or tags.keywords):
                item.update(action=(tags.action or "").strip() or None,
                            state_change=(tags.state_change or "").strip() or None,
                            subjects=tags.subjects,
                            keywords=",".join(dict.fromkeys(k.strip().lower() for k in tags.keywords if k.strip())),
                            embedding=None)
            else:
                events.attempt_failures += 1
        text = _event_semantic_text(item.get("action"), item.get("state_change"), item.get("keywords") or "")
        if text and not item.get("embedding"):
            pending.append((len(events), text))
        events.append(item)
    if pending:
        vectors = embed_texts([text for _, text in pending])
        if vectors and len(vectors) == len(pending):
            for (index, _), vector in zip(pending, vectors):
                events[index]["embedding"] = json.dumps(vector)
    return events

def _find_video_files(scan_dir: Path, *, offset: int = 0) -> tuple[list[Path], bool]:
    """Find supported videos and report whether the scan was truncated."""
    exts = {"." + e.strip().lower().lstrip(".") for e in settings.SUPPORTED_VIDEO_FORMATS}
    found: list[Path] = []
    truncated = False
    skipped = 0
    def scan_error(error):
        raise error
    for dirpath, dirnames, filenames in os.walk(scan_dir, followlinks=False, onerror=scan_error):
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in exts:
                if skipped < offset:
                    skipped += 1
                    continue
                found.append(Path(dirpath) / name)
                if len(found) > _MAX_INGEST_FILES:
                    truncated = True
                    break
        if truncated:
            break
    return found[:_MAX_INGEST_FILES], truncated


def _semantic_text(description: str | None, keywords: str, mood: str | None) -> str:
    """Build the text embedded for semantic search."""
    parts = []
    if description:
        parts.append(description)
    if keywords:
        parts.append(keywords.replace(",", " "))
    if mood:
        parts.append(f"{mood} mood")
    return ". ".join(parts).strip()


def ingest_footage(directory: str = None, project_id: int = 1,
                   analyze_content: bool = True, generate_proxies: bool = False,
                   analyze_events: bool = True, force_reanalyze: bool = False,
                   synchronize: bool = False, verify_content: bool = False,
                   batch_offset: int | None = None) -> IngestResult:
    """Incrementally index videos, preserving existing clip and event identities.

    Args:
        directory: Folder to import; defaults to RAW_FOOTAGE_DIR.
        project_id: Explicit positive project identifier.
        analyze_content: Fill missing visual tags and embeddings.
        generate_proxies: Generate missing editing proxies.
        analyze_events: Fill missing temporal windows/tags when content analysis is enabled.
        force_reanalyze: Reanalyse explicitly; use only at the user's request.
        synchronize: Mark missing files under this root offline after a COMPLETE scan.
            Default is append/update. Never deletes clips or their history.
        verify_content: Hash even unchanged-size/mtime files for strict verification.
        batch_offset: Opt into deterministic pages of up to 200 files; follow next_offset.
            Default refuses an oversized import before writing. Paged imports cannot synchronize.

    Returns:
        A result owned by this invocation, with counts, status, and continuation offset.
    """
    pid = project_id_value(project_id)
    with project_ingest_lock(pid):
        result = _ingest_directory(Path(directory or settings.RAW_FOOTAGE_DIR), pid, analyze_content,
                                   generate_proxies, analyze_events, force_reanalyze,
                                   synchronize, verify_content, batch_offset=batch_offset)
        return result.model_copy(update={"run_id": current_run_id() or uuid.uuid4().hex})


def _ingest_directory(scan_dir, project_id, analyze_content, generate_proxies,
                      analyze_events, force_reanalyze, synchronize, verify_content, *, batch_offset=None):
    def fail(message, truncated=False):
        return IngestResult(status="failure", project_id=project_id,
            truncated=truncated, message=message, errors=[message])
    if not scan_dir.is_dir():
        return fail(f"Directory not found: {scan_dir}")
    if batch_offset is not None and (isinstance(batch_offset, bool) or not isinstance(batch_offset, int) or batch_offset < 0):
        return fail("batch_offset must be a nonnegative integer")
    if synchronize and batch_offset is not None:
        return fail("Synchronization requires a complete scan; paged imports only append/update")
    try:
        found, truncated = _find_video_files(scan_dir, offset=batch_offset or 0)
    except OSError as exc:
        return fail(f"Scan could not finish ({type(exc).__name__}); existing catalogue retained.")
    if truncated and batch_offset is None:
        return fail(f"Scan exceeds the {_MAX_INGEST_FILES}-file limit; import a smaller batch. "
                    "Existing catalogue was not changed.", truncated=True)
    if not found:
        return fail(f"No supported video files found in {scan_dir}; existing catalogue retained.")
    cached_by_path = {canonical_path(k): v for k, v in get_catalogued_clips(project_id).items()}
    events_by_path = {canonical_path(k): v for k, v in get_catalogued_events(project_id).items()}
    do_vision = analyze_content and bool(settings.OPENAI_API_KEY)
    do_events = do_vision and analyze_events
    indexed = reused = unreadable = event_count = 0
    warnings, errors, observed = [], [], set()
    for f in found:
        path = canonical_path(f)
        if path in observed:
            continue
        observed.add(path)
        cached = cached_by_path.get(path)
        unchanged = False
        try:
            before = fingerprint(path)
            unchanged = bool(cached and same_stat(cached, before))
            digest = cached.get("content_hash") if cached else None
            if verify_content or not unchanged:
                measured_hash = content_hash(path)
                if cached and digest:
                    unchanged = measured_hash == digest
                digest = measured_hash
            previous_steps = get_analysis_steps(project_id, cached["clip_id"], cached["source_revision"]) if cached and unchanged else {}
            def done(name):
                return previous_steps.get(name, {}).get("status") == "complete"
            row = dict(cached) if unchanged else {
                "shot_type": "unclassified", "description": None, "keywords": "",
                "people_count": None, "mood": None, "embedding": None}
            row.update(file_path=path, **before, content_hash=digest, online=1)
            steps = {}
            needs_probe = force_reanalyze or not unchanged or not done("probe")
            if needs_probe:
                if not check_ffprobe_installed():
                    raise ValueError("ffprobe is not installed; no new metadata was published")
                meta = probe_video_metadata(path)
                if not meta["ok"]:
                    raise ValueError("Video could not be probed; previous analysis was retained")
                row.update({key: meta.get(key) for key in (
                    "width", "height", "duration_seconds", "orientation", "fps", "codec", "has_audio",
                    "audio_channels", "audio_sample_rate", "audio_bit_depth")})
                steps["probe"] = "complete"
            old_events = events_by_path.get(path, []) if unchanged else []
            needs_tags = do_vision and (force_reanalyze or not done("clip_tags"))
            needs_events = do_events and (force_reanalyze or not done("events") or
                not old_events or any(not (e.get("action") or e.get("keywords")) for e in old_events))
            needs_event_vectors = do_events and any(
                (e.get("action") or e.get("keywords")) and not e.get("embedding") for e in old_events)
            scene_details = previous_steps.get("scene_cuts", {}).get("details") or {}
            cuts = scene_details.get("cut_times", [])
            if needs_probe or (needs_events and (not old_events or force_reanalyze)):
                scene = detect_scene_cuts(path, duration_seconds=row["duration_seconds"])
                cuts = scene["cut_times"]
                row["scene_count"] = scene["scene_count"]
                steps["scene_cuts"] = {"status": "complete" if scene.get("complete") else "partial",
                                      "details": {"cut_times": cuts}}
            if needs_tags:
                count = choose_frame_count(row["duration_seconds"], row.get("scene_count"))
                frames, timestamps = extract_scene_representative_frames_b64(
                    path, row["duration_seconds"], cuts, max_frames=count)
                if not frames:
                    frames = extract_sample_frames_b64(path, row["duration_seconds"], count=count)
                    timestamps = []
                tags = analyze_frames(frames, duration_seconds=row["duration_seconds"],
                    scene_count=row.get("scene_count"), sampled_timestamps=timestamps,
                    sampling_strategy="scene-midpoint" if cuts else "adaptive-even") if frames else None
                if tags is None:
                    steps["clip_tags"] = {"status": "failed", "config_version": f"{settings.VISION_MODEL}:clip-v1"}
                    warnings.append(f"{f.name}: clip tagging incomplete")
                else:
                    keywords = [*tags.keywords, *tags.objects, tags.setting]
                    row.update(shot_type=tags.shot_type or "unclassified",
                               keywords=",".join(dict.fromkeys(k.strip().lower() for k in keywords
                                                             if k.strip() and k != "unknown")),
                               description=tags.scene_description or None,
                               people_count=tags.people_count,
                               mood=tags.mood if tags.mood != "unknown" else None, embedding=None)
                    steps["clip_tags"] = {"status": "complete", "config_version": f"{settings.VISION_MODEL}:clip-v1"}
            semantic = _semantic_text(row.get("description"), row.get("keywords") or "", row.get("mood"))
            if do_vision and semantic and (force_reanalyze or not row.get("embedding")):
                vectors = embed_texts([semantic])
                if vectors:
                    row["embedding"] = json.dumps(vectors[0])
                steps["clip_embedding"] = {"status": "complete" if vectors else "failed",
                                           "config_version": settings.EMBEDDING_MODEL}
            new_events = None
            complete = True
            if needs_events or needs_event_vectors:
                new_events = _build_clip_events(path, row["duration_seconds"], cuts,
                    existing_events=old_events, force=force_reanalyze)
                complete = new_events.complete
                steps["events"] = {"status": "complete" if complete else "partial",
                                   "config_version": f"{settings.VISION_MODEL}:events-v1"}
                vector_complete = all(e.get("embedding") for e in new_events if e.get("action") or e.get("keywords"))
                steps["event_embedding"] = {"status": "complete" if vector_complete else "partial",
                                            "config_version": settings.EMBEDDING_MODEL}
                if not complete:
                    warnings.append(f"{f.name}: some event windows need analysis")
            if generate_proxies and (force_reanalyze or not done("proxy")):
                try:
                    proxy = generate_proxy(path)
                    steps["proxy"] = {"status": "complete", "details": {"path": proxy}}
                except Exception:
                    steps["proxy"] = "failed"
                    warnings.append(f"{f.name}: proxy generation failed")
            if fingerprint(path) != before:
                raise ValueError("Source changed during analysis; result discarded")
            saved = cached if unchanged and not steps and row == cached else publish_clip(
                project_id, row, events=new_events, steps=steps, events_complete=complete,
                expected_revision=cached["source_revision"] if cached else 0)
            cached_by_path[path] = saved
            indexed += 1
            if unchanged and not any(key not in ("probe",) for key in steps):
                reused += 1
            event_count += len(new_events if new_events is not None else old_events)
        except Exception as exc:
            unreadable += 1
            errors.append(f"{f.name}: {type(exc).__name__}: {exc}")
            if cached:
                try:
                    still_usable = unchanged and same_stat(cached, fingerprint(path))
                except OSError:
                    still_usable = False
                if not still_usable:
                    mark_offline(project_id, [path])
    if synchronize and not errors:
        root = Path(canonical_path(scan_dir))
        missing = [p for p in cached_by_path if Path(p).is_relative_to(root) and p not in observed]
        mark_offline(project_id, missing)
    status = "failure" if not indexed else ("partial_success" if errors or warnings else "success")
    message = (f"Ingest {status}: {indexed} clip(s) indexed; {reused} reused; "
               f"{unreadable} failed; {event_count} active event(s) in this import. "
               "Existing IDs and unrelated catalogue entries are retained.")
    return IngestResult(status=status, project_id=project_id,
        indexed_count=indexed, reused_count=reused, unreadable_count=unreadable,
        event_count=event_count, message=message, warnings=warnings, errors=errors,
        next_offset=(batch_offset + len(found)) if truncated and batch_offset is not None else None,
        truncated=truncated)
