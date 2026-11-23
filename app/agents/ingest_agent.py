"""Ingest Agent — footage ingestion and metadata extraction."""

import json
import os
from datetime import datetime
from pathlib import Path

from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState
from sqlalchemy import text as _sql

from app.config import settings
from app.models.state import ProductionState
from app.models.schemas import IngestResult
from app.services.openai_service import llm, embed_texts
from app.services.database_service import (
    db, get_catalogued_paths, get_catalogued_shots, get_catalogued_events,
    publish_clip, get_analysis_steps, project_ingest_lock, mark_offline,
)
from app.services.media_identity import canonical_path, fingerprint, same_stat, content_hash
from app.services.ffmpeg_service import (
    generate_proxy, get_video_duration, check_ffmpeg_installed,
    check_ffprobe_installed, probe_video_metadata, detect_scene_cuts,
    choose_frame_count,
    extract_scene_representative_frames_b64, extract_sample_frames_b64,
    build_event_windows, extract_frames_in_window_b64,
)
from app.services.vision_service import analyze_frames, analyze_event
from app.utils.logger import get_logger

logger = get_logger("ingest_agent")


# Used by the UI to gate later stages.
_LAST_INGEST_RESULT: IngestResult | None = None


def _record_ingest_result(result: IngestResult) -> IngestResult:
    global _LAST_INGEST_RESULT
    _LAST_INGEST_RESULT = result
    return result


def reset_last_ingest_result() -> None:
    """Clear the previous ingest result before a new run."""
    global _LAST_INGEST_RESULT
    _LAST_INGEST_RESULT = None


def get_last_ingest_result() -> IngestResult | None:
    """Return the most recent ingest result."""
    return _LAST_INGEST_RESULT


def _scan_dir(directory, state) -> Path:
    """Resolve the footage directory from the run state or default."""
    return Path(directory or (state or {}).get("footage_dir") or settings.RAW_FOOTAGE_DIR)


def _resolve_pid(state, project_id: int) -> int:
    """Resolve project ID, preferring injected state."""
    pid = (state or {}).get("project_id")
    if pid is None:
        pid = project_id
    try:
        return int(pid)
    except (TypeError, ValueError):
        return 1


# ── Tools: ingest / media preparation ─────────────────────────────────────────

# Scan safety limits
_MAX_SCAN_FILES = 2000    
_MAX_SCAN_DIRS = 50_000   
_SCAN_PREVIEW = 50        


@tool
def scan_footage_directory(directory: str = None) -> str:
    """Scan a directory for supported video files (bounded, single traversal).

    Walks the tree ONCE with ``os.walk`` and does NOT follow directory
    symlinks/junctions, so symlink cycles cannot cause unbounded recursion. The
    walk stops at a file/directory safety limit, and the returned summary is
    truncated, so a huge footage folder cannot blow up CPU/RAM or the agent's
    message history.

    Args:
        directory: Path to scan. Defaults to RAW_FOOTAGE_DIR.

    Returns:
        A bounded summary of the discovered video files.
    """
    scan_dir = Path(directory or settings.RAW_FOOTAGE_DIR)
    if not scan_dir.exists():
        return f"Directory not found: {scan_dir}"

    exts = {"." + e.strip().lower().lstrip(".") for e in settings.SUPPORTED_VIDEO_FORMATS}

    found_files: list[Path] = []
    dirs_seen = 0
    truncated = False

    for dirpath, _dirnames, filenames in os.walk(scan_dir, followlinks=False):
        dirs_seen += 1
        if dirs_seen > _MAX_SCAN_DIRS:
            truncated = True
            break
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                found_files.append(Path(dirpath) / name)
                if len(found_files) >= _MAX_SCAN_FILES:
                    truncated = True
                    break
        if truncated:
            break

    if not found_files:
        return f"No supported video files found in {scan_dir}. Supported formats: {sorted(exts)}"

    lines = []
    for f in found_files[:_SCAN_PREVIEW]:
        try:
            size_mb = f.stat().st_size / (1024 * 1024)
            lines.append(f"  {f.name} ({size_mb:.1f} MB) — {f.parent}")
        except OSError:
            lines.append(f"  {f.name} — {f.parent}")

    summary = f"Found {len(found_files)} video file(s)"
    if truncated:
        summary += " (scan stopped at a safety limit — there may be more)"
    summary += ":\n" + "\n".join(lines)
    if len(found_files) > _SCAN_PREVIEW:
        summary += f"\n  … and {len(found_files) - _SCAN_PREVIEW} more (not listed)."
    return summary


@tool
def verify_file_integrity(file_path: str) -> str:
    """Verify a video file can be read by FFmpeg (basic integrity check).

    Args:
        file_path: Path to the video file.

    Returns:
        Verification result with duration if valid.
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"

    duration = get_video_duration(file_path)
    if duration > 0:
        return f"✓ Valid: {path.name} (duration: {duration:.1f}s)"
    else:
        return f"✗ Invalid or unreadable: {path.name}"


@tool
def create_project_structure(project_name: str) -> str:
    """Create the standard MAPO project folder structure.

    Args:
        project_name: Name for the project directory.

    Returns:
        Confirmation of created directories.
    """
    base = settings.PROCESSED_OUTPUT_DIR / project_name
    subdirs = ["proxies", "frames", "metadata", "reports", "exports"]

    created = []
    for sub in subdirs:
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
        created.append(str(d))

    return f"Project structure created at {base}:" + "".join(f"  {d}" for d in created)


@tool
def generate_proxy_for_file(file_path: str) -> str:
    """Generate a low-resolution proxy for a video file.

    Args:
        file_path: Path to the original footage.

    Returns:
        Path to the generated proxy file.
    """
    if not check_ffmpeg_installed():
        return "Error: FFmpeg is not installed. Please install FFmpeg first."

    try:
        proxy_path = generate_proxy(file_path)
        return f"Proxy generated: {proxy_path}"
    except Exception as e:
        return f"Proxy generation failed: {e}"


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

def _find_video_files(scan_dir: Path) -> tuple[list[Path], bool]:
    """Find supported videos and report whether the scan was truncated."""
    exts = {"." + e.strip().lower().lstrip(".") for e in settings.SUPPORTED_VIDEO_FORMATS}
    found: list[Path] = []
    truncated = False
    def scan_error(error):
        raise error
    for dirpath, _dirnames, filenames in os.walk(scan_dir, followlinks=False, onerror=scan_error):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                found.append(Path(dirpath) / name)
                if len(found) > _MAX_INGEST_FILES:
                    truncated = True
                    break
        if truncated:
            break
    return found[:_MAX_INGEST_FILES], truncated


@tool
def detect_new_footage(directory: str = None, project_id: int = 1,
                       state: Annotated[dict, InjectedState] = None) -> str:
    """Detect footage on disk that is NOT yet in the project's catalogue.

    Read-only. Compares the video files in the directory against the catalogued
    file paths and reports which are new — useful before deciding to (re)ingest.

    Args:
        directory: Footage directory to check. Defaults to the run's footage directory.
        project_id: Project whose catalogue to compare against.
    """
    scan_dir = _scan_dir(directory, state)
    project_id = _resolve_pid(state, project_id)
    if not scan_dir.exists():
        return f"Directory not found: {scan_dir}"

    on_disk, truncated = _find_video_files(scan_dir)
    known = get_catalogued_paths(project_id)
    known = {canonical_path(p) for p in known}
    new_files = [f for f in on_disk if canonical_path(f) not in known]

    cap_note = (f"\n⚠ Only the first {_MAX_INGEST_FILES} files were scanned (per-run "
                "capacity limit) — there may be more.") if truncated else ""

    if not on_disk:
        return f"No supported video files found in {scan_dir}."
    if not new_files:
        return (f"All {len(on_disk)} file(s) in {scan_dir} are already catalogued for "
                f"project {project_id}. Nothing new to ingest.{cap_note}")

    preview = "\n".join(f"  • {f.name}" for f in new_files[:30])
    more = "" if len(new_files) <= 30 else f"\n  … and {len(new_files) - 30} more."
    return (f"{len(new_files)} new file(s) (of {len(on_disk)} on disk) not yet "
            f"catalogued for project {project_id}:\n{preview}{more}{cap_note}")


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


@tool
def ingest_footage(directory: str = None, project_id: int = 1,
                   analyze_content: bool = True, generate_proxies: bool = False,
                   analyze_events: bool = True, force_reanalyze: bool = False,
                   synchronize: bool = False, verify_content: bool = False,
                   state: Annotated[dict, InjectedState] = None) -> str:
    """Incrementally index videos, preserving existing clip and event identities.

    Args:
        directory: Folder to import; defaults to the directory in injected state.
        project_id: Project fallback; injected project state is authoritative.
        analyze_content: Fill missing visual tags and embeddings.
        generate_proxies: Generate missing editing proxies.
        analyze_events: Fill missing temporal windows/tags when content analysis is enabled.
        force_reanalyze: Reanalyse explicitly; use only at the user's request.
        synchronize: Mark missing files under this root offline after a COMPLETE scan.
            Default is append/update. Never deletes clips or their history.
        verify_content: Hash even unchanged-size/mtime files for strict verification.
        state: Injected project and footage directory.

    Returns:
        Structured status is recorded for the UI; text reports indexed/reused/failed files.
    """
    pid = _resolve_pid(state, project_id)
    with project_ingest_lock(pid):
        return _ingest_directory(_scan_dir(directory, state), pid, analyze_content,
                                 generate_proxies, analyze_events, force_reanalyze,
                                 synchronize, verify_content)


def _ingest_directory(scan_dir, project_id, analyze_content, generate_proxies,
                      analyze_events, force_reanalyze, synchronize, verify_content):
    def fail(message, truncated=False):
        _record_ingest_result(IngestResult(status="failure", project_id=project_id,
            truncated=truncated, message=message, errors=[message]))
        return message
    if not scan_dir.is_dir():
        return fail(f"Directory not found: {scan_dir}")
    try:
        found, truncated = _find_video_files(scan_dir)
    except OSError as exc:
        return fail(f"Scan could not finish ({type(exc).__name__}); existing catalogue retained.")
    if truncated:
        return fail(f"Scan exceeds the {_MAX_INGEST_FILES}-file limit; import a smaller batch. "
                    "Existing catalogue was not changed.", truncated=True)
    if not found:
        return fail(f"No supported video files found in {scan_dir}; existing catalogue retained.")
    cached_by_path = {canonical_path(k): v for k, v in get_catalogued_shots(project_id).items()}
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
            previous_steps = get_analysis_steps(cached["clip_id"], cached["source_revision"]) if cached and unchanged else {}
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
    _record_ingest_result(IngestResult(status=status, project_id=project_id,
        indexed_count=indexed, reused_count=reused, unreadable_count=unreadable,
        event_count=event_count, message=message, warnings=warnings, errors=errors))
    return message

# ── Tools: shot classification / catalogue queries ─────────────────────────────


@tool
def get_shots_by_type(shot_type: str) -> str:
    """Retrieve all shots of a specific type from the database.

    Shot types: wide_shot, close_up, establishing, medium_shot,
    over_shoulder, pov, drone, gimbal, handheld

    Args:
        shot_type: Type of shot to search for.
    """
    return db.run(
        _sql("""SELECT s.clip_id AS shot_id, s.file_path, s.shot_type,
                       s.duration_seconds, s.keywords
                FROM clips s
                WHERE s.shot_type LIKE :stype
                ORDER BY s.clip_id"""),
        parameters={"stype": f"%{shot_type}%"},
        include_columns=True,
    )


@tool
def classify_shot_attributes(shot_id: int) -> str:
    """Get full metadata for a specific shot.

    Args:
        shot_id: Unique shot identifier.
    """
    return db.run(
        _sql("SELECT * FROM clips WHERE clip_id = :sid"),
        parameters={"sid": shot_id},
        include_columns=True,
    )


# ── Tool: metadata export ─────────────────────────────────────────────────────────


@tool
def export_metadata_json(project_id: int = 1) -> str:
    """Export all project metadata as a JSON file (the "store in JSON" deliverable).

    Args:
        project_id: Project identifier.

    Returns:
        Path to the exported JSON file.
    """
    shots = db.run(
        _sql("SELECT * FROM clips WHERE project_id = :pid"),
        parameters={"pid": project_id},
        include_columns=True,
    )

    output_dir = settings.PROCESSED_OUTPUT_DIR / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / f"metadata_export_{project_id}.json"

    export_data = {
        "project_id": project_id,
        "exported_at": datetime.now().isoformat(),
        "shots_raw": shots,
    }

    with open(export_path, "w") as f:
        json.dump(export_data, f, indent=2)

    return f"Metadata exported: {export_path}"


# ── Agent Assembly ───────────────────────────────────────────────────────────

ingest_tools = [
    scan_footage_directory,
    detect_new_footage,
    ingest_footage,
    verify_file_integrity,
    create_project_structure,
    generate_proxy_for_file,
    get_shots_by_type,
    classify_shot_attributes,
    export_metadata_json,
]

llm_with_ingest = llm.bind_tools(ingest_tools)
ingest_tool_node = ToolNode(ingest_tools)

INGEST_PROMPT = """You are the INGEST AGENT in MAPO.
Your role is to build the searchable catalogue used by all later stages.
Ingest is the first indexing stage; Search and Selection use its output.

RUN INGEST:
When asked to ingest or index a project, call `ingest_footage` on the footage directory.
The tool:
- verifies media readability
- extracts technical metadata
- detects scene cuts with FFmpeg
- uses GPT-5.5 Vision to tag sampled frames
- extracts temporal events for moment-level search
- creates semantic embeddings for catalogue search
The tool rebuilds the project's catalogue from the current footage.

You may call `detect_new_footage` before ingestion or `export_metadata_json` afterwards.

INCREMENTAL INGEST:
Unchanged files are reused automatically based on their cached metadata.
Only new or modified files are analysed.
Use `force_reanalyze=True` only when the user explicitly requests a full re-analysis.

GROUNDING:
Report only facts returned by the tools or observed by the vision model.
Do not identify people.
Treat unavailable or unclassified fields as unknown.
Never invent filenames, metadata, keywords, counts, or scores.

OUTPUT:
Relay the ingest summary, including clips indexed, orientation breakdown,
vision-tagged clips, and search readiness.
Do not make editorial judgements.

Prior user preferences: {memory}"""


def ingest_assistant(state: ProductionState, config: RunnableConfig):
    """Ingest Agent reasoning node."""
    memory = state.get("loaded_preferences", "None")
    prompt = INGEST_PROMPT.format(memory=memory)
    response = llm_with_ingest.invoke(
        [SystemMessage(prompt)] + state["messages"]
    )
    return {"messages": [response]}


def should_continue_ingest(state: ProductionState, config: RunnableConfig) -> str:
    """Router for the Ingest Agent ReAct loop."""
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "end"
    return "continue"
