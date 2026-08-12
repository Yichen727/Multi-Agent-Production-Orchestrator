"""Ingest Agent — footage ingestion and metadata extraction.

The FIRST stage of the MAPO pipeline. The Ingest Agent builds the searchable
knowledge base that Search and Selection depend on:

    - Scan the footage directory for supported media
    - Verify file integrity and prepare proxies / project structure
    - Extract per-shot metadata (shot type, duration, dimensions)
    - Apply semantic tags / keywords and detect shot/scene cuts (FFmpeg)
    - Persist the catalogue to the metadata database (SQLite) and export JSON

It owns all media-ingest and shot-cataloguing tools directly — the single source
of truth for that responsibility. Search and Selection stay disabled until the
Ingest Agent has completed (enforced by the UI).
"""

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
    db, replace_project_shots, get_catalogued_paths, get_catalogued_shots,
    replace_project_events, get_catalogued_events,
)
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


# ── Structured ingest outcome (audit C-08) ─────────────────────────────────────
#
# ingest_footage still returns a human-readable summary for the LLM, but it ALSO records
# a structured IngestResult here. The UI reads it (in-process, single-user) to gate the
# later phases on real success + indexed_count > 0, instead of trusting that the agent
# returned some text — an error string like "Directory not found" is non-empty too.
_LAST_INGEST_RESULT: IngestResult | None = None


def _record_ingest_result(result: IngestResult) -> IngestResult:
    global _LAST_INGEST_RESULT
    _LAST_INGEST_RESULT = result
    return result


def reset_last_ingest_result() -> None:
    """Clear the recorded result. The UI calls this BEFORE a run so a stale success from
    an earlier ingest can't unlock the pipeline when this run never reached the tool."""
    global _LAST_INGEST_RESULT
    _LAST_INGEST_RESULT = None


def get_last_ingest_result() -> IngestResult | None:
    """Structured result of the most recent ingest_footage run this session, or None."""
    return _LAST_INGEST_RESULT


def _scan_dir(directory, state) -> Path:
    """Resolve the footage directory: explicit arg → injected state → global default.

    Reading ``state['footage_dir']`` (seeded per-run by the orchestrator) instead of a
    mutated global ``settings.RAW_FOOTAGE_DIR`` keeps concurrent sessions isolated
    (audit H-07). The global is only the last-resort fallback.
    """
    return Path(directory or (state or {}).get("footage_dir") or settings.RAW_FOOTAGE_DIR)


def _resolve_pid(state, project_id: int) -> int:
    """Project id: injected graph state wins over the model-supplied arg, else default 1."""
    pid = (state or {}).get("project_id")
    if pid is None:
        pid = project_id
    try:
        return int(pid)
    except (TypeError, ValueError):
        return 1


# ── Tools: ingest / media preparation ─────────────────────────────────────────


# Hard bounds for the directory scan. They guarantee the scan always terminates in
# bounded time and memory — even for a huge tree or a symlink/junction cycle, which
# would otherwise make a symlink-following rglob recurse forever and peg the CPU.
_MAX_SCAN_FILES = 2000     # stop collecting once this many matches are found
_MAX_SCAN_DIRS = 50_000    # stop walking once this many directories are visited
_SCAN_PREVIEW = 50         # cap the number of files listed in the returned text


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

    # followlinks=False (the default, made explicit): never descend into symlinks.
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


# Probing + vision per file is the slow part of ingest; cap it so a huge folder
# can't hang the agent or run up an unbounded API bill.
_MAX_INGEST_FILES = 200

# Temporal event extraction bounds. Each event window is a separate GPT-5.4 call, so the
# per-clip window count is budgeted — but the budget SCALES WITH DURATION instead of being
# a flat 12: a fixed cap forced a long clip's windows to be coarsened far more than a short
# one's, for no reason other than the constant. Windows over budget are merged (never
# dropped) by build_event_windows, so coverage is always complete; the budget only sets
# how fine the granularity gets.
_MIN_EVENTS_PER_CLIP = 12        # floor, so even a short clip gets a usable event layer
_MAX_EVENTS_PER_CLIP = 40        # ceiling, so one long clip can't dominate the API bill
_SECONDS_PER_EVENT = 6.0         # target event granularity
_FRAMES_PER_EVENT = 3


def _event_budget(duration_seconds: float | None) -> int:
    """How many event windows this clip's duration justifies (clamped to the bounds)."""
    if not duration_seconds or duration_seconds <= 0:
        return _MIN_EVENTS_PER_CLIP
    want = int(round(float(duration_seconds) / _SECONDS_PER_EVENT))
    return max(_MIN_EVENTS_PER_CLIP, min(want, _MAX_EVENTS_PER_CLIP))


# Columns copied verbatim when REUSING a previously-analysed clip's events (write
# contract for replace_project_events, minus the ids it resolves itself).
_EVENT_REUSE_COLUMNS = (
    "file_path", "event_order", "start_seconds", "end_seconds", "duration_seconds",
    "action", "state_change", "subjects", "keywords", "embedding",
)


def _event_semantic_text(action: str | None, state_change: str | None,
                         keywords: str) -> str:
    """Assemble the per-event text that gets embedded for moment-level search.

    Combines the action sentence, the state change, and the event keywords. Returns ""
    when there is nothing real to embed so no vector is stored (never an empty string).
    """
    parts = []
    if action:
        parts.append(action)
    if state_change:
        parts.append(state_change)
    if keywords:
        parts.append(keywords.replace(",", " "))
    return ". ".join(parts).strip()


def _build_clip_events(path: str, duration: float, cut_times: list[float]) -> list[dict]:
    """Extract ordered temporal events for one clip (the 'what happens' layer).

    Segments the clip into event windows (scene cuts + long-scene sub-division, budgeted
    by :func:`_event_budget`), sends several ordered frames per window to GPT-5.4, and
    returns event dicts ready for ``replace_project_events`` — each with a real start/end,
    an action description, and its own semantic embedding. Only windows the model actually
    described are kept; a window that fails analysis is skipped rather than stored as a
    fabricated event.

    All of the clip's event embeddings are fetched in ONE batched call (the embeddings API
    is batch-native) rather than one call per event.
    """
    windows = build_event_windows(duration, cut_times,
                                  max_events=_event_budget(duration))
    if not windows:
        return []
    events: list[dict] = []
    pending: list[tuple[int, str]] = []   # (index in events, text to embed)
    order = 0
    for (start, end) in windows:
        frames = extract_frames_in_window_b64(path, start, end, count=_FRAMES_PER_EVENT)
        if not frames:
            continue
        tags = analyze_event(frames, start_seconds=start, end_seconds=end,
                             clip_duration=duration)
        if tags is None:
            continue
        kws = [k for k in (s.strip().lower() for s in tags.keywords) if k and k != "unknown"]
        keywords = ",".join(dict.fromkeys(kws))
        action = (tags.action or "").strip()
        state_change = (tags.state_change or "").strip()
        # Skip an empty event — the model saw the frames but reported nothing usable.
        if not action and not keywords:
            continue
        order += 1
        events.append({
            "file_path": path,
            "event_order": order,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(max(0.0, end - start), 3),
            "action": action or None,
            "state_change": state_change or None,
            "subjects": tags.subjects or [],
            "keywords": keywords,
            "embedding": None,
        })
        sem = _event_semantic_text(action, state_change, keywords)
        if sem:
            pending.append((len(events) - 1, sem))

    # One embedding call for the whole clip's events.
    if pending:
        vecs = embed_texts([text for _, text in pending])
        if vecs and len(vecs) == len(pending):
            for (idx, _), vec in zip(pending, vecs):
                events[idx]["embedding"] = json.dumps(vec)
        elif vecs:
            # Never risk pairing a vector with the wrong event — store none instead.
            logger.error(f"Event embedding count mismatch for {path} "
                         f"({len(pending)} text(s) → {len(vecs)} vector(s)); "
                         "storing no event vectors for this clip.")
    return events


def _find_video_files(scan_dir: Path) -> tuple[list[Path], bool]:
    """Bounded, single-pass, symlink-safe walk for supported video files.

    Returns ``(files, truncated)``. ``truncated`` is True when the scan hit
    ``_MAX_INGEST_FILES``. The caller MUST NOT perform the destructive catalogue rewrite
    when truncated (audit C-07): ``replace_project_shots`` deletes the project's rows and
    writes only what it is given, so writing a silently-capped first-N set would erase
    the rest of the catalogue and report success.
    """
    exts = {"." + e.strip().lower().lstrip(".") for e in settings.SUPPORTED_VIDEO_FORMATS}
    found: list[Path] = []
    truncated = False
    for dirpath, _dirnames, filenames in os.walk(scan_dir, followlinks=False):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                found.append(Path(dirpath) / name)
                if len(found) >= _MAX_INGEST_FILES:
                    truncated = True
                    break
        if truncated:
            break
    return found, truncated


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
    new_files = [f for f in on_disk if str(f) not in known]

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
    """Assemble the text that gets embedded for semantic search.

    Combines the vision description, keyword tags, and mood into one editor-facing
    sentence. Returns "" when there is nothing real to embed, so the caller stores no
    vector rather than embedding an empty/fabricated string.
    """
    parts = []
    if description:
        parts.append(description)
    if keywords:
        parts.append(keywords.replace(",", " "))
    if mood:
        parts.append(f"{mood} mood")
    return ". ".join(parts).strip()


def _file_fingerprint(path: Path) -> tuple[float | None, int | None]:
    """Return (mtime, size) for a file, or (None, None) if it can't be stat'd."""
    try:
        st = path.stat()
        return st.st_mtime, st.st_size
    except OSError:
        return None, None


def _can_reuse(cached: dict, mtime: float | None, size: int | None) -> bool:
    """Decide whether a catalogued row can be reused instead of re-analysed.

    Reuse when the file's content looks unchanged: size matches and mtime is within
    1s of what was stored. If the stored fingerprint is missing (older ingest), we
    still reuse on a path match — the user asked to skip re-analysis for already
    analysed paths — unless the size we can read now clearly differs.
    """
    c_size = cached.get("source_size")
    c_mtime = cached.get("source_mtime")
    if c_size is not None and size is not None and c_size != size:
        return False
    if c_mtime is not None and mtime is not None and abs(c_mtime - mtime) > 1.0:
        return False
    return True


@tool
def ingest_footage(directory: str = None, project_id: int = 1,
                   analyze_content: bool = True, generate_proxies: bool = False,
                   analyze_events: bool = True, force_reanalyze: bool = False,
                   state: Annotated[dict, InjectedState] = None) -> str:
    """Run the ingest pipeline on a directory and (re)build the catalogue.

    INCREMENTAL BY DEFAULT: a file whose path is already catalogued and whose
    content is unchanged (same size + modification time) is REUSED as-is — its
    ffprobe metadata and GPT-5.4 Vision tags are kept, so no FFmpeg or LLM work runs
    for it again. Only new or modified files go through full analysis. This keeps
    repeated ingests fast and avoids re-spending Vision API calls.

    For each file needing analysis it performs, in order:
      1. VERIFY the media is readable (ffprobe).
      2. TECHNICAL METADATA — true dimensions, display orientation
         (portrait/landscape/square, rotation-aware), fps, codec, audio presence,
         duration.
      3. SHOT/SCENE DETECTION — FFmpeg scene-change cut count.
      4. VISION TAGGING (if analyze_content) — GPT-5.4 watches sampled frames and
         returns a description, shot type, objects, searchable keywords, and an
         approximate people count.
      5. TEMPORAL EVENT EXTRACTION (if analyze_events) — the clip is split into event
         windows (scene cuts + long-scene sub-division) and GPT-5.4 describes WHAT
         HAPPENS in each (action + change), stored as ordered ``clip_events`` with real
         start/end timecodes and their own embeddings for moment-level search.
      6. (optional) PROXY generation for each clip.
    The merged results (reused + freshly analysed) REPLACE the project's catalogue,
    so files deleted from disk drop out too.

    Only measured / model-observed facts are stored. When a step cannot run (no
    FFmpeg, no API key, unreadable file), the corresponding fields are left
    unclassified rather than guessed.

    Args:
        directory: Footage directory to ingest. Defaults to the run's footage directory.
        project_id: Project to (re)build the catalogue for.
        analyze_content: Run GPT-5.4 Vision tagging on sampled frames (default True).
        generate_proxies: Also generate a low-res proxy per clip (slow; default False).
        analyze_events: Extract temporal 'what happens' events per clip (default True;
            requires analyze_content / an API key — skipped otherwise).
        force_reanalyze: Ignore the cache and re-analyse every file (default False).

    Returns:
        A summary: clips indexed, how many were reused vs newly analysed, the
        orientation breakdown, and how many were vision-tagged.
    """
    scan_dir = _scan_dir(directory, state)
    project_id = _resolve_pid(state, project_id)
    if not scan_dir.exists():
        msg = f"Directory not found: {scan_dir}"
        _record_ingest_result(IngestResult(status="failure", project_id=project_id,
                                            message=msg, errors=[msg]))
        return msg
    if not check_ffprobe_installed():
        msg = ("Error: ffprobe (FFmpeg) is not on PATH, so real metadata cannot be "
               "probed. Install FFmpeg and try again — nothing was catalogued.")
        _record_ingest_result(IngestResult(status="failure", project_id=project_id,
                                            message=msg, errors=["ffprobe not on PATH"]))
        return msg

    found, truncated = _find_video_files(scan_dir)
    if not found:
        exts = sorted("." + e.strip().lower().lstrip(".") for e in settings.SUPPORTED_VIDEO_FORMATS)
        msg = f"No supported video files found in {scan_dir}. Supported: {exts}"
        _record_ingest_result(IngestResult(status="failure", project_id=project_id,
                                            message=msg, errors=["no supported files"]))
        return msg
    if truncated:
        # C-07: refuse rather than delete the catalogue and keep only the first N files.
        msg = (f"Refusing to ingest: {scan_dir} holds more than {_MAX_INGEST_FILES} "
               "supported video files, which exceeds this build's per-run capacity. "
               "Ingesting anyway would DELETE the existing catalogue and keep only the "
               f"first {_MAX_INGEST_FILES}, so NOTHING was changed. Split the footage "
               "into smaller batches (or raise the limit) and re-run. No rows were "
               "written or deleted.")
        _record_ingest_result(IngestResult(
            status="failure", project_id=project_id, truncated=True, message=msg,
            errors=[f"scan exceeded the {_MAX_INGEST_FILES}-file per-run cap"]))
        return msg

    cache = {} if force_reanalyze else get_catalogued_shots(project_id)
    events_cache = {} if force_reanalyze else get_catalogued_events(project_id)
    do_vision = analyze_content and bool(settings.OPENAI_API_KEY)
    do_events = do_vision and analyze_events
    all_events: list[dict] = []      # temporal events across all clips (reused + fresh)
    clips_with_events = 0            # clips that carry at least one event
    rows: list[dict] = []
    orientation_counts = {"portrait": 0, "landscape": 0, "square": 0, "unknown": 0}
    unreadable = 0
    tagged = 0
    reused = 0
    proxies = 0
    scene_sampled = 0        # clips whose frames came from scene midpoints
    even_sampled = 0         # clips that fell back to adaptive even sampling
    frames_sent = 0          # total frames sent to the vision model
    scene_incomplete = 0     # clips whose scene detection timed out (partial cut list)

    for f in found:
        path = str(f)
        mtime, size = _file_fingerprint(f)

        # ── Reuse path: skip all probing / LLM work for unchanged, known files ──
        cached = cache.get(path)
        if cached is not None and _can_reuse(cached, mtime, size):
            row = {col: cached.get(col) for col in (
                "file_path", "shot_type",
                "duration_seconds",
                "keywords", "width", "height", "orientation", "fps",
                "codec", "has_audio", "scene_count", "description", "people_count",
                "mood", "embedding",
                "audio_channels", "audio_sample_rate", "audio_bit_depth",
            )}
            # Refresh the fingerprint in case the previous ingest stored none.
            row["source_mtime"] = mtime if mtime is not None else cached.get("source_mtime")
            row["source_size"] = size if size is not None else cached.get("source_size")
            rows.append(row)
            reused += 1
            orientation_counts[row.get("orientation") or "unknown"] = (
                orientation_counts.get(row.get("orientation") or "unknown", 0) + 1
            )
            # Carry this clip's previously-extracted events forward unchanged (no new
            # per-window GPT-5.4 calls), so a re-ingest keeps the temporal layer intact.
            cached_events = events_cache.get(path) or []
            if cached_events:
                for ev in cached_events:
                    all_events.append({c: ev.get(c) for c in _EVENT_REUSE_COLUMNS})
                clips_with_events += 1
            continue

        # ── Analyse path: new or changed file ──────────────────────────────────
        meta = probe_video_metadata(path)
        if not meta["ok"]:
            unreadable += 1
        orientation_counts[meta["orientation"]] = orientation_counts.get(meta["orientation"], 0) + 1

        # Shot/scene cut detection (only meaningful for readable clips) — ONE decode
        # pass. The previous version re-ran the entire detection via detect_scene_count()
        # whenever no cuts were found, doubling the cost of exactly the clips (single
        # continuous shots, and anything that timed out) where it added nothing. We keep
        # the cut TIMESTAMPS so vision sampling can hit one frame per scene, and the
        # timeout is now sized from the real duration.
        scene_info = (detect_scene_cuts(path, duration_seconds=meta["duration_seconds"])
                      if meta["ok"] else None)
        cut_times = scene_info["cut_times"] if scene_info else []
        scene_count = scene_info["scene_count"] if scene_info else 0
        if scene_info and scene_info["ran"] and not scene_info["complete"]:
            # A partial scan is kept (the cuts it found are real), but the tail has no
            # detected boundaries — say so instead of passing it off as a full scan.
            scene_incomplete += 1
            covered = (f"up to {max(cut_times):.0f}s of {meta['duration_seconds']:.0f}s"
                       if cut_times else "no cuts at all")
            logger.warning(f"{f.name}: scene detection ended early, covering {covered}; "
                           f"kept {len(cut_times)} cut(s). Event boundaries past that point "
                           "fall back to even sub-division.")

        # Vision tagging from sampled frames.
        shot_type = "unclassified"
        keywords = ""
        description = None
        people_count = None
        mood = None
        if do_vision and meta["ok"]:
            duration = meta["duration_seconds"]
            # Adaptive frame budget: more frames for longer / multi-scene clips.
            frame_count = choose_frame_count(duration, scene_count)
            # Prefer one representative frame per detected scene (midpoint).
            frames, sampled_timestamps = extract_scene_representative_frames_b64(
                path, duration, cut_times, max_frames=frame_count)
            sampling_strategy = "scene-midpoint" if cut_times else "adaptive-even"
            if not frames:
                # Scene sampling produced nothing usable → adaptive even sampling.
                frames = extract_sample_frames_b64(path, duration, count=frame_count)
                sampled_timestamps = []
                sampling_strategy = "adaptive-even"
            if frames:
                frames_sent += len(frames)
                if sampling_strategy == "scene-midpoint":
                    scene_sampled += 1
                else:
                    even_sampled += 1
            tags = analyze_frames(
                frames, duration_seconds=duration, scene_count=scene_count or None,
                sampled_timestamps=sampled_timestamps or None,
                sampling_strategy=sampling_strategy,
            ) if frames else None
            if tags is not None:
                tagged += 1
                shot_type = tags.shot_type or "unclassified"
                kw = list(dict.fromkeys([*tags.keywords, *tags.objects, tags.setting]))
                keywords = ",".join(k for k in (s.strip().lower() for s in kw) if k and k != "unknown")
                description = tags.scene_description or None
                people_count = tags.people_count
                # Semantic dimension — store only what the model actually reported.
                mood = tags.mood if tags.mood != "unknown" else None

        # Build a semantic embedding from what the vision model saw. This is the
        # vector-search layer: only embed when there is real observed content, so we
        # never index a fabricated / empty string.
        embedding = None
        semantic_text = _semantic_text(description, keywords, mood)
        if semantic_text:
            vecs = embed_texts([semantic_text])
            if vecs:
                embedding = json.dumps(vecs[0])

        if generate_proxies and meta["ok"] and check_ffmpeg_installed():
            try:
                generate_proxy(path)
                proxies += 1
            except Exception as e:
                logger.error(f"Proxy generation failed for {f.name}: {e}")

        rows.append({
            "file_path": path,
            "shot_type": shot_type,
            "duration_seconds": meta["duration_seconds"],
            "keywords": keywords,
            "width": meta["width"],
            "height": meta["height"],
            "orientation": meta["orientation"],
            "fps": meta["fps"],
            "codec": meta["codec"],
            "has_audio": 1 if meta["has_audio"] else 0,
            "scene_count": scene_count,
            "description": description,
            "people_count": people_count,
            "mood": mood,
            "embedding": embedding,
            "source_mtime": mtime,
            "source_size": size,
            # Real source audio params for a faithful Premiere export (audit H-10).
            "audio_channels": meta.get("audio_channels"),
            "audio_sample_rate": meta.get("audio_sample_rate"),
            "audio_bit_depth": meta.get("audio_bit_depth"),
        })

        # Temporal event extraction — the 'what happens' layer. Runs only on readable
        # clips with a known duration; degrades to no events (never fabricated) otherwise.
        if do_events and meta["ok"] and meta["duration_seconds"] > 0:
            clip_events = _build_clip_events(path, meta["duration_seconds"], cut_times)
            if clip_events:
                all_events.extend(clip_events)
                clips_with_events += 1

    count = replace_project_shots(project_id, rows)
    # Persist the temporal events AFTER the shots exist (they resolve their shot_id from
    # the freshly written shots; replace_project_shots cascade-cleared the old events).
    events_indexed = replace_project_events(project_id, all_events)
    analysed = count - reused
    logger.info(f"Ingested {count} file(s) into project {project_id} from {scan_dir} "
                f"({reused} reused, {analysed} analysed, {unreadable} unreadable)")

    # Status (audit C-08 / L-03): a run with unreadable files is NOT a clean success, and
    # a run where EVERY file was unreadable (nothing usable indexed) is a failure.
    if count == 0:
        status = "failure"
    elif unreadable == 0:
        status = "success"
    elif unreadable < count:
        status = "partial_success"
    else:
        status = "failure"

    warnings: list[str] = []
    if unreadable:
        warnings.append(f"{unreadable} file(s) could not be probed and were stored as 'unknown'")
    if do_vision and tagged < analysed:
        warnings.append(f"{analysed - tagged} newly analysed clip(s) were not vision-tagged")
    if scene_incomplete:
        warnings.append(
            f"{scene_incomplete} clip(s) hit the scene-detection timeout — their cut list "
            "covers only the analysed head, so scene_count is a lower bound and later "
            "event boundaries are evenly sub-divided rather than cut-aligned"
        )

    breakdown = ", ".join(f"{k}: {v}" for k, v in orientation_counts.items() if v)
    headline = {
        "success": "Ingest completed successfully.",
        "partial_success": "Ingest completed with warnings (some files were unreadable).",
        "failure": "Ingest FAILED — no usable clips were indexed.",
    }[status]
    summary = (
        f"{headline} Indexed {count} clip(s) from {scan_dir} "
        f"into project {project_id} — {analysed} newly analysed, {reused} reused "
        f"from previous analysis.\n"
        f"Orientation — {breakdown}."
    )
    if do_vision:
        summary += f"\nVision-tagged {tagged} newly analysed clip(s) with GPT-5.4."
        summary += (
            f"\nFrame sampling — {scene_sampled} clip(s) via scene midpoints, "
            f"{even_sampled} via adaptive even sampling; "
            f"{frames_sent} frame(s) sent to the vision model."
        )
    if do_events:
        summary += (
            f"\nTemporal events — extracted {events_indexed} 'what happens' event(s) "
            f"across {clips_with_events} clip(s) for moment-level search."
        )
    elif analyze_content:
        summary += "\nVision tagging skipped (no OPENAI_API_KEY)."
    else:
        summary += "\nVision tagging disabled for this run."
    if generate_proxies:
        summary += f"\nGenerated {proxies} proxy file(s)."
    if unreadable:
        summary += f"\n({unreadable} file(s) could not be probed and were stored as 'unknown'.)"
    if scene_incomplete:
        summary += (f"\n({scene_incomplete} clip(s) hit the scene-detection timeout — the "
                    "cuts found were kept, but their scene count is a lower bound.)")
    if status != "failure":
        summary += "\nFootage indexed and ready for search."

    _record_ingest_result(IngestResult(
        status=status, project_id=project_id, indexed_count=count,
        reused_count=reused, unreadable_count=unreadable, event_count=events_indexed,
        truncated=False, message=headline, warnings=warnings,
    ))
    return summary


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
        _sql("""SELECT s.shot_id, s.file_path, s.shot_type,
                       s.duration_seconds, s.keywords
                FROM shots s
                WHERE s.shot_type LIKE :stype
                ORDER BY s.shot_id"""),
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
        _sql("SELECT * FROM shots WHERE shot_id = :sid"),
        parameters={"sid": shot_id},
        include_columns=True,
    )


# ── Tool: persist catalogue to JSON ───────────────────────────────────────────


@tool
def export_metadata_json(project_id: int = 1) -> str:
    """Export all project metadata as a JSON file (the "store in JSON" deliverable).

    Args:
        project_id: Project identifier.

    Returns:
        Path to the exported JSON file.
    """
    shots = db.run(
        _sql("SELECT * FROM shots WHERE project_id = :pid"),
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
    # Ingest / media prep
    scan_footage_directory,
    detect_new_footage,
    ingest_footage,
    verify_file_integrity,
    create_project_structure,
    generate_proxy_for_file,
    # Shot classification / catalogue
    get_shots_by_type,
    classify_shot_attributes,
    # Persistence
    export_metadata_json,
]

llm_with_ingest = llm.bind_tools(ingest_tools)
ingest_tool_node = ToolNode(ingest_tools)

INGEST_PROMPT = """You are the INGEST AGENT in the MAPO (Multi-Agent Production
Orchestrator) system. You build the searchable knowledge base that every later stage
depends on. You are the FIRST and ONLY indexing stage — Search and Selection run after
you, on the catalogue you produce.

RUN INGEST ANALYSIS — when asked to ingest/index a project, call the `ingest_footage`
tool on the footage directory. One call runs the whole pipeline per clip:
- verify the media is readable
- extract REAL technical metadata (dimensions, orientation, fps, codec, audio,
  duration)
- detect shot/scene cuts (FFmpeg)
- VISION TAGGING: GPT-5.4 watches sampled frames and returns a semantic description,
  shot type, objects, searchable keywords, an approximate people count, plus the mood
- SEMANTIC EMBEDDING: the description + keywords + mood are embedded into a vector so
  Search can do semantic recall, not just literal keyword matching
It then replaces the project's catalogue with those rows. You may also call
`detect_new_footage` first to see what is new, and `export_metadata_json` afterwards.

INCREMENTAL — ingest is cheap to re-run: files already analysed and unchanged are
REUSED automatically (no re-probing, no GPT-5.4 calls), and only new or modified files
are analysed. Do NOT pass force_reanalyze unless the user explicitly wants every clip
re-analysed from scratch.

CRITICAL — only report what the tools actually measured or the vision model actually
saw. The pipeline records true technical facts plus GPT-5.4's frame observations; it
does NOT identify *who* people are, and any step that cannot run leaves its fields
'unclassified'. NEVER invent file names, shot types, keywords, counts, or scores that
the tools did not return.

When ingestion is complete, relay the tool's summary: clips indexed, orientation
breakdown, how many were vision-tagged, and that the footage is ready for search. Do
not make editorial judgements; that is Selection's job.

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
