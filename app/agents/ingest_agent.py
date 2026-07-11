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

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.models.state import ProductionState
from app.services.openai_service import llm, embed_texts
from app.services.database_service import (
    db, replace_project_shots, get_catalogued_paths, get_catalogued_shots,
)
from app.services.ffmpeg_service import (
    generate_proxy, get_video_duration, check_ffmpeg_installed,
    check_ffprobe_installed, probe_video_metadata, detect_scene_count,
    extract_sample_frames_b64,
)
from app.services.vision_service import analyze_frames
from app.utils.logger import get_logger

logger = get_logger("ingest_agent")


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


def _find_video_files(scan_dir: Path) -> list[Path]:
    """Bounded, single-pass, symlink-safe walk for supported video files."""
    exts = {"." + e.strip().lower().lstrip(".") for e in settings.SUPPORTED_VIDEO_FORMATS}
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(scan_dir, followlinks=False):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                found.append(Path(dirpath) / name)
                if len(found) >= _MAX_INGEST_FILES:
                    return found
    return found


@tool
def detect_new_footage(directory: str = None, project_id: int = 1) -> str:
    """Detect footage on disk that is NOT yet in the project's catalogue.

    Read-only. Compares the video files in the directory against the catalogued
    file paths and reports which are new — useful before deciding to (re)ingest.

    Args:
        directory: Footage directory to check. Defaults to RAW_FOOTAGE_DIR.
        project_id: Project whose catalogue to compare against.
    """
    scan_dir = Path(directory or settings.RAW_FOOTAGE_DIR)
    if not scan_dir.exists():
        return f"Directory not found: {scan_dir}"

    on_disk = _find_video_files(scan_dir)
    known = get_catalogued_paths(project_id)
    new_files = [f for f in on_disk if str(f) not in known]

    if not on_disk:
        return f"No supported video files found in {scan_dir}."
    if not new_files:
        return (f"All {len(on_disk)} file(s) in {scan_dir} are already catalogued for "
                f"project {project_id}. Nothing new to ingest.")

    preview = "\n".join(f"  • {f.name}" for f in new_files[:30])
    more = "" if len(new_files) <= 30 else f"\n  … and {len(new_files) - 30} more."
    return (f"{len(new_files)} new file(s) (of {len(on_disk)} on disk) not yet "
            f"catalogued for project {project_id}:\n{preview}{more}")


def _semantic_text(description: str | None, keywords: str, mood: str | None,
                   lighting: str | None) -> str:
    """Assemble the text that gets embedded for semantic search.

    Combines the vision description, keyword tags, and mood/lighting into one
    editor-facing sentence. Returns "" when there is nothing real to embed, so the
    caller stores no vector rather than embedding an empty/fabricated string.
    """
    parts = []
    if description:
        parts.append(description)
    if keywords:
        parts.append(keywords.replace(",", " "))
    if mood:
        parts.append(f"{mood} mood")
    if lighting:
        parts.append(f"{lighting} lighting")
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
                   force_reanalyze: bool = False) -> str:
    """Run the ingest pipeline on a directory and (re)build the catalogue.

    INCREMENTAL BY DEFAULT: a file whose path is already catalogued and whose
    content is unchanged (same size + modification time) is REUSED as-is — its
    ffprobe metadata and GPT-4o Vision tags are kept, so no FFmpeg or LLM work runs
    for it again. Only new or modified files go through full analysis. This keeps
    repeated ingests fast and avoids re-spending Vision API calls.

    For each file needing analysis it performs, in order:
      1. VERIFY the media is readable (ffprobe).
      2. TECHNICAL METADATA — true dimensions, display orientation
         (portrait/landscape/square, rotation-aware), fps, codec, audio presence,
         duration.
      3. SHOT/SCENE DETECTION — FFmpeg scene-change cut count.
      4. VISION TAGGING (if analyze_content) — GPT-4o watches sampled frames and
         returns a description, shot type, objects, searchable keywords, and an
         approximate people count.
      5. (optional) PROXY generation for each clip.
    The merged results (reused + freshly analysed) REPLACE the project's catalogue,
    so files deleted from disk drop out too.

    Only measured / model-observed facts are stored. When a step cannot run (no
    FFmpeg, no API key, unreadable file), the corresponding fields are left
    unclassified rather than guessed.

    Args:
        directory: Footage directory to ingest. Defaults to RAW_FOOTAGE_DIR.
        project_id: Project to (re)build the catalogue for.
        analyze_content: Run GPT-4o Vision tagging on sampled frames (default True).
        generate_proxies: Also generate a low-res proxy per clip (slow; default False).
        force_reanalyze: Ignore the cache and re-analyse every file (default False).

    Returns:
        A summary: clips indexed, how many were reused vs newly analysed, the
        orientation breakdown, and how many were vision-tagged.
    """
    scan_dir = Path(directory or settings.RAW_FOOTAGE_DIR)
    if not scan_dir.exists():
        return f"Directory not found: {scan_dir}"
    if not check_ffprobe_installed():
        return ("Error: ffprobe (FFmpeg) is not on PATH, so real metadata cannot be "
                "probed. Install FFmpeg and try again — nothing was catalogued.")

    found = _find_video_files(scan_dir)
    if not found:
        exts = sorted("." + e.strip().lower().lstrip(".") for e in settings.SUPPORTED_VIDEO_FORMATS)
        return f"No supported video files found in {scan_dir}. Supported: {exts}"

    cache = {} if force_reanalyze else get_catalogued_shots(project_id)
    do_vision = analyze_content and bool(settings.OPENAI_API_KEY)
    rows: list[dict] = []
    orientation_counts = {"portrait": 0, "landscape": 0, "square": 0, "unknown": 0}
    unreadable = 0
    tagged = 0
    reused = 0
    proxies = 0

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
                "camera_motion", "lighting", "mood", "subject_position", "embedding",
            )}
            # Refresh the fingerprint in case the previous ingest stored none.
            row["source_mtime"] = mtime if mtime is not None else cached.get("source_mtime")
            row["source_size"] = size if size is not None else cached.get("source_size")
            rows.append(row)
            reused += 1
            orientation_counts[row.get("orientation") or "unknown"] = (
                orientation_counts.get(row.get("orientation") or "unknown", 0) + 1
            )
            continue

        # ── Analyse path: new or changed file ──────────────────────────────────
        meta = probe_video_metadata(path)
        if not meta["ok"]:
            unreadable += 1
        orientation_counts[meta["orientation"]] = orientation_counts.get(meta["orientation"], 0) + 1

        # Shot/scene cut detection (only meaningful for readable clips).
        scene_count = detect_scene_count(path) if meta["ok"] else 0

        # Vision tagging from sampled frames.
        shot_type = "unclassified"
        keywords = ""
        description = None
        people_count = None
        camera_motion = None
        lighting = None
        mood = None
        subject_position = None
        if do_vision and meta["ok"]:
            frames = extract_sample_frames_b64(path, meta["duration_seconds"])
            tags = analyze_frames(frames) if frames else None
            if tags is not None:
                tagged += 1
                shot_type = tags.shot_type or "unclassified"
                kw = list(dict.fromkeys([*tags.keywords, *tags.objects, tags.setting]))
                keywords = ",".join(k for k in (s.strip().lower() for s in kw) if k and k != "unknown")
                description = tags.scene_description or None
                people_count = tags.people_count
                # Semantic dimensions — store only what the model actually reported.
                camera_motion = tags.camera_motion if tags.camera_motion != "unknown" else None
                lighting = tags.lighting if tags.lighting != "unknown" else None
                mood = tags.mood if tags.mood != "unknown" else None
                subject_position = (
                    tags.subject_position if tags.subject_position != "unknown" else None
                )

        # Build a semantic embedding from what the vision model saw. This is the
        # vector-search layer: only embed when there is real observed content, so we
        # never index a fabricated / empty string.
        embedding = None
        semantic_text = _semantic_text(description, keywords, mood, lighting)
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
            "camera_motion": camera_motion,
            "lighting": lighting,
            "mood": mood,
            "subject_position": subject_position,
            "embedding": embedding,
            "source_mtime": mtime,
            "source_size": size,
        })

    count = replace_project_shots(project_id, rows)
    analysed = count - reused
    logger.info(f"Ingested {count} file(s) into project {project_id} from {scan_dir} "
                f"({reused} reused, {analysed} analysed)")

    breakdown = ", ".join(f"{k}: {v}" for k, v in orientation_counts.items() if v)
    summary = (
        f"Ingest completed successfully. Indexed {count} clip(s) from {scan_dir} "
        f"into project {project_id} — {analysed} newly analysed, {reused} reused "
        f"from previous analysis.\n"
        f"Orientation — {breakdown}."
    )
    if do_vision:
        summary += f"\nVision-tagged {tagged} newly analysed clip(s) with GPT-4o."
    elif analyze_content:
        summary += "\nVision tagging skipped (no OPENAI_API_KEY)."
    else:
        summary += "\nVision tagging disabled for this run."
    if generate_proxies:
        summary += f"\nGenerated {proxies} proxy file(s)."
    if unreadable:
        summary += f"\n({unreadable} file(s) could not be probed and were stored as 'unknown'.)"
    summary += "\nFootage indexed and ready for search."
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
        f"""SELECT s.shot_id, s.file_path, s.shot_type,
                   s.duration_seconds, s.keywords
            FROM shots s
            WHERE s.shot_type LIKE '%{shot_type}%'
            ORDER BY s.shot_id;""",
        include_columns=True,
    )


@tool
def classify_shot_attributes(shot_id: int) -> str:
    """Get full metadata for a specific shot.

    Args:
        shot_id: Unique shot identifier.
    """
    return db.run(
        f"""SELECT * FROM shots WHERE shot_id = {shot_id};""",
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
        f"SELECT * FROM shots WHERE project_id = {project_id};",
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
- VISION TAGGING: GPT-4o watches sampled frames and returns a semantic description,
  shot type, objects, searchable keywords, an approximate people count, plus editing-
  oriented dimensions (camera motion, lighting, mood, subject position)
- SEMANTIC EMBEDDING: the description + keywords + mood/lighting are embedded into a
  vector so Search can do semantic recall, not just literal keyword matching
It then replaces the project's catalogue with those rows. You may also call
`detect_new_footage` first to see what is new, and `export_metadata_json` afterwards.

INCREMENTAL — ingest is cheap to re-run: files already analysed and unchanged are
REUSED automatically (no re-probing, no GPT-4o calls), and only new or modified files
are analysed. Do NOT pass force_reanalyze unless the user explicitly wants every clip
re-analysed from scratch.

CRITICAL — only report what the tools actually measured or the vision model actually
saw. The pipeline records true technical facts plus GPT-4o's frame observations; it
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
