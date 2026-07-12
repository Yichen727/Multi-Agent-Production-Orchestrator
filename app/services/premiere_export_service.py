"""Premiere export service — compile an ordered edit timeline into a Premiere
Pro–importable project file.

This is the compiler behind the DELIVERY stage of the MAPO pipeline
(Ingest → Search → Selection → Delivery). It is a pure PROJECT COMPILER: it does
NOT rank, re-order, or make any creative decision. It takes the ordered clips the
Selection Agent laid out and turns them into:

    1. a structured JSON intermediate (``build_timeline``) — the neutral, tool-agnostic
       representation of the sequence, convertible into any NLE format later;
    2. a valid **FCP7 XML** document (``to_fcp7_xml``) in the ``xmeml`` v5 dialect, which
       Adobe Premiere Pro imports NATIVELY (File ▸ Import) with no plugin — so the
       timeline opens with correct media linking and track layout and needs no manual
       restructuring.

Design rules honoured here (see CLAUDE.md):
    - ORDER IS PRESERVED EXACTLY as provided. No re-ranking, ever.
    - NEVER fabricate. Every clip must carry a real ``file_path`` from ingestion; media
      are referenced by their absolute path. Attributes that were not measured
      (fps, has_audio) fall back to explicit, documented sequence defaults — never a
      guessed per-clip value dressed up as measured.
    - Track layout follows the delivery spec:
        V1  → main video footage
        A1  → ambient / original audio (present when the clip has audio)
        A2  → optional secondary / crowd audio (only when a 2nd audio stream exists)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("premiere_export_service")

# Sequence defaults used only when a value was never measured. Kept explicit so the
# output is honest about what is a real measurement vs. a documented fallback.
DEFAULT_FPS = 24.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

# Tolerance (seconds) when comparing an explicit out-point against the measured source
# duration — absorbs float rounding so a legitimately full-length trim isn't rejected.
_RANGE_EPS = 1e-3


class InvalidSegmentRange(ValueError):
    """An explicit in/out trim is illegal (zero/negative length or outside the source).

    Raised by :func:`build_timeline` rather than silently rewriting the interval to the
    full clip (audit C-05). Carries the 1-based ``order`` and the clip name so the caller
    (and the UI) can name exactly which segment failed and why.
    """

    def __init__(self, order: int, name: str, reason: str):
        self.order = order
        self.name = name
        self.reason = reason
        super().__init__(f"Segment #{order} ({name}): {reason}")


class MediaValidationError(ValueError):
    """Pre-export validation failed — a clip's media is missing/unreadable on disk (H-12).

    A catalogue row is not proof the file still exists, so compilation validates every
    clip against the filesystem first and refuses rather than emit a "successful" XML that
    points at media Premiere cannot relink. Carries the structured ``report``.
    """

    def __init__(self, report: dict):
        self.report = report
        problems = "; ".join(f"#{p['order']} {p['name']}: {p['error']}"
                             for p in report.get("problems", []))
        super().__init__(f"Media validation failed for {len(report.get('problems', []))} "
                         f"clip(s): {problems}")


def validate_media(clips: list[dict]) -> dict:
    """Check every clip's media exists and is a readable file (audit H-12).

    Returns a structured report ``{"ok": bool, "problems": [{order, name, file_path,
    error}]}``. Existence/readability only — the in/out RANGE against the measured source
    duration is validated separately by :func:`build_timeline` (`InvalidSegmentRange`).
    """
    problems: list[dict] = []
    for order, clip in enumerate(clips, start=1):
        fp = clip.get("file_path")
        name = Path(fp).name if fp else f"clip #{order}"
        if not fp:
            problems.append({"order": order, "name": name, "file_path": fp,
                             "error": "no file_path"})
            continue
        p = Path(fp)
        if not p.exists():
            problems.append({"order": order, "name": name, "file_path": str(fp),
                             "error": "file not found on disk"})
        elif not p.is_file():
            problems.append({"order": order, "name": name, "file_path": str(fp),
                             "error": "path is not a file"})
        elif not os.access(p, os.R_OK):
            problems.append({"order": order, "name": name, "file_path": str(fp),
                             "error": "file is not readable"})
    return {"ok": not problems, "problems": problems}


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically: write a temp file, then ``os.replace``.

    A crash mid-write leaves the temp file, never a truncated/half-written target (H-13).
    """
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# Documented audio fallbacks used only when ingest did not measure the real value.
DEFAULT_AUDIO_CHANNELS = 2
DEFAULT_AUDIO_SAMPLE_RATE = 48000
DEFAULT_AUDIO_BIT_DEPTH = 16


# ── JSON intermediate (the neutral, convertible representation) ─────────────────


def _seconds_to_frames(seconds: float, timebase: int) -> int:
    """Convert a duration in seconds to whole frames at ``timebase`` fps."""
    return int(round(max(0.0, float(seconds or 0.0)) * timebase))


def _timebase_ntsc(fps: float) -> tuple[int, bool]:
    """Map a real frame rate to an FCP7 ``(timebase:int, ntsc:bool)`` pair (audit H-11).

    FCP7/Premiere express fractional NTSC rates as an INTEGER timebase plus an ``ntsc``
    (1.001 pulldown) flag, so the exact rational is captured without frame drift:

        23.976 → (24, True)   29.97 → (30, True)   59.94 → (60, True)
        24 / 25 / 30 / 50 / 60 (exact integers) → ntsc = False

    A fractional (non-integer) rate is treated as NTSC; an exact integer rate is not.
    Falls back to the default fps for a missing/zero rate.
    """
    f = float(fps or 0.0)
    if f <= 0:
        f = DEFAULT_FPS
    timebase = max(1, int(round(f)))
    ntsc = abs(f - timebase) > 1e-3
    return timebase, ntsc


def build_timeline(
    clips: list[dict],
    *,
    sequence_name: str = "MAPO Edit",
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Build the structured JSON timeline from ordered, resolved clips.

    Ordering is taken EXACTLY as given — element 0 is the first clip on the timeline.
    Time is mapped sequentially: unless a clip supplies explicit ``in_point`` /
    ``out_point`` (seconds), the whole clip is used, and each clip starts where the
    previous one ended (a straight assembly cut, no gaps, no overlaps).

    Args:
        clips: Ordered list of resolved clip dicts. Each MUST have ``file_path``. Optional
            keys read here: ``duration_seconds``, ``fps``, ``has_audio``,
            ``audio_streams`` (int count of audio streams if probed), ``role``,
            ``shot_id``, ``in_point`` / ``out_point`` (seconds).
        sequence_name: Name of the resulting Premiere sequence.
        fps: Sequence frame rate (timebase). Falls back to the first clip's fps, then
            ``DEFAULT_FPS``.
        width / height: Sequence raster size; falls back to the first clip's, then the
            default 1920×1080.

    Returns:
        A dict with ``sequence`` metadata and an ordered ``clips`` list, each entry
        carrying both seconds and frame-accurate in/out and sequence start/end. This is
        the intermediate that ``to_fcp7_xml`` (or any other exporter) consumes.
    """
    first = clips[0] if clips else {}
    seq_fps = float(fps or first.get("fps") or DEFAULT_FPS) or DEFAULT_FPS
    timebase, seq_ntsc = _timebase_ntsc(seq_fps)
    seq_w = int(width or first.get("width") or DEFAULT_WIDTH)
    seq_h = int(height or first.get("height") or DEFAULT_HEIGHT)

    entries: list[dict] = []
    playhead_frames = 0  # cumulative sequence position, in frames

    for order, clip in enumerate(clips, start=1):
        file_path = clip.get("file_path")
        if not file_path:
            # Never emit a clip without a real media reference.
            raise ValueError(f"Clip #{order} has no file_path — refusing to fabricate a media reference.")

        duration = float(clip.get("duration_seconds") or 0.0)
        # Trim handling (audit C-05): the full clip is used ONLY when NEITHER an in- nor
        # an out-point is supplied. When the caller provides an explicit interval it is
        # VALIDATED, not silently repaired — an illegal range (zero/negative length, or
        # outside the measured source) raises InvalidSegmentRange naming the segment,
        # rather than being quietly rewritten back to the full/remaining source duration.
        has_in = clip.get("in_point") is not None
        has_out = clip.get("out_point") is not None
        name = Path(file_path).name
        if not has_in and not has_out:
            in_s, out_s = 0.0, duration
        else:
            in_s = float(clip.get("in_point") or 0.0)
            out_s = float(clip["out_point"]) if has_out else duration
            if in_s < 0:
                raise InvalidSegmentRange(order, name, f"in_point {in_s:g}s is negative")
            if duration > 0 and in_s > duration + _RANGE_EPS:
                raise InvalidSegmentRange(
                    order, name,
                    f"in_point {in_s:g}s is beyond the source duration {duration:g}s")
            if duration > 0 and out_s > duration + _RANGE_EPS:
                raise InvalidSegmentRange(
                    order, name,
                    f"out_point {out_s:g}s exceeds the source duration {duration:g}s")
            if out_s <= in_s + _RANGE_EPS:
                raise InvalidSegmentRange(
                    order, name,
                    f"out_point {out_s:g}s <= in_point {in_s:g}s (zero-length segment)")
        used = max(0.0, out_s - in_s)

        in_f = _seconds_to_frames(in_s, timebase)
        out_f = _seconds_to_frames(out_s, timebase)
        clip_frames = max(1, out_f - in_f)  # at least one frame so the clip is valid
        seq_start_f = playhead_frames
        seq_end_f = seq_start_f + clip_frames
        playhead_frames = seq_end_f

        has_audio = bool(clip.get("has_audio"))
        audio_streams = clip.get("audio_streams")
        if audio_streams is None:
            audio_streams = 1 if has_audio else 0

        # Real per-source media parameters (audit H-10) — the SOURCE <file> element uses
        # these, kept separate from the SEQUENCE raster/rate. Fall back to the sequence
        # size / documented audio defaults only when ingest did not measure a value.
        src_w = int(clip.get("width") or seq_w)
        src_h = int(clip.get("height") or seq_h)
        src_fps = float(clip.get("fps") or seq_fps)
        audio_channels = int(clip.get("audio_channels") or 0) or (
            DEFAULT_AUDIO_CHANNELS if has_audio else 0)
        audio_sample_rate = int(clip.get("audio_sample_rate") or 0) or (
            DEFAULT_AUDIO_SAMPLE_RATE if has_audio else 0)
        audio_bit_depth = int(clip.get("audio_bit_depth") or 0) or (
            DEFAULT_AUDIO_BIT_DEPTH if has_audio else 0)

        abs_path = _absolute_path(file_path)
        entries.append({
            "order": order,
            "role": clip.get("role") or "",
            "shot_id": clip.get("shot_id"),
            "name": Path(file_path).name,
            "file_path": str(file_path),
            "absolute_path": str(abs_path),
            "pathurl": _path_to_url(abs_path),
            "fps": src_fps,
            "width": src_w,
            "height": src_h,
            "has_audio": has_audio,
            "audio_streams": int(audio_streams),
            "audio_channels": audio_channels,
            "audio_sample_rate": audio_sample_rate,
            "audio_bit_depth": audio_bit_depth,
            "in_seconds": round(in_s, 3),
            "out_seconds": round(out_s, 3),
            "used_seconds": round(used, 3),
            "in_frame": in_f,
            "out_frame": out_f,
            "seq_start_seconds": round(seq_start_f / timebase, 3),
            "seq_end_seconds": round(seq_end_f / timebase, 3),
            "seq_start_frame": seq_start_f,
            "seq_end_frame": seq_end_f,
        })

    return {
        "sequence": {
            "name": sequence_name,
            "fps": seq_fps,
            "timebase": timebase,
            "ntsc": seq_ntsc,
            "width": seq_w,
            "height": seq_h,
            "total_frames": playhead_frames,
            "total_seconds": round(playhead_frames / timebase, 3),
            "clip_count": len(entries),
        },
        "clips": entries,
    }


# ── Path / URL helpers ──────────────────────────────────────────────────────


def _absolute_path(file_path: str) -> Path:
    """Resolve a catalogue path to an absolute path (no disk access required)."""
    p = Path(file_path)
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        return p if p.is_absolute() else (Path.cwd() / p)


def _path_to_url(abs_path: Path) -> str:
    """Render an absolute path as an FCP7 ``file://localhost/...`` media URL.

    Premiere/FCP7 expect a ``file://localhost/`` prefixed, percent-encoded URL.
    """
    try:
        uri = abs_path.as_uri()  # e.g. file:///C:/footage/goal.mov
    except ValueError:
        # Not absolute (shouldn't happen after _absolute_path) — best-effort.
        uri = "file://localhost/" + str(abs_path).replace("\\", "/").lstrip("/")
        return uri
    return uri.replace("file:///", "file://localhost/", 1)


# ── FCP7 (xmeml v5) XML generation ─────────────────────────────────────────


def _rate_xml(timebase: int, ntsc: bool, indent: str) -> str:
    return (
        f"{indent}<rate>\n"
        f"{indent}  <timebase>{timebase}</timebase>\n"
        f"{indent}  <ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc>\n"
        f"{indent}</rate>"
    )


def _file_element(clip: dict, file_id: str, define: bool, indent: str) -> str:
    """A <file> element carrying the clip's REAL source media parameters (audit H-10).

    The source ``<file>`` describes the ORIGINAL media, independent of the sequence:
      - video ``<rate>`` and ``<width>``/``<height>`` come from the clip's own fps and
        pixel dimensions (not the sequence raster/rate);
      - audio ``<channelcount>`` / ``<samplerate>`` / ``<depth>`` are the measured values
        (documented fallbacks only when ingest did not capture them). ``channelcount`` is
        the number of CHANNELS — not the number of audio streams (which drives A1/A2).

    FCP7 requires each media file be fully described the first time its id appears;
    later clipitems (e.g. the audio track linked to the same file) reference the id
    with an empty ``<file id="..."/>`` so Premiere links them to one master clip.
    """
    if not define:
        return f'{indent}<file id="{file_id}"/>'

    name = escape(clip["name"])
    src_tb, src_ntsc = _timebase_ntsc(clip.get("fps"))
    src_w = int(clip.get("width") or 0)
    src_h = int(clip.get("height") or 0)
    total_dur_frames = _seconds_to_frames(
        clip["out_seconds"] if clip["out_seconds"] > 0 else clip["used_seconds"], src_tb)
    total_dur_frames = max(total_dur_frames, clip["out_frame"], 1)

    video_char = f"{_rate_xml(src_tb, src_ntsc, indent + '        ')}\n"
    if src_w > 0 and src_h > 0:
        video_char += (f"{indent}        <width>{src_w}</width>\n"
                       f"{indent}        <height>{src_h}</height>\n")

    audio_block = ""
    if clip["audio_streams"] > 0:
        channels = int(clip.get("audio_channels") or DEFAULT_AUDIO_CHANNELS)
        samplerate = int(clip.get("audio_sample_rate") or DEFAULT_AUDIO_SAMPLE_RATE)
        depth = int(clip.get("audio_bit_depth") or DEFAULT_AUDIO_BIT_DEPTH)
        audio_block = (
            f"{indent}    <audio>\n"
            f"{indent}      <samplecharacteristics>\n"
            f"{indent}        <samplerate>{samplerate}</samplerate>\n"
            f"{indent}        <depth>{depth}</depth>\n"
            f"{indent}      </samplecharacteristics>\n"
            f"{indent}      <channelcount>{channels}</channelcount>\n"
            f"{indent}    </audio>\n"
        )
    return (
        f'{indent}<file id="{file_id}">\n'
        f"{indent}  <name>{name}</name>\n"
        f"{indent}  <pathurl>{escape(clip['pathurl'])}</pathurl>\n"
        f"{_rate_xml(src_tb, src_ntsc, indent + '  ')}\n"
        f"{indent}  <duration>{total_dur_frames}</duration>\n"
        f"{indent}  <media>\n"
        f"{indent}    <video>\n"
        f"{indent}      <samplecharacteristics>\n"
        f"{video_char}"
        f"{indent}      </samplecharacteristics>\n"
        f"{indent}    </video>\n"
        f"{audio_block}"
        f"{indent}  </media>\n"
        f"{indent}</file>"
    )


def _link_xml(refs: list[tuple[str, str, int]], clip_index: int, indent: str) -> str:
    """Build the <link> group shared by a clip's video + audio items.

    In FCP7 XML a linked master clip is recorded by giving EACH member clipitem one
    <link> child per group member (video + its audio items), so Premiere moves and
    trims them together. ``refs`` is the list of (clipitem_id, mediatype, trackindex)
    members; ``clip_index`` is the clip's shared position in the group. One link per
    member — no duplication.
    """
    blocks = []
    for ref_id, ref_type, ref_tindex in refs:
        blocks.append(
            f"{indent}<link>\n"
            f"{indent}  <linkclipref>{ref_id}</linkclipref>\n"
            f"{indent}  <mediatype>{ref_type}</mediatype>\n"
            f"{indent}  <trackindex>{ref_tindex}</trackindex>\n"
            f"{indent}  <clipindex>{clip_index}</clipindex>\n"
            f"{indent}</link>"
        )
    return "\n".join(blocks)


def to_fcp7_xml(timeline: dict) -> str:
    """Render a ``build_timeline`` dict as a valid FCP7 (xmeml v5) XML document.

    The sequence has one video track (V1) and up to two audio tracks (A1 ambient/
    original, A2 secondary/crowd). Clip order and timing come straight from the
    timeline dict — this function makes NO editorial decision.
    """
    seq = timeline["sequence"]
    clips = timeline["clips"]
    tb = int(seq["timebase"])
    ntsc = bool(seq["ntsc"])
    w, h = int(seq["width"]), int(seq["height"])
    total = int(seq["total_frames"])

    # Assign stable ids. One file id per unique source path (referenced, not redefined).
    file_ids: dict[str, str] = {}
    defined_files: set[str] = set()
    next_file = [1]

    def file_id_for(path: str) -> str:
        if path not in file_ids:
            file_ids[path] = f"file-{next_file[0]}"
            next_file[0] += 1
        return file_ids[path]

    video_items: list[str] = []
    a1_items: list[str] = []
    a2_items: list[str] = []
    link_sections: dict[str, list[str]] = {}

    for clip in clips:
        fid = file_id_for(clip["absolute_path"])
        n = clip["order"]
        name = escape(clip["name"])
        role = escape(clip["role"]) if clip["role"] else ""
        comment = (f"        <comments>\n          <mastercomment1>{role}</mastercomment1>\n"
                   f"        </comments>\n") if role else ""

        v_id = f"clipitem-v{n}"
        a1_id = f"clipitem-a1-{n}"
        a2_id = f"clipitem-a2-{n}"

        # Which items exist for this clip → drives linking.
        refs: list[tuple[str, str, int]] = [(v_id, "video", 1)]
        if clip["audio_streams"] >= 1:
            refs.append((a1_id, "audio", 1))
        if clip["audio_streams"] >= 2:
            refs.append((a2_id, "audio", 2))
        # A single-item clip (video only) needs no link group.
        links_for_clip = _link_xml(refs, n, "        ") if len(refs) > 1 else ""

        # ── Video clipitem (V1): defines the file the first time it appears ──
        define_here = clip["absolute_path"] not in defined_files
        defined_files.add(clip["absolute_path"])
        file_xml = _file_element(clip, fid, define_here, "        ")
        video_items.append(
            f'      <clipitem id="{v_id}">\n'
            f"        <name>{name}</name>\n"
            f"        <enabled>TRUE</enabled>\n"
            f"        <duration>{max(1, clip['out_frame'] - clip['in_frame'])}</duration>\n"
            f"{_rate_xml(tb, ntsc, '        ')}\n"
            f"        <start>{clip['seq_start_frame']}</start>\n"
            f"        <end>{clip['seq_end_frame']}</end>\n"
            f"        <in>{clip['in_frame']}</in>\n"
            f"        <out>{clip['out_frame']}</out>\n"
            f"{comment}"
            f"{file_xml}\n"
            f"{links_for_clip}\n"
            f"      </clipitem>"
        )

        # ── Audio clipitems: reference the same file id (already defined) ──
        if clip["audio_streams"] >= 1:
            a1_items.append(
                _audio_clipitem(a1_id, name, clip, tb, ntsc, fid, sourcetrack=1, indent="      ",
                                links=links_for_clip))
        if clip["audio_streams"] >= 2:
            a2_items.append(
                _audio_clipitem(a2_id, name, clip, tb, ntsc, fid, sourcetrack=2, indent="      ",
                                links=links_for_clip))

    # Audio tracks: A1 always rendered when any clip has audio; A2 only if any clip
    # actually has a second audio stream ("if present" — never fabricated).
    audio_tracks = []
    if a1_items:
        audio_tracks.append("    <track>\n" + "\n".join(a1_items) + "\n    </track>")
    if a2_items:
        audio_tracks.append("    <track>\n" + "\n".join(a2_items) + "\n    </track>")
    audio_block = ""
    if audio_tracks:
        audio_block = "  <audio>\n" + "\n".join(audio_tracks) + "\n  </audio>\n"

    video_block = (
        "  <video>\n"
        "    <format>\n"
        "      <samplecharacteristics>\n"
        f"{_rate_xml(tb, ntsc, '        ')}\n"
        f"        <width>{w}</width>\n"
        f"        <height>{h}</height>\n"
        "      </samplecharacteristics>\n"
        "    </format>\n"
        "    <track>\n"
        + "\n".join(video_items) +
        "\n    </track>\n"
        "  </video>\n"
    )

    seq_name = escape(seq["name"])
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE xmeml>\n"
        '<xmeml version="5">\n'
        '  <sequence id="sequence-1">\n'
        f"    <name>{seq_name}</name>\n"
        f"    <duration>{total}</duration>\n"
        f"{_rate_xml(tb, ntsc, '    ')}\n"
        "    <timecode>\n"
        f"{_rate_xml(tb, ntsc, '      ')}\n"
        "      <string>00:00:00:00</string>\n"
        "      <frame>0</frame>\n"
        "      <displayformat>NDF</displayformat>\n"
        "    </timecode>\n"
        "    <media>\n"
        + _reindent(video_block, "    ")
        + (_reindent(audio_block, "    ") if audio_block else "")
        + "    </media>\n"
        "  </sequence>\n"
        "</xmeml>\n"
    )
    return xml


def _audio_clipitem(cid: str, name: str, clip: dict, tb: int, ntsc: bool, fid: str,
                    sourcetrack: int, indent: str, links: str) -> str:
    return (
        f'{indent}<clipitem id="{cid}">\n'
        f"{indent}  <name>{name}</name>\n"
        f"{indent}  <enabled>TRUE</enabled>\n"
        f"{indent}  <duration>{max(1, clip['out_frame'] - clip['in_frame'])}</duration>\n"
        f"{_rate_xml(tb, ntsc, indent + '  ')}\n"
        f"{indent}  <start>{clip['seq_start_frame']}</start>\n"
        f"{indent}  <end>{clip['seq_end_frame']}</end>\n"
        f"{indent}  <in>{clip['in_frame']}</in>\n"
        f"{indent}  <out>{clip['out_frame']}</out>\n"
        f'{indent}  <file id="{fid}"/>\n'
        f"{indent}  <sourcetrack>\n"
        f"{indent}    <mediatype>audio</mediatype>\n"
        f"{indent}    <trackindex>{sourcetrack}</trackindex>\n"
        f"{indent}  </sourcetrack>\n"
        f"{links}\n"
        f"{indent}</clipitem>"
    )


def _reindent(block: str, prefix: str) -> str:
    """Prefix every non-empty line of a block (keeps the XML tidy under <media>)."""
    return "".join(
        (prefix + line if line.strip() else line) + "\n"
        for line in block.rstrip("\n").split("\n")
    ) if block else ""


# ── Top-level compile + write ──────────────────────────────────────────────


def compile_project(
    clips: list[dict],
    *,
    sequence_name: str = "MAPO Edit",
    project_id: int = 1,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    write: bool = True,
    verify_media: bool = True,
) -> dict:
    """Compile ordered clips into the JSON intermediate + FCP7 XML, and (optionally) write both.

    Args:
        clips: Ordered, resolved clip dicts (see ``build_timeline``).
        sequence_name: Name of the Premiere sequence.
        project_id: Project id (used for the output filename).
        fps / width / height: Sequence overrides; otherwise inferred from the clips.
        write: When True, write the ``.xml`` and ``.json`` to
            ``settings.PROCESSED_OUTPUT_DIR / "exports"`` (versioned + atomic).
        verify_media: When True (default), validate every clip's media exists on disk
            BEFORE compiling and raise :class:`MediaValidationError` if not (audit H-12) —
            so a moved/deleted file fails loudly instead of producing a dead-link XML.

    Returns:
        Dict with ``timeline`` (JSON intermediate), ``xml`` (the FCP7 document string),
        and, when written, ``xml_path`` / ``json_path``.
    """
    if not clips:
        raise ValueError("No clips to compile — nothing to deliver.")

    if verify_media:
        report = validate_media(clips)
        if not report["ok"]:
            raise MediaValidationError(report)

    timeline = build_timeline(clips, sequence_name=sequence_name, fps=fps,
                              width=width, height=height)
    xml = to_fcp7_xml(timeline)

    result = {"timeline": timeline, "xml": xml}

    if write:
        output_dir = settings.PROCESSED_OUTPUT_DIR / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sequence_name).strip("_") or "edit"
        # Versioned filename (timestamp + short uuid) so a re-export of the same sequence
        # never silently overwrites a prior export — each is a trackable version (H-13).
        version = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        stem = f"premiere_{project_id}_{safe}_{version}"
        xml_path = output_dir / f"{stem}.xml"
        json_path = output_dir / f"{stem}.json"
        _atomic_write(xml_path, xml)
        _atomic_write(json_path, json.dumps(timeline, indent=2))
        result["xml_path"] = str(xml_path)
        result["json_path"] = str(json_path)
        result["version"] = version
        logger.info(f"Compiled Premiere project -> {xml_path}")

    return result
