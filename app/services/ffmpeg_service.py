"""FFmpeg service for video processing — proxy generation and frame extraction."""

import base64
import json
import re
import subprocess
import shutil
from pathlib import Path
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("ffmpeg_service")


def check_ffmpeg_installed() -> bool:
    """Verify FFmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def check_ffprobe_installed() -> bool:
    """Verify ffprobe is available on the system."""
    return shutil.which("ffprobe") is not None


def _parse_frame_rate(value: str) -> float:
    """Convert an ffprobe frame-rate value to decimal FPS."""
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den = float(den)
            return round(float(num) / den, 6) if den else 0.0
        return round(float(value), 6)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe_video_metadata(input_path: str) -> dict:
    """Extract video, audio, and container metadata using ffprobe."""
    blank = {"width": 0, "height": 0, "duration_seconds": 0.0,
             "orientation": "unknown", "fps": 0.0, "codec": "unknown",
             "has_audio": False, "audio_channels": 0, "audio_sample_rate": 0,
             "audio_bit_depth": 0, "ok": False}

    if not check_ffprobe_installed():
        logger.error("ffprobe not found — cannot probe real metadata.")
        return blank

    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json",
        str(input_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True, timeout=60)
        data = json.loads(result.stdout or "{}")
    except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Could not probe {input_path}: {e}")
        return blank

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    has_audio = audio is not None

    def _int(v) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    # Preserve source audio metadata when available.
    audio_meta = {
        "audio_channels": _int(audio.get("channels")) if audio else 0,
        "audio_sample_rate": _int(audio.get("sample_rate")) if audio else 0,
        "audio_bit_depth": (_int(audio.get("bits_per_raw_sample")
                                 or audio.get("bits_per_sample")) if audio else 0),
    }

    if video is None:
        logger.error(f"No video stream found in {input_path}")
        return {**blank, "has_audio": has_audio, **audio_meta}

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    codec = video.get("codec_name") or "unknown"
    fps = _parse_frame_rate(video.get("avg_frame_rate") or video.get("r_frame_rate") or "")

    # Account for phone footage stored with a rotation tag.
    rotation = 0
    tags = video.get("tags") or {}
    if "rotate" in tags:
        try:
            rotation = int(tags["rotate"])
        except (TypeError, ValueError):
            rotation = 0
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rotation = int(sd["rotation"])
            except (TypeError, ValueError):
                pass

    # A quarter-turn swaps the displayed dimensions.
    if abs(rotation) % 180 == 90:
        width, height = height, width

    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    if width == 0 or height == 0:
        orientation = "unknown"
    elif height > width:
        orientation = "portrait"
    elif width > height:
        orientation = "landscape"
    else:
        orientation = "square"

    return {"width": width, "height": height, "duration_seconds": duration,
            "orientation": orientation, "fps": fps, "codec": codec,
            "has_audio": has_audio, "ok": True, **audio_meta}


# Scene detection timeout scales with source duration.
_SCENE_TIMEOUT_BASE = 60.0
_SCENE_TIMEOUT_PER_SECOND = 0.5
_SCENE_TIMEOUT_MAX = 900.0


def scene_detection_timeout(duration_seconds: float | None) -> int:
    """Return a duration-aware scene-detection timeout."""
    if not duration_seconds or duration_seconds <= 0:
        return int(_SCENE_TIMEOUT_BASE * 2)
    budget = _SCENE_TIMEOUT_BASE + float(duration_seconds) * _SCENE_TIMEOUT_PER_SECOND
    return int(max(_SCENE_TIMEOUT_BASE, min(budget, _SCENE_TIMEOUT_MAX)))


def _decode_stream(value) -> str:
    """Convert subprocess output to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _parse_cut_times(stderr: str) -> list[float]:
    """Extract and sort scene-cut timestamps from FFmpeg output."""
    times: list[float] = []
    for m in re.finditer(r"pts_time:([0-9]+\.?[0-9]*)", stderr or ""):
        try:
            times.append(float(m.group(1)))
        except ValueError:
            continue
    return sorted(times)


def detect_scene_cuts(input_path: str, threshold: float = 0.4,
                      timeout: int | None = None,
                      duration_seconds: float | None = None) -> dict:
    """Detect scene cuts and report whether the scan completed.

    Partial cuts are retained when FFmpeg times out or exits mid-scan.
    """
    blank = {"ran": False, "complete": False, "cut_times": [], "scene_count": 0}
    if not check_ffmpeg_installed():
        return blank

    limit = int(timeout) if timeout else scene_detection_timeout(duration_seconds)
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-map", "0:v:0",   
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-an", "-sn", "-dn", "-f", "null", "-",
    ]

    complete = True
    returncode = 0
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
        stderr, returncode = result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        # Keep cuts detected before the timeout.
        stderr, complete = _decode_stream(e.stderr), False
    except (subprocess.SubprocessError, OSError) as e:
        logger.error(f"Scene detection failed for {input_path}: {e}")
        return blank

    cut_times = _parse_cut_times(stderr)

    if returncode != 0:
        if not cut_times:
            logger.error(f"Scene detection failed for {input_path} "
                         f"(ffmpeg exit {returncode}): {(stderr or '').strip()[-300:]}")
            return blank
        complete = False
        logger.warning(f"Scene detection ended early for {input_path} (ffmpeg exit "
                       f"{returncode}); keeping the {len(cut_times)} cut(s) found.")
    elif not complete:
        covered = f"up to {max(cut_times):.1f}s" if cut_times else "no cuts yet"
        logger.warning(
            f"Scene detection timed out after {limit}s for {input_path} ({covered}). "
            f"Keeping the {len(cut_times)} cut(s) found so far; the rest of the clip is "
            "unanalysed, so the scene count is a lower bound."
        )

    # A successful scan with no cuts represents one continuous shot.
    return {"ran": True, "complete": complete, "cut_times": cut_times,
            "scene_count": len(cut_times) + 1 if cut_times else 1}


def detect_scene_cut_times(input_path: str, threshold: float = 0.4,
                           timeout: int | None = None,
                           duration_seconds: float | None = None) -> list[float]:
    """Return detected scene-cut timestamps."""
    return detect_scene_cuts(input_path, threshold, timeout, duration_seconds)["cut_times"]


def detect_scene_count(input_path: str, threshold: float = 0.4,
                       timeout: int | None = None,
                       duration_seconds: float | None = None) -> int:
    """Return the detected scene count, or zero on failure."""
    return detect_scene_cuts(input_path, threshold, timeout, duration_seconds)["scene_count"]


def choose_frame_count(duration_seconds: float, scene_count: int = None,
                       min_frames: int = 3, max_frames: int = 24) -> int:
    """Adaptively choose frame count from duration and scene count."""
    count = min_frames
    if duration_seconds and duration_seconds > 0:
        count = max(count, round(duration_seconds / 5.0))
    if scene_count and scene_count > 0:
        count = max(count, scene_count)
    return max(min_frames, min(count, max_frames))


def _grab_frame_b64(input_path: str, ts: float, max_width: int = 768) -> str | None:
    """Extract one downscaled JPEG frame as base64."""
    vf = f"scale='min({max_width},iw)':-2"
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(max(0.0, ts)),
        "-i", str(input_path),
        "-frames:v", "1",
        "-vf", vf,
        "-f", "image2", "-c:v", "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            return base64.b64encode(result.stdout).decode("ascii")
    except (subprocess.SubprocessError, OSError) as e:
        logger.error(f"Frame grab at {ts}s failed for {input_path}: {e}")
    return None


def extract_frames_at_timestamps_b64(input_path: str, timestamps: list[float],
                                     max_width: int = 768) -> list[str]:
    """Extract downscaled JPEG frames at the given timestamps."""
    if not check_ffmpeg_installed():
        return []
    frames: list[str] = []
    for ts in timestamps:
        b64 = _grab_frame_b64(input_path, ts, max_width)
        if b64:
            frames.append(b64)
    return frames


def extract_scene_representative_frames_b64(
    input_path: str, duration_seconds: float, cut_times: list[float],
    max_frames: int = 12,
) -> tuple[list[str], list[float]]:
    """Sample representative frames from detected scenes.

    Scene midpoints are used when cuts are available; otherwise sampling
    falls back to evenly spaced timestamps.
    """
    if not check_ffmpeg_installed():
        return [], []

    cap = max(1, max_frames)
    timestamps: list[float] = []

    valid_cuts = sorted(
        t for t in (cut_times or [])
        if t and t > 0 and (not duration_seconds or t < duration_seconds)
    )
    if valid_cuts and duration_seconds and duration_seconds > 0:
        boundaries = [0.0] + valid_cuts + [float(duration_seconds)]
        midpoints = [
            round((boundaries[i] + boundaries[i + 1]) / 2.0, 2)
            for i in range(len(boundaries) - 1)
            if boundaries[i + 1] > boundaries[i]
        ]
        if len(midpoints) > cap:
            step = len(midpoints) / cap
            midpoints = [midpoints[int(i * step)] for i in range(cap)]
        timestamps = midpoints
    else:
        if duration_seconds and duration_seconds > 0:
            timestamps = [round(duration_seconds * (i + 1) / (cap + 1), 2)
                          for i in range(cap)]
        else:
            timestamps = [0.0]

    frames: list[str] = []
    sampled: list[float] = []
    for ts in timestamps:
        b64 = _grab_frame_b64(input_path, ts)
        if b64:
            frames.append(b64)
            sampled.append(ts)
    return frames, sampled


# Marks windows that span a real scene boundary.
_SPANS_A_CUT = -1


def _merge_windows_to_cap(windows: list[list], cap: int) -> list[list]:
    """Merge adjacent windows to meet the cap while preserving coverage."""
    work = [list(w) for w in windows]
    while len(work) > cap:
        best_i, best_key = 0, None
        for i in range(len(work) - 1):
            a, b = work[i], work[i + 1]
            intra = a[2] == b[2] and a[2] != _SPANS_A_CUT
            key = (0 if intra else 1, b[1] - a[0])
            if best_key is None or key < best_key:
                best_i, best_key = i, key
        a, b = work[best_i], work[best_i + 1]
        scene = a[2] if a[2] == b[2] else _SPANS_A_CUT
        work[best_i:best_i + 2] = [[a[0], b[1], scene]]
    return work


def build_event_windows(duration_seconds: float, cut_times: list[float],
                        max_events: int = 12, min_event_seconds: float = 1.0,
                        long_scene_seconds: float = 8.0) -> list[tuple[float, float]]:
    """Build ordered event windows from scene boundaries.

    Long scenes are subdivided, short windows are merged, and excess
    windows are merged rather than dropped to preserve full coverage.
    """
    if not duration_seconds or duration_seconds <= 0:
        return []

    dur = float(duration_seconds)
    valid_cuts = sorted(t for t in (cut_times or []) if t and 0.0 < t < dur)
    boundaries = [0.0] + valid_cuts + [dur]

    windows: list[list] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        span = end - start
        if span <= 0:
            continue
        if span > long_scene_seconds:
            parts = max(2, int(round(span / long_scene_seconds)))
            step = span / parts
            for p in range(parts):
                windows.append([round(start + p * step, 3),
                                round(start + (p + 1) * step, 3), i])
        else:
            windows.append([round(start, 3), round(end, 3), i])

    # Merge very short windows into the preceding window.
    merged: list[list] = []
    for w in windows:
        if merged and (w[1] - w[0]) < min_event_seconds:
            merged[-1][1] = w[1]
            if merged[-1][2] != w[2]:
                merged[-1][2] = _SPANS_A_CUT
        else:
            merged.append(list(w))

    # Merge rather than drop windows to preserve temporal coverage.
    cap = max(1, max_events)
    if len(merged) > cap:
        before = len(merged)
        merged = _merge_windows_to_cap(merged, cap)
        logger.info(f"Event windows: merged {before} → {len(merged)} to fit the "
                    f"{cap}-event budget (full temporal coverage preserved).")

    return [(w[0], w[1]) for w in merged]


def extract_frames_in_window_b64(input_path: str, start: float, end: float,
                                 count: int = 3, max_width: int = 768) -> list[str]:
    """Extract temporally ordered frames from an event window."""
    if not check_ffmpeg_installed():
        return []
    count = max(1, count)
    span = max(0.0, end - start)
    if span <= 0:
        timestamps = [max(0.0, start)]
    else:
        # Sample inside the window rather than at its boundaries.
        timestamps = [round(start + span * (i + 1) / (count + 1), 3)
                      for i in range(count)]
    frames: list[str] = []
    for ts in timestamps:
        b64 = _grab_frame_b64(input_path, ts, max_width)
        if b64:
            frames.append(b64)
    return frames


def extract_sample_frames_b64(input_path: str, duration_seconds: float = 0.0,
                              count: int = 3, max_width: int = 768) -> list[str]:
    """Extract evenly spaced frames across a clip."""
    if not check_ffmpeg_installed():
        return []

    count = max(1, count)
    if duration_seconds and duration_seconds > 0:
        # Spread across the clip: e.g. count=3 → 0.25, 0.5, 0.75 of duration.
        fractions = [(i + 1) / (count + 1) for i in range(count)]
        timestamps = [round(duration_seconds * f, 2) for f in fractions]
    else:
        timestamps = [0.0] * count

    frames: list[str] = []
    for ts in timestamps:
        b64 = _grab_frame_b64(input_path, ts, max_width)
        if b64:
            frames.append(b64)
    return frames


def generate_proxy(input_path: str, output_dir: str = None, scale: str = "1280:720") -> str:
    """Generate a low-resolution H.264 proxy for editing."""
    output_dir = Path(output_dir or settings.PROXY_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = Path(input_path)
    output_path = output_dir / f"{input_file.stem}_proxy.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"scale={scale}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path),
    ]

    logger.info(f"Generating proxy: {input_file.name} → {output_path.name}")

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        logger.info(f"Proxy generated: {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg proxy generation failed: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        logger.error("FFmpeg not found. Install FFmpeg: https://ffmpeg.org/download.html")
        raise


def extract_frames(
    input_path: str,
    output_dir: str = None,
    fps: float = 1.0,
    max_frames: int = 50,
) -> list[str]:
    """Extract up to ``max_frames`` JPEG frames at the requested FPS."""
    output_dir = Path(output_dir or settings.PROCESSED_OUTPUT_DIR) / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = Path(input_path)
    frame_pattern = output_dir / f"{input_file.stem}_frame_%04d.jpg"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(frame_pattern),
    ]

    logger.info(f"Extracting frames from: {input_file.name} (fps={fps}, max={max_frames})")

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    except subprocess.CalledProcessError as e:
        logger.error(f"Frame extraction failed: {e.stderr.decode()}")
        raise

    # Collect generated frame paths
    frames = sorted(output_dir.glob(f"{input_file.stem}_frame_*.jpg"))
    logger.info(f"Extracted {len(frames)} frames")
    return [str(f) for f in frames]


def get_video_duration(input_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Could not get duration for {input_path}: {e}")
        return 0.0


def count_audio_streams(input_path: str) -> int:
    """Count source audio streams for Delivery track assignment."""
    if not check_ffprobe_installed():
        return 0
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(input_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True, timeout=30)
        return len([line for line in result.stdout.splitlines() if line.strip()])
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Could not count audio streams for {input_path}: {e}")
        return 0