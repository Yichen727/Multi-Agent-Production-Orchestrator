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
    """Parse an ffprobe frame-rate string like '30000/1001' into fps.

    Kept at 6 decimals (not 3) so the rational is preserved faithfully — e.g.
    ``30000/1001`` → ``29.97003`` rather than ``29.97`` — which lets the exporter map
    NTSC rates to the correct ``timebase`` + ``ntsc`` flag without frame drift (H-11).
    """
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
    """Probe REAL technical metadata for a video file via ffprobe.

    Inspects every stream and the container to extract the first video stream's
    pixel dimensions, rotation, codec and frame rate, whether an audio track is
    present, and the duration — then derives the *display* orientation. Phone
    "vertical" videos are commonly stored as landscape pixels with a 90/270°
    rotation tag; this accounts for that, so orientation reflects what the viewer
    actually sees.

    Args:
        input_path: Path to the video file.

    Returns:
        Dict with keys: width, height (display, after rotation), duration_seconds,
        orientation ('portrait' | 'landscape' | 'square' | 'unknown'), fps, codec,
        has_audio (bool), and ``ok`` (bool). On failure (missing ffprobe, unreadable
        file) ``ok`` is False and fields are zeroed / 'unknown' — nothing is guessed.
    """
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

    # Real audio params (first audio stream) for a faithful source <file> in the export
    # (audit H-10). bits_per_raw_sample/bits_per_sample are often absent for compressed
    # codecs → 0 here, and the exporter substitutes a documented default.
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

    # Rotation may live in stream tags ("rotate") or side data ("rotation").
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


def detect_scene_cut_times(input_path: str, threshold: float = 0.4,
                           timeout: int = 120) -> list[float]:
    """Detect the TIMESTAMPS (seconds) of shot/scene cuts using FFmpeg.

    Runs the ``select='gt(scene,threshold)',showinfo`` filter and parses the
    ``pts_time:`` of every selected (cut) frame out of stderr. This is real (if
    coarse) shot-boundary detection — no content is fabricated. The timestamps
    let ingest sample one representative frame per detected scene instead of a
    fixed set of evenly-spaced frames.

    Args:
        input_path: Path to the video file.
        threshold: Scene-change sensitivity (0-1; higher = fewer cuts).
        timeout: Hard cap in seconds so a long clip can't hang ingest.

    Returns:
        Sorted list of cut timestamps in seconds. Empty when FFmpeg is
        unavailable, detection failed, or no cuts were found.
    """
    if not check_ffmpeg_installed():
        return []
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-an", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as e:
        logger.error(f"Scene detection failed for {input_path}: {e}")
        return []
    # showinfo prints one line per selected (cut) frame, each with 'pts_time:<sec>'.
    times: list[float] = []
    for m in re.finditer(r"pts_time:([0-9]+\.?[0-9]*)", result.stderr):
        try:
            times.append(float(m.group(1)))
        except ValueError:
            continue
    return sorted(times)


def detect_scene_count(input_path: str, threshold: float = 0.4,
                       timeout: int = 120) -> int:
    """Count shot/scene cuts in a clip (compatibility wrapper).

    Backed by :func:`detect_scene_cut_times`; the number of distinct shots is
    cuts + 1. Kept for callers that only need the count.

    Args:
        input_path: Path to the video file.
        threshold: Scene-change sensitivity (0-1; higher = fewer cuts).
        timeout: Hard cap in seconds so a long clip can't hang ingest.

    Returns:
        Estimated number of shots (>= 1) when detection ran, or 0 if FFmpeg is
        unavailable.
    """
    if not check_ffmpeg_installed():
        return 0
    cuts = detect_scene_cut_times(input_path, threshold, timeout)
    # No cuts but FFmpeg ran fine → the clip is a single continuous shot.
    return len(cuts) + 1 if cuts else 1


def choose_frame_count(duration_seconds: float, scene_count: int = None,
                       min_frames: int = 3, max_frames: int = 12) -> int:
    """Pick how many frames to sample for vision tagging, adaptively.

    A fixed 3-frame sample under-represents long or multi-scene clips. This
    scales the sample up with both the clip's duration (roughly one frame per 5
    seconds) and the number of detected scenes, clamped to ``[min_frames,
    max_frames]``.

    Args:
        duration_seconds: Clip duration from probe (0/None → treated as short).
        scene_count: Detected number of shots, if known.
        min_frames: Floor (short single-shot clips still get this many).
        max_frames: Ceiling (bounds Vision cost / latency).

    Returns:
        Frame count in ``[min_frames, max_frames]``.
    """
    count = min_frames
    if duration_seconds and duration_seconds > 0:
        count = max(count, round(duration_seconds / 5.0))
    if scene_count and scene_count > 0:
        count = max(count, scene_count)
    return max(min_frames, min(count, max_frames))


def _grab_frame_b64(input_path: str, ts: float, max_width: int = 768) -> str | None:
    """Grab a single frame at ``ts`` seconds as a base64 JPEG (downscaled).

    Downscales to at most ``max_width`` px wide (never upscales, keeps aspect
    ratio, even height) so a 4K source is not sent to the vision model at full
    resolution. Returns None on any failure — never raises.
    """
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
    """Grab frames at the given timestamps as base64 JPEGs (in-memory).

    Each frame is downscaled to ``max_width`` px wide. A frame that fails to
    grab is skipped without affecting the others.

    Args:
        input_path: Path to the video file.
        timestamps: Seconds into the clip to sample.
        max_width: Max frame width sent to the vision model.

    Returns:
        List of base64-encoded JPEG strings (may be shorter than ``timestamps``;
        empty if FFmpeg is unavailable).
    """
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
    """Sample one representative frame per detected scene (its midpoint).

    Builds scene boundaries from ``[0] + cut_times + [duration]`` and samples the
    midpoint of each scene, so every detected shot is represented rather than a
    fixed set of evenly-spaced points. If there are more scenes than
    ``max_frames``, the midpoints are uniformly down-sampled. When there are no
    usable cut times (scene detection found nothing or failed) it falls back to
    adaptive even sampling across the whole clip.

    Args:
        input_path: Path to the video file.
        duration_seconds: Clip duration from probe.
        cut_times: Scene-cut timestamps from :func:`detect_scene_cut_times`.
        max_frames: Maximum number of frames to sample.

    Returns:
        ``(frames_b64, sampled_timestamps)`` — aligned lists (a frame and the
        second it was taken at). Both empty if FFmpeg is unavailable / all grabs
        failed.
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
            # Uniform down-sample to ``cap`` midpoints, preserving spread.
            step = len(midpoints) / cap
            midpoints = [midpoints[int(i * step)] for i in range(cap)]
        timestamps = midpoints
    else:
        # Fallback: adaptive even sampling across the clip.
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


def extract_sample_frames_b64(input_path: str, duration_seconds: float = 0.0,
                              count: int = 3, max_width: int = 768) -> list[str]:
    """Grab evenly spaced frames as base64 JPEGs (in-memory, no temp files).

    Used to feed representative frames to the vision model for tagging. Seeks to
    fractions of the clip's duration so the samples span the whole shot. This is
    the adaptive-even sampling fallback when scene detection yields no cuts.

    Args:
        input_path: Path to the video file.
        duration_seconds: Clip duration (from probe); if 0, samples near the start.
        count: Number of frames to extract.
        max_width: Max frame width sent to the vision model (downscaled, never
            upscaled) so a 4K source isn't handed over at full resolution.

    Returns:
        List of base64-encoded JPEG strings (may be shorter than ``count`` if some
        seeks fail; empty if FFmpeg is unavailable).
    """
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
    """Generate a low-resolution proxy file for faster editing.

    Args:
        input_path: Path to the original footage file.
        output_dir: Directory for proxy output. Defaults to settings.PROXY_OUTPUT_DIR.
        scale: Resolution scale (width:height).

    Returns:
        Path to the generated proxy file.
    """
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
    """Extract frames from a video at a given FPS rate.

    Used by ShotTagger for scene analysis via vision models.

    Args:
        input_path: Path to the video file.
        output_dir: Directory for extracted frames.
        fps: Frames per second to extract (default: 1 frame/sec).
        max_frames: Maximum number of frames to extract.

    Returns:
        List of paths to extracted frame images.
    """
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
    """Get video duration in seconds using ffprobe.

    Args:
        input_path: Path to the video file.

    Returns:
        Duration in seconds.
    """
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
    """Count the audio streams in a file via ffprobe (for A1/A2 track assignment).

    The Delivery Agent puts a clip's original audio on A1 and only creates a secondary
    A2 track when a file genuinely carries a SECOND audio stream (e.g. a separate
    ambient / crowd mic). This returns the real count so A2 is never fabricated.

    Returns:
        The number of audio streams, or 0 when ffprobe is missing / the file is
        unreadable — never a guess.
    """
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