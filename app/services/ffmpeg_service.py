"""FFmpeg service for video processing — proxy generation and frame extraction."""

import json
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
    """Parse an ffprobe frame-rate string like '30000/1001' into fps."""
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den = float(den)
            return round(float(num) / den, 3) if den else 0.0
        return round(float(value), 3)
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
             "has_audio": False, "ok": False}

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
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if video is None:
        logger.error(f"No video stream found in {input_path}")
        return {**blank, "has_audio": has_audio}

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
            "has_audio": has_audio, "ok": True}


def detect_scene_count(input_path: str, threshold: float = 0.4,
                       timeout: int = 120) -> int:
    """Count shot/scene cuts in a clip using FFmpeg's scene-change detector.

    Runs the ``select='gt(scene,threshold)'`` filter and counts detected cut
    frames; the number of distinct shots is cuts + 1. This is real (if coarse)
    shot detection — no content is fabricated.

    Args:
        input_path: Path to the video file.
        threshold: Scene-change sensitivity (0-1; higher = fewer cuts).
        timeout: Hard cap in seconds so a long clip can't hang ingest.

    Returns:
        Estimated number of shots (>= 1), or 0 if detection could not run.
    """
    if not check_ffmpeg_installed():
        return 0
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
        return 0
    # showinfo prints one line per selected (cut) frame, each containing 'pts_time'.
    cuts = result.stderr.count("pts_time:")
    return cuts + 1


def extract_sample_frames_b64(input_path: str, duration_seconds: float = 0.0,
                              count: int = 3) -> list[str]:
    """Grab evenly spaced frames as base64 JPEGs (in-memory, no temp files).

    Used to feed representative frames to the vision model for tagging. Seeks to
    fractions of the clip's duration so the samples span the whole shot.

    Args:
        input_path: Path to the video file.
        duration_seconds: Clip duration (from probe); if 0, samples near the start.
        count: Number of frames to extract.

    Returns:
        List of base64-encoded JPEG strings (may be shorter than ``count`` if some
        seeks fail; empty if FFmpeg is unavailable).
    """
    import base64

    if not check_ffmpeg_installed():
        return []

    count = max(1, count)
    if duration_seconds and duration_seconds > 0:
        # Spread across the clip: e.g. count=3 → 0.2, 0.5, 0.8 of duration.
        fractions = [(i + 1) / (count + 1) for i in range(count)]
        timestamps = [round(duration_seconds * f, 2) for f in fractions]
    else:
        timestamps = [0.0] * count

    frames: list[str] = []
    for ts in timestamps:
        cmd = [
            "ffmpeg", "-v", "error",
            "-ss", str(ts),
            "-i", str(input_path),
            "-frames:v", "1",
            "-f", "image2", "-c:v", "mjpeg",
            "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and result.stdout:
                frames.append(base64.b64encode(result.stdout).decode("ascii"))
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Frame grab at {ts}s failed for {input_path}: {e}")
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