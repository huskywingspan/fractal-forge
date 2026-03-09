"""Video encoding -- assemble frame sequences into video files via FFmpeg.

Supports multiple codecs and quality presets. Uses ffmpeg-python for
clean subprocess management.
"""

import shutil
import subprocess
from pathlib import Path

import ffmpeg


# --- Encoding presets ---

ENCODE_PRESETS = {
    "preview": {
        "label": "Preview (fast, smaller files)",
        "vcodec": "libx264",
        "crf": "23",
        "preset": "fast",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
    },
    "quality": {
        "label": "High quality (slower, larger files)",
        "vcodec": "libx264",
        "crf": "18",
        "preset": "slow",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
    },
    "lossless": {
        "label": "Lossless (very large files)",
        "vcodec": "libx264",
        "crf": "0",
        "preset": "veryslow",
        "pix_fmt": "yuv444p",
        "ext": ".mp4",
    },
    "prores": {
        "label": "ProRes 422 HQ (editing-friendly)",
        "vcodec": "prores_ks",
        "profile:v": "3",
        "pix_fmt": "yuv422p10le",
        "ext": ".mov",
    },
    "youtube": {
        "label": "YouTube upload (high quality, H.264 High Profile)",
        "vcodec": "libx264",
        "crf": "16",
        "preset": "slow",
        "pix_fmt": "yuv420p",
        "profile:v": "high",
        "extra": ["-bf", "2", "-g", "120", "-movflags", "+faststart"],
        "ext": ".mp4",
    },
}


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def encode_video(
    frames_dir: Path,
    output_path: Path,
    fps: int = 60,
    preset: str = "quality",
    overwrite: bool = False,
) -> Path:
    """Encode a directory of frame PNGs into a video file.

    Frames must be named frame_000000.png, frame_000001.png, etc.

    Args:
        frames_dir: Directory containing sequentially-named frame PNGs.
        output_path: Output video file path.
        fps: Frames per second.
        preset: Encoding preset name (preview, quality, lossless, prores).
        overwrite: Overwrite output file if it exists.

    Returns:
        The output video file path.

    Raises:
        FileNotFoundError: If frames_dir doesn't exist or has no frames.
        ValueError: If preset name is unknown.
        RuntimeError: If FFmpeg is not installed or encoding fails.
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)

    if not check_ffmpeg():
        raise RuntimeError("FFmpeg not found. Install FFmpeg and ensure it's on PATH.")

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Check for at least one frame
    first_frame = frames_dir / "frame_000000.png"
    if not first_frame.exists():
        raise FileNotFoundError(f"No frames found in {frames_dir} (expected frame_000000.png)")

    if preset not in ENCODE_PRESETS:
        available = ", ".join(sorted(ENCODE_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    settings = ENCODE_PRESETS[preset]

    # Ensure output has correct extension for codec
    if output_path.suffix != settings["ext"]:
        output_path = output_path.with_suffix(settings["ext"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    input_pattern = str(frames_dir / "frame_%06d.png")

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", settings["vcodec"],
        "-pix_fmt", settings["pix_fmt"],
    ]

    # Codec-specific options
    if "crf" in settings:
        cmd.extend(["-crf", settings["crf"]])
    if "preset" in settings:
        cmd.extend(["-preset", settings["preset"]])
    if "profile:v" in settings:
        cmd.extend(["-profile:v", settings["profile:v"]])
    if "extra" in settings:
        cmd.extend(settings["extra"])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed:\n{result.stderr}")

    return output_path


def get_video_info(path: Path) -> dict:
    """Get basic info about an encoded video file."""
    try:
        probe = ffmpeg.probe(str(path))
        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"), None
        )
        if video_stream:
            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "codec": video_stream.get("codec_name", "unknown"),
                "duration": float(probe["format"].get("duration", 0)),
                "size_bytes": int(probe["format"].get("size", 0)),
                "fps": video_stream.get("r_frame_rate", "unknown"),
            }
    except Exception:
        pass
    return {}
