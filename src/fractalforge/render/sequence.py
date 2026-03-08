"""Frame sequence renderer -- renders all frames for a zoom path.

Supports progress tracking, checkpointing (skip already-rendered frames),
and configurable output format.
"""

import time
from pathlib import Path

from PIL import Image

from fractalforge.artist.palette import get_palette
from fractalforge.artist.zoompath import ZoomPath
from fractalforge.engine.coloring import smooth_to_image
from fractalforge.engine.mandelbrot import render_frame


def render_sequence(
    zoom_path: ZoomPath,
    output_dir: Path,
    use_gpu: bool | None = None,
    skip_existing: bool = True,
    supersampling: int = 1,
    on_progress: callable = None,
) -> list[Path]:
    """Render all frames in a zoom path to individual PNG files.

    Args:
        zoom_path: The zoom path defining keyframes and interpolation.
        output_dir: Directory to write frame PNGs (frame_000000.png, etc.).
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None).
        skip_existing: Skip frames that already exist on disk (checkpoint resume).
        supersampling: Supersampling factor (1=off, 2=4x SSAA, 3=9x).
        on_progress: Callback(frame_idx, total_frames, elapsed, fps) called after each frame.

    Returns:
        List of output file paths in frame order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = zoom_path.total_frames
    paths: list[Path] = []
    rendered_count = 0
    skipped_count = 0
    start_time = time.perf_counter()

    ss = max(1, supersampling)
    render_w = zoom_path.width * ss
    render_h = zoom_path.height * ss

    for frame_idx in range(total):
        frame_path = output_dir / f"frame_{frame_idx:06d}.png"
        paths.append(frame_path)

        # Checkpoint: skip if already rendered
        if skip_existing and frame_path.exists():
            skipped_count += 1
            if on_progress:
                elapsed = time.perf_counter() - start_time
                on_progress(frame_idx, total, elapsed, 0.0, skipped=True)
            continue

        # Interpolate parameters at this frame
        params = zoom_path.interpolate(frame_idx)
        palette = get_palette(params["palette"])

        # Render at supersampled resolution
        smooth_data = render_frame(
            center_re=params["center_re"],
            center_im=params["center_im"],
            zoom=params["zoom"],
            width=render_w,
            height=render_h,
            max_iter=params["max_iter"],
            use_gpu=use_gpu,
        )

        img = smooth_to_image(smooth_data, palette)

        # Downsample if supersampled
        if ss > 1:
            img = img.resize((zoom_path.width, zoom_path.height), Image.LANCZOS)

        img.save(frame_path, format="PNG")
        rendered_count += 1

        if on_progress:
            elapsed = time.perf_counter() - start_time
            fps = rendered_count / elapsed if elapsed > 0 else 0.0
            on_progress(frame_idx, total, elapsed, fps, skipped=False)

    return paths


def get_sequence_status(output_dir: Path, total_frames: int) -> dict:
    """Check how many frames have been rendered in a sequence directory.

    Args:
        output_dir: The frame output directory.
        total_frames: Expected total frame count.

    Returns:
        Dict with rendered, missing, total, and complete status.
    """
    output_dir = Path(output_dir)
    rendered = 0
    missing = []

    for i in range(total_frames):
        frame_path = output_dir / f"frame_{i:06d}.png"
        if frame_path.exists():
            rendered += 1
        else:
            missing.append(i)

    return {
        "rendered": rendered,
        "missing_count": len(missing),
        "missing_first_10": missing[:10],
        "total": total_frames,
        "complete": rendered == total_frames,
        "percent": (rendered / total_frames * 100) if total_frames > 0 else 0.0,
    }
