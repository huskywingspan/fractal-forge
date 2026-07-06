"""Frame sequence renderer -- renders all frames for a zoom path.

Supports progress tracking, checkpointing (skip already-rendered frames),
and configurable output format. Automatically switches to perturbation
theory for deep zoom frames (zoom >= 1e13).
"""

import time
from pathlib import Path

from PIL import Image

from fractalforge.artist.palette import get_palette
from fractalforge.artist.zoompath import ZoomPath
from fractalforge.engine.coloring import smooth_to_image
from fractalforge.engine.mandelbrot import render_frame
from fractalforge.render.frame_renderer import needs_perturbation


def render_sequence(
    zoom_path: ZoomPath,
    output_dir: Path,
    use_gpu: bool | None = None,
    skip_existing: bool = True,
    supersampling: int = 1,
    on_progress: callable = None,
    histogram: bool = False,
    slope_shading: bool = False,
    cycle_speed: float = 0.0,
    log_scaling: bool = False,
    vignette: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
    bloom: float = 0.0,
    bloom_threshold: float = 0.6,
    bloom_radius: float = 20.0,
    halation: float = 0.0,
    tone_map: str = "none",
    exposure: float = 1.0,
    color_mode: str | None = None,
) -> list[Path]:
    """Render all frames in a zoom path to individual PNG files.

    Args:
        zoom_path: The zoom path defining keyframes and interpolation.
        output_dir: Directory to write frame PNGs (frame_000000.png, etc.).
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None).
        skip_existing: Skip frames that already exist on disk (checkpoint resume).
        supersampling: Supersampling factor (1=off, 2=4x SSAA, 3=9x).
        on_progress: Callback(frame_idx, total_frames, elapsed, fps) called after each frame.
        histogram: If True, apply histogram equalization for even color distribution.
        vignette: Vignette strength (0.0=off).
        contrast: Contrast multiplier (1.0=unchanged).
        saturation: Saturation multiplier (1.0=unchanged).
        brightness: Brightness multiplier (1.0=unchanged).
        bloom: HDR bloom intensity (0.0=off).
        bloom_threshold: Brightness threshold for bloom extraction.
        bloom_radius: Bloom blur radius in pixels.
        halation: Film halation intensity (0.0=off).
        tone_map: Tone mapping operator ("none", "aces", "reinhard").
        exposure: Exposure multiplier for tone mapping.
        color_mode: Palette mapping ("default" | "histogram" | "normalized");
            None falls back to the histogram flag.

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
        frame_zoom = params["zoom"]  # float, or a string for depths > 1e308
        ftype = params.get("fractal_type", "mandelbrot")

        if ftype == "julia":
            from fractalforge.engine.julia import render_frame_julia
            smooth_data = render_frame_julia(
                c_re=params.get("julia_re") or -0.7269,
                c_im=params.get("julia_im") or 0.1889,
                center_re=params.get("center_re_hp") or params["center_re"],
                center_im=params.get("center_im_hp") or params["center_im"],
                zoom=frame_zoom,
                width=render_w,
                height=render_h,
                max_iter=params["max_iter"],
                use_gpu=use_gpu,
            )
        elif ftype == "burning_ship":
            from fractalforge.engine.burning_ship import render_frame_burning_ship
            smooth_data = render_frame_burning_ship(
                center_re=params["center_re"],
                center_im=params["center_im"],
                zoom=frame_zoom,
                width=render_w,
                height=render_h,
                max_iter=params["max_iter"],
                use_gpu=use_gpu,
            )
        elif needs_perturbation(frame_zoom, render_h):
            from fractalforge.engine.perturbation import render_frame_perturbation
            # Use hp strings when available for full precision at deep zoom.
            # frame_zoom may be a string (e.g. "1e500") for unbounded depth;
            # render_frame_perturbation accepts string zoom and routes to the
            # floatexp deep kernel automatically.
            re_str = params.get("center_re_hp") or str(params["center_re"])
            im_str = params.get("center_im_hp") or str(params["center_im"])
            smooth_data = render_frame_perturbation(
                center_re=re_str,
                center_im=im_str,
                zoom=frame_zoom,
                width=render_w,
                height=render_h,
                max_iter=params["max_iter"],
                use_gpu=use_gpu,
            )
        else:
            smooth_data = render_frame(
                center_re=params["center_re"],
                center_im=params["center_im"],
                zoom=frame_zoom,
                width=render_w,
                height=render_h,
                max_iter=params["max_iter"],
                use_gpu=use_gpu,
            )

        # Color cycling: shift palette offset each frame
        cycle_offset = frame_idx * cycle_speed if cycle_speed != 0.0 else 0.0

        img = smooth_to_image(
            smooth_data, palette,
            histogram=histogram,
            slope_shading=slope_shading,
            cycle_offset=cycle_offset,
            log_scaling=log_scaling,
            color_mode=color_mode,
        )

        # Downsample if supersampled
        if ss > 1:
            img = img.resize((zoom_path.width, zoom_path.height), Image.LANCZOS)

        # Post-processing (color grading, HDR bloom, halation, tone mapping)
        if (vignette > 0 or contrast != 1.0 or saturation != 1.0
                or brightness != 1.0 or bloom > 0 or halation > 0
                or tone_map != "none"):
            from fractalforge.engine.postprocess import postprocess
            img = postprocess(
                img, vignette=vignette, contrast=contrast,
                saturation=saturation, brightness=brightness,
                bloom=bloom, bloom_threshold=bloom_threshold,
                bloom_radius=bloom_radius, halation=halation,
                tone_map=tone_map, exposure=exposure,
            )

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
