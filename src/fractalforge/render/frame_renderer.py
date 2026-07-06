"""Single frame renderer -- orchestrates kernel -> coloring -> image output.

Auto-selects between standard double-precision and perturbation theory
based on zoom level:
  zoom < 1e13:  standard float64 (mandelbrot.py)
  zoom >= 1e13: perturbation theory (perturbation.py)
"""

import math
from pathlib import Path

from PIL import Image

from fractalforge.engine.coloring import smooth_to_image
from fractalforge.engine.mandelbrot import render_frame
from fractalforge.engine.precision import zoom_to_log10
from fractalforge.artist.palette import get_palette

# Minimum pixel spacing (in the complex plane) the standard float64 engine
# can resolve. The standard kernel computes c = center + offset in ABSOLUTE
# coordinates, where one ulp near |c| ~ 2 is ~4.4e-16; below ~100 ulps of
# spacing, adjacent pixels start collapsing onto identical c values and the
# image develops duplicate-column streaks. Perturbation avoids this entirely
# (its dc grid is relative to the center, so tiny spacings stay exact).
# The threshold is therefore RESOLUTION-AWARE: a 1080-row preview leaves
# float64 around zoom ~1.4e11, a small thumbnail around ~2e12.
_MIN_STD_PIXEL_SPACING = 2e-14


def needs_perturbation(zoom: float | str, height: int) -> bool:
    """True when the standard float64 engine can no longer resolve pixels.

    Accepts zoom as a string for depths beyond float64 range (e.g. "1e500").
    """
    log10_zoom = zoom_to_log10(zoom)
    # pixel spacing = (3 / zoom) / height; compare in log10 space so string
    # zooms beyond 1e308 never hit float arithmetic.
    log10_spacing = math.log10(3.0) - log10_zoom - math.log10(max(height, 1))
    return log10_spacing < math.log10(_MIN_STD_PIXEL_SPACING)


# Backward-compatible alias (some callers only have the zoom value; assume a
# 1080-row frame, the common preview/production height).
def _needs_perturbation(zoom: float | str, height: int = 1080) -> bool:
    return needs_perturbation(zoom, height)


def render_single(
    center_re: float | str = -0.75,
    center_im: float | str = 0.0,
    zoom: float | str = 1.0,
    width: int = 1920,
    height: int = 1080,
    max_iter: int = 1000,
    palette_name: str = "ocean",
    interior_color: tuple[int, int, int] = (0, 0, 0),
    use_gpu: bool | None = None,
    supersampling: int = 1,
    histogram: bool = False,
    slope_shading: bool = False,
    light_angle: float = 2.356,
    light_elevation: float = 0.6,
    cycle_offset: float = 0.0,
    log_scaling: bool = False,
    distance_coloring: bool = False,
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
    fractal_type: str = "mandelbrot",
    julia_re: float | None = None,
    julia_im: float | None = None,
    color_mode: str | None = None,
) -> Image.Image:
    """Render a single Mandelbrot frame as a PIL Image.

    Automatically selects perturbation theory for deep zooms (>= 1e13).
    Coordinates can be passed as strings to preserve precision at deep zoom.
    color_mode selects the value-to-palette mapping ("default", "histogram",
    "normalized"); None falls back to the histogram flag.
    """
    palette = get_palette(palette_name)

    # Render at supersampled resolution
    ss = max(1, supersampling)
    render_w = width * ss
    render_h = height * ss
    dist_data = None

    if fractal_type == "julia":
        from fractalforge.engine.julia import render_frame_julia
        # Pass center as-is: strings preserve precision for deep Julia zoom.
        smooth_data = render_frame_julia(
            c_re=julia_re or -0.7269,
            c_im=julia_im or 0.1889,
            center_re=center_re,
            center_im=center_im,
            zoom=float(zoom),
            width=render_w,
            height=render_h,
            max_iter=max_iter,
            use_gpu=use_gpu,
        )
    elif fractal_type == "burning_ship":
        from fractalforge.engine.burning_ship import render_frame_burning_ship
        smooth_data = render_frame_burning_ship(
            center_re=float(center_re),
            center_im=float(center_im),
            zoom=float(zoom),
            width=render_w,
            height=render_h,
            max_iter=max_iter,
            use_gpu=use_gpu,
        )
    elif needs_perturbation(zoom, render_h):
        from fractalforge.engine.perturbation import render_frame_perturbation
        smooth_data = render_frame_perturbation(
            center_re=str(center_re),
            center_im=str(center_im),
            zoom=zoom,
            width=render_w,
            height=render_h,
            max_iter=max_iter,
            use_gpu=use_gpu,
        )
    else:
        use_de = distance_coloring and fractal_type == "mandelbrot"
        result = render_frame(
            float(center_re), float(center_im), float(zoom),
            render_w, render_h, max_iter, use_gpu=use_gpu, distance=use_de,
        )
        if use_de:
            smooth_data, dist_data = result
        else:
            smooth_data = result
            dist_data = None

    img = smooth_to_image(
        smooth_data, palette, interior_color,
        histogram=histogram,
        slope_shading=slope_shading,
        light_angle=light_angle,
        light_elevation=light_elevation,
        cycle_offset=cycle_offset,
        log_scaling=log_scaling,
        distance_data=dist_data,
        color_mode=color_mode,
    )

    # Downsample if supersampled (box filter = proper area average)
    if ss > 1:
        img = img.resize((width, height), Image.LANCZOS)

    # Post-processing (color grading, HDR bloom, halation, tone mapping, vignette)
    has_postprocess = (vignette > 0 or contrast != 1.0 or saturation != 1.0
                       or brightness != 1.0 or bloom > 0 or halation > 0
                       or tone_map != "none")
    if has_postprocess:
        from fractalforge.engine.postprocess import postprocess
        img = postprocess(
            img, vignette=vignette, contrast=contrast, saturation=saturation,
            brightness=brightness, bloom=bloom, bloom_threshold=bloom_threshold,
            bloom_radius=bloom_radius, halation=halation, tone_map=tone_map,
            exposure=exposure,
        )

    return img


def render_and_save(
    output_path: Path,
    center_re: float | str = -0.75,
    center_im: float | str = 0.0,
    zoom: float | str = 1.0,
    width: int = 1920,
    height: int = 1080,
    max_iter: int = 1000,
    palette_name: str = "ocean",
    interior_color: tuple[int, int, int] = (0, 0, 0),
    use_gpu: bool | None = None,
    supersampling: int = 1,
    histogram: bool = False,
    slope_shading: bool = False,
    light_angle: float = 2.356,
    light_elevation: float = 0.6,
    cycle_offset: float = 0.0,
    log_scaling: bool = False,
    distance_coloring: bool = False,
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
    fractal_type: str = "mandelbrot",
    julia_re: float | None = None,
    julia_im: float | None = None,
    color_mode: str | None = None,
) -> Path:
    """Render a single frame and save to disk. See render_single() for args."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = render_single(
        center_re=center_re,
        center_im=center_im,
        zoom=zoom,
        width=width,
        height=height,
        max_iter=max_iter,
        palette_name=palette_name,
        interior_color=interior_color,
        use_gpu=use_gpu,
        supersampling=supersampling,
        histogram=histogram,
        slope_shading=slope_shading,
        light_angle=light_angle,
        light_elevation=light_elevation,
        cycle_offset=cycle_offset,
        log_scaling=log_scaling,
        distance_coloring=distance_coloring,
        vignette=vignette,
        contrast=contrast,
        saturation=saturation,
        brightness=brightness,
        bloom=bloom,
        bloom_threshold=bloom_threshold,
        bloom_radius=bloom_radius,
        halation=halation,
        tone_map=tone_map,
        exposure=exposure,
        fractal_type=fractal_type,
        julia_re=julia_re,
        julia_im=julia_im,
        color_mode=color_mode,
    )
    img.save(output_path)
    return output_path
