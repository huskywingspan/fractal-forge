"""Single frame renderer -- orchestrates kernel -> coloring -> image output.

Auto-selects between standard double-precision and perturbation theory
based on zoom level:
  zoom < 1e13:  standard float64 (mandelbrot.py)
  zoom >= 1e13: perturbation theory (perturbation.py)
"""

from pathlib import Path

from PIL import Image

from fractalforge.engine.coloring import smooth_to_image
from fractalforge.engine.mandelbrot import render_frame
from fractalforge.artist.palette import get_palette

# Zoom threshold for switching to perturbation theory
_DEEP_ZOOM_THRESHOLD = 1e13


def _needs_perturbation(zoom: float) -> bool:
    """Return True if zoom level requires perturbation theory."""
    return zoom >= _DEEP_ZOOM_THRESHOLD


def render_single(
    center_re: float | str = -0.75,
    center_im: float | str = 0.0,
    zoom: float = 1.0,
    width: int = 1920,
    height: int = 1080,
    max_iter: int = 1000,
    palette_name: str = "ocean",
    interior_color: tuple[int, int, int] = (0, 0, 0),
    use_gpu: bool | None = None,
    supersampling: int = 1,
    histogram: bool = False,
    vignette: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
    fractal_type: str = "mandelbrot",
    julia_re: float | None = None,
    julia_im: float | None = None,
) -> Image.Image:
    """Render a single Mandelbrot frame as a PIL Image.

    Automatically selects perturbation theory for deep zooms (>= 1e13).
    Coordinates can be passed as strings to preserve precision at deep zoom.

    Args:
        center_re: Real part of center coordinate (float or string).
        center_im: Imaginary part of center coordinate (float or string).
        zoom: Zoom level.
        width: Frame width in pixels.
        height: Frame height in pixels.
        max_iter: Maximum iterations.
        palette_name: Name of a built-in palette.
        interior_color: RGB for interior (non-escaping) points.
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None).
        supersampling: Supersampling factor (1=off, 2=4x SSAA, 3=9x).
            Renders at factor*width x factor*height then downsamples with
            a box filter, averaging colors to eliminate aliasing noise.
        histogram: If True, apply histogram equalization for even color distribution.
        vignette: Vignette strength (0.0=off, 0.5=moderate, 1.0=strong edge darkening).
        contrast: Contrast multiplier (1.0=unchanged).
        saturation: Saturation multiplier (1.0=unchanged).
        brightness: Brightness multiplier (1.0=unchanged).

    Returns:
        PIL Image (RGB) at the requested width x height.
    """
    palette = get_palette(palette_name)

    # Render at supersampled resolution
    ss = max(1, supersampling)
    render_w = width * ss
    render_h = height * ss

    if fractal_type == "julia":
        from fractalforge.engine.julia import render_frame_julia
        smooth_data = render_frame_julia(
            c_re=julia_re or -0.7269,
            c_im=julia_im or 0.1889,
            center_re=float(center_re),
            center_im=float(center_im),
            zoom=zoom,
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
            zoom=zoom,
            width=render_w,
            height=render_h,
            max_iter=max_iter,
            use_gpu=use_gpu,
        )
    elif _needs_perturbation(zoom):
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
        smooth_data = render_frame(
            float(center_re), float(center_im), zoom,
            render_w, render_h, max_iter, use_gpu=use_gpu,
        )

    img = smooth_to_image(smooth_data, palette, interior_color, histogram=histogram)

    # Downsample if supersampled (box filter = proper area average)
    if ss > 1:
        img = img.resize((width, height), Image.LANCZOS)

    # Post-processing (vignette, color grading)
    if vignette > 0 or contrast != 1.0 or saturation != 1.0 or brightness != 1.0:
        from fractalforge.engine.postprocess import postprocess
        img = postprocess(img, vignette, contrast, saturation, brightness)

    return img


def render_and_save(
    output_path: Path,
    center_re: float | str = -0.75,
    center_im: float | str = 0.0,
    zoom: float = 1.0,
    width: int = 1920,
    height: int = 1080,
    max_iter: int = 1000,
    palette_name: str = "ocean",
    interior_color: tuple[int, int, int] = (0, 0, 0),
    use_gpu: bool | None = None,
    supersampling: int = 1,
    histogram: bool = False,
    vignette: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
    fractal_type: str = "mandelbrot",
    julia_re: float | None = None,
    julia_im: float | None = None,
) -> Path:
    """Render a single frame and save to disk.

    Args:
        output_path: Path to save the image file.
        center_re: Real part of center (float or string for deep zoom).
        center_im: Imaginary part of center (float or string for deep zoom).
        Other args: See render_single().

    Returns:
        The output path.
    """
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
        vignette=vignette,
        contrast=contrast,
        saturation=saturation,
        brightness=brightness,
        fractal_type=fractal_type,
        julia_re=julia_re,
        julia_im=julia_im,
    )
    img.save(output_path)
    return output_path
