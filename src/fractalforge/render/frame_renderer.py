"""Single frame renderer — orchestrates kernel → coloring → image output."""

from pathlib import Path

import numpy as np
from PIL import Image

from fractalforge.engine.mandelbrot import render_frame
from fractalforge.engine.coloring import smooth_to_image
from fractalforge.artist.palette import get_palette, PALETTE_OCEAN


def render_single(
    center_re: float = -0.75,
    center_im: float = 0.0,
    zoom: float = 1.0,
    width: int = 1920,
    height: int = 1080,
    max_iter: int = 1000,
    palette_name: str = "ocean",
    interior_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Render a single Mandelbrot frame as a PIL Image.

    Args:
        center_re: Real part of center coordinate.
        center_im: Imaginary part of center coordinate.
        zoom: Zoom level.
        width: Frame width in pixels.
        height: Frame height in pixels.
        max_iter: Maximum iterations.
        palette_name: Name of a built-in palette.
        interior_color: RGB for interior (non-escaping) points.

    Returns:
        PIL Image (RGB).
    """
    palette = get_palette(palette_name)
    smooth_data = render_frame(center_re, center_im, zoom, width, height, max_iter)
    return smooth_to_image(smooth_data, palette, interior_color)


def render_and_save(
    output_path: Path,
    center_re: float = -0.75,
    center_im: float = 0.0,
    zoom: float = 1.0,
    width: int = 1920,
    height: int = 1080,
    max_iter: int = 1000,
    palette_name: str = "ocean",
    interior_color: tuple[int, int, int] = (0, 0, 0),
) -> Path:
    """Render a single frame and save to disk.

    Args:
        output_path: Path to save the PNG file.
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
    )
    img.save(output_path, format="PNG")
    return output_path
