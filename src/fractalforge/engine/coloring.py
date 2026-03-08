"""Coloring algorithms — map smooth iteration counts to RGB images.

Supports multiple coloring modes and palette-based gradient mapping.
"""

import numpy as np
from PIL import Image


def apply_palette(
    smooth_data: np.ndarray,
    palette: np.ndarray,
    interior_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Map smooth iteration data to RGB using a color palette.

    Args:
        smooth_data: 2D array of smooth iteration counts (-1.0 for interior).
        palette: Nx3 uint8 array defining the color gradient.
        interior_color: RGB tuple for interior (non-escaping) points.

    Returns:
        3D uint8 array (height × width × 3) of RGB pixel data.
    """
    height, width = smooth_data.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    palette_len = len(palette)

    # Mask for interior vs exterior
    interior = smooth_data < 0
    exterior = ~interior

    if np.any(exterior):
        # Normalize smooth values to palette indices with interpolation
        ext_vals = smooth_data[exterior]
        # Map to palette position (cyclic)
        t = ext_vals % palette_len
        idx0 = t.astype(np.int32) % palette_len
        idx1 = (idx0 + 1) % palette_len
        frac = (t - np.floor(t)).reshape(-1, 1)

        # Linear interpolation between adjacent palette colors
        color0 = palette[idx0].astype(np.float64)
        color1 = palette[idx1].astype(np.float64)
        interpolated = (color0 * (1.0 - frac) + color1 * frac).astype(np.uint8)

        rgb[exterior] = interpolated

    # Interior color
    rgb[interior] = interior_color

    return rgb


def smooth_to_image(
    smooth_data: np.ndarray,
    palette: np.ndarray,
    interior_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Convert smooth iteration data to a PIL Image.

    Args:
        smooth_data: 2D array of smooth iteration counts.
        palette: Nx3 uint8 color palette array.
        interior_color: RGB tuple for interior points.

    Returns:
        PIL Image in RGB mode.
    """
    rgb = apply_palette(smooth_data, palette, interior_color)
    return Image.fromarray(rgb, mode="RGB")
