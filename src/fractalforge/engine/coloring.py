"""Coloring algorithms — map smooth iteration counts to RGB images.

Supports multiple coloring modes and palette-based gradient mapping.
"""

import numpy as np
from PIL import Image


def histogram_equalize(
    smooth_data: np.ndarray,
    palette_len: int,
    num_bins: int = 4096,
) -> np.ndarray:
    """Remap smooth iteration counts via histogram equalization.

    Redistributes iteration values so the full color palette is used evenly,
    preventing large washed-out single-color regions.

    Args:
        smooth_data: 2D array of smooth iteration counts (-1.0 for interior).
        palette_len: Length of the color palette (output mapped to [0, palette_len)).
        num_bins: Number of histogram bins (higher = finer redistribution).

    Returns:
        2D array with remapped values; interior pixels (-1.0) unchanged.
    """
    exterior = smooth_data >= 0
    if not np.any(exterior):
        return smooth_data

    ext_vals = smooth_data[exterior]

    # Build histogram and CDF
    hist, bin_edges = np.histogram(ext_vals, bins=num_bins)
    cdf = np.cumsum(hist).astype(np.float64)
    cdf = cdf / cdf[-1]  # normalize to [0, 1]

    # Map each value through the CDF
    # Find which bin each value falls into
    bin_indices = np.searchsorted(bin_edges[:-1], ext_vals) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Remap to [0, palette_len) via CDF
    result = smooth_data.copy()
    result[exterior] = cdf[bin_indices] * palette_len

    return result


def apply_palette(
    smooth_data: np.ndarray,
    palette: np.ndarray,
    interior_color: tuple[int, int, int] = (0, 0, 0),
    histogram: bool = False,
) -> np.ndarray:
    """Map smooth iteration data to RGB using a color palette.

    Args:
        smooth_data: 2D array of smooth iteration counts (-1.0 for interior).
        palette: Nx3 uint8 array defining the color gradient.
        interior_color: RGB tuple for interior (non-escaping) points.
        histogram: If True, apply histogram equalization before palette mapping
            to distribute colors evenly and eliminate washed-out regions.

    Returns:
        3D uint8 array (height × width × 3) of RGB pixel data.
    """
    height, width = smooth_data.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    palette_len = len(palette)

    # Optionally remap values via histogram equalization
    if histogram:
        smooth_data = histogram_equalize(smooth_data, palette_len)

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
    histogram: bool = False,
) -> Image.Image:
    """Convert smooth iteration data to a PIL Image.

    Args:
        smooth_data: 2D array of smooth iteration counts.
        palette: Nx3 uint8 color palette array.
        interior_color: RGB tuple for interior points.
        histogram: If True, apply histogram equalization before coloring.

    Returns:
        PIL Image in RGB mode.
    """
    rgb = apply_palette(smooth_data, palette, interior_color, histogram=histogram)
    return Image.fromarray(rgb, mode="RGB")
