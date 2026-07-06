"""Coloring algorithms — map smooth iteration counts to RGB images.

Supports multiple coloring modes, palette-based gradient mapping,
slope shading (3D lighting effect), and color cycling.
"""

import math

import numpy as np
from numba import njit, prange
from PIL import Image


def histogram_equalize(
    smooth_data: np.ndarray,
    palette_len: int,
    num_bins: int = 4096,
) -> np.ndarray:
    """Remap smooth iteration counts via continuous histogram equalization.

    Redistributes iteration values so the full color palette is used evenly,
    preventing large washed-out single-color regions.

    The mapping interpolates the CDF between bin centers rather than assigning
    each bin a single output value. The old step-function mapping quantized
    deep-zoom frames — whose narrow iteration range concentrates many pixels
    per bin — into hard-edged posterized color bands (the "shattered polygon"
    artifact). A piecewise-linear CDF is continuous and monotone, so smooth
    input gradients stay smooth.

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

    hist, bin_edges = np.histogram(ext_vals, bins=num_bins)
    cdf = np.cumsum(hist).astype(np.float64)
    cdf = cdf / cdf[-1]  # normalize to [0, 1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    result = smooth_data.copy()
    result[exterior] = np.interp(ext_vals, bin_centers, cdf) * palette_len
    return result


def normalize_range(
    smooth_data: np.ndarray,
    palette_len: int,
    cycles: float = 3.0,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> np.ndarray:
    """Map smooth iterations linearly over their percentile range.

    An alternative to histogram EQ with a very different character: instead of
    equal palette area per pixel population, the palette sweeps linearly from
    the p_lo to the p_hi percentile of the frame. Dense filament regions get a
    glowing, high-contrast look while sparse regions stay dark and moody —
    especially striking on deep-zoom frames.

    Args:
        smooth_data: 2D array of smooth iteration counts (-1.0 for interior).
        palette_len: Palette length (output spans cycles * palette_len).
        cycles: Number of palette sweeps across the value range.
        p_lo, p_hi: Percentile window of exterior values to span.

    Returns:
        2D array with remapped values; interior pixels (-1.0) unchanged.
    """
    exterior = smooth_data >= 0
    if not np.any(exterior):
        return smooth_data

    ext_vals = smooth_data[exterior]
    lo = np.percentile(ext_vals, p_lo)
    hi = np.percentile(ext_vals, p_hi)
    span = max(hi - lo, 1e-9)

    result = smooth_data.copy()
    # Values below lo clamp to 0 so they stay at the palette start (dark end).
    result[exterior] = np.maximum(ext_vals - lo, 0.0) / span * palette_len * cycles
    return result


@njit(parallel=True)
def _compute_slope_lighting(
    smooth: np.ndarray,
    light_angle: float,
    light_elev: float,
    ambient: float,
    diffuse: float,
    specular: float,
    shininess: float,
) -> np.ndarray:
    """Compute per-pixel lighting from numerical gradient of smooth iterations.

    Uses finite differences to estimate surface normals, then applies
    Blinn-Phong shading with a directional light source.
    """
    h, w = smooth.shape
    lighting = np.ones((h, w), dtype=np.float64)

    # Light direction vector from angle and elevation
    cos_a = math.cos(light_angle)
    sin_a = math.sin(light_angle)
    cos_e = math.cos(light_elev)
    sin_e = math.sin(light_elev)
    light_x = cos_a * cos_e
    light_y = sin_a * cos_e
    light_z = sin_e

    # View direction (straight down onto the fractal)
    view_z = 1.0

    # Half vector for specular (Blinn-Phong)
    hx = light_x
    hy = light_y
    hz = light_z + view_z
    h_len = math.sqrt(hx * hx + hy * hy + hz * hz)
    if h_len > 0:
        hx /= h_len
        hy /= h_len
        hz /= h_len

    for row in prange(h):
        for col in range(w):
            if smooth[row, col] < 0:
                continue

            # Finite differences for gradient (central where possible)
            if col == 0:
                dx = smooth[row, min(col + 1, w - 1)] - smooth[row, col]
            elif col == w - 1:
                dx = smooth[row, col] - smooth[row, col - 1]
            else:
                dx = (smooth[row, col + 1] - smooth[row, col - 1]) * 0.5

            if row == 0:
                dy = smooth[min(row + 1, h - 1), col] - smooth[row, col]
            elif row == h - 1:
                dy = smooth[row, col] - smooth[row - 1, col]
            else:
                dy = (smooth[row + 1, col] - smooth[row - 1, col]) * 0.5

            # Skip interior neighbor contamination
            if col > 0 and smooth[row, col - 1] < 0:
                dx = 0.0
            if col < w - 1 and smooth[row, col + 1] < 0:
                dx = 0.0
            if row > 0 and smooth[row - 1, col] < 0:
                dy = 0.0
            if row < h - 1 and smooth[row + 1, col] < 0:
                dy = 0.0

            # Surface normal from gradient: n = (-dx, -dy, 1), normalized
            nx = -dx
            ny = -dy
            nz = 1.0
            n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= n_len
            ny /= n_len
            nz /= n_len

            # Diffuse: N dot L
            n_dot_l = nx * light_x + ny * light_y + nz * light_z
            diff = max(0.0, n_dot_l) * diffuse

            # Specular: (N dot H)^shininess
            n_dot_h = nx * hx + ny * hy + nz * hz
            spec = 0.0
            if n_dot_h > 0:
                spec = (n_dot_h ** shininess) * specular

            val = ambient + diff + spec
            # Clamp to [0, 2] to allow highlights to brighten
            lighting[row, col] = min(max(val, 0.0), 2.0)

    return lighting


def compute_slope_shading(
    smooth_data: np.ndarray,
    light_angle: float = 2.356,
    light_elevation: float = 0.6,
    ambient: float = 0.35,
    diffuse: float = 0.6,
    specular: float = 0.3,
    shininess: float = 15.0,
) -> np.ndarray:
    """Compute slope-based lighting for 3D depth effect.

    Uses numerical gradients of the smooth iteration field as a height map,
    then applies Blinn-Phong directional lighting.

    Args:
        smooth_data: 2D array of smooth iteration counts (-1.0 for interior).
        light_angle: Light direction angle in radians (0=right, pi/2=up).
            Default 2.356 (~135 deg, upper-left).
        light_elevation: Light elevation angle in radians (0=horizon, pi/2=overhead).
        ambient: Ambient light intensity [0, 1].
        diffuse: Diffuse light intensity [0, 1].
        specular: Specular highlight intensity [0, 1].
        shininess: Specular exponent (higher = tighter highlights).

    Returns:
        2D float64 array of per-pixel lighting multipliers.
    """
    return _compute_slope_lighting(
        smooth_data, light_angle, light_elevation,
        ambient, diffuse, specular, shininess,
    )


def apply_palette(
    smooth_data: np.ndarray,
    palette: np.ndarray,
    interior_color: tuple[int, int, int] = (0, 0, 0),
    histogram: bool = False,
    slope_shading: bool = False,
    light_angle: float = 2.356,
    light_elevation: float = 0.6,
    ambient: float = 0.35,
    diffuse: float = 0.6,
    specular: float = 0.3,
    shininess: float = 15.0,
    cycle_offset: float = 0.0,
    log_scaling: bool = False,
    distance_data: np.ndarray | None = None,
    color_mode: str | None = None,
) -> np.ndarray:
    """Map smooth iteration data to RGB using a color palette.

    Args:
        smooth_data: 2D array of smooth iteration counts (-1.0 for interior).
        palette: Nx3 uint8 array defining the color gradient.
        interior_color: RGB tuple for interior (non-escaping) points.
        histogram: If True, apply histogram equalization before palette mapping.
            (Legacy flag — equivalent to color_mode="histogram".)
        slope_shading: If True, apply 3D slope-based lighting.
        light_angle: Light direction angle in radians (for slope shading).
        light_elevation: Light elevation angle in radians (for slope shading).
        ambient: Ambient light intensity (for slope shading).
        diffuse: Diffuse light intensity (for slope shading).
        specular: Specular highlight intensity (for slope shading).
        shininess: Specular exponent (for slope shading).
        cycle_offset: Palette offset for color cycling animation (0.0 = no shift).
        color_mode: Value-to-palette mapping: "default" (raw cycling),
            "histogram" (continuous EQ — balanced contrast everywhere), or
            "normalized" (linear percentile sweep — dark, glowing look).
            None falls back to the histogram flag.

    Returns:
        3D uint8 array (height x width x 3) of RGB pixel data.
    """
    height, width = smooth_data.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    palette_len = len(palette)

    if color_mode is None:
        color_mode = "histogram" if histogram else "default"

    # Compute slope shading before histogram EQ (needs raw iteration topology)
    lighting = None
    if slope_shading:
        lighting = compute_slope_shading(
            smooth_data, light_angle, light_elevation,
            ambient, diffuse, specular, shininess,
        )

    # Distance coloring: use log(distance) for perfectly smooth gradients
    if distance_data is not None:
        exterior_mask = smooth_data >= 0
        dist_exterior = (distance_data > 0) & exterior_mask
        if np.any(dist_exterior):
            smooth_data = smooth_data.copy()
            log_dist = np.log(distance_data[dist_exterior])
            # Normalize to palette range: map log distance to [0, palette_len * N]
            d_min, d_max = log_dist.min(), log_dist.max()
            if d_max > d_min:
                smooth_data[dist_exterior] = (
                    (log_dist - d_min) / (d_max - d_min) * palette_len * 4.0
                )

    # Log scaling: compresses wide bands at high iteration, expands near boundary
    if log_scaling:
        exterior_mask = smooth_data >= 0
        if np.any(exterior_mask):
            smooth_data = smooth_data.copy()
            smooth_data[exterior_mask] = np.log1p(smooth_data[exterior_mask]) * (
                palette_len / np.log1p(smooth_data[exterior_mask].max())
            )

    # Remap values to palette space per the selected color mode
    if color_mode == "histogram":
        smooth_data = histogram_equalize(smooth_data, palette_len)
    elif color_mode == "normalized":
        smooth_data = normalize_range(smooth_data, palette_len)

    # Process in row chunks to avoid OOM on large SSAA renders.
    target_pixels = 8_000_000
    rows_per_chunk = max(1, target_pixels // width)

    for row_start in range(0, height, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, height)
        chunk = smooth_data[row_start:row_end]
        interior = chunk < 0
        exterior = ~interior

        if np.any(exterior):
            ext_vals = chunk[exterior]
            # Apply color cycling offset.
            # Normalize cycling speed: map iterations through a fixed-size
            # window (256) regardless of palette length. This ensures sharp
            # detail at boundaries for all palettes — without it, 2048-entry
            # sandwich palettes cycle 8x slower and look blurry at edges.
            shifted = ext_vals + cycle_offset
            cycle_len = min(palette_len, 256)
            t = (shifted % cycle_len) * (palette_len / cycle_len) % palette_len
            idx0 = t.astype(np.int32) % palette_len
            idx1 = (idx0 + 1) % palette_len
            frac = (t - np.floor(t)).reshape(-1, 1)

            color0 = palette[idx0].astype(np.float64)
            color1 = palette[idx1].astype(np.float64)
            interpolated = color0 * (1.0 - frac) + color1 * frac

            # Apply slope shading lighting
            if lighting is not None:
                light_chunk = lighting[row_start:row_end]
                light_vals = light_chunk[exterior].reshape(-1, 1)
                interpolated = interpolated * light_vals

            rgb[row_start:row_end][exterior] = np.clip(
                interpolated, 0, 255
            ).astype(np.uint8)

        rgb[row_start:row_end][interior] = interior_color

    return rgb


def smooth_to_image(
    smooth_data: np.ndarray,
    palette: np.ndarray,
    interior_color: tuple[int, int, int] = (0, 0, 0),
    histogram: bool = False,
    slope_shading: bool = False,
    light_angle: float = 2.356,
    light_elevation: float = 0.6,
    ambient: float = 0.15,
    diffuse: float = 0.7,
    specular: float = 0.4,
    shininess: float = 20.0,
    cycle_offset: float = 0.0,
    log_scaling: bool = False,
    distance_data: np.ndarray | None = None,
    color_mode: str | None = None,
) -> Image.Image:
    """Convert smooth iteration data to a PIL Image.

    Args:
        smooth_data: 2D array of smooth iteration counts.
        palette: Nx3 uint8 color palette array.
        interior_color: RGB tuple for interior points.
        histogram: If True, apply histogram equalization before coloring.
        slope_shading: If True, apply 3D slope-based lighting.
        light_angle: Light direction angle in radians.
        light_elevation: Light elevation angle in radians.
        ambient: Ambient light intensity.
        diffuse: Diffuse light intensity.
        specular: Specular highlight intensity.
        shininess: Specular exponent.
        cycle_offset: Palette offset for color cycling.
        log_scaling: If True, apply log scaling to compress wide iteration bands.

    Returns:
        PIL Image in RGB mode.
    """
    rgb = apply_palette(
        smooth_data, palette, interior_color,
        histogram=histogram,
        slope_shading=slope_shading,
        light_angle=light_angle,
        light_elevation=light_elevation,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        shininess=shininess,
        cycle_offset=cycle_offset,
        log_scaling=log_scaling,
        distance_data=distance_data,
        color_mode=color_mode,
    )
    return Image.fromarray(rgb, mode="RGB")
