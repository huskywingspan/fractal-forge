"""Color palette system — gradient design, preset palettes, interpolation.

Palettes are represented as Nx3 uint8 arrays. The coloring system maps
smooth iteration counts into these palettes with linear interpolation.
"""

import json
from pathlib import Path

import numpy as np


def linear_gradient(colors: list[tuple[int, int, int]], steps: int = 256) -> np.ndarray:
    """Create a smooth gradient palette from a list of color stops.

    Args:
        colors: List of (R, G, B) color stops to interpolate between.
        steps: Total number of entries in the resulting palette.

    Returns:
        Nx3 uint8 array.
    """
    if len(colors) < 2:
        raise ValueError("Need at least 2 colors for a gradient")

    palette = np.zeros((steps, 3), dtype=np.uint8)
    n_segments = len(colors) - 1
    steps_per_segment = steps / n_segments

    for i in range(steps):
        seg = min(int(i / steps_per_segment), n_segments - 1)
        t = (i - seg * steps_per_segment) / steps_per_segment

        c0 = np.array(colors[seg], dtype=np.float64)
        c1 = np.array(colors[seg + 1], dtype=np.float64)
        palette[i] = (c0 * (1.0 - t) + c1 * t).astype(np.uint8)

    return palette


# --- Built-in palette presets ---

PALETTE_OCEAN = linear_gradient([
    (0, 7, 100),
    (32, 107, 203),
    (237, 255, 255),
    (255, 170, 0),
    (0, 2, 0),
], steps=256)

PALETTE_FIRE = linear_gradient([
    (0, 0, 0),
    (128, 0, 0),
    (255, 128, 0),
    (255, 255, 100),
    (255, 255, 255),
], steps=256)

PALETTE_ELECTRIC = linear_gradient([
    (0, 0, 0),
    (0, 0, 180),
    (0, 200, 255),
    (255, 255, 255),
    (200, 100, 255),
    (60, 0, 120),
    (0, 0, 0),
], steps=256)

PALETTE_MONOCHROME = linear_gradient([
    (0, 0, 0),
    (255, 255, 255),
], steps=256)

PALETTE_NEBULA = linear_gradient([
    (10, 0, 20),
    (60, 0, 100),
    (180, 40, 200),
    (255, 150, 220),
    (255, 255, 255),
    (100, 200, 255),
    (0, 50, 120),
    (10, 0, 20),
], steps=256)


BUILTIN_PALETTES: dict[str, np.ndarray] = {
    "ocean": PALETTE_OCEAN,
    "fire": PALETTE_FIRE,
    "electric": PALETTE_ELECTRIC,
    "monochrome": PALETTE_MONOCHROME,
    "nebula": PALETTE_NEBULA,
}


def get_palette(name: str) -> np.ndarray:
    """Get a built-in palette by name.

    Args:
        name: Palette name (ocean, fire, electric, monochrome, nebula).

    Returns:
        Nx3 uint8 palette array.

    Raises:
        KeyError: If palette name is not found.
    """
    if name not in BUILTIN_PALETTES:
        available = ", ".join(sorted(BUILTIN_PALETTES.keys()))
        raise KeyError(f"Unknown palette '{name}'. Available: {available}")
    return BUILTIN_PALETTES[name]


def save_palette(palette: np.ndarray, path: Path) -> None:
    """Save a palette to a JSON file."""
    data = {"colors": palette.tolist()}
    path.write_text(json.dumps(data, indent=2))


def load_palette(path: Path) -> np.ndarray:
    """Load a palette from a JSON file."""
    data = json.loads(path.read_text())
    return np.array(data["colors"], dtype=np.uint8)
