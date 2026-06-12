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


# --- Cyclic palettes (wrap seamlessly for color cycling) ---
# Higher resolution (1024 entries) with multiple hue cycles for rich detail.

PALETTE_DEEP_BLUE = linear_gradient([
    (0, 10, 60),
    (0, 50, 180),
    (80, 140, 255),
    (200, 220, 255),
    (255, 255, 255),
    (200, 180, 255),
    (100, 60, 200),
    (30, 10, 100),
    (0, 10, 60),
], steps=1024)

PALETTE_INFERNO = linear_gradient([
    (20, 0, 0),
    (120, 10, 0),
    (220, 60, 0),
    (255, 160, 20),
    (255, 255, 120),
    (255, 200, 80),
    (200, 80, 10),
    (100, 20, 0),
    (20, 0, 0),
], steps=1024)

PALETTE_ARCTIC = linear_gradient([
    (10, 10, 40),
    (30, 60, 140),
    (60, 160, 220),
    (180, 230, 255),
    (255, 255, 255),
    (200, 240, 255),
    (80, 180, 240),
    (20, 80, 160),
    (10, 10, 40),
], steps=1024)

PALETTE_PRISM = linear_gradient([
    (255, 40, 40),
    (255, 180, 20),
    (200, 255, 40),
    (20, 255, 120),
    (20, 200, 255),
    (60, 80, 255),
    (180, 40, 255),
    (255, 40, 160),
    (255, 40, 40),
], steps=1024)

PALETTE_TWILIGHT = linear_gradient([
    (10, 5, 30),
    (40, 10, 80),
    (120, 30, 160),
    (200, 80, 180),
    (255, 180, 200),
    (255, 220, 180),
    (200, 160, 120),
    (80, 60, 80),
    (10, 5, 30),
], steps=1024)


# --- Sandwich palettes (dark/colored bands for striped wave effects) ---
# These create dramatic banded visuals. With color cycling, bands sweep
# across the fractal surface like waves. Built from explicit color stop
# sequences at high resolution (2048+ entries) for smooth results.

# Deep ocean theme: navy/cobalt bands with white crests, black separators
PALETTE_OCEAN_WAVES = linear_gradient([
    (0, 0, 0),
    (10, 20, 80), (40, 80, 200), (120, 180, 255), (200, 230, 255),
    (120, 180, 255), (40, 80, 200), (10, 20, 80),
    (0, 0, 0),
    (5, 10, 40), (20, 40, 120), (60, 120, 220), (160, 200, 255),
    (60, 120, 220), (20, 40, 120), (5, 10, 40),
    (0, 0, 0),
    (15, 30, 100), (80, 140, 240), (180, 220, 255), (255, 255, 255),
    (180, 220, 255), (80, 140, 240), (15, 30, 100),
    (0, 0, 0),
    (8, 15, 60), (30, 60, 160), (90, 150, 230), (140, 190, 255),
    (90, 150, 230), (30, 60, 160), (8, 15, 60),
    (0, 0, 0),
], steps=2048)

# Volcanic: black/red/orange/gold bands like lava flows
PALETTE_VOLCANIC = linear_gradient([
    (0, 0, 0),
    (60, 0, 0), (160, 20, 0), (255, 80, 0), (255, 180, 40),
    (255, 80, 0), (160, 20, 0), (60, 0, 0),
    (0, 0, 0),
    (40, 10, 0), (120, 40, 0), (220, 120, 0), (255, 220, 80),
    (255, 255, 160),
    (255, 220, 80), (220, 120, 0), (120, 40, 0), (40, 10, 0),
    (0, 0, 0),
    (80, 0, 0), (200, 40, 0), (255, 100, 10), (255, 160, 30),
    (255, 100, 10), (200, 40, 0), (80, 0, 0),
    (0, 0, 0),
    (30, 5, 0), (100, 30, 0), (180, 80, 10), (255, 200, 100),
    (180, 80, 10), (100, 30, 0), (30, 5, 0),
    (0, 0, 0),
], steps=2048)

# Aurora borealis: green/cyan/purple bands on dark sky
PALETTE_AURORA = linear_gradient([
    (0, 5, 15),
    (0, 40, 20), (0, 160, 80), (40, 255, 140), (120, 255, 200),
    (40, 255, 140), (0, 160, 80), (0, 40, 20),
    (0, 5, 15),
    (20, 0, 40), (80, 0, 160), (160, 40, 255), (200, 120, 255),
    (160, 40, 255), (80, 0, 160), (20, 0, 40),
    (0, 5, 15),
    (0, 20, 30), (0, 100, 120), (0, 220, 200), (80, 255, 240),
    (0, 220, 200), (0, 100, 120), (0, 20, 30),
    (0, 5, 15),
    (10, 0, 20), (60, 0, 100), (120, 20, 200), (180, 80, 255),
    (120, 20, 200), (60, 0, 100), (10, 0, 20),
    (0, 5, 15),
], steps=2048)

# Neon city: electric pink/cyan/yellow bands on black
PALETTE_NEON_CITY = linear_gradient([
    (0, 0, 0),
    (80, 0, 60), (255, 0, 180), (255, 100, 220), (255, 180, 240),
    (255, 100, 220), (255, 0, 180), (80, 0, 60),
    (0, 0, 0),
    (0, 40, 60), (0, 180, 255), (80, 220, 255), (180, 240, 255),
    (80, 220, 255), (0, 180, 255), (0, 40, 60),
    (0, 0, 0),
    (60, 60, 0), (255, 255, 0), (255, 255, 120), (255, 255, 200),
    (255, 255, 120), (255, 255, 0), (60, 60, 0),
    (0, 0, 0),
    (60, 0, 80), (200, 0, 255), (240, 100, 255), (255, 180, 255),
    (240, 100, 255), (200, 0, 255), (60, 0, 80),
    (0, 0, 0),
], steps=2048)

# Copper patina: warm copper/teal/verdigris bands
PALETTE_PATINA = linear_gradient([
    (10, 5, 0),
    (60, 30, 10), (160, 80, 30), (220, 140, 60), (255, 200, 120),
    (220, 140, 60), (160, 80, 30), (60, 30, 10),
    (10, 5, 0),
    (0, 20, 20), (0, 80, 70), (20, 160, 140), (80, 220, 200),
    (20, 160, 140), (0, 80, 70), (0, 20, 20),
    (10, 5, 0),
    (80, 40, 15), (180, 100, 40), (240, 180, 100), (255, 230, 180),
    (240, 180, 100), (180, 100, 40), (80, 40, 15),
    (10, 5, 0),
    (0, 30, 25), (0, 100, 80), (40, 180, 160), (100, 230, 210),
    (40, 180, 160), (0, 100, 80), (0, 30, 25),
    (10, 5, 0),
], steps=2048)

# Royal: gold/purple/crimson bands on deep black
PALETTE_ROYAL = linear_gradient([
    (0, 0, 0),
    (60, 40, 0), (200, 160, 0), (255, 220, 40), (255, 245, 140),
    (255, 220, 40), (200, 160, 0), (60, 40, 0),
    (0, 0, 0),
    (30, 0, 50), (100, 0, 160), (160, 40, 220), (200, 100, 255),
    (160, 40, 220), (100, 0, 160), (30, 0, 50),
    (0, 0, 0),
    (50, 0, 0), (160, 0, 20), (220, 20, 40), (255, 80, 80),
    (220, 20, 40), (160, 0, 20), (50, 0, 0),
    (0, 0, 0),
    (40, 30, 0), (140, 120, 0), (220, 200, 40), (255, 240, 120),
    (220, 200, 40), (140, 120, 0), (40, 30, 0),
    (0, 0, 0),
], steps=2048)

# Abyssal: deep sea bioluminescence — dark teal with bright accents
PALETTE_ABYSSAL = linear_gradient([
    (0, 5, 10),
    (0, 20, 40), (0, 60, 80), (0, 120, 140), (40, 200, 200),
    (120, 255, 240),
    (40, 200, 200), (0, 120, 140), (0, 60, 80), (0, 20, 40),
    (0, 5, 10),
    (5, 0, 10), (20, 0, 40), (60, 0, 100), (140, 40, 200),
    (200, 120, 255),
    (140, 40, 200), (60, 0, 100), (20, 0, 40), (5, 0, 10),
    (0, 5, 10),
    (0, 15, 30), (0, 50, 60), (0, 100, 110), (20, 180, 180),
    (80, 240, 220),
    (20, 180, 180), (0, 100, 110), (0, 50, 60), (0, 15, 30),
    (0, 5, 10),
], steps=2048)

# Solar flare: white-hot center bands fading through orange to black
PALETTE_SOLAR_FLARE = linear_gradient([
    (0, 0, 0),
    (40, 5, 0), (120, 20, 0), (220, 80, 0), (255, 180, 40),
    (255, 240, 160), (255, 255, 255),
    (255, 240, 160), (255, 180, 40), (220, 80, 0), (120, 20, 0), (40, 5, 0),
    (0, 0, 0),
    (30, 10, 0), (100, 40, 0), (200, 100, 20), (255, 200, 80),
    (255, 255, 200),
    (255, 200, 80), (200, 100, 20), (100, 40, 0), (30, 10, 0),
    (0, 0, 0),
    (50, 10, 0), (160, 50, 0), (240, 120, 10), (255, 210, 60),
    (255, 250, 180), (255, 255, 240),
    (255, 250, 180), (255, 210, 60), (240, 120, 10), (160, 50, 0), (50, 10, 0),
    (0, 0, 0),
], steps=2048)

# Frozen: ice blue/white bands with subtle purple on dark navy
PALETTE_FROZEN = linear_gradient([
    (5, 5, 20),
    (20, 30, 80), (60, 80, 180), (140, 160, 240), (220, 230, 255),
    (255, 255, 255),
    (220, 230, 255), (140, 160, 240), (60, 80, 180), (20, 30, 80),
    (5, 5, 20),
    (15, 10, 30), (40, 20, 80), (100, 60, 180), (180, 140, 240),
    (230, 210, 255),
    (180, 140, 240), (100, 60, 180), (40, 20, 80), (15, 10, 30),
    (5, 5, 20),
    (15, 20, 60), (40, 60, 140), (100, 120, 220), (180, 200, 255),
    (240, 245, 255),
    (180, 200, 255), (100, 120, 220), (40, 60, 140), (15, 20, 60),
    (5, 5, 20),
], steps=2048)

# Jungle: dark greens with gold and amber highlights
PALETTE_JUNGLE = linear_gradient([
    (0, 5, 0),
    (0, 30, 10), (0, 80, 20), (20, 160, 40), (80, 220, 80),
    (160, 255, 120),
    (80, 220, 80), (20, 160, 40), (0, 80, 20), (0, 30, 10),
    (0, 5, 0),
    (10, 5, 0), (40, 30, 0), (120, 80, 0), (200, 160, 20),
    (255, 220, 80),
    (200, 160, 20), (120, 80, 0), (40, 30, 0), (10, 5, 0),
    (0, 5, 0),
    (0, 20, 5), (0, 60, 15), (10, 120, 30), (40, 180, 60),
    (100, 240, 100),
    (40, 180, 60), (10, 120, 30), (0, 60, 15), (0, 20, 5),
    (0, 5, 0),
    (5, 5, 0), (30, 20, 0), (80, 60, 0), (160, 120, 10),
    (220, 180, 40),
    (160, 120, 10), (80, 60, 0), (30, 20, 0), (5, 5, 0),
    (0, 5, 0),
], steps=2048)

# Vaporwave: retro pink/cyan/purple aesthetic
PALETTE_VAPORWAVE = linear_gradient([
    (10, 0, 20),
    (60, 0, 80), (180, 0, 200), (255, 60, 220), (255, 150, 240),
    (255, 60, 220), (180, 0, 200), (60, 0, 80),
    (10, 0, 20),
    (0, 20, 40), (0, 80, 140), (0, 180, 220), (80, 240, 255),
    (180, 255, 255),
    (80, 240, 255), (0, 180, 220), (0, 80, 140), (0, 20, 40),
    (10, 0, 20),
    (40, 0, 60), (120, 0, 140), (200, 40, 200), (240, 120, 240),
    (255, 200, 255),
    (240, 120, 240), (200, 40, 200), (120, 0, 140), (40, 0, 60),
    (10, 0, 20),
    (0, 10, 30), (0, 40, 80), (0, 120, 160), (40, 200, 220),
    (120, 255, 240),
    (40, 200, 220), (0, 120, 160), (0, 40, 80), (0, 10, 30),
    (10, 0, 20),
], steps=2048)

# Stained glass: jewel tones with dark lead lines
PALETTE_STAINED_GLASS = linear_gradient([
    (5, 5, 5),
    (100, 0, 0), (200, 20, 20), (255, 80, 60), (255, 160, 120),
    (255, 80, 60), (200, 20, 20), (100, 0, 0),
    (5, 5, 5),
    (0, 0, 100), (20, 20, 200), (60, 80, 255), (140, 160, 255),
    (60, 80, 255), (20, 20, 200), (0, 0, 100),
    (5, 5, 5),
    (0, 80, 0), (20, 160, 20), (80, 220, 60), (160, 255, 120),
    (80, 220, 60), (20, 160, 20), (0, 80, 0),
    (5, 5, 5),
    (80, 60, 0), (180, 140, 0), (240, 200, 40), (255, 240, 120),
    (240, 200, 40), (180, 140, 0), (80, 60, 0),
    (5, 5, 5),
    (60, 0, 80), (140, 0, 180), (200, 60, 240), (230, 140, 255),
    (200, 60, 240), (140, 0, 180), (60, 0, 80),
    (5, 5, 5),
], steps=2048)

# Midnight rose: deep reds/pinks with dark blue separators
PALETTE_MIDNIGHT_ROSE = linear_gradient([
    (5, 0, 15),
    (30, 0, 10), (100, 0, 20), (180, 20, 60), (240, 80, 120),
    (255, 160, 180),
    (240, 80, 120), (180, 20, 60), (100, 0, 20), (30, 0, 10),
    (5, 0, 15),
    (10, 0, 30), (30, 0, 80), (60, 20, 140), (100, 60, 200),
    (160, 120, 240),
    (100, 60, 200), (60, 20, 140), (30, 0, 80), (10, 0, 30),
    (5, 0, 15),
    (40, 0, 15), (120, 10, 40), (200, 40, 80), (255, 100, 140),
    (255, 180, 200), (255, 220, 230),
    (255, 180, 200), (255, 100, 140), (200, 40, 80), (120, 10, 40), (40, 0, 15),
    (5, 0, 15),
], steps=2048)

# Supernova: white-hot cores with colorful nebula bands
PALETTE_SUPERNOVA = linear_gradient([
    (0, 0, 5),
    (40, 0, 60), (100, 0, 160), (180, 40, 220), (240, 140, 255),
    (255, 220, 255), (255, 255, 255),
    (255, 220, 255), (240, 140, 255), (180, 40, 220), (100, 0, 160), (40, 0, 60),
    (0, 0, 5),
    (0, 20, 40), (0, 60, 120), (0, 140, 200), (40, 220, 255),
    (160, 250, 255), (255, 255, 255),
    (160, 250, 255), (40, 220, 255), (0, 140, 200), (0, 60, 120), (0, 20, 40),
    (0, 0, 5),
    (30, 10, 0), (100, 40, 0), (200, 100, 0), (255, 200, 40),
    (255, 245, 180), (255, 255, 255),
    (255, 245, 180), (255, 200, 40), (200, 100, 0), (100, 40, 0), (30, 10, 0),
    (0, 0, 5),
], steps=2048)


BUILTIN_PALETTES: dict[str, np.ndarray] = {
    "ocean": PALETTE_OCEAN,
    "fire": PALETTE_FIRE,
    "electric": PALETTE_ELECTRIC,
    "monochrome": PALETTE_MONOCHROME,
    "nebula": PALETTE_NEBULA,
    "deep_blue": PALETTE_DEEP_BLUE,
    "inferno": PALETTE_INFERNO,
    "arctic": PALETTE_ARCTIC,
    "prism": PALETTE_PRISM,
    "twilight": PALETTE_TWILIGHT,
    "ocean_waves": PALETTE_OCEAN_WAVES,
    "volcanic": PALETTE_VOLCANIC,
    "aurora": PALETTE_AURORA,
    "neon_city": PALETTE_NEON_CITY,
    "patina": PALETTE_PATINA,
    "royal": PALETTE_ROYAL,
    "abyssal": PALETTE_ABYSSAL,
    "solar_flare": PALETTE_SOLAR_FLARE,
    "frozen": PALETTE_FROZEN,
    "jungle": PALETTE_JUNGLE,
    "vaporwave": PALETTE_VAPORWAVE,
    "stained_glass": PALETTE_STAINED_GLASS,
    "midnight_rose": PALETTE_MIDNIGHT_ROSE,
    "supernova": PALETTE_SUPERNOVA,
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
