"""Infinite Descent brand configuration and font helpers.

Centralizes the channel's visual identity -- colors, fonts, sizes -- so that
title cards, thumbnails, and any future publishing tools stay consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import ImageFont


# ---------------------------------------------------------------------------
# Brand identity
# ---------------------------------------------------------------------------

BRAND = {
    "channel_name": "Infinite Descent",
    "tagline": "Falling forever into infinite detail.",
    "bg_color": (10, 14, 26),          # #0a0e1a  Deep Space Navy
    "accent_color": (0, 212, 255),      # #00d4ff  Electric Cyan
    "accent2_color": (168, 85, 247),    # #a855f7  Soft Violet
    "text_color": (240, 240, 240),      # #f0f0f0  White
    "text_secondary": (148, 163, 184),  # #94a3b8  Silver
}


# ---------------------------------------------------------------------------
# Font resolution
# ---------------------------------------------------------------------------

# Project root: walk up from this file to src/fractalforge/publish -> src/fractalforge -> src -> project
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Font search paths -- tried in order for each family.
_FONT_PATHS = {
    "montserrat": [
        _PROJECT_ROOT / "assets" / "fonts" / "Montserrat-SemiBold.ttf",
        Path("C:/Windows/Fonts/Montserrat-SemiBold.ttf"),
        Path("C:/Windows/Fonts/montserrat-semibold.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),             # Windows fallback
    ],
    "montserrat-bold": [
        _PROJECT_ROOT / "assets" / "fonts" / "Montserrat-ExtraBold.ttf",
        Path("C:/Windows/Fonts/Montserrat-ExtraBold.ttf"),
        Path("C:/Windows/Fonts/montserrat-extrabold.ttf"),
        _PROJECT_ROOT / "assets" / "fonts" / "Montserrat-Bold.ttf",
        Path("C:/Windows/Fonts/Montserrat-Bold.ttf"),
        Path("C:/Windows/Fonts/montserrat-bold.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),             # Windows fallback (bold)
    ],
    "montserrat-light": [
        _PROJECT_ROOT / "assets" / "fonts" / "Montserrat-Light.ttf",
        Path("C:/Windows/Fonts/Montserrat-Light.ttf"),
        Path("C:/Windows/Fonts/montserrat-light.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ],
    "rajdhani": [
        _PROJECT_ROOT / "assets" / "fonts" / "Rajdhani-Bold.ttf",
        Path("C:/Windows/Fonts/Rajdhani-Bold.ttf"),
        Path("C:/Windows/Fonts/rajdhani-bold.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),             # Windows fallback (bold)
    ],
    "rajdhani-medium": [
        _PROJECT_ROOT / "assets" / "fonts" / "Rajdhani-Medium.ttf",
        Path("C:/Windows/Fonts/Rajdhani-Medium.ttf"),
        Path("C:/Windows/Fonts/rajdhani-medium.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ],
}


def get_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a brand font by family name at the requested pixel size.

    Tries project-local fonts first, then system-installed fonts, then falls
    back to Pillow's built-in bitmap font (which ignores *size*).

    Args:
        name: Font family key -- one of ``"montserrat"``, ``"montserrat-bold"``,
              ``"montserrat-light"``, ``"rajdhani"``, ``"rajdhani-medium"``.
        size: Desired font size in pixels.

    Returns:
        A PIL ``ImageFont`` instance.
    """
    candidates = _FONT_PATHS.get(name.lower(), [])
    for font_path in candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except (OSError, IOError):
                continue

    # Last resort: Pillow default bitmap font (size cannot be changed).
    return ImageFont.load_default()


def scale_size(base_size: int, target_height: int, reference_height: int = 1080) -> int:
    """Scale a font/element size proportionally to output resolution.

    Args:
        base_size: Size in pixels at the reference resolution.
        target_height: Actual output height in pixels.
        reference_height: Height the base_size was designed for (default 1080).

    Returns:
        Proportionally scaled integer size.
    """
    return max(1, round(base_size * target_height / reference_height))
