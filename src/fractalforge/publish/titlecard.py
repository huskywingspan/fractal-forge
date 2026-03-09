"""Title card overlay renderer for DaVinci Resolve compositing.

Produces an RGBA PNG with transparent background, containing:
- Semi-transparent dark gradient in the top ~35% (text readability)
- Channel name in accent cyan (top-left)
- Video title centered in the top third (white, semibold)
- Optional subtitle below the title (silver)

The overlay is designed to be placed on a track above the fractal video
and faded out over ~4 seconds in Resolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from fractalforge.publish.brand import BRAND, get_font, scale_size


def render_title_card(
    title: str,
    subtitle: str = "",
    width: int = 1920,
    height: int = 1080,
    output_path: Optional[Path] = None,
) -> Image.Image:
    """Render a title card overlay as an RGBA image.

    Args:
        title: Main video title text.
        subtitle: Optional subtitle (e.g. zoom depth or location).
        width: Output width in pixels.
        height: Output height in pixels.
        output_path: If provided, save the image to this path.

    Returns:
        The rendered ``PIL.Image.Image`` in RGBA mode.
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ------------------------------------------------------------------
    # Gradient overlay: black at alpha ~180 at the top, fading to
    # alpha 0 at ~35% of the frame height.
    # ------------------------------------------------------------------
    gradient_bottom = int(height * 0.35)
    for y in range(gradient_bottom):
        # Linear fade from alpha 180 (top) to 0 (bottom of gradient band)
        alpha = int(180 * (1 - y / gradient_bottom))
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

    # ------------------------------------------------------------------
    # Font sizes -- scale proportionally with resolution
    # ------------------------------------------------------------------
    channel_size = scale_size(24, height)
    title_size = scale_size(64, height)
    subtitle_size = scale_size(32, height)

    font_channel = get_font("rajdhani", channel_size)
    font_title = get_font("montserrat", title_size)
    font_subtitle = get_font("montserrat-light", subtitle_size)

    # ------------------------------------------------------------------
    # Channel name -- top-left corner
    # ------------------------------------------------------------------
    margin_x = scale_size(40, height)
    margin_y = scale_size(30, height)

    draw.text(
        (margin_x, margin_y),
        BRAND["channel_name"],
        fill=(*BRAND["accent_color"], 255),
        font=font_channel,
    )

    # ------------------------------------------------------------------
    # Video title -- centered horizontally in the top third
    # ------------------------------------------------------------------
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]

    title_x = (width - title_w) // 2
    # Vertically center in the top third of the frame
    top_third = height // 3
    title_y = (top_third - title_h) // 2 + scale_size(20, height)

    draw.text(
        (title_x, title_y),
        title,
        fill=(*BRAND["text_color"], 255),
        font=font_title,
    )

    # ------------------------------------------------------------------
    # Subtitle -- below the title, centered
    # ------------------------------------------------------------------
    if subtitle:
        sub_bbox = draw.textbbox((0, 0), subtitle, font=font_subtitle)
        sub_w = sub_bbox[2] - sub_bbox[0]

        sub_x = (width - sub_w) // 2
        sub_y = title_y + title_h + scale_size(16, height)

        draw.text(
            (sub_x, sub_y),
            subtitle,
            fill=(*BRAND["text_secondary"], 255),
            font=font_subtitle,
        )

    # ------------------------------------------------------------------
    # Save if requested
    # ------------------------------------------------------------------
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), "PNG")

    return img
