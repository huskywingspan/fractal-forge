"""Thumbnail auto-sampler for YouTube publishing.

Selects visually interesting frames from a rendered zoom sequence (biased
toward the deeper end), resizes to 1280x720, and composites brand elements:
- Semi-transparent gradient strip along the bottom
- Zoom depth text (large, bold) bottom-left
- Channel watermark bottom-right in accent cyan

Outputs multiple candidates so the creator can pick the best one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from fractalforge.publish.brand import BRAND, get_font, scale_size


# ---------------------------------------------------------------------------
# YouTube thumbnail dimensions
# ---------------------------------------------------------------------------

THUMB_WIDTH = 1280
THUMB_HEIGHT = 720


# ---------------------------------------------------------------------------
# Zoom formatting
# ---------------------------------------------------------------------------

def format_zoom(zoom: float) -> str:
    """Format a zoom level as a human-readable string with magnitude suffix.

    Examples:
        >>> format_zoom(1e10)
        '10 BILLION x'
        >>> format_zoom(2.5e6)
        '3 MILLION x'
        >>> format_zoom(1500)
        '2K x'
        >>> format_zoom(42)
        '42x'
    """
    if zoom >= 1e12:
        return f"{zoom / 1e12:.0f} TRILLION x"
    if zoom >= 1e9:
        return f"{zoom / 1e9:.0f} BILLION x"
    if zoom >= 1e6:
        return f"{zoom / 1e6:.0f} MILLION x"
    if zoom >= 1e3:
        return f"{zoom / 1e3:.0f}K x"
    return f"{zoom:.0f}x"


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

# Sample positions as fractions of total_frames -- biased toward the deep end
# where visuals are typically most interesting.
_SAMPLE_POSITIONS = [0.40, 0.55, 0.70, 0.85, 0.95]


def _pick_frame_indices(total_frames: int, num_samples: int) -> list[int]:
    """Choose *num_samples* frame indices, biased toward the end.

    Uses fixed percentage positions for the first 5; if more are requested
    the extras are evenly spaced between 40% and 95%.
    """
    if num_samples <= len(_SAMPLE_POSITIONS):
        positions = _SAMPLE_POSITIONS[:num_samples]
    else:
        positions = [
            0.40 + (0.55 * i / (num_samples - 1))
            for i in range(num_samples)
        ]

    indices = []
    for p in positions:
        idx = min(int(p * total_frames), total_frames - 1)
        indices.append(max(0, idx))
    return indices


# ---------------------------------------------------------------------------
# Single thumbnail compositing
# ---------------------------------------------------------------------------

def _composite_thumbnail(
    frame_img: Image.Image,
    zoom_text: Optional[str] = None,
    title_text: Optional[str] = None,
) -> Image.Image:
    """Resize a frame and overlay brand elements for a YouTube thumbnail.

    Args:
        frame_img: Source fractal frame (any resolution).
        zoom_text: Zoom depth string (e.g. ``"10 BILLION x"``).
        title_text: Optional title text placed above the zoom text.

    Returns:
        An RGB PIL Image at exactly 1280x720.
    """
    # Resize to thumbnail dimensions
    thumb = frame_img.convert("RGB").resize(
        (THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS
    )

    draw = ImageDraw.Draw(thumb)

    # ------------------------------------------------------------------
    # Bottom gradient strip (~20% of height)
    # ------------------------------------------------------------------
    gradient_top = int(THUMB_HEIGHT * 0.80)
    for y in range(gradient_top, THUMB_HEIGHT):
        progress = (y - gradient_top) / (THUMB_HEIGHT - gradient_top)
        alpha = int(200 * progress)
        draw.line(
            [(0, y), (THUMB_WIDTH, y)],
            fill=(0, 0, 0, alpha),
        )

    # To apply alpha blending properly on an RGB image, composite via RGBA.
    overlay = Image.new("RGBA", (THUMB_WIDTH, THUMB_HEIGHT), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for y in range(gradient_top, THUMB_HEIGHT):
        progress = (y - gradient_top) / (THUMB_HEIGHT - gradient_top)
        alpha = int(200 * progress)
        overlay_draw.line([(0, y), (THUMB_WIDTH, y)], fill=(0, 0, 0, alpha))

    thumb_rgba = thumb.convert("RGBA")
    thumb_rgba = Image.alpha_composite(thumb_rgba, overlay)

    draw = ImageDraw.Draw(thumb_rgba)

    margin = scale_size(24, THUMB_HEIGHT, reference_height=720)

    # ------------------------------------------------------------------
    # Zoom depth text -- bottom-left, large bold font
    # ------------------------------------------------------------------
    if zoom_text:
        zoom_font_size = scale_size(72, THUMB_HEIGHT, reference_height=720)
        font_zoom = get_font("montserrat-bold", zoom_font_size)

        zoom_bbox = draw.textbbox((0, 0), zoom_text, font=font_zoom)
        zoom_h = zoom_bbox[3] - zoom_bbox[1]

        zoom_x = margin
        zoom_y = THUMB_HEIGHT - zoom_h - margin

        # Text shadow for readability
        shadow_offset = max(1, scale_size(2, THUMB_HEIGHT, reference_height=720))
        draw.text(
            (zoom_x + shadow_offset, zoom_y + shadow_offset),
            zoom_text,
            fill=(0, 0, 0, 200),
            font=font_zoom,
        )
        draw.text(
            (zoom_x, zoom_y),
            zoom_text,
            fill=(*BRAND["text_color"], 255),
            font=font_zoom,
        )

    # ------------------------------------------------------------------
    # Optional title text -- above zoom text
    # ------------------------------------------------------------------
    if title_text:
        title_font_size = scale_size(28, THUMB_HEIGHT, reference_height=720)
        font_title = get_font("montserrat", title_font_size)

        title_bbox = draw.textbbox((0, 0), title_text, font=font_title)
        title_h = title_bbox[3] - title_bbox[1]

        title_x = margin
        title_y = THUMB_HEIGHT - (zoom_h + margin if zoom_text else 0) - title_h - margin
        if zoom_text:
            title_y -= scale_size(8, THUMB_HEIGHT, reference_height=720)

        draw.text(
            (title_x, title_y),
            title_text,
            fill=(*BRAND["text_color"], 230),
            font=font_title,
        )

    # ------------------------------------------------------------------
    # Channel watermark -- bottom-right, accent cyan
    # ------------------------------------------------------------------
    wm_font_size = scale_size(18, THUMB_HEIGHT, reference_height=720)
    font_wm = get_font("rajdhani-medium", wm_font_size)

    wm_text = BRAND["channel_name"]
    wm_bbox = draw.textbbox((0, 0), wm_text, font=font_wm)
    wm_w = wm_bbox[2] - wm_bbox[0]
    wm_h = wm_bbox[3] - wm_bbox[1]

    wm_x = THUMB_WIDTH - wm_w - margin
    wm_y = THUMB_HEIGHT - wm_h - margin

    draw.text(
        (wm_x, wm_y),
        wm_text,
        fill=(*BRAND["accent_color"], 220),
        font=font_wm,
    )

    # Convert back to RGB for final PNG
    return thumb_rgba.convert("RGB")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_thumbnail_samples(
    frames_dir: Path,
    total_frames: int,
    num_samples: int = 5,
    output_dir: Optional[Path] = None,
    title_text: Optional[str] = None,
    zoom_text: Optional[str] = None,
) -> list[Path]:
    """Generate thumbnail candidates from a rendered frame sequence.

    Selects frames biased toward the deeper zoom end, composites brand
    elements, and saves as ``thumb_001.png`` through ``thumb_NNN.png``.

    Args:
        frames_dir: Directory containing ``frame_000000.png`` etc.
        total_frames: Total number of frames in the sequence.
        num_samples: How many thumbnail candidates to generate.
        output_dir: Where to save thumbnails (default: ``frames_dir / "thumbnails"``).
        title_text: Optional title text overlaid on each thumbnail.
        zoom_text: Zoom depth string (e.g. ``"10 BILLION x"``). If ``None``,
                   no zoom text is drawn.

    Returns:
        List of paths to the generated thumbnail PNGs.
    """
    frames_dir = Path(frames_dir)
    if output_dir is None:
        output_dir = frames_dir / "thumbnails"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = _pick_frame_indices(total_frames, num_samples)
    output_paths: list[Path] = []

    for sample_num, frame_idx in enumerate(frame_indices, start=1):
        # Standard frame naming convention: frame_000000.png
        frame_file = frames_dir / f"frame_{frame_idx:06d}.png"

        if not frame_file.exists():
            # Try nearby frames if the exact one is missing
            found = False
            for offset in range(-5, 6):
                alt = frames_dir / f"frame_{max(0, frame_idx + offset):06d}.png"
                if alt.exists():
                    frame_file = alt
                    found = True
                    break
            if not found:
                continue  # Skip this sample -- no frames available

        frame_img = Image.open(frame_file)
        thumb = _composite_thumbnail(
            frame_img,
            zoom_text=zoom_text,
            title_text=title_text,
        )

        out_path = output_dir / f"thumb_{sample_num:03d}.png"
        thumb.save(str(out_path), "PNG")
        output_paths.append(out_path)

    return output_paths
