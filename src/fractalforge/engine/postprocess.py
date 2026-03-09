"""Post-processing effects -- vignette, color grading.

Applied to PIL Images after coloring, before save.
"""

import numpy as np
from PIL import Image, ImageEnhance


def apply_vignette(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """Apply a radial vignette (darken edges) to a PIL Image.

    Args:
        img: Input RGB image.
        strength: Vignette intensity from 0.0 (none) to 1.0 (edges go to black).

    Returns:
        New PIL Image with vignette applied.
    """
    if strength <= 0:
        return img

    strength = min(strength, 1.0)
    w, h = img.size

    # Build radial distance mask from center, normalized to [0, 1]
    cx, cy = w / 2, h / 2
    # Max distance is from center to corner
    max_dist = np.sqrt(cx**2 + cy**2)

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist

    # Vignette curve: smooth falloff using cosine-based curve
    # At center (dist=0): mask=1.0, at corners (dist=1): mask=1-strength
    mask = 1.0 - strength * (dist**2)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    # Apply mask to each channel
    arr = np.array(img, dtype=np.float32)
    arr *= mask[:, :, np.newaxis]
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode=img.mode)


def apply_color_grade(
    img: Image.Image,
    contrast: float = 1.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
) -> Image.Image:
    """Apply basic color grading adjustments.

    Args:
        img: Input RGB image.
        contrast: Contrast multiplier (1.0 = unchanged, >1 = more contrast).
        saturation: Saturation multiplier (1.0 = unchanged, 0 = grayscale).
        brightness: Brightness multiplier (1.0 = unchanged, >1 = brighter).

    Returns:
        New PIL Image with adjustments applied.
    """
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    return img


def postprocess(
    img: Image.Image,
    vignette: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
) -> Image.Image:
    """Apply all post-processing effects in order.

    Args:
        img: Input RGB image.
        vignette: Vignette strength (0 = off).
        contrast: Contrast multiplier (1.0 = unchanged).
        saturation: Saturation multiplier (1.0 = unchanged).
        brightness: Brightness multiplier (1.0 = unchanged).

    Returns:
        Post-processed PIL Image.
    """
    if contrast != 1.0 or saturation != 1.0 or brightness != 1.0:
        img = apply_color_grade(img, contrast, saturation, brightness)
    if vignette > 0:
        img = apply_vignette(img, vignette)
    return img
