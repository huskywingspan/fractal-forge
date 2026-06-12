"""Post-processing effects -- vignette, color grading, HDR bloom, tone mapping.

Applied to PIL Images after coloring, before save. The HDR pipeline operates
in linear float32 space: bloom/halation extract and blur bright regions, then
tone mapping compresses the HDR range to displayable [0, 255].

Pipeline order:
  1. Color grading (contrast, saturation, brightness)
  2. HDR bloom (additive glow around bright areas)
  3. Halation (warm-tinted light bleed with per-channel blur radii)
  4. Tone mapping (ACES filmic or Reinhard)
  5. Vignette (edge darkening)
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


def apply_bloom(
    img: Image.Image,
    intensity: float = 0.3,
    threshold: float = 0.6,
    radius: float = 20.0,
    num_passes: int = 3,
) -> Image.Image:
    """Apply HDR bloom (glow around bright areas).

    Extracts pixels above a brightness threshold, applies multi-pass Gaussian
    blur at increasing radii, and blends back additively. Creates the luminous
    glow effect seen in high-end fractal renders.

    Args:
        img: Input RGB image.
        intensity: Bloom strength (0 = off, 0.3 = subtle, 1.0 = heavy glow).
        threshold: Brightness cutoff for bloom extraction (0-1, lower = more glow).
        radius: Base blur radius in pixels. Each pass doubles the radius.
        num_passes: Number of blur passes (more = wider, softer glow).

    Returns:
        New PIL Image with bloom applied.
    """
    if intensity <= 0:
        return img

    from scipy.ndimage import gaussian_filter

    arr = np.array(img, dtype=np.float32) / 255.0

    # Extract bright pixels above threshold
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    bright_mask = np.maximum(luminance - threshold, 0.0)
    bright_mask /= max(bright_mask.max(), 1e-6)  # normalize
    bright = arr * bright_mask[:, :, np.newaxis]

    # Multi-pass bloom: each pass uses a wider blur for a natural falloff
    bloom = np.zeros_like(arr)
    for p in range(num_passes):
        r = radius * (2 ** p)
        weight = 1.0 / (p + 1)  # diminishing contribution per pass
        for c in range(3):
            bloom[:, :, c] += gaussian_filter(bright[:, :, c], sigma=r) * weight

    # Normalize bloom passes
    if num_passes > 1:
        total_weight = sum(1.0 / (p + 1) for p in range(num_passes))
        bloom /= total_weight

    # Additive blend
    result = arr + bloom * intensity
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode=img.mode)


def apply_halation(
    img: Image.Image,
    intensity: float = 0.15,
    threshold: float = 0.7,
) -> Image.Image:
    """Apply halation (warm-tinted light bleed simulating film exposure).

    Unlike bloom which is uniform, halation blurs each color channel with
    different radii: red bleeds widest, green medium, blue narrowest. This
    creates the warm-halo effect seen in analog film and high-end renders.

    Args:
        img: Input RGB image.
        intensity: Halation strength (0 = off, 0.15 = subtle, 0.5 = heavy).
        threshold: Brightness cutoff for halation extraction.

    Returns:
        New PIL Image with halation applied.
    """
    if intensity <= 0:
        return img

    from scipy.ndimage import gaussian_filter

    arr = np.array(img, dtype=np.float32) / 255.0

    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    bright_mask = np.maximum(luminance - threshold, 0.0)
    bright_mask /= max(bright_mask.max(), 1e-6)
    bright = arr * bright_mask[:, :, np.newaxis]

    # Per-channel blur with different radii (red widest, blue narrowest)
    halation = np.zeros_like(arr)
    radii = [30.0, 18.0, 10.0]  # R, G, B
    for c, r in enumerate(radii):
        halation[:, :, c] = gaussian_filter(bright[:, :, c], sigma=r)

    # Warm tint: boost red/green channels of the halation
    halation[:, :, 0] *= 1.3  # warm red
    halation[:, :, 1] *= 1.1  # slight green warmth

    result = arr + halation * intensity
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode=img.mode)


def apply_tone_mapping(
    img: Image.Image,
    method: str = "aces",
    exposure: float = 1.0,
) -> Image.Image:
    """Apply HDR tone mapping to compress dynamic range.

    Args:
        img: Input RGB image (can have values pushed above 255 by bloom).
        method: Tone mapping curve — "aces" (filmic, industry standard) or
            "reinhard" (simpler, softer look).
        exposure: Exposure multiplier applied before tone mapping. >1 brightens.

    Returns:
        New PIL Image with tone mapping applied.
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    arr *= exposure

    if method == "aces":
        # ACES Filmic (Narkowicz approximation, CC0 license)
        # Adds natural hue shift toward white for very bright values
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        arr = (arr * (a * arr + b)) / (arr * (c * arr + d) + e)
    elif method == "reinhard":
        # Reinhard simple: mapped = x / (x + 1)
        arr = arr / (arr + 1.0)
    # else: no tone mapping (linear clamp)

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def postprocess(
    img: Image.Image,
    vignette: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
    bloom: float = 0.0,
    bloom_threshold: float = 0.6,
    bloom_radius: float = 20.0,
    halation: float = 0.0,
    tone_map: str = "none",
    exposure: float = 1.0,
) -> Image.Image:
    """Apply all post-processing effects in pipeline order.

    Pipeline: color grade -> bloom -> halation -> tone map -> vignette.

    Args:
        img: Input RGB image.
        vignette: Vignette strength (0 = off).
        contrast: Contrast multiplier (1.0 = unchanged).
        saturation: Saturation multiplier (1.0 = unchanged).
        brightness: Brightness multiplier (1.0 = unchanged).
        bloom: Bloom/glow intensity (0 = off, 0.3 = subtle, 1.0 = heavy).
        bloom_threshold: Brightness cutoff for bloom extraction (0-1).
        bloom_radius: Base blur radius in pixels for bloom.
        halation: Halation intensity (0 = off, 0.15 = subtle warm bleed).
        tone_map: Tone mapping method ("none", "aces", "reinhard").
        exposure: Exposure multiplier for tone mapping.

    Returns:
        Post-processed PIL Image.
    """
    if contrast != 1.0 or saturation != 1.0 or brightness != 1.0:
        img = apply_color_grade(img, contrast, saturation, brightness)
    if bloom > 0:
        img = apply_bloom(img, intensity=bloom, threshold=bloom_threshold,
                          radius=bloom_radius)
    if halation > 0:
        img = apply_halation(img, intensity=halation)
    if tone_map in ("aces", "reinhard"):
        img = apply_tone_mapping(img, method=tone_map, exposure=exposure)
    if vignette > 0:
        img = apply_vignette(img, vignette)
    return img
