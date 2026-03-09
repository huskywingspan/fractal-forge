"""YouTube Shorts generator -- crops 16:9 frames to 9:16 vertical video.

Selects a segment of an existing rendered sequence, center-crops each frame
to 9:16 portrait, and encodes to a Shorts-ready MP4.
"""

from pathlib import Path

from PIL import Image


def crop_frame_to_portrait(
    img: Image.Image,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Image.Image:
    """Center-crop a landscape frame to portrait aspect ratio.

    Takes the tallest possible 9:16 crop centered in the source frame,
    then resizes to the target dimensions.

    Args:
        img: Source image (any size, typically 16:9 landscape).
        target_width: Output width (default 1080 for Shorts).
        target_height: Output height (default 1920 for Shorts).

    Returns:
        Cropped and resized PIL Image.
    """
    src_w, src_h = img.size
    target_ratio = target_width / target_height  # 0.5625 for 9:16

    # Calculate the largest 9:16 rectangle that fits in the source
    # Option A: full height, calculate width
    crop_h = src_h
    crop_w = int(crop_h * target_ratio)

    if crop_w > src_w:
        # Option B: full width, calculate height
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)

    # Center the crop box
    left = (src_w - crop_w) // 2
    top = (src_h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = img.crop((left, top, right, bottom))
    if cropped.size != (target_width, target_height):
        cropped = cropped.resize((target_width, target_height), Image.LANCZOS)

    return cropped


def generate_short_frames(
    frames_dir: Path,
    output_dir: Path,
    start_frame: int,
    end_frame: int,
    target_width: int = 1080,
    target_height: int = 1920,
) -> list[Path]:
    """Crop a range of landscape frames to portrait for Shorts encoding.

    Args:
        frames_dir: Directory with source frames (frame_000000.png, etc.).
        output_dir: Directory to write cropped portrait frames.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (exclusive).
        target_width: Output width.
        target_height: Output height.

    Returns:
        List of output frame paths, sequentially numbered from 000000.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    out_idx = 0

    for src_idx in range(start_frame, end_frame):
        src_path = frames_dir / f"frame_{src_idx:06d}.png"
        if not src_path.exists():
            continue

        img = Image.open(src_path)
        cropped = crop_frame_to_portrait(img, target_width, target_height)

        out_path = output_dir / f"frame_{out_idx:06d}.png"
        cropped.save(out_path, format="PNG")
        paths.append(out_path)
        out_idx += 1

    return paths
