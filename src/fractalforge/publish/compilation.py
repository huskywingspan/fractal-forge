"""Short compilation pipeline -- stitch multiple zoom clips with crossfade transitions.

Takes a list of clip specifications (each referencing a rendered zoom path preset),
extracts the relevant frame ranges, and assembles them into a single numbered
frame sequence with crossfade transitions between clips.  The assembled frames
can then be encoded to video via ``render.video.encode_video``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from fractalforge.artist.zoompath import ZoomPath

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ClipSpec(BaseModel):
    """Specification for a single clip segment in a compilation."""

    preset: str            # Path to zoom path JSON
    start_sec: float       # Start time within the zoom path (seconds)
    duration_sec: float    # Clip duration (seconds)


class CompilationSpec(BaseModel):
    """Full specification for a multi-clip compilation video."""

    name: str = "compilation"
    fps: int = 60
    transition_frames: int = 60    # Frames of crossfade between clips (1 s at 60 fps)
    encode_preset: str = "youtube"
    clips: list[ClipSpec]

    @classmethod
    def load(cls, path: Path) -> "CompilationSpec":
        """Load a compilation spec from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A ``CompilationSpec`` instance.
        """
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save this compilation spec to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_frames_dir(preset_path: str) -> tuple[Path, int]:
    """Return (frames_dir, total_frames) for a rendered preset.

    Loads the ``ZoomPath`` from *preset_path* to discover the canonical frames
    directory (``output/{name}_frames``) and frame count.

    Args:
        preset_path: Path to the zoom path JSON preset file.

    Returns:
        Tuple of (frames_dir, total_frames).

    Raises:
        FileNotFoundError: If the preset or its frames directory does not exist.
    """
    preset = Path(preset_path)
    if not preset.exists():
        raise FileNotFoundError(f"Preset file not found: {preset}")

    zoom_path = ZoomPath.load(preset)
    frames_dir = Path("output") / f"{zoom_path.name}_frames"

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir}  "
            f"(render the preset first with `fractalforge sequence`)"
        )

    total = zoom_path.total_frames
    return frames_dir, total


# ---------------------------------------------------------------------------
# Crossfade blending
# ---------------------------------------------------------------------------

def render_crossfade(
    frames_a: list[Path],
    frames_b: list[Path],
    output_dir: Path,
    start_index: int,
) -> list[Path]:
    """Render crossfade transition frames by alpha-blending two frame lists.

    For each transition frame *i*, the output is::

        output = A[i] * (1 - t) + B[i] * t

    where ``t = i / (N - 1)`` and *N* is the number of transition frames.

    Args:
        frames_a: Last N frames of the outgoing clip.
        frames_b: First N frames of the incoming clip.
        output_dir: Directory to write blended transition frames.
        start_index: Starting frame number for output file naming.

    Returns:
        List of output frame paths, one per transition frame.

    Raises:
        ValueError: If the two frame lists differ in length or are empty.
    """
    if len(frames_a) != len(frames_b):
        raise ValueError(
            f"Frame lists must be the same length, got {len(frames_a)} and {len(frames_b)}"
        )
    n = len(frames_a)
    if n == 0:
        raise ValueError("Frame lists must not be empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    for i in range(n):
        t = i / (n - 1) if n > 1 else 1.0

        img_a = np.asarray(Image.open(frames_a[i]), dtype=np.float32)
        img_b = np.asarray(Image.open(frames_b[i]), dtype=np.float32)

        blended = img_a * (1.0 - t) + img_b * t
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        out_path = output_dir / f"frame_{start_index + i:06d}.png"
        Image.fromarray(blended).save(out_path, format="PNG")
        output_paths.append(out_path)

    return output_paths


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_compilation(
    spec: CompilationSpec,
    output_dir: Path,
) -> Path:
    """Assemble a compilation from rendered clip frames and crossfade transitions.

    For each clip the function:
    1. Locates the pre-rendered frames directory for the clip's preset.
    2. Copies the clip's frame range into *output_dir*, renumbered sequentially.
    3. Between consecutive clips, renders a crossfade transition where the last
       ``transition_frames`` of clip A overlap with the first ``transition_frames``
       of clip B.

    The first clip starts without a leading transition and the last clip ends
    without a trailing one.

    Args:
        spec: The compilation specification.
        output_dir: Directory to write the assembled (renumbered) frame sequence.

    Returns:
        The *output_dir* path containing all assembled frames.

    Raises:
        FileNotFoundError: If a preset or its frames directory is missing.
        ValueError: If a clip's frame range exceeds available frames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_clips = len(spec.clips)
    out_index = 0  # Global output frame counter

    for clip_idx, clip in enumerate(spec.clips):
        frames_dir, total_available = _resolve_frames_dir(clip.preset)

        start_frame = int(clip.start_sec * spec.fps)
        frame_count = int(clip.duration_sec * spec.fps)
        end_frame = start_frame + frame_count  # exclusive

        if end_frame > total_available:
            raise ValueError(
                f"Clip {clip_idx} ({clip.preset}): requested frames {start_frame}..{end_frame - 1} "
                f"but only {total_available} frames available"
            )

        is_first = clip_idx == 0
        is_last = clip_idx == num_clips - 1
        tf = spec.transition_frames

        # -----------------------------------------------------------
        # Determine which source frames are "unique" (not part of a
        # crossfade) and which participate in transitions.
        # -----------------------------------------------------------

        # Leading transition: the first `tf` frames of this clip overlap
        # with the previous clip's trailing frames (handled when the
        # *previous* clip was processed).  So we skip them here -- they
        # will be written by render_crossfade.
        body_start = start_frame if is_first else start_frame + tf

        # Trailing transition: the last `tf` frames of this clip will
        # overlap with the next clip.  We don't copy them as unique
        # frames -- they'll be blended in the crossfade step below.
        body_end = end_frame if is_last else end_frame - tf

        # Copy unique body frames
        for src_idx in range(body_start, body_end):
            src_path = frames_dir / f"frame_{src_idx:06d}.png"
            dst_path = output_dir / f"frame_{out_index:06d}.png"
            shutil.copy2(src_path, dst_path)
            out_index += 1

        # -----------------------------------------------------------
        # Crossfade transition into the NEXT clip
        # -----------------------------------------------------------
        if not is_last:
            next_clip = spec.clips[clip_idx + 1]
            next_frames_dir, next_total = _resolve_frames_dir(next_clip.preset)
            next_start_frame = int(next_clip.start_sec * spec.fps)

            # Outgoing: last `tf` frames of current clip
            tail_paths = [
                frames_dir / f"frame_{end_frame - tf + i:06d}.png"
                for i in range(tf)
            ]

            # Incoming: first `tf` frames of next clip
            head_paths = [
                next_frames_dir / f"frame_{next_start_frame + i:06d}.png"
                for i in range(tf)
            ]

            render_crossfade(tail_paths, head_paths, output_dir, out_index)
            out_index += tf

    return output_dir
