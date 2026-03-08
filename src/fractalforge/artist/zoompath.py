"""Zoom path system -- keyframe-based camera trajectories through fractal space.

A zoom path defines a sequence of keyframes, each specifying:
- center_re, center_im: position in the complex plane
- zoom: zoom level (exponential)
- max_iter: iteration depth
- palette: color palette name
- rotation: rotation angle (future)

Position interpolation is zoom-weighted: the offset from the target keyframe
scales as 1/zoom, so the target stays locked in the viewport as zoom increases.
This prevents the "drift off-screen" problem with naive linear interpolation.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import math


@dataclass
class Keyframe:
    """A single keyframe in a zoom path."""

    frame: int  # Frame number where this keyframe occurs
    center_re: float = -0.75
    center_im: float = 0.0
    zoom: float = 1.0
    max_iter: int = 1000
    palette: str = "ocean"
    rotation: float = 0.0  # Degrees, future use


@dataclass
class ZoomPath:
    """A sequence of keyframes defining a zoom video."""

    name: str = "untitled"
    fps: int = 60
    width: int = 1920
    height: int = 1080
    keyframes: list[Keyframe] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        """Total frame count (from first to last keyframe)."""
        if not self.keyframes:
            return 0
        return self.keyframes[-1].frame + 1

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.total_frames / self.fps if self.fps > 0 else 0.0

    def interpolate(self, frame: int) -> dict:
        """Interpolate parameters at a given frame number.

        Uses zoom-weighted interpolation for position: the offset from the
        end keyframe's center scales as (zoom_start / zoom_current), keeping
        the target locked in the viewport as zoom increases. Zoom itself is
        interpolated exponentially. Max_iter is linear.

        Args:
            frame: The frame number to interpolate at.

        Returns:
            Dict with center_re, center_im, zoom, max_iter, palette.
        """
        if not self.keyframes:
            raise ValueError("No keyframes defined")

        # Clamp to keyframe range
        if frame <= self.keyframes[0].frame:
            kf = self.keyframes[0]
            return {
                "center_re": kf.center_re,
                "center_im": kf.center_im,
                "zoom": kf.zoom,
                "max_iter": kf.max_iter,
                "palette": kf.palette,
            }

        if frame >= self.keyframes[-1].frame:
            kf = self.keyframes[-1]
            return {
                "center_re": kf.center_re,
                "center_im": kf.center_im,
                "zoom": kf.zoom,
                "max_iter": kf.max_iter,
                "palette": kf.palette,
            }

        # Find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            kf0 = self.keyframes[i]
            kf1 = self.keyframes[i + 1]

            if kf0.frame <= frame <= kf1.frame:
                # Compute interpolation factor [0, 1]
                span = kf1.frame - kf0.frame
                t = (frame - kf0.frame) / span if span > 0 else 0.0

                # Exponential interpolation for zoom (computed first -- position depends on it)
                log_zoom0 = math.log(kf0.zoom)
                log_zoom1 = math.log(kf1.zoom)
                zoom = math.exp(log_zoom0 + t * (log_zoom1 - log_zoom0))

                # Zoom-weighted position interpolation:
                # offset from target scales as (start_zoom / current_zoom)
                # At t=0, zoom=zoom0: center = kf0.center (full offset)
                # At t=1, zoom=zoom1: center = kf1.center (zero offset)
                # The key insight: viewport width ~ 1/zoom, so an offset that
                # fills the screen at zoom0 should shrink proportionally.
                zoom_ratio = kf0.zoom / zoom  # 1.0 at start, ~0 at deep zoom
                center_re = kf1.center_re + (kf0.center_re - kf1.center_re) * zoom_ratio
                center_im = kf1.center_im + (kf0.center_im - kf1.center_im) * zoom_ratio

                # Linear interpolation for max_iter
                max_iter = int(kf0.max_iter + t * (kf1.max_iter - kf0.max_iter))

                # Use kf0's palette until we pass the midpoint
                palette = kf0.palette if t < 0.5 else kf1.palette

                return {
                    "center_re": center_re,
                    "center_im": center_im,
                    "zoom": zoom,
                    "max_iter": max_iter,
                    "palette": palette,
                }

        # Should not reach here
        raise ValueError(f"Frame {frame} not in keyframe range")

    def save(self, path: Path) -> None:
        """Save zoom path to JSON."""
        data = {
            "name": self.name,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "keyframes": [
                {
                    "frame": kf.frame,
                    "center_re": kf.center_re,
                    "center_im": kf.center_im,
                    "zoom": kf.zoom,
                    "max_iter": kf.max_iter,
                    "palette": kf.palette,
                    "rotation": kf.rotation,
                }
                for kf in self.keyframes
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ZoomPath":
        """Load zoom path from JSON."""
        data = json.loads(path.read_text())
        keyframes = [Keyframe(**kf) for kf in data["keyframes"]]
        return cls(
            name=data.get("name", "untitled"),
            fps=data.get("fps", 60),
            width=data.get("width", 1920),
            height=data.get("height", 1080),
            keyframes=keyframes,
        )
