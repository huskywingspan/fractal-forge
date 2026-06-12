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
    fractal_type: str = "mandelbrot"  # mandelbrot, julia, burning_ship
    julia_re: float | None = None
    julia_im: float | None = None
    easing: str = "ease_in_out"  # Easing for transition INTO this keyframe
    tension: float = 0.5  # Spline tension (0=sharp corner, 1=maximum smoothing)
    # High-precision coordinate strings for deep zoom (zoom >= 1e13)
    center_re_hp: str | None = None
    center_im_hp: str | None = None


@dataclass
class ZoomPath:
    """A sequence of keyframes defining a zoom video."""

    name: str = "untitled"
    fps: int = 60
    width: int = 1920
    height: int = 1080
    keyframes: list[Keyframe] = field(default_factory=list)
    interpolation: str = "legacy"  # "legacy" or "cinematic"

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

        Dispatches to legacy or cinematic interpolation based on the
        ``interpolation`` field.

        Args:
            frame: The frame number to interpolate at.

        Returns:
            Dict with center_re, center_im, zoom, max_iter, palette, etc.
        """
        if self.interpolation == "cinematic":
            return self._interpolate_cinematic(frame)
        return self._interpolate_legacy(frame)

    def _keyframe_result(self, kf: "Keyframe") -> dict:
        """Build result dict from a single keyframe."""
        result = {
            "center_re": kf.center_re,
            "center_im": kf.center_im,
            "zoom": kf.zoom,
            "max_iter": kf.max_iter,
            "palette": kf.palette,
            "fractal_type": kf.fractal_type,
            "julia_re": kf.julia_re,
            "julia_im": kf.julia_im,
        }
        if kf.center_re_hp is not None:
            result["center_re_hp"] = kf.center_re_hp
        if kf.center_im_hp is not None:
            result["center_im_hp"] = kf.center_im_hp
        return result

    def _interpolate_legacy(self, frame: int) -> dict:
        """Original piecewise zoom-weighted interpolation.

        For deep zoom (target zoom >= 1e13), uses mpmath for coordinate math
        to preserve precision. The hp string coordinates are passed through
        so the renderer can use them for perturbation theory.
        """
        if not self.keyframes:
            raise ValueError("No keyframes defined")

        if frame <= self.keyframes[0].frame:
            return self._keyframe_result(self.keyframes[0])
        if frame >= self.keyframes[-1].frame:
            return self._keyframe_result(self.keyframes[-1])

        for i in range(len(self.keyframes) - 1):
            kf0 = self.keyframes[i]
            kf1 = self.keyframes[i + 1]

            if kf0.frame <= frame <= kf1.frame:
                span = kf1.frame - kf0.frame
                t = (frame - kf0.frame) / span if span > 0 else 0.0

                log_zoom0 = math.log(kf0.zoom)
                log_zoom1 = math.log(kf1.zoom)
                zoom = math.exp(log_zoom0 + t * (log_zoom1 - log_zoom0))

                # Use mpmath for coordinate interpolation at deep zoom
                # to prevent float64 precision loss
                target_has_hp = (kf1.center_re_hp is not None
                                 and kf1.center_im_hp is not None)
                if zoom >= 1e13 and target_has_hp:
                    center_re, center_im, hp_re, hp_im = (
                        self._interp_coords_hp(kf0, kf1, zoom)
                    )
                else:
                    zoom_ratio = kf0.zoom / zoom
                    center_re = kf1.center_re + (kf0.center_re - kf1.center_re) * zoom_ratio
                    center_im = kf1.center_im + (kf0.center_im - kf1.center_im) * zoom_ratio
                    hp_re = None
                    hp_im = None

                max_iter = int(kf0.max_iter + t * (kf1.max_iter - kf0.max_iter))
                palette = kf0.palette if t < 0.5 else kf1.palette

                fractal_type = kf0.fractal_type
                julia_re = kf0.julia_re
                julia_im = kf0.julia_im
                if (kf0.julia_re is not None and kf1.julia_re is not None
                        and kf0.julia_im is not None and kf1.julia_im is not None):
                    julia_re = kf0.julia_re + t * (kf1.julia_re - kf0.julia_re)
                    julia_im = kf0.julia_im + t * (kf1.julia_im - kf0.julia_im)

                result = {
                    "center_re": center_re,
                    "center_im": center_im,
                    "zoom": zoom,
                    "max_iter": max_iter,
                    "palette": palette,
                    "fractal_type": fractal_type,
                    "julia_re": julia_re,
                    "julia_im": julia_im,
                }
                if hp_re is not None:
                    result["center_re_hp"] = hp_re
                    result["center_im_hp"] = hp_im
                return result

        raise ValueError(f"Frame {frame} not in keyframe range")

    @staticmethod
    def _interp_coords_hp(kf0, kf1, zoom):
        """Interpolate coordinates using mpmath for deep zoom precision.

        The zoom-weighted formula: pos = target + (start - target) * (start_zoom / zoom)
        At deep zoom, (start - target) is large and zoom is huge, so the offset
        shrinks to sub-float64 scale. mpmath keeps the needed precision.
        """
        import mpmath
        # Use enough precision for the target zoom
        digits = max(20, int(math.log10(zoom)) + 20)
        mpmath.mp.dps = digits

        # Use hp strings when available, fall back to float
        re0 = mpmath.mpf(kf0.center_re_hp or str(kf0.center_re))
        im0 = mpmath.mpf(kf0.center_im_hp or str(kf0.center_im))
        re1 = mpmath.mpf(kf1.center_re_hp)
        im1 = mpmath.mpf(kf1.center_im_hp)

        zoom_ratio = mpmath.mpf(kf0.zoom) / mpmath.mpf(zoom)
        center_re_mp = re1 + (re0 - re1) * zoom_ratio
        center_im_mp = im1 + (im0 - im1) * zoom_ratio

        hp_re = mpmath.nstr(center_re_mp, digits, strip_zeros=False)
        hp_im = mpmath.nstr(center_im_mp, digits, strip_zeros=False)
        return float(center_re_mp), float(center_im_mp), hp_re, hp_im

    def _interpolate_cinematic(self, frame: int) -> dict:
        """Cinematic interpolation with spline position and eased zoom.

        Uses Catmull-Rom splines in zoom-scaled screen space for C1-continuous
        position, and easing functions for smooth zoom acceleration. This
        eliminates velocity discontinuities at keyframe boundaries.

        For 2-keyframe paths, falls back to legacy (which is already optimal
        for single-segment dives).
        """
        from fractalforge.artist.easing import get_easing
        from fractalforge.artist.spline import catmull_rom_2d, smooth_zoom_path

        if not self.keyframes:
            raise ValueError("No keyframes defined")

        if frame <= self.keyframes[0].frame:
            return self._keyframe_result(self.keyframes[0])
        if frame >= self.keyframes[-1].frame:
            return self._keyframe_result(self.keyframes[-1])

        kfs = self.keyframes
        n = len(kfs)

        # For 2-keyframe paths, legacy is already optimal
        if n <= 2:
            return self._interpolate_legacy(frame)

        # --- Zoom: eased exponential interpolation ---
        # Find which segment we're in for easing
        seg_idx = 0
        for i in range(n - 1):
            if kfs[i].frame <= frame <= kfs[i + 1].frame:
                seg_idx = i
                break

        kf0 = kfs[seg_idx]
        kf1 = kfs[seg_idx + 1]
        span = kf1.frame - kf0.frame
        t_raw = (frame - kf0.frame) / span if span > 0 else 0.0

        # Apply easing to the segment t for smooth acceleration
        easing_fn = get_easing(kf1.easing)
        t_eased = easing_fn(t_raw)

        # Exponential zoom with easing
        log_zoom0 = math.log(kf0.zoom)
        log_zoom1 = math.log(kf1.zoom)
        zoom = math.exp(log_zoom0 + t_eased * (log_zoom1 - log_zoom0))

        # --- Position: Catmull-Rom spline in zoom-scaled screen space ---
        # Transform keyframe positions to screen space relative to final target
        ref_re = kfs[-1].center_re
        ref_im = kfs[-1].center_im

        screen_re = [(kf.center_re - ref_re) * kf.zoom for kf in kfs]
        screen_im = [(kf.center_im - ref_im) * kf.zoom for kf in kfs]

        # Compute global spline parameter: map frame to [0, n-1]
        # Each segment maps to a unit interval on the spline
        t_global = float(seg_idx) + t_eased

        # Evaluate spline in screen space
        scr_re, scr_im = catmull_rom_2d(screen_re, screen_im, t_global)

        # Convert back to complex plane: position = ref + screen_pos / zoom
        center_re = ref_re + scr_re / zoom
        center_im = ref_im + scr_im / zoom

        # --- Other parameters: same as legacy ---
        max_iter = int(kf0.max_iter + t_raw * (kf1.max_iter - kf0.max_iter))

        # Palette: smooth crossfade window (use kf0 until 40%, kf1 after 60%)
        palette = kf0.palette if t_raw < 0.5 else kf1.palette

        fractal_type = kf0.fractal_type
        julia_re = kf0.julia_re
        julia_im = kf0.julia_im
        if (kf0.julia_re is not None and kf1.julia_re is not None
                and kf0.julia_im is not None and kf1.julia_im is not None):
            julia_re = kf0.julia_re + t_raw * (kf1.julia_re - kf0.julia_re)
            julia_im = kf0.julia_im + t_raw * (kf1.julia_im - kf0.julia_im)

        return {
            "center_re": center_re,
            "center_im": center_im,
            "zoom": zoom,
            "max_iter": max_iter,
            "palette": palette,
            "fractal_type": fractal_type,
            "julia_re": julia_re,
            "julia_im": julia_im,
        }

    def save(self, path: Path) -> None:
        """Save zoom path to JSON."""
        # Required keys always included; optional keys only when non-default
        REQUIRED = {"frame", "center_re", "center_im", "zoom", "max_iter", "palette"}
        DEFAULTS = {
            "rotation": 0.0,
            "fractal_type": "mandelbrot",
            "julia_re": None,
            "julia_im": None,
            "easing": "ease_in_out",
            "tension": 0.5,
            "center_re_hp": None,
            "center_im_hp": None,
        }

        data = {
            "name": self.name,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "keyframes": [
                {
                    k: v for k, v in {
                        "frame": kf.frame,
                        "center_re": kf.center_re,
                        "center_im": kf.center_im,
                        "zoom": kf.zoom,
                        "max_iter": kf.max_iter,
                        "palette": kf.palette,
                        "rotation": kf.rotation,
                        "fractal_type": kf.fractal_type,
                        "julia_re": kf.julia_re,
                        "julia_im": kf.julia_im,
                        "easing": kf.easing,
                        "tension": kf.tension,
                        "center_re_hp": kf.center_re_hp,
                        "center_im_hp": kf.center_im_hp,
                    }.items()
                    if k in REQUIRED or (v is not None and v != DEFAULTS.get(k))
                }
                for kf in self.keyframes
            ],
        }
        # Only include interpolation if non-default
        if self.interpolation != "legacy":
            data["interpolation"] = self.interpolation
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
            interpolation=data.get("interpolation", "legacy"),
        )
