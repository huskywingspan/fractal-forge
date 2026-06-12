"""Viewer state -- single mutable dataclass holding all viewer state.

Every component reads from and writes to this object. This avoids scattered
state and makes serialization (save/load session) trivial.

Coordinates are stored as high-precision strings (center_re_hp, center_im_hp)
to support perturbation theory at zoom >= 1e13. The float fields (center_re,
center_im) are derived and used for display and low-zoom mouse math.
"""

import math
from dataclasses import dataclass, field

# Zoom threshold for switching to perturbation theory / high-precision math
DEEP_ZOOM_THRESHOLD = 1e13


@dataclass
class ViewerState:
    """All mutable state for the interactive viewer."""

    # Viewport position (float -- derived from hp strings, used for display)
    center_re: float = -0.75
    center_im: float = 0.0
    zoom: float = 1.0
    max_iter: int = 1000

    # High-precision coordinates (authoritative at deep zoom)
    center_re_hp: str = "-0.75"
    center_im_hp: str = "0.0"

    # Fractal parameters
    fractal_type: str = "mandelbrot"  # mandelbrot | julia | burning_ship
    palette_name: str = "ocean"
    julia_re: float = -0.7269
    julia_im: float = 0.1889

    # Preview settings
    preview_width: int = 640
    preview_height: int = 360

    # Post-processing
    histogram: bool = False
    slope_shading: bool = False
    vignette: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    brightness: float = 1.0

    # Auto max-iter: automatically scale iterations with zoom depth
    auto_max_iter: bool = True

    # Render status
    render_pending: bool = True  # Start with a render queued
    render_queued: bool = False  # Another render waiting behind in-flight
    last_render_ms: float = 0.0

    # Bookmarks
    bookmarks: list = field(default_factory=list)

    @property
    def needs_perturbation(self) -> bool:
        """True if current zoom requires perturbation theory."""
        return self.zoom >= DEEP_ZOOM_THRESHOLD and self.fractal_type == "mandelbrot"

    @property
    def precision_digits(self) -> int:
        """Required decimal digits for current zoom level."""
        if self.zoom <= 1:
            return 20
        return max(20, int(math.log10(self.zoom)) + 20)

    def set_center(self, re, im):
        """Set center coordinates, syncing both float and hp fields.

        Args:
            re: Real part (float, str, or mpmath.mpf).
            im: Imaginary part (float, str, or mpmath.mpf).
        """
        self.center_re_hp = str(re)
        self.center_im_hp = str(im)
        self.center_re = float(re)
        self.center_im = float(im)

    def set_center_float(self, re: float, im: float):
        """Set center from floats (low zoom only -- hp derived from float)."""
        self.center_re = re
        self.center_im = im
        self.center_re_hp = f"{re:.17g}"
        self.center_im_hp = f"{im:.17g}"

    def request_render(self):
        """Mark that a new render is needed."""
        self.render_pending = True

    def copy_location(self) -> str:
        """Return a CLI command string for the current location."""
        # Use hp strings for deep zoom precision
        re_str = self.center_re_hp if self.needs_perturbation else str(self.center_re)
        im_str = self.center_im_hp if self.needs_perturbation else str(self.center_im)
        cmd = (
            f"fractalforge render"
            f" -x \"{re_str}\""
            f" -y \"{im_str}\""
            f" -z {self.zoom}"
            f" -i {self.max_iter}"
            f" -p {self.palette_name}"
            f" --fractal {self.fractal_type}"
        )
        if self.fractal_type == "julia":
            cmd += f" --julia-re {self.julia_re} --julia-im {self.julia_im}"
        return cmd

    def add_bookmark(self):
        """Save current location as a bookmark."""
        self.bookmarks.append({
            "center_re": self.center_re,
            "center_im": self.center_im,
            "center_re_hp": self.center_re_hp,
            "center_im_hp": self.center_im_hp,
            "zoom": self.zoom,
            "max_iter": self.max_iter,
            "fractal_type": self.fractal_type,
            "palette_name": self.palette_name,
            "julia_re": self.julia_re,
            "julia_im": self.julia_im,
        })

    def load_bookmark(self, idx: int):
        """Restore state from a bookmark."""
        if 0 <= idx < len(self.bookmarks):
            bm = self.bookmarks[idx]
            self.center_re = bm["center_re"]
            self.center_im = bm["center_im"]
            self.center_re_hp = bm.get("center_re_hp", f"{bm['center_re']:.17g}")
            self.center_im_hp = bm.get("center_im_hp", f"{bm['center_im']:.17g}")
            self.zoom = bm["zoom"]
            self.max_iter = bm["max_iter"]
            self.fractal_type = bm["fractal_type"]
            self.palette_name = bm["palette_name"]
            self.julia_re = bm["julia_re"]
            self.julia_im = bm["julia_im"]
            self.request_render()
