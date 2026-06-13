"""Viewer state -- single mutable dataclass holding all viewer state.

Every component reads from and writes to this object. This avoids scattered
state and makes serialization (save/load session) trivial.

Zoom is stored as ``log10_zoom`` (a plain float -- log10(1e500) is just 500.0)
so depth is never capped by float64's 1e308 ceiling. The ``zoom`` property is
a convenience float for shallow math/display; deep paths use ``zoom_str`` (an
mpmath-formatted string the renderer accepts) and ``log10_zoom`` directly.

Coordinates are stored as high-precision strings (center_re_hp, center_im_hp)
to support perturbation theory at depth. The float fields (center_re,
center_im) are derived for display and low-zoom mouse math.
"""

import math
from dataclasses import dataclass, field

import mpmath

# log10(zoom) thresholds for engine selection (mirror the render pipeline).
DEEP_ZOOM_LOG10 = 13.0   # perturbation theory kicks in
DEEP_FXP_LOG10 = 18.0    # floatexp deep kernel takes over


@dataclass
class ViewerState:
    """All mutable state for the interactive viewer."""

    # Viewport position (float -- derived from hp strings, used for display)
    center_re: float = -0.75
    center_im: float = 0.0
    log10_zoom: float = 0.0  # zoom = 10 ** log10_zoom; 0.0 => zoom 1.0
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
    # Set by the render bridge so the HUD can show which engine ran.
    last_engine: str = "float64"

    # Bookmarks
    bookmarks: list = field(default_factory=list)

    # ---- zoom helpers -------------------------------------------------------

    @property
    def zoom(self) -> float:
        """Zoom as a float for shallow math/display.

        Clamped just below the float64 ceiling so display code never sees inf;
        deep rendering and coordinate math use ``zoom_str`` / ``log10_zoom``.
        """
        if self.log10_zoom >= 307.0:
            return 1e307
        return 10.0 ** self.log10_zoom

    @zoom.setter
    def zoom(self, value: float):
        self.log10_zoom = math.log10(value) if value > 0 else 0.0

    @property
    def zoom_str(self) -> str:
        """Zoom as an mpmath-parseable string (valid at any depth)."""
        if self.log10_zoom < 290.0:
            return repr(10.0 ** self.log10_zoom)
        with mpmath.workdps(20):
            return mpmath.nstr(mpmath.power(10, self.log10_zoom), 12)

    @property
    def zoom_mp(self):
        """Zoom as an mpmath.mpf at the current precision."""
        return mpmath.power(10, mpmath.mpf(self.log10_zoom))

    def zoom_by(self, factor: float):
        """Multiply zoom by ``factor`` (factor < 1 zooms out). Floor at 0.1x."""
        self.log10_zoom = max(math.log10(0.1), self.log10_zoom + math.log10(factor))

    # ---- engine / precision -------------------------------------------------

    @property
    def needs_perturbation(self) -> bool:
        """True if current zoom requires perturbation theory."""
        return (self.log10_zoom >= DEEP_ZOOM_LOG10
                and self.fractal_type == "mandelbrot")

    @property
    def needs_deep_fxp(self) -> bool:
        """True if current zoom routes to the floatexp deep kernel."""
        return (self.log10_zoom >= DEEP_FXP_LOG10
                and self.fractal_type == "mandelbrot")

    @property
    def precision_digits(self) -> int:
        """Required decimal digits for current zoom level."""
        if self.log10_zoom <= 0:
            return 20
        return max(20, int(1.5 * self.log10_zoom) + 30)

    # ---- mutation -----------------------------------------------------------

    def set_center(self, re, im):
        """Set center coordinates, syncing both float and hp fields."""
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

    def reset_view(self):
        """Return to the default home view."""
        self.set_center_float(-0.75, 0.0)
        self.log10_zoom = 0.0
        self.request_render()

    def request_render(self):
        """Mark that a new render is needed."""
        self.render_pending = True

    def copy_location(self) -> str:
        """Return a CLI command string for the current location."""
        re_str = self.center_re_hp if self.needs_perturbation else str(self.center_re)
        im_str = self.center_im_hp if self.needs_perturbation else str(self.center_im)
        cmd = (
            f"fractalforge render"
            f" -x \"{re_str}\""
            f" -y \"{im_str}\""
            f" -z {self.zoom_str}"
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
            "log10_zoom": self.log10_zoom,
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
            # Back-compat: older bookmarks stored raw zoom.
            if "log10_zoom" in bm:
                self.log10_zoom = bm["log10_zoom"]
            else:
                self.log10_zoom = math.log10(max(bm.get("zoom", 1.0), 1e-9))
            self.max_iter = bm["max_iter"]
            self.fractal_type = bm["fractal_type"]
            self.palette_name = bm["palette_name"]
            self.julia_re = bm["julia_re"]
            self.julia_im = bm["julia_im"]
            self.request_render()
