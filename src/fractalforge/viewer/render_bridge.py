"""Render bridge -- async dispatch from viewer to GPU render pipeline.

Uses a single-threaded ThreadPoolExecutor so the GPU render runs on a
dedicated thread, keeping the Dear PyGui main loop responsive. Only one
render is in-flight at a time; if a new render is requested while one is
running, the latest state queues and the stale result is discarded.

Supports progressive rendering: a ``scale`` < 1 renders at reduced resolution
(snappy during interaction) and upsamples to the preview size; the app then
re-renders at full scale once interaction settles.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from fractalforge.viewer.state import ViewerState, DEEP_ZOOM_LOG10, DEEP_FXP_LOG10


def auto_max_iter(log10_zoom: float, user_max_iter: int, auto: bool = True) -> int:
    """Effective max_iter for a zoom depth.

    Shallow zooms need head-room for long-period orbits (300/decade). Beyond
    the perturbation threshold the dominant cost is embedded-Julia escape depth,
    which grows only modestly with log-zoom, so we use a gentler 60/decade to
    keep extreme renders tractable.

    The result is quantized upward to coarse steps so that consecutive zoom
    levels share the same iteration budget — this is what lets the
    reference-orbit cache hit while click-zooming into a fixed point
    (a continuously growing max_iter would invalidate the orbit every step).
    """
    if not auto or log10_zoom <= 0.0:
        return user_max_iter
    if log10_zoom < DEEP_FXP_LOG10:
        floor = int(500 + 300 * log10_zoom)
        step = 1000
    else:
        floor = int(2000 + 60 * log10_zoom)
        step = 2000
    floor = ((floor + step - 1) // step) * step
    return max(user_max_iter, floor)


def engine_label(state: ViewerState) -> str:
    """Short badge for which engine the current zoom will use."""
    if state.fractal_type != "mandelbrot":
        return state.fractal_type
    if state.log10_zoom >= DEEP_FXP_LOG10:
        return "FXP"      # floatexp deep kernel
    if state.log10_zoom >= DEEP_ZOOM_LOG10:
        return "PT"       # perturbation theory (float64 deltas)
    return "STD"          # standard float64


def _do_render(state: ViewerState, scale: float = 1.0) -> tuple[np.ndarray, float, str]:
    """Execute a render on the worker thread.

    Returns (flat float32 RGBA array sized to the preview, render_ms, engine).
    """
    from fractalforge.render.frame_renderer import render_single
    from PIL import Image

    start = time.perf_counter()

    out_w, out_h = state.preview_width, state.preview_height
    render_w = max(16, int(out_w * scale))
    render_h = max(16, int(out_h * scale))

    # Deep zoom: pass string coords + string zoom to preserve precision.
    if state.needs_perturbation:
        center_re = state.center_re_hp
        center_im = state.center_im_hp
        zoom = state.zoom_str
    else:
        center_re = state.center_re
        center_im = state.center_im
        zoom = state.zoom

    effective_max_iter = auto_max_iter(state.log10_zoom, state.max_iter,
                                       state.auto_max_iter)

    # Auto-enable histogram EQ at deep zoom -- the narrow iteration band would
    # otherwise map to a single flat color.
    use_histogram = state.histogram or state.needs_perturbation
    color_mode = None if state.color_mode == "auto" else state.color_mode

    img = render_single(
        center_re=center_re,
        center_im=center_im,
        zoom=zoom,
        width=render_w,
        height=render_h,
        max_iter=effective_max_iter,
        palette_name=state.palette_name,
        fractal_type=state.fractal_type,
        julia_re=state.julia_re if state.fractal_type == "julia" else None,
        julia_im=state.julia_im if state.fractal_type == "julia" else None,
        histogram=use_histogram,
        color_mode=color_mode,
        slope_shading=state.slope_shading,
        vignette=state.vignette,
        contrast=state.contrast,
        saturation=state.saturation,
        brightness=state.brightness,
        supersampling=1,
        use_gpu=True,
    )

    # Upsample a reduced-scale render back to the preview size. Bilinear
    # looks far less blocky than nearest during interaction; the full-res
    # pass replaces it as soon as the view settles.
    if (render_w, render_h) != (out_w, out_h):
        img = img.resize((out_w, out_h), Image.BILINEAR)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    arr = np.array(img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = arr
    return rgba.flatten(), elapsed_ms, engine_label(state)


class RenderBridge:
    """Manages async rendering from the viewer's main loop."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._pending: tuple[ViewerState, float] | None = None

    def submit(self, state: ViewerState, scale: float = 1.0):
        """Submit a render request (optionally at reduced scale)."""
        if self._future is not None and not self._future.done():
            self._pending = (state, scale)
            return
        self._pending = None
        self._future = self._executor.submit(_do_render, state, scale)

    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def check(self, state: ViewerState) -> np.ndarray | None:
        """Check if a render completed. Call every frame."""
        if self._future is None or not self._future.done():
            return None
        try:
            result, elapsed_ms, engine = self._future.result()
            state.last_render_ms = elapsed_ms
            state.last_engine = engine
        except Exception as e:
            print(f"Render error: {e}")
            self._future = None
            return None
        self._future = None
        if self._pending is not None:
            pending_state, pending_scale = self._pending
            self._pending = None
            self._future = self._executor.submit(_do_render, pending_state, pending_scale)
        return result

    def shutdown(self):
        self._executor.shutdown(wait=False)
