"""Render bridge -- async dispatch from viewer to GPU render pipeline.

Uses a single-threaded ThreadPoolExecutor so the GPU render runs on a
dedicated thread, keeping the Dear PyGui main loop responsive. Only one
render is in-flight at a time; if a new render is requested while one is
running, it queues up and the stale result is discarded.
"""

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from fractalforge.viewer.state import ViewerState


def _auto_max_iter(zoom: float, user_max_iter: int, auto: bool = True) -> int:
    """Compute effective max_iter, ensuring it's sufficient for the zoom depth.

    When auto=True, scales iterations with zoom depth:
    - 500 base + 300 per decade of zoom
    - Returns the max of computed floor and user's manual setting
    When auto=False, returns the user's manual setting unchanged.
    """
    if not auto or zoom <= 1.0:
        return user_max_iter
    zoom_floor = int(500 + 300 * math.log10(zoom))
    return max(user_max_iter, zoom_floor)


def _do_render(state: ViewerState) -> tuple[np.ndarray, float]:
    """Execute a render on the worker thread.

    Returns:
        Tuple of (flat float32 RGBA array for DPG texture, render_ms).
    """
    from fractalforge.render.frame_renderer import render_single

    start = time.perf_counter()

    # Use high-precision string coordinates for perturbation theory
    if state.needs_perturbation:
        center_re = state.center_re_hp
        center_im = state.center_im_hp
    else:
        center_re = state.center_re
        center_im = state.center_im

    # Auto-scale max_iter based on zoom depth
    effective_max_iter = _auto_max_iter(state.zoom, state.max_iter, state.auto_max_iter)

    # Auto-enable histogram EQ at deep zoom -- without it, the narrow iteration
    # range maps to a single palette color and the image looks flat/monochrome
    use_histogram = state.histogram or state.needs_perturbation

    img = render_single(
        center_re=center_re,
        center_im=center_im,
        zoom=state.zoom,
        width=state.preview_width,
        height=state.preview_height,
        max_iter=effective_max_iter,
        palette_name=state.palette_name,
        fractal_type=state.fractal_type,
        julia_re=state.julia_re if state.fractal_type == "julia" else None,
        julia_im=state.julia_im if state.fractal_type == "julia" else None,
        histogram=use_histogram,
        slope_shading=state.slope_shading,
        vignette=state.vignette,
        contrast=state.contrast,
        saturation=state.saturation,
        brightness=state.brightness,
        supersampling=1,
        use_gpu=True,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Convert PIL Image to flat float32 RGBA array [0, 1] for DPG raw texture
    arr = np.array(img, dtype=np.float32) / 255.0
    # Add alpha channel (DPG raw textures need RGBA)
    h, w, _ = arr.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = arr
    return rgba.flatten(), elapsed_ms


class RenderBridge:
    """Manages async rendering from the viewer's main loop."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._pending_state: ViewerState | None = None

    def submit(self, state: ViewerState):
        """Submit a render request.

        If a render is already in-flight, queues the latest state so
        the result updates as soon as the current render completes.
        """
        if self._future is not None and not self._future.done():
            # Queue the latest state; discard intermediate requests
            self._pending_state = state
            return

        # Snapshot the state values we need (avoid race conditions)
        self._pending_state = None
        self._future = self._executor.submit(_do_render, state)

    def check(self, state: ViewerState) -> np.ndarray | None:
        """Check if a render completed. Call this every frame.

        Returns:
            Flat float32 RGBA array if a render completed, else None.
        """
        if self._future is None:
            return None

        if not self._future.done():
            return None

        try:
            result, elapsed_ms = self._future.result()
            state.last_render_ms = elapsed_ms
        except Exception as e:
            print(f"Render error: {e}")
            self._future = None
            return None

        self._future = None

        # If another render was queued while this one was running, start it
        if self._pending_state is not None:
            pending = self._pending_state
            self._pending_state = None
            self._future = self._executor.submit(_do_render, pending)

        return result

    def shutdown(self):
        """Clean up the thread pool."""
        self._executor.shutdown(wait=False)
