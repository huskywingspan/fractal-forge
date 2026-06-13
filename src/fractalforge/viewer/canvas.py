"""Fractal canvas -- displays the rendered fractal and handles interaction.

Mouse:
- Left-click: re-center on the clicked point and zoom in
- Scroll wheel: zoom in/out toward the cursor
- Right/middle-drag: pan the viewport (1:1 with the cursor)

Keyboard:
- Arrow keys: pan; +/-: zoom to center; R: reset view; B: bookmark

All coordinate math uses mpmath so navigation stays exact at any depth, up to
1e1000+. The viewport is mapped with zoom = 10**state.log10_zoom, which never
overflows the way a raw float zoom would.
"""

import array

import dearpygui.dearpygui as dpg
import mpmath
import numpy as np

from fractalforge.viewer.state import ViewerState

# Zoom factor per scroll notch / click / key press.
ZOOM_FACTOR = 2.0


def _view_extent_mp(canvas_w, canvas_h, state: ViewerState):
    """Return (view_w, view_h, zoom) as mpmath values at current precision."""
    mpmath.mp.dps = state.precision_digits
    zoom = state.zoom_mp
    aspect = mpmath.mpf(canvas_w) / mpmath.mpf(canvas_h)
    view_h = mpmath.mpf(3) / zoom
    view_w = view_h * aspect
    return view_w, view_h, zoom


def _pixel_to_complex_hp(mouse_x, mouse_y, canvas_w, canvas_h, state):
    """Convert a canvas pixel to a complex coordinate (mpmath, full precision)."""
    view_w, view_h, _ = _view_extent_mp(canvas_w, canvas_h, state)
    center_re = mpmath.mpf(state.center_re_hp)
    center_im = mpmath.mpf(state.center_im_hp)
    frac_x = mpmath.mpf(mouse_x) / mpmath.mpf(canvas_w) - mpmath.mpf("0.5")
    frac_y = mpmath.mpf(mouse_y) / mpmath.mpf(canvas_h) - mpmath.mpf("0.5")
    return center_re + frac_x * view_w, center_im + frac_y * view_h


class FractalCanvas:
    """Manages the fractal display texture and interaction."""

    def __init__(self, state: ViewerState):
        self.state = state
        self.texture_tag = None
        self.image_tag = None
        self.window_tag = None
        self._dragging = False
        self._drag_start_center = (None, None)  # mpf snapshot at press
        self._tex_width = 0
        self._tex_height = 0

    def setup(self, parent_tag):
        """Create the texture, image widget, and input handlers."""
        self.window_tag = parent_tag
        self._create_texture_and_image(parent_tag)

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self._on_click)
            dpg.add_mouse_wheel_handler(callback=self._on_scroll)
            for btn in (dpg.mvMouseButton_Right, dpg.mvMouseButton_Middle):
                dpg.add_mouse_drag_handler(button=btn, callback=self._on_drag)
                dpg.add_mouse_release_handler(button=btn, callback=self._on_drag_release)
            # Keyboard navigation
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=lambda: self._pan_keys(0, -1))
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=lambda: self._pan_keys(0, 1))
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=lambda: self._pan_keys(-1, 0))
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=lambda: self._pan_keys(1, 0))
            dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda: self.state.reset_view())
            dpg.add_key_press_handler(dpg.mvKey_B, callback=self._on_key_bookmark)
            for k in (dpg.mvKey_Plus, dpg.mvKey_Add):
                dpg.add_key_press_handler(k, callback=lambda: self._zoom_center(ZOOM_FACTOR))
            for k in (dpg.mvKey_Minus, dpg.mvKey_Subtract):
                dpg.add_key_press_handler(k, callback=lambda: self._zoom_center(1.0 / ZOOM_FACTOR))

    # ---- texture ------------------------------------------------------------

    def _create_texture_and_image(self, parent_tag):
        """Create (or recreate) the raw texture and image widget."""
        w = self.state.preview_width
        h = self.state.preview_height

        if self.image_tag is not None:
            dpg.delete_item(self.image_tag)
            self.image_tag = None
        if self.texture_tag is not None:
            dpg.delete_item(self.texture_tag)
            self.texture_tag = None

        initial_data = array.array('f', [0.0] * (w * h * 4))
        for i in range(w * h):
            initial_data[i * 4 + 3] = 1.0  # alpha

        with dpg.texture_registry():
            self.texture_tag = dpg.add_raw_texture(
                width=w, height=h, default_value=initial_data,
                format=dpg.mvFormat_Float_rgba,
            )

        self.image_tag = dpg.add_image(self.texture_tag, parent=parent_tag,
                                       width=w, height=h)
        self._tex_width = w
        self._tex_height = h

    def update_texture(self, data: np.ndarray):
        """Update the display texture with new render data."""
        if self.texture_tag is None:
            return
        w = self.state.preview_width
        h = self.state.preview_height
        if w != self._tex_width or h != self._tex_height:
            self._create_texture_and_image(self.window_tag)
        raw = array.array('f', data.astype(np.float32).tobytes())
        dpg.set_value(self.texture_tag, raw)

    # ---- helpers ------------------------------------------------------------

    def _get_mouse_on_canvas(self):
        """Mouse position relative to the canvas image, or None if outside."""
        if self.image_tag is None or self.window_tag is None:
            return None
        mouse_pos = dpg.get_mouse_pos(local=False)
        img_pos = dpg.get_item_pos(self.image_tag)
        win_pos = dpg.get_item_pos(self.window_tag)
        cx = mouse_pos[0] - win_pos[0] - img_pos[0]
        cy = mouse_pos[1] - win_pos[1] - img_pos[1]
        if 0 <= cx < self._tex_width and 0 <= cy < self._tex_height:
            return cx, cy
        return None

    # ---- mouse --------------------------------------------------------------

    def _on_click(self, sender, app_data):
        """Left-click: recenter on the clicked point and zoom in."""
        if app_data != dpg.mvMouseButton_Left:
            return
        pos = self._get_mouse_on_canvas()
        if pos is None:
            return
        re, im = _pixel_to_complex_hp(pos[0], pos[1], self._tex_width,
                                      self._tex_height, self.state)
        self.state.zoom_by(ZOOM_FACTOR)
        self.state.set_center(re, im)
        self.state.request_render()

    def _on_scroll(self, sender, app_data):
        """Scroll: zoom toward the cursor, keeping the point under it fixed."""
        pos = self._get_mouse_on_canvas()
        if pos is None:
            return
        cx, cy = pos
        w, h = self._tex_width, self._tex_height
        re, im = _pixel_to_complex_hp(cx, cy, w, h, self.state)
        self.state.zoom_by(ZOOM_FACTOR if app_data > 0 else 1.0 / ZOOM_FACTOR)
        re_after, im_after = _pixel_to_complex_hp(cx, cy, w, h, self.state)
        center_re = mpmath.mpf(self.state.center_re_hp) + (re - re_after)
        center_im = mpmath.mpf(self.state.center_im_hp) + (im - im_after)
        self.state.set_center(center_re, center_im)
        self.state.request_render()

    def _on_drag(self, sender, app_data):
        """Pan: apply the cumulative drag delta against the press-time center.

        DPG reports drag delta as cumulative pixels since the button went down,
        so we snapshot the center once at drag start and offset from it -- this
        avoids the runaway acceleration of re-applying the growing delta to an
        already-moved center every frame.
        """
        if len(app_data) < 3:
            return
        dx_pixels, dy_pixels = app_data[1], app_data[2]
        w, h = self._tex_width, self._tex_height

        if not self._dragging:
            self._dragging = True
            self._drag_start_center = (mpmath.mpf(self.state.center_re_hp),
                                       mpmath.mpf(self.state.center_im_hp))

        view_w, view_h, _ = _view_extent_mp(w, h, self.state)
        start_re, start_im = self._drag_start_center
        center_re = start_re - mpmath.mpf(dx_pixels) * view_w / w
        center_im = start_im - mpmath.mpf(dy_pixels) * view_h / h
        self.state.set_center(center_re, center_im)
        self.state.request_render()

    def _on_drag_release(self, sender, app_data):
        """End the current pan."""
        self._dragging = False
        self._drag_start_center = (None, None)

    # ---- keyboard -----------------------------------------------------------

    def _pan_keys(self, dx_frac, dy_frac):
        """Pan by a fraction of the viewport (arrow keys)."""
        step = 0.12
        view_w, view_h, _ = _view_extent_mp(self._tex_width, self._tex_height,
                                            self.state)
        center_re = mpmath.mpf(self.state.center_re_hp) + dx_frac * step * view_w
        center_im = mpmath.mpf(self.state.center_im_hp) + dy_frac * step * view_h
        self.state.set_center(center_re, center_im)
        self.state.request_render()

    def _zoom_center(self, factor):
        """Zoom toward the center of the view (keyboard +/-)."""
        self.state.zoom_by(factor)
        self.state.request_render()

    def _on_key_bookmark(self):
        self.state.add_bookmark()
