"""Fractal canvas -- displays the rendered fractal and handles mouse interaction.

Supports:
- Left-click: re-center on clicked coordinate
- Scroll wheel: zoom in/out toward cursor position
- Right-click drag: pan the viewport

At deep zoom (>= 1e13), coordinate math uses mpmath for arbitrary precision
so navigation remains accurate at zoom levels up to 1e200+.
"""

import array
import math

import dearpygui.dearpygui as dpg
import numpy as np

from fractalforge.viewer.state import ViewerState

# Zoom factor per scroll notch
ZOOM_FACTOR = 2.0

# Always use mpmath for coordinate math so precision accumulates from
# the very first click. This ensures that by the time perturbation theory
# kicks in at zoom 1e13, the hp coordinate strings have full precision
# built up incrementally from every click/scroll/pan operation.
# The mpmath overhead is ~1-2ms per operation, imperceptible to the user.
_HP_MATH_THRESHOLD = 1.0


def _pixel_to_complex(
    mouse_x: float, mouse_y: float,
    canvas_w: int, canvas_h: int,
    state: ViewerState,
) -> tuple[float, float]:
    """Convert pixel coordinates on the canvas to complex plane coordinates.

    Uses float64 math -- suitable for zoom < 1e13.
    """
    aspect = canvas_w / canvas_h
    view_h = 3.0 / state.zoom
    view_w = view_h * aspect

    re = state.center_re + (mouse_x / canvas_w - 0.5) * view_w
    im = state.center_im + (mouse_y / canvas_h - 0.5) * view_h
    return re, im


def _pixel_to_complex_hp(
    mouse_x: float, mouse_y: float,
    canvas_w: int, canvas_h: int,
    state: ViewerState,
):
    """Convert pixel coordinates to complex plane using mpmath.

    Returns (re_mpf, im_mpf) as mpmath.mpf objects with full precision.
    """
    import mpmath
    mpmath.mp.dps = state.precision_digits

    center_re = mpmath.mpf(state.center_re_hp)
    center_im = mpmath.mpf(state.center_im_hp)
    zoom = mpmath.mpf(state.zoom)

    aspect = mpmath.mpf(canvas_w) / mpmath.mpf(canvas_h)
    view_h = mpmath.mpf(3) / zoom
    view_w = view_h * aspect

    frac_x = mpmath.mpf(mouse_x) / mpmath.mpf(canvas_w) - mpmath.mpf('0.5')
    frac_y = mpmath.mpf(mouse_y) / mpmath.mpf(canvas_h) - mpmath.mpf('0.5')

    re = center_re + frac_x * view_w
    im = center_im + frac_y * view_h
    return re, im


class FractalCanvas:
    """Manages the fractal display texture and mouse interaction."""

    def __init__(self, state: ViewerState):
        self.state = state
        self.texture_tag = None
        self.image_tag = None
        self.window_tag = None
        self._dragging = False
        self._drag_start = (0, 0)
        self._drag_center_start = (0.0, 0.0)
        self._tex_width = 0
        self._tex_height = 0

    def setup(self, parent_tag):
        """Create the texture and image display widget."""
        self.window_tag = parent_tag
        self._create_texture_and_image(parent_tag)

        # Mouse handlers
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self._on_click)
            dpg.add_mouse_wheel_handler(callback=self._on_scroll)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Right,
                callback=self._on_drag,
            )
            dpg.add_mouse_release_handler(
                button=dpg.mvMouseButton_Right,
                callback=self._on_drag_release,
            )

    def _create_texture_and_image(self, parent_tag):
        """Create (or recreate) the raw texture and image widget."""
        w = self.state.preview_width
        h = self.state.preview_height

        # Clean up old texture/image if they exist
        if self.image_tag is not None:
            dpg.delete_item(self.image_tag)
            self.image_tag = None
        if self.texture_tag is not None:
            dpg.delete_item(self.texture_tag)
            self.texture_tag = None

        # Create initial black texture (RGBA float32)
        initial_data = array.array('f', [0.0] * (w * h * 4))
        for i in range(w * h):
            initial_data[i * 4 + 3] = 1.0  # alpha = 1

        with dpg.texture_registry():
            self.texture_tag = dpg.add_raw_texture(
                width=w, height=h,
                default_value=initial_data,
                format=dpg.mvFormat_Float_rgba,
            )

        # Image display
        self.image_tag = dpg.add_image(
            self.texture_tag, parent=parent_tag,
            width=w, height=h,
        )
        self._tex_width = w
        self._tex_height = h

    def update_texture(self, data: np.ndarray):
        """Update the display texture with new render data."""
        if self.texture_tag is None:
            return

        # Check if texture needs recreation (resolution changed)
        w = self.state.preview_width
        h = self.state.preview_height
        if w != self._tex_width or h != self._tex_height:
            self._create_texture_and_image(self.window_tag)

        raw = array.array('f', data.astype(np.float32).tobytes())
        dpg.set_value(self.texture_tag, raw)

    def _get_mouse_on_canvas(self) -> tuple[float, float] | None:
        """Get mouse position relative to the canvas image, or None if outside."""
        if self.image_tag is None or self.window_tag is None:
            return None

        mouse_pos = dpg.get_mouse_pos(local=False)
        # Get image screen position
        img_pos = dpg.get_item_pos(self.image_tag)
        win_pos = dpg.get_item_pos(self.window_tag)

        # Canvas-relative position
        cx = mouse_pos[0] - win_pos[0] - img_pos[0]
        cy = mouse_pos[1] - win_pos[1] - img_pos[1]

        w = self._tex_width
        h = self._tex_height

        if 0 <= cx < w and 0 <= cy < h:
            return cx, cy
        return None

    def _on_click(self, sender, app_data):
        """Left-click: re-center on the clicked coordinate and zoom in."""
        if app_data != dpg.mvMouseButton_Left:
            return

        pos = self._get_mouse_on_canvas()
        if pos is None:
            return

        cx, cy = pos
        w = self._tex_width
        h = self._tex_height

        if self.state.zoom >= _HP_MATH_THRESHOLD:
            re, im = _pixel_to_complex_hp(cx, cy, w, h, self.state)
            self.state.zoom *= ZOOM_FACTOR
            self.state.set_center(re, im)
        else:
            re, im = _pixel_to_complex(cx, cy, w, h, self.state)
            self.state.zoom *= ZOOM_FACTOR
            self.state.set_center_float(re, im)

        self.state.request_render()

    def _on_scroll(self, sender, app_data):
        """Scroll: zoom toward cursor position."""
        pos = self._get_mouse_on_canvas()
        if pos is None:
            return

        cx, cy = pos
        w = self._tex_width
        h = self._tex_height

        if self.state.zoom >= _HP_MATH_THRESHOLD:
            self._scroll_hp(cx, cy, w, h, app_data)
        else:
            self._scroll_float(cx, cy, w, h, app_data)

        self.state.request_render()

    def _scroll_float(self, cx, cy, w, h, direction):
        """Scroll zoom using float math (zoom < 1e13)."""
        # Complex coord under cursor before zoom
        re, im = _pixel_to_complex(cx, cy, w, h, self.state)

        if direction > 0:
            self.state.zoom *= ZOOM_FACTOR
        else:
            self.state.zoom /= ZOOM_FACTOR
            self.state.zoom = max(0.1, self.state.zoom)

        # Complex coord under cursor after zoom (at old center)
        re_after, im_after = _pixel_to_complex(cx, cy, w, h, self.state)

        # Adjust center so the point under cursor stays fixed
        new_re = self.state.center_re + re - re_after
        new_im = self.state.center_im + im - im_after
        self.state.set_center_float(new_re, new_im)

    def _scroll_hp(self, cx, cy, w, h, direction):
        """Scroll zoom using mpmath (zoom >= 1e13)."""
        import mpmath
        mpmath.mp.dps = self.state.precision_digits

        re, im = _pixel_to_complex_hp(cx, cy, w, h, self.state)

        if direction > 0:
            self.state.zoom *= ZOOM_FACTOR
        else:
            self.state.zoom /= ZOOM_FACTOR
            self.state.zoom = max(0.1, self.state.zoom)

        re_after, im_after = _pixel_to_complex_hp(cx, cy, w, h, self.state)

        center_re = mpmath.mpf(self.state.center_re_hp) + (re - re_after)
        center_im = mpmath.mpf(self.state.center_im_hp) + (im - im_after)
        self.state.set_center(center_re, center_im)

    def _on_drag(self, sender, app_data):
        """Right-click drag: pan the viewport."""
        # app_data = [button, dx, dy] from drag handler
        if len(app_data) < 3:
            return

        dx_pixels = app_data[1]
        dy_pixels = app_data[2]

        w = self._tex_width
        h = self._tex_height

        if self.state.zoom >= _HP_MATH_THRESHOLD:
            import mpmath
            mpmath.mp.dps = self.state.precision_digits
            aspect = mpmath.mpf(w) / mpmath.mpf(h)
            view_h = mpmath.mpf(3) / mpmath.mpf(self.state.zoom)
            view_w = view_h * aspect
            center_re = mpmath.mpf(self.state.center_re_hp) - mpmath.mpf(dx_pixels) * view_w / w
            center_im = mpmath.mpf(self.state.center_im_hp) - mpmath.mpf(dy_pixels) * view_h / h
            self.state.set_center(center_re, center_im)
        else:
            aspect = w / h
            view_h = 3.0 / self.state.zoom
            view_w = view_h * aspect
            new_re = self.state.center_re - dx_pixels * view_w / w
            new_im = self.state.center_im - dy_pixels * view_h / h
            self.state.set_center_float(new_re, new_im)

        self.state.request_render()

    def _on_drag_release(self, sender, app_data):
        """Right-click release: stop panning."""
        pass
