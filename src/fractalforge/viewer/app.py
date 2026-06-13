"""FractalForge Viewer -- application lifecycle and main loop.

Premium Dear PyGui explorer: themed layout with a canvas, a unified sidebar
(controls / coordinates / video), and a bottom status bar. Rendering is
progressive -- a fast reduced-scale pass during interaction, replaced by a
full-resolution pass once the view settles -- so deep zooms stay responsive.
"""

import time

import dearpygui.dearpygui as dpg

from fractalforge.viewer.state import ViewerState
from fractalforge.viewer.render_bridge import RenderBridge, engine_label
from fractalforge.viewer.canvas import FractalCanvas
from fractalforge.viewer.controls import ControlPanel
from fractalforge.viewer.coordinate_panel import CoordinatePanel
from fractalforge.viewer.video_panel import VideoRenderPanel
from fractalforge.viewer.theme import apply_theme, CYAN, VIOLET, TEXT_DIM


class ViewerApp:
    """Main application class for the interactive fractal viewer."""

    INITIAL_WIDTH = 1360
    INITIAL_HEIGHT = 860
    SIDEBAR_WIDTH = 340
    STATUS_HEIGHT = 30

    # Progressive rendering
    PREVIEW_SCALE = 0.4       # reduced-scale pass during interaction
    IDLE_DEBOUNCE = 0.18      # seconds of stillness before the full-res pass

    def __init__(self):
        self.state = ViewerState()
        self.bridge = RenderBridge()
        self.canvas = FractalCanvas(self.state)
        self.controls = ControlPanel(self.state)
        self.coords = CoordinatePanel(self.state)
        self.video = VideoRenderPanel(self.state)
        self._last_vp_width = 0
        self._last_vp_height = 0
        self._pending_full = False
        self._last_change = 0.0

    def run(self):
        """Launch the viewer application."""
        dpg.create_context()
        dpg.create_viewport(
            title="FractalForge - Infinite Descent",
            width=self.INITIAL_WIDTH,
            height=self.INITIAL_HEIGHT,
            min_width=900,
            min_height=640,
            small_icon="", large_icon="",
        )
        apply_theme()
        self._build_layout()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_resize_callback(self._on_viewport_resize)

        # Initial render (full scale)
        self.bridge.submit(self.state, scale=1.0)
        self.state.render_pending = False

        while dpg.is_dearpygui_running():
            self._tick()
            dpg.render_dearpygui_frame()

        self._shutdown()

    # ---- layout -------------------------------------------------------------

    def _build_layout(self):
        vp_w = self.INITIAL_WIDTH
        vp_h = self.INITIAL_HEIGHT
        sidebar = self.SIDEBAR_WIDTH
        status = self.STATUS_HEIGHT
        canvas_w = vp_w - sidebar
        body_h = vp_h - status

        with dpg.window(tag="canvas_window", no_close=True, no_collapse=True,
                        no_title_bar=True, no_move=True, no_resize=True,
                        no_scrollbar=True, width=canvas_w, height=body_h,
                        pos=[0, 0]):
            self.canvas.setup("canvas_window")

        # Unified sidebar with collapsing sections
        with dpg.window(tag="sidebar_window", no_close=True, no_collapse=True,
                        no_title_bar=True, no_move=True, no_resize=True,
                        width=sidebar, height=body_h, pos=[canvas_w, 0]):
            dpg.add_text("FRACTALFORGE", color=CYAN)
            dpg.add_text("Infinite Descent  -  deep zoom explorer", color=TEXT_DIM)
            dpg.add_separator()
            with dpg.collapsing_header(label="Controls", default_open=True):
                self.controls.setup(dpg.last_item())
            with dpg.collapsing_header(label="Coordinates & Discovery",
                                       default_open=True):
                self.coords.setup(dpg.last_item())
            with dpg.collapsing_header(label="Video Render", default_open=False):
                self.video.setup(dpg.last_item())

        # Status bar
        with dpg.window(tag="status_window", no_close=True, no_collapse=True,
                        no_title_bar=True, no_move=True, no_resize=True,
                        no_scrollbar=True, width=vp_w, height=status,
                        pos=[0, body_h]):
            with dpg.group(horizontal=True):
                dpg.add_text("", tag="status_zoom")
                dpg.add_text("|", color=TEXT_DIM)
                dpg.add_text("", tag="status_engine", color=CYAN)
                dpg.add_text("|", color=TEXT_DIM)
                dpg.add_text("", tag="status_iter", color=TEXT_DIM)
                dpg.add_text("|", color=TEXT_DIM)
                dpg.add_text("", tag="status_prec", color=TEXT_DIM)
                dpg.add_text("|", color=TEXT_DIM)
                dpg.add_text("", tag="status_time", color=VIOLET)

        self._last_vp_width = vp_w
        self._last_vp_height = vp_h

    def _on_viewport_resize(self, sender, app_data):
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()
        if vp_w == self._last_vp_width and vp_h == self._last_vp_height:
            return
        if vp_w < 200 or vp_h < 200:
            return
        self._last_vp_width = vp_w
        self._last_vp_height = vp_h

        sidebar = self.SIDEBAR_WIDTH
        status = self.STATUS_HEIGHT
        canvas_w = max(240, vp_w - sidebar)
        body_h = max(200, vp_h - status)
        dpg.configure_item("canvas_window", width=canvas_w, height=body_h, pos=[0, 0])
        dpg.configure_item("sidebar_window", width=sidebar, height=body_h,
                           pos=[canvas_w, 0])
        dpg.configure_item("status_window", width=vp_w, height=status,
                           pos=[0, body_h])

    # ---- main loop ----------------------------------------------------------

    def _tick(self):
        now = time.perf_counter()

        # Progressive rendering: fast reduced-scale pass on any change, then a
        # full-resolution pass once the view has been still for IDLE_DEBOUNCE.
        if self.state.render_pending:
            self.state.render_pending = False
            self._last_change = now
            self._pending_full = True
            self.bridge.submit(self.state, scale=self.PREVIEW_SCALE)
        elif (self._pending_full and not self.bridge.busy()
              and (now - self._last_change) > self.IDLE_DEBOUNCE):
            self._pending_full = False
            self.bridge.submit(self.state, scale=1.0)

        result = self.bridge.check(self.state)
        if result is not None:
            self.canvas.update_texture(result)

        self.coords.update()
        self.video.update()
        self._update_status()

    def _update_status(self):
        s = self.state
        lz = s.log10_zoom
        if lz < 4:
            zoom_str = f"Zoom {10.0 ** lz:,.1f}x"
        elif lz < 290:
            zoom_str = f"Zoom {10.0 ** lz:.3e}x"
        else:
            zoom_str = f"Zoom 10^{lz:.1f}"
        dpg.set_value("status_zoom", zoom_str)
        dpg.set_value("status_engine", f"Engine {s.last_engine}")

        from fractalforge.viewer.render_bridge import auto_max_iter
        eff = auto_max_iter(lz, s.max_iter, s.auto_max_iter)
        dpg.set_value("status_iter", f"Iter {eff}")
        dpg.set_value("status_prec",
                      f"Prec {s.precision_digits}d" if s.needs_perturbation else "Prec f64")
        ms = s.last_render_ms
        t = f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"
        dpg.set_value("status_time", f"Render {t}")

    def _shutdown(self):
        self.bridge.shutdown()
        dpg.destroy_context()
