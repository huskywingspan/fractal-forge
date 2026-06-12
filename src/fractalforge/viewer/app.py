"""FractalForge Viewer -- application lifecycle and main loop.

Creates the Dear PyGui application with a responsive layout that adapts
to viewport resizing. Canvas fills the left area, sidebar panels stack
on the right.
"""

import dearpygui.dearpygui as dpg

from fractalforge.viewer.state import ViewerState
from fractalforge.viewer.render_bridge import RenderBridge
from fractalforge.viewer.canvas import FractalCanvas
from fractalforge.viewer.controls import ControlPanel
from fractalforge.viewer.coordinate_panel import CoordinatePanel
from fractalforge.viewer.video_panel import VideoRenderPanel


class ViewerApp:
    """Main application class for the interactive fractal viewer."""

    INITIAL_WIDTH = 1280
    INITIAL_HEIGHT = 800
    SIDEBAR_WIDTH = 320

    def __init__(self):
        self.state = ViewerState()
        self.bridge = RenderBridge()
        self.canvas = FractalCanvas(self.state)
        self.controls = ControlPanel(self.state)
        self.coords = CoordinatePanel(self.state)
        self.video = VideoRenderPanel(self.state)
        self._last_vp_width = 0
        self._last_vp_height = 0

    def run(self):
        """Launch the viewer application."""
        dpg.create_context()
        dpg.create_viewport(
            title="FractalForge Viewer",
            width=self.INITIAL_WIDTH,
            height=self.INITIAL_HEIGHT,
            min_width=800,
            min_height=600,
        )

        self._build_layout()

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Set viewport resize callback
        dpg.set_viewport_resize_callback(self._on_viewport_resize)

        # Trigger the initial render
        self.bridge.submit(self.state)

        # Main loop
        while dpg.is_dearpygui_running():
            self._tick()
            dpg.render_dearpygui_frame()

        self._shutdown()

    def _build_layout(self):
        """Create the window layout with canvas and side panels."""
        vp_w = self.INITIAL_WIDTH
        vp_h = self.INITIAL_HEIGHT
        sidebar = self.SIDEBAR_WIDTH
        canvas_w = vp_w - sidebar

        # Main canvas window (left side, fills remaining space)
        with dpg.window(
            label="Fractal",
            tag="canvas_window",
            no_close=True,
            no_collapse=True,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
            width=canvas_w,
            height=vp_h,
            pos=[0, 0],
        ):
            self.canvas.setup("canvas_window")

        # Controls panel (right side, top third)
        panel_h = vp_h // 3
        with dpg.window(
            label="Controls",
            tag="controls_window",
            no_close=True,
            no_move=True,
            no_resize=True,
            width=sidebar,
            height=panel_h,
            pos=[canvas_w, 0],
        ):
            self.controls.setup("controls_window")

        # Coordinates panel (right side, middle third)
        with dpg.window(
            label="Coordinates",
            tag="coords_window",
            no_close=True,
            no_move=True,
            no_resize=True,
            width=sidebar,
            height=panel_h,
            pos=[canvas_w, panel_h],
        ):
            self.coords.setup("coords_window")

        # Video render panel (right side, bottom third)
        with dpg.window(
            label="Video Render",
            tag="video_window",
            no_close=True,
            no_move=True,
            no_resize=True,
            width=sidebar,
            height=panel_h,
            pos=[canvas_w, panel_h * 2],
        ):
            self.video.setup("video_window")

        self._last_vp_width = vp_w
        self._last_vp_height = vp_h

    def _on_viewport_resize(self, sender, app_data):
        """Reposition and resize all windows when viewport changes."""
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()

        if vp_w == self._last_vp_width and vp_h == self._last_vp_height:
            return
        if vp_w < 100 or vp_h < 100:
            return

        self._last_vp_width = vp_w
        self._last_vp_height = vp_h

        sidebar = self.SIDEBAR_WIDTH
        canvas_w = max(200, vp_w - sidebar)
        panel_h = vp_h // 3

        dpg.configure_item("canvas_window", width=canvas_w, height=vp_h, pos=[0, 0])
        dpg.configure_item("controls_window", width=sidebar, height=panel_h,
                           pos=[canvas_w, 0])
        dpg.configure_item("coords_window", width=sidebar, height=panel_h,
                           pos=[canvas_w, panel_h])
        dpg.configure_item("video_window", width=sidebar, height=panel_h,
                           pos=[canvas_w, panel_h * 2])

    def _tick(self):
        """Called every frame -- check for render results, update HUD."""
        # Submit new render if state changed
        if self.state.render_pending:
            self.state.render_pending = False
            self.bridge.submit(self.state)

        # Check for completed render
        result = self.bridge.check(self.state)
        if result is not None:
            self.canvas.update_texture(result)

        # Update coordinate display
        self.coords.update()

        # Update video panel
        self.video.update()

    def _shutdown(self):
        """Clean up resources."""
        self.bridge.shutdown()
        dpg.destroy_context()
