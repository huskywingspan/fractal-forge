"""Control panel -- parameter sliders and dropdowns for the viewer."""

import dearpygui.dearpygui as dpg

from fractalforge.viewer.state import ViewerState

FRACTAL_TYPES = ["mandelbrot", "julia", "burning_ship"]
PALETTE_NAMES = [
    "ocean", "electric", "fire", "nebula", "monochrome",
    "deep_blue", "inferno", "arctic", "prism", "twilight",
    "ocean_waves", "volcanic", "aurora", "neon_city", "patina",
    "royal", "abyssal", "solar_flare", "frozen", "jungle",
    "vaporwave", "stained_glass", "midnight_rose", "supernova",
]
PREVIEW_SIZES = {
    "160x90": (160, 90),
    "320x180": (320, 180),
    "480x270": (480, 270),
    "640x360": (640, 360),
    "960x540": (960, 540),
    "1280x720": (1280, 720),
    "1920x1080": (1920, 1080),
}


class ControlPanel:
    """Side panel with fractal parameter controls."""

    def __init__(self, state: ViewerState):
        self.state = state
        self._julia_group = None

    def setup(self, parent_tag):
        """Build the control panel UI."""
        with dpg.group(parent=parent_tag):
            dpg.add_text("Fractal Type")
            dpg.add_radio_button(
                items=FRACTAL_TYPES,
                default_value=self.state.fractal_type,
                callback=self._on_fractal_type,
                horizontal=True,
            )

            dpg.add_separator()

            # Julia parameters (only visible for julia type)
            self._julia_group = dpg.add_group(show=self.state.fractal_type == "julia")
            with dpg.group(parent=self._julia_group):
                dpg.add_text("Julia c-parameter")
                dpg.add_input_float(
                    label="Re(c)", default_value=self.state.julia_re,
                    callback=self._on_julia_re, width=200,
                    step=0.01, format="%.6f",
                    tag="julia_re_input",
                )
                dpg.add_input_float(
                    label="Im(c)", default_value=self.state.julia_im,
                    callback=self._on_julia_im, width=200,
                    step=0.01, format="%.6f",
                    tag="julia_im_input",
                )
                dpg.add_separator()

            dpg.add_text("Palette")
            dpg.add_combo(
                items=PALETTE_NAMES,
                default_value=self.state.palette_name,
                callback=self._on_palette,
                width=200,
            )

            dpg.add_text("Color Mapping")
            dpg.add_combo(
                items=["auto", "default", "histogram", "normalized"],
                default_value=self.state.color_mode,
                callback=self._on_color_mode,
                width=200,
                tag="color_mode_combo",
            )

            dpg.add_separator()
            dpg.add_text("Max Iterations")
            dpg.add_input_int(
                default_value=self.state.max_iter,
                min_value=50, max_value=500000,
                min_clamped=True, max_clamped=True,
                step=500, step_fast=5000,
                callback=self._on_max_iter,
                width=200,
                tag="max_iter_input",
            )
            dpg.add_checkbox(
                label="Auto (scale with zoom)",
                default_value=self.state.auto_max_iter,
                callback=self._on_auto_max_iter,
                tag="auto_max_iter_cb",
            )

            dpg.add_separator()
            dpg.add_text("Preview Resolution")
            dpg.add_combo(
                items=list(PREVIEW_SIZES.keys()),
                default_value="640x360",
                callback=self._on_preview_size,
                width=200,
            )

            dpg.add_separator()
            dpg.add_text("Post-Processing")
            dpg.add_checkbox(
                label="Histogram EQ",
                default_value=self.state.histogram,
                callback=self._on_histogram,
            )
            dpg.add_checkbox(
                label="Slope Shading (3D)",
                default_value=self.state.slope_shading,
                callback=self._on_slope_shading,
            )
            dpg.add_slider_float(
                label="Vignette",
                default_value=self.state.vignette,
                min_value=0.0, max_value=1.0,
                callback=self._on_vignette,
                width=200,
            )
            dpg.add_slider_float(
                label="Contrast",
                default_value=self.state.contrast,
                min_value=0.5, max_value=2.0,
                callback=self._on_contrast,
                width=200,
            )
            dpg.add_slider_float(
                label="Saturation",
                default_value=self.state.saturation,
                min_value=0.0, max_value=2.0,
                callback=self._on_saturation,
                width=200,
            )
            dpg.add_slider_float(
                label="Brightness",
                default_value=self.state.brightness,
                min_value=0.5, max_value=2.0,
                callback=self._on_brightness,
                width=200,
            )

    def _on_fractal_type(self, sender, app_data):
        self.state.fractal_type = app_data
        if self._julia_group is not None:
            dpg.configure_item(self._julia_group, show=(app_data == "julia"))
        self.state.request_render()

    def _on_julia_re(self, sender, app_data):
        self.state.julia_re = app_data
        self.state.request_render()

    def _on_julia_im(self, sender, app_data):
        self.state.julia_im = app_data
        self.state.request_render()

    def _on_palette(self, sender, app_data):
        self.state.palette_name = app_data
        self.state.request_render()

    def _on_color_mode(self, sender, app_data):
        self.state.color_mode = app_data
        self.state.request_render()

    def _on_max_iter(self, sender, app_data):
        self.state.max_iter = app_data
        self.state.request_render()

    def _on_auto_max_iter(self, sender, app_data):
        self.state.auto_max_iter = app_data
        self.state.request_render()

    def _on_preview_size(self, sender, app_data):
        w, h = PREVIEW_SIZES[app_data]
        self.state.preview_width = w
        self.state.preview_height = h
        self.state.request_render()

    def _on_histogram(self, sender, app_data):
        self.state.histogram = app_data
        self.state.request_render()

    def _on_slope_shading(self, sender, app_data):
        self.state.slope_shading = app_data
        self.state.request_render()

    def _on_vignette(self, sender, app_data):
        self.state.vignette = app_data
        self.state.request_render()

    def _on_contrast(self, sender, app_data):
        self.state.contrast = app_data
        self.state.request_render()

    def _on_saturation(self, sender, app_data):
        self.state.saturation = app_data
        self.state.request_render()

    def _on_brightness(self, sender, app_data):
        self.state.brightness = app_data
        self.state.request_render()
