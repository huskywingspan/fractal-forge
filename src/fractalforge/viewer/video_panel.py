"""Video render panel -- launch deep zoom video renders from the viewer.

Generates a two-keyframe zoom path from the overview (zoom=1) to the
current viewer position, then renders the frame sequence in a background
thread and encodes to video via FFmpeg.
"""

import math
import threading
import time
from pathlib import Path

import dearpygui.dearpygui as dpg

from fractalforge.viewer.state import ViewerState

RESOLUTIONS = {
    "1280x720 (720p)": (1280, 720),
    "1920x1080 (1080p)": (1920, 1080),
    "2560x1440 (1440p)": (2560, 1440),
    "3840x2160 (4K)": (3840, 2160),
    "5120x1440 (Ultrawide)": (5120, 1440),
    "2560x720 (UW draft)": (2560, 720),
    "720x1280 (Shorts draft)": (720, 1280),
    "1080x1920 (Shorts 9:16)": (1080, 1920),
}

PRESETS = ["preview", "quality", "youtube", "lossless"]

# Creator presets: Format picks the aspect/venue, Quality picks the
# speed/fidelity tradeoff. Together they drive resolution, SSAA, and the
# encode preset — every widget stays editable afterwards, presets just set
# sensible values.
FORMATS = ["YouTube Long (16:9)", "YouTube Shorts (9:16)", "Ultrawide (32:9)"]
QUALITIES = ["Draft", "Standard", "Production"]

# (resolution key, ssaa widget value, encode preset)
_PRESET_MATRIX = {
    ("YouTube Long (16:9)", "Draft"): ("1280x720 (720p)", "1 (none)", "preview"),
    ("YouTube Long (16:9)", "Standard"): ("1920x1080 (1080p)", "1 (none)", "quality"),
    ("YouTube Long (16:9)", "Production"): ("3840x2160 (4K)", "2 (4x SSAA)", "youtube"),
    ("YouTube Shorts (9:16)", "Draft"): ("720x1280 (Shorts draft)", "1 (none)", "preview"),
    ("YouTube Shorts (9:16)", "Standard"): ("1080x1920 (Shorts 9:16)", "1 (none)", "quality"),
    ("YouTube Shorts (9:16)", "Production"): ("1080x1920 (Shorts 9:16)", "2 (4x SSAA)", "youtube"),
    ("Ultrawide (32:9)", "Draft"): ("2560x720 (UW draft)", "1 (none)", "preview"),
    ("Ultrawide (32:9)", "Standard"): ("5120x1440 (Ultrawide)", "1 (none)", "quality"),
    ("Ultrawide (32:9)", "Production"): ("5120x1440 (Ultrawide)", "2 (4x SSAA)", "youtube"),
}

# Default duration when switching format (Shorts must stay under 60s)
_FORMAT_DURATION = {
    "YouTube Long (16:9)": 60,
    "YouTube Shorts (9:16)": 45,
    "Ultrawide (32:9)": 60,
}


class VideoRenderPanel:
    """Side panel for configuring and launching video renders."""

    def __init__(self, state: ViewerState):
        self.state = state
        self._worker_thread: threading.Thread | None = None
        self._cancel_flag = threading.Event()
        self._progress = 0.0
        self._status_text = "Ready"
        self._rendering = False
        self._last_output_path = ""

    def setup(self, parent_tag):
        """Build the video render panel UI."""
        with dpg.group(parent=parent_tag):
            dpg.add_text("Video Render", color=(0, 212, 255))
            dpg.add_separator()

            dpg.add_text("Format")
            dpg.add_combo(
                items=FORMATS,
                default_value=FORMATS[0],
                tag="video_format",
                width=-1,
                callback=self._on_preset_change,
            )
            dpg.add_text("Quality")
            dpg.add_combo(
                items=QUALITIES,
                default_value="Standard",
                tag="video_quality",
                width=-1,
                callback=self._on_preset_change,
            )

            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_text("Output Resolution")
            dpg.add_combo(
                items=list(RESOLUTIONS.keys()),
                default_value="1920x1080 (1080p)",
                tag="video_resolution",
                width=-1,
            )

            dpg.add_spacer(height=4)
            dpg.add_text("Duration (seconds)")
            dpg.add_input_int(
                default_value=60,
                min_value=5, max_value=600,
                min_clamped=True, max_clamped=True,
                tag="video_duration",
                width=-1,
            )

            dpg.add_spacer(height=4)
            dpg.add_text("FPS")
            dpg.add_combo(
                items=["30", "60"],
                default_value="60",
                tag="video_fps",
                width=-1,
            )

            dpg.add_spacer(height=4)
            dpg.add_text("Encode Preset")
            dpg.add_combo(
                items=PRESETS,
                default_value="quality",
                tag="video_preset",
                width=-1,
            )

            dpg.add_spacer(height=4)
            dpg.add_text("Supersampling")
            dpg.add_combo(
                items=["1 (none)", "2 (4x SSAA)"],
                default_value="1 (none)",
                tag="video_ssaa",
                width=-1,
            )

            dpg.add_spacer(height=4)
            dpg.add_checkbox(
                label="Histogram EQ",
                default_value=True,
                tag="video_histogram",
            )
            dpg.add_checkbox(
                label="Slope Shading (3D)",
                default_value=True,
                tag="video_slope_shading",
            )
            dpg.add_checkbox(
                label="Log Scaling (smooth bands)",
                default_value=True,
                tag="video_log_scaling",
            )

            dpg.add_spacer(height=4)
            dpg.add_text("Color Cycling Speed")
            dpg.add_slider_float(
                default_value=2.0,
                min_value=0.0, max_value=20.0,
                tag="video_cycle_speed",
                width=-1,
                format="%.1f",
            )

            dpg.add_spacer(height=8)
            dpg.add_separator()

            # Target info (read-only, shows current viewer position)
            dpg.add_text("Target Point", color=(168, 85, 247))
            dpg.add_text("", tag="video_target_info", wrap=280)

            dpg.add_spacer(height=8)
            dpg.add_separator()

            # Render button
            dpg.add_button(
                label="Render Video",
                callback=self._on_render,
                tag="video_render_btn",
                width=-1,
                height=32,
            )

            dpg.add_spacer(height=4)
            dpg.add_button(
                label="Cancel",
                callback=self._on_cancel,
                tag="video_cancel_btn",
                width=-1,
                show=False,
            )

            # Progress
            dpg.add_spacer(height=4)
            dpg.add_progress_bar(
                default_value=0.0,
                tag="video_progress",
                width=-1,
                show=False,
            )
            dpg.add_text("", tag="video_status", wrap=280)

    def update(self):
        """Called every frame to refresh target info and progress."""
        # Update target info
        lz = self.state.log10_zoom
        if lz < 4:
            zoom_str = f"{10.0 ** lz:.1f}"
        elif lz < 290:
            zoom_str = f"{10.0 ** lz:.2e}"
        else:
            zoom_str = f"10^{lz:.1f}"
        if self.state.needs_perturbation:
            re_str = self.state.center_re_hp
            im_str = self.state.center_im_hp
            # Truncate for display
            if len(re_str) > 24:
                re_str = re_str[:24] + "..."
            if len(im_str) > 24:
                im_str = im_str[:24] + "..."
        else:
            re_str = f"{self.state.center_re:.12g}"
            im_str = f"{self.state.center_im:.12g}"
        info = f"Re: {re_str}\nIm: {im_str}\nZoom: {zoom_str}"
        dpg.set_value("video_target_info", info)

        # Update progress
        if self._rendering:
            dpg.set_value("video_progress", self._progress)
            dpg.set_value("video_status", self._status_text)
            dpg.configure_item("video_progress", show=True)
            dpg.configure_item("video_cancel_btn", show=True)
            dpg.configure_item("video_render_btn", enabled=False)
        else:
            dpg.set_value("video_status", self._status_text)
            dpg.configure_item("video_cancel_btn", show=False)
            dpg.configure_item("video_render_btn", enabled=True)
            if self._progress >= 1.0:
                dpg.configure_item("video_progress", show=True)

    def _on_preset_change(self, sender=None, app_data=None):
        """Apply the Format x Quality preset to the concrete widgets."""
        fmt = dpg.get_value("video_format")
        quality = dpg.get_value("video_quality")
        entry = _PRESET_MATRIX.get((fmt, quality))
        if entry is None:
            return
        res_key, ssaa_val, encode = entry
        dpg.set_value("video_resolution", res_key)
        dpg.set_value("video_ssaa", ssaa_val)
        dpg.set_value("video_preset", encode)
        if sender == "video_format":
            dpg.set_value("video_duration", _FORMAT_DURATION.get(fmt, 60))

    def _on_render(self, sender=None, app_data=None):
        """Start video render in background thread."""
        if self._rendering:
            return

        res_key = dpg.get_value("video_resolution")
        width, height = RESOLUTIONS.get(res_key, (1920, 1080))
        ssaa_key = dpg.get_value("video_ssaa")

        # Full snapshot: render settings from the panel widgets, target and
        # LOOK from the live viewer state -- what you see while exploring is
        # exactly what the video renders. zoom_str is deep-safe (can exceed
        # float64's 1e308 ceiling).
        s = self.state
        cfg = {
            "width": width,
            "height": height,
            "duration": dpg.get_value("video_duration"),
            "fps": int(dpg.get_value("video_fps")),
            "preset": dpg.get_value("video_preset"),
            "ssaa": 2 if "2" in ssaa_key else 1,
            "histogram": dpg.get_value("video_histogram"),
            "slope_shading": dpg.get_value("video_slope_shading"),
            "log_scaling": dpg.get_value("video_log_scaling"),
            "cycle_speed": dpg.get_value("video_cycle_speed"),
            "target_re": s.center_re,
            "target_im": s.center_im,
            "target_re_hp": s.center_re_hp,
            "target_im_hp": s.center_im_hp,
            "target_zoom": s.zoom_str,
            "target_log10": s.log10_zoom,
            "target_max_iter": s.max_iter,
            "palette": s.palette_name,
            "fractal_type": s.fractal_type,
            "julia_re": s.julia_re,
            "julia_im": s.julia_im,
            # Look: everything you tuned live carries into the render
            "color_mode": None if s.color_mode == "auto" else s.color_mode,
            "vignette": s.vignette,
            "contrast": s.contrast,
            "saturation": s.saturation,
            "brightness": s.brightness,
            "bloom": s.bloom,
            "halation": s.halation,
            "tone_map": s.tone_map,
            "exposure": s.exposure,
        }

        self._cancel_flag.clear()
        self._rendering = True
        self._progress = 0.0
        self._status_text = "Preparing..."

        self._worker_thread = threading.Thread(
            target=self._render_worker, args=(cfg,), daemon=True,
        )
        self._worker_thread.start()

    def _on_cancel(self, sender=None, app_data=None):
        """Signal the worker to stop."""
        self._cancel_flag.set()
        self._status_text = "Cancelling..."

    def _render_worker(self, cfg: dict):
        """Background thread: render frames + encode video."""
        width = cfg["width"]
        height = cfg["height"]
        duration = cfg["duration"]
        fps = cfg["fps"]
        preset = cfg["preset"]
        ssaa = cfg["ssaa"]
        histogram = cfg["histogram"]
        slope_shading = cfg["slope_shading"]
        log_scaling = cfg["log_scaling"]
        cycle_speed = cfg["cycle_speed"]
        target_re = cfg["target_re"]
        target_im = cfg["target_im"]
        target_re_hp = cfg["target_re_hp"]
        target_im_hp = cfg["target_im_hp"]
        target_zoom = cfg["target_zoom"]
        target_log10 = cfg["target_log10"]
        target_max_iter = cfg["target_max_iter"]
        palette = cfg["palette"]
        fractal_type = cfg["fractal_type"]
        julia_re = cfg["julia_re"]
        julia_im = cfg["julia_im"]
        from fractalforge.artist.zoompath import ZoomPath, Keyframe
        from fractalforge.render.sequence import render_sequence
        from fractalforge.render.video import encode_video, check_ffmpeg

        try:
            total_frames = duration * fps

            # Auto-compute max_iter for the start keyframe
            start_max_iter = max(500, target_max_iter // 4)
            # Ensure the deep end keyframe has enough iterations for its depth
            # (the final-frame structure needs ~linear-in-depth iterations).
            end_max_iter = max(target_max_iter,
                               min(int(800 + 400 * target_log10), 120000))

            # Build a simple two-keyframe zoom path:
            # Frame 0: overview at zoom=1, centered on Mandelbrot
            # Frame N: target position at target zoom
            kf_start = Keyframe(
                frame=0,
                center_re=target_re,
                center_im=target_im,
                zoom=1.0,
                max_iter=start_max_iter,
                palette=palette,
                fractal_type=fractal_type,
                julia_re=julia_re if fractal_type == "julia" else None,
                julia_im=julia_im if fractal_type == "julia" else None,
                center_re_hp=target_re_hp,
                center_im_hp=target_im_hp,
            )

            kf_end = Keyframe(
                frame=total_frames - 1,
                center_re=target_re,
                center_im=target_im,
                zoom=target_zoom,
                max_iter=end_max_iter,
                palette=palette,
                fractal_type=fractal_type,
                julia_re=julia_re if fractal_type == "julia" else None,
                julia_im=julia_im if fractal_type == "julia" else None,
                center_re_hp=target_re_hp,
                center_im_hp=target_im_hp,
            )

            zoom_path = ZoomPath(
                name="viewer_render",
                fps=fps,
                width=width,
                height=height,
                keyframes=[kf_start, kf_end],
            )

            # Output paths
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output") / f"video_{timestamp}"
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            self._status_text = f"Rendering {total_frames} frames..."

            def on_progress(frame_idx, total, elapsed, render_fps, skipped=False):
                if self._cancel_flag.is_set():
                    raise InterruptedError("Render cancelled")
                self._progress = (frame_idx + 1) / total
                eta = (elapsed / (frame_idx + 1)) * (total - frame_idx - 1) if frame_idx > 0 else 0
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)
                self._status_text = (
                    f"Frame {frame_idx + 1}/{total} "
                    f"({render_fps:.1f} fps, ETA {eta_min}m{eta_sec:02d}s)"
                )

            render_sequence(
                zoom_path=zoom_path,
                output_dir=frames_dir,
                skip_existing=True,
                supersampling=ssaa,
                on_progress=on_progress,
                histogram=histogram,
                slope_shading=slope_shading,
                cycle_speed=cycle_speed,
                log_scaling=log_scaling,
                color_mode=cfg["color_mode"],
                vignette=cfg["vignette"],
                contrast=cfg["contrast"],
                saturation=cfg["saturation"],
                brightness=cfg["brightness"],
                bloom=cfg["bloom"],
                halation=cfg["halation"],
                tone_map=cfg["tone_map"],
                exposure=cfg["exposure"],
            )

            if self._cancel_flag.is_set():
                self._status_text = "Cancelled"
                self._rendering = False
                return

            # Encode video
            self._status_text = "Encoding video..."
            self._progress = 0.95

            if check_ffmpeg():
                video_path = output_dir / f"fractal_dive.mp4"
                encode_video(
                    frames_dir=frames_dir,
                    output_path=video_path,
                    fps=fps,
                    preset=preset,
                    overwrite=True,
                )
                self._last_output_path = str(video_path)
                self._status_text = f"Done! {video_path}"
            else:
                self._status_text = f"Frames saved to {frames_dir} (FFmpeg not found)"

            self._progress = 1.0

            # Save the zoom path for reference
            zoom_path.save(output_dir / "zoom_path.json")

        except InterruptedError:
            self._status_text = "Cancelled"
        except Exception as e:
            self._status_text = f"Error: {e}"
        finally:
            self._rendering = False
