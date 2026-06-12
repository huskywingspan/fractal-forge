"""Coordinate panel -- displays current position and manages bookmarks."""

import threading

import dearpygui.dearpygui as dpg

from fractalforge.viewer.state import ViewerState


class CoordinatePanel:
    """Shows current coordinates, copy-to-clipboard, bookmarks, and Newton discovery."""

    def __init__(self, state: ViewerState):
        self.state = state
        self._coord_text = None
        self._zoom_text = None
        self._render_time_text = None
        self._engine_text = None
        self._bookmark_group = None
        self._discover_group = None
        self._discover_status = None
        self._discover_btn = None
        self._discovering = False
        self._discover_results = []

    def setup(self, parent_tag):
        """Build the coordinate panel UI."""
        with dpg.group(parent=parent_tag):
            dpg.add_text("Location")
            self._coord_text = dpg.add_text("Re: -0.75\nIm: 0.0")
            self._zoom_text = dpg.add_text("Zoom: 1.0x")
            self._render_time_text = dpg.add_text("Render: --")
            self._engine_text = dpg.add_text("Engine: float64")

            dpg.add_separator()

            dpg.add_button(label="Copy as CLI command", callback=self._on_copy)
            dpg.add_button(label="Bookmark this location", callback=self._on_bookmark)

            dpg.add_separator()

            dpg.add_text("Go to coordinates")
            self._input_re = dpg.add_input_text(
                label="Re", default_value=str(self.state.center_re),
                width=200,
            )
            self._input_im = dpg.add_input_text(
                label="Im", default_value=str(self.state.center_im),
                width=200,
            )
            self._input_zoom = dpg.add_input_text(
                label="Zoom", default_value=str(self.state.zoom),
                width=200,
            )
            dpg.add_button(label="Go", callback=self._on_goto)

            dpg.add_separator()
            dpg.add_text("Bookmarks")
            self._bookmark_group = dpg.add_group()

            dpg.add_separator()
            dpg.add_text("Newton-Raphson Discovery")
            self._discover_btn = dpg.add_button(
                label="Discover boundary points",
                callback=self._on_discover,
            )
            self._discover_status = dpg.add_text("")
            self._discover_group = dpg.add_group()

    def _on_copy(self):
        """Copy current location as a CLI command to clipboard."""
        cmd = self.state.copy_location()
        try:
            import subprocess
            process = subprocess.Popen(
                ["clip"], stdin=subprocess.PIPE, shell=True
            )
            process.communicate(cmd.encode())
        except Exception:
            pass  # Clipboard not available
        print(f"Copied: {cmd}")

    def _on_bookmark(self):
        """Save current location as a bookmark."""
        self.state.add_bookmark()
        self._rebuild_bookmarks()

    def _on_goto(self):
        """Navigate to the entered coordinates (supports full-precision strings)."""
        re_str = dpg.get_value(self._input_re).strip()
        im_str = dpg.get_value(self._input_im).strip()
        zoom_str = dpg.get_value(self._input_zoom).strip()
        try:
            self.state.zoom = float(zoom_str)
            self.state.set_center(re_str, im_str)
            self.state.request_render()
        except (ValueError, TypeError) as e:
            print(f"Invalid coordinates: {e}")

    def _rebuild_bookmarks(self):
        """Rebuild the bookmark list UI."""
        if self._bookmark_group is None:
            return

        # Clear existing
        dpg.delete_item(self._bookmark_group, children_only=True)

        for i, bm in enumerate(self.state.bookmarks):
            zoom = bm["zoom"]
            label = f"#{i+1}: ({bm['center_re']:.6g}, {bm['center_im']:.6g}) z={zoom:.2g}"
            dpg.add_button(
                label=label,
                parent=self._bookmark_group,
                callback=lambda s, a, idx=i: self._goto_bookmark(idx),
            )

    def _goto_bookmark(self, idx: int):
        """Navigate to a bookmarked location."""
        self.state.load_bookmark(idx)

    def _rebuild_discover_results(self):
        """Rebuild discovery results as navigable buttons."""
        if self._discover_group is None:
            return
        dpg.delete_item(self._discover_group, children_only=True)

        from fractalforge.engine.newton import _format_angle

        for i, bp in enumerate(self._discover_results):
            if not bp.converged:
                continue
            angle_str = _format_angle(bp.internal_angle)
            zoom_str = f"{bp.suggested_zoom:.0e}"
            label = f"{angle_str} -> zoom {zoom_str}"
            dpg.add_button(
                label=label,
                parent=self._discover_group,
                callback=lambda s, a, idx=i: self._goto_discover(idx),
            )

    def _goto_discover(self, idx: int):
        """Navigate to a discovered boundary point."""
        if 0 <= idx < len(self._discover_results):
            bp = self._discover_results[idx]
            self.state.set_center(bp.c_re, bp.c_im)
            self.state.zoom = min(bp.suggested_zoom, 1e20)
            self.state.request_render()

    def _on_discover(self):
        """Run Newton-Raphson discovery on the current location."""
        if self._discovering:
            return

        self._discovering = True
        dpg.configure_item(self._discover_btn, enabled=False)
        dpg.set_value(self._discover_status, "Detecting period...")

        # Snapshot current position
        re_str = self.state.center_re_hp
        im_str = self.state.center_im_hp

        # Run in background thread so UI stays responsive
        thread = threading.Thread(
            target=self._discover_worker,
            args=(re_str, im_str),
            daemon=True,
        )
        thread.start()

    def _discover_worker(self, re_str: str, im_str: str):
        """Background worker for Newton-Raphson discovery."""
        try:
            from fractalforge.engine.newton import discover_coordinates

            results = discover_coordinates(
                c_re_str=re_str,
                c_im_str=im_str,
                precision=100,
                angles=[0.0, 0.5, 1/3, 1/4],
                max_period=10000,
                verbose=False,
            )

            self._discover_results = results
            if results:
                ok = sum(1 for r in results if r.converged)
                self._discover_done_msg = f"Found {ok} boundary points (period {results[0].period})"
            else:
                self._discover_done_msg = "No components found near this point"
        except Exception as e:
            self._discover_results = []
            self._discover_done_msg = f"Error: {e}"

        self._discovering = False

    def update(self):
        """Update the coordinate display (call every frame)."""
        # Check if discovery just finished
        if not self._discovering and self._discover_btn is not None:
            dpg.configure_item(self._discover_btn, enabled=True)
            if hasattr(self, '_discover_done_msg') and self._discover_done_msg:
                dpg.set_value(self._discover_status, self._discover_done_msg)
                self._discover_done_msg = None
                self._rebuild_discover_results()

        if self._coord_text is not None:
            if self.state.needs_perturbation:
                re_s = self.state.center_re_hp
                im_s = self.state.center_im_hp
                if len(re_s) > 40:
                    re_s = re_s[:40] + "..."
                if len(im_s) > 40:
                    im_s = im_s[:40] + "..."
                dpg.set_value(self._coord_text, f"Re: {re_s}\nIm: {im_s}")
            else:
                dpg.set_value(
                    self._coord_text,
                    f"Re: {self.state.center_re:.15g}\nIm: {self.state.center_im:.15g}",
                )
        if self._zoom_text is not None:
            zoom = self.state.zoom
            if zoom >= 1e6:
                dpg.set_value(self._zoom_text, f"Zoom: {zoom:.2e}")
            else:
                dpg.set_value(self._zoom_text, f"Zoom: {zoom:.1f}x")
        if self._render_time_text is not None:
            ms = self.state.last_render_ms
            if ms >= 1000:
                dpg.set_value(self._render_time_text, f"Render: {ms / 1000:.1f}s")
            else:
                dpg.set_value(self._render_time_text, f"Render: {ms:.0f}ms")
        if self._engine_text is not None:
            from fractalforge.viewer.render_bridge import _auto_max_iter
            engine = "perturbation" if self.state.needs_perturbation else "float64"
            eff_iter = _auto_max_iter(
                self.state.zoom, self.state.max_iter, self.state.auto_max_iter
            )
            extras = []
            if eff_iter > self.state.max_iter:
                extras.append(f"iter:{eff_iter}")
            if self.state.needs_perturbation and not self.state.histogram:
                extras.append("hist:auto")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            dpg.set_value(self._engine_text, f"Engine: {engine}{extra_str}")
