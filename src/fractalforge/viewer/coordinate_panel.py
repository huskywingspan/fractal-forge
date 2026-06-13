"""Coordinate panel -- position readout, navigation, bookmarks, discovery."""

import threading

import dearpygui.dearpygui as dpg
import mpmath

from fractalforge.viewer.state import ViewerState


def _fmt_zoom(log10_zoom: float) -> str:
    """Human-readable zoom: '1.0x', '3.4e07x', or '10^123'."""
    if log10_zoom < 6:
        return f"{10.0 ** log10_zoom:,.1f}x"
    if log10_zoom < 290:
        return f"{10.0 ** log10_zoom:.2e}x"
    return f"10^{log10_zoom:.1f}"


class CoordinatePanel:
    """Current location, go-to navigation, bookmarks, Newton/Misiurewicz find."""

    def __init__(self, state: ViewerState):
        self.state = state
        self._coord_text = None
        self._bookmark_group = None
        self._discover_group = None
        self._discover_status = None
        self._discover_btn = None
        self._misi_btn = None
        self._discovering = False
        self._discover_results = []
        self._async_msg = None

    def setup(self, parent_tag):
        with dpg.group(parent=parent_tag):
            dpg.add_text("LOCATION", color=(0, 212, 255))
            self._coord_text = dpg.add_text("Re: -0.75\nIm: 0.0")

            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Copy CLI", callback=self._on_copy, width=92)
                dpg.add_button(label="Bookmark", callback=self._on_bookmark, width=92)
                dpg.add_button(label="Reset", callback=lambda: self.state.reset_view(),
                               width=70)

            dpg.add_separator()
            dpg.add_text("GO TO", color=(0, 212, 255))
            self._input_re = dpg.add_input_text(
                label="Re", default_value=str(self.state.center_re), width=210)
            self._input_im = dpg.add_input_text(
                label="Im", default_value=str(self.state.center_im), width=210)
            self._input_zoom = dpg.add_input_text(
                label="Zoom", default_value="1.0", width=210,
                hint="e.g. 1e30, 1e200")
            dpg.add_button(label="Go", callback=self._on_goto, width=210)

            dpg.add_separator()
            dpg.add_text("BOOKMARKS", color=(0, 212, 255))
            self._bookmark_group = dpg.add_group()

            dpg.add_separator()
            dpg.add_text("DEEP TARGET FINDER", color=(168, 85, 247))
            with dpg.group(horizontal=True):
                self._discover_btn = dpg.add_button(
                    label="Boundary pts", callback=self._on_discover, width=110)
                self._misi_btn = dpg.add_button(
                    label="Misiurewicz", callback=self._on_misiurewicz, width=100)
            self._discover_status = dpg.add_text("")
            self._discover_group = dpg.add_group()

    # ---- actions ------------------------------------------------------------

    def _on_copy(self):
        cmd = self.state.copy_location()
        try:
            import subprocess
            p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
            p.communicate(cmd.encode())
        except Exception:
            pass
        print(f"Copied: {cmd}")

    def _on_bookmark(self):
        self.state.add_bookmark()
        self._rebuild_bookmarks()

    def _on_goto(self):
        re_str = dpg.get_value(self._input_re).strip()
        im_str = dpg.get_value(self._input_im).strip()
        zoom_str = dpg.get_value(self._input_zoom).strip()
        try:
            with mpmath.workdps(30):
                self.state.log10_zoom = float(mpmath.log10(mpmath.mpf(zoom_str)))
            self.state.set_center(re_str, im_str)
            self.state.request_render()
        except (ValueError, TypeError) as e:
            print(f"Invalid coordinates: {e}")

    def _rebuild_bookmarks(self):
        if self._bookmark_group is None:
            return
        dpg.delete_item(self._bookmark_group, children_only=True)
        for i, bm in enumerate(self.state.bookmarks):
            lz = bm.get("log10_zoom")
            if lz is None:
                import math
                lz = math.log10(max(bm.get("zoom", 1.0), 1e-9))
            label = f"#{i+1}: ({bm['center_re']:.5g}, {bm['center_im']:.5g}) {_fmt_zoom(lz)}"
            dpg.add_button(label=label, parent=self._bookmark_group,
                           callback=lambda s, a, idx=i: self.state.load_bookmark(idx))

    # ---- discovery (background threads) -------------------------------------

    def _on_discover(self):
        if self._discovering:
            return
        self._discovering = True
        dpg.configure_item(self._discover_btn, enabled=False)
        dpg.configure_item(self._misi_btn, enabled=False)
        dpg.set_value(self._discover_status, "Detecting period...")
        re_str, im_str = self.state.center_re_hp, self.state.center_im_hp
        threading.Thread(target=self._discover_worker, args=(re_str, im_str),
                         daemon=True).start()

    def _discover_worker(self, re_str, im_str):
        try:
            from fractalforge.engine.newton import discover_coordinates
            results = discover_coordinates(
                c_re_str=re_str, c_im_str=im_str, precision=100,
                angles=[0.0, 0.5, 1 / 3, 1 / 4], max_period=10000, verbose=False)
            self._discover_results = results
            if results:
                ok = sum(1 for r in results if r.converged)
                self._async_msg = f"Found {ok} boundary pts (period {results[0].period})"
            else:
                self._async_msg = "No components found here"
        except Exception as e:
            self._discover_results = []
            self._async_msg = f"Error: {e}"
        self._discovering = False

    def _on_misiurewicz(self):
        if self._discovering:
            return
        self._discovering = True
        dpg.configure_item(self._discover_btn, enabled=False)
        dpg.configure_item(self._misi_btn, enabled=False)
        dpg.set_value(self._discover_status, "Searching for Misiurewicz point...")
        re_str, im_str = self.state.center_re_hp, self.state.center_im_hp
        threading.Thread(target=self._misi_worker, args=(re_str, im_str),
                         daemon=True).start()

    def _misi_worker(self, re_str, im_str):
        try:
            from fractalforge.engine.newton import find_misiurewicz
            mp = find_misiurewicz(re_str, im_str, precision=160, verbose=False)
            if mp is not None:
                self._misi_target = mp
                self._async_msg = (f"Misiurewicz M({mp.preperiod},{mp.period}) "
                                   f"-- click to dive")
            else:
                self._misi_target = None
                self._async_msg = "No Misiurewicz point near here"
        except Exception as e:
            self._misi_target = None
            self._async_msg = f"Error: {e}"
        self._discovering = False

    def _rebuild_discover_results(self):
        if self._discover_group is None:
            return
        dpg.delete_item(self._discover_group, children_only=True)
        from fractalforge.engine.newton import _format_angle
        for i, bp in enumerate(self._discover_results):
            if not bp.converged:
                continue
            label = f"{_format_angle(bp.internal_angle)} -> {bp.suggested_zoom:.0e}"
            dpg.add_button(label=label, parent=self._discover_group,
                           callback=lambda s, a, idx=i: self._goto_discover(idx))

    def _goto_discover(self, idx):
        if 0 <= idx < len(self._discover_results):
            bp = self._discover_results[idx]
            self.state.set_center(bp.c_re, bp.c_im)
            import math
            self.state.log10_zoom = math.log10(max(bp.suggested_zoom, 10.0))
            self.state.request_render()

    def _show_misi_button(self):
        mp = getattr(self, "_misi_target", None)
        if mp is None:
            return
        dpg.delete_item(self._discover_group, children_only=True)
        dpg.add_button(
            label=f"Dive to M({mp.preperiod},{mp.period}) @ 1e60",
            parent=self._discover_group,
            callback=lambda: self._dive_misi(mp))

    def _dive_misi(self, mp):
        self.state.set_center(mp.c_re, mp.c_im)
        self.state.log10_zoom = 60.0
        self.state.request_render()

    # ---- per-frame update ---------------------------------------------------

    def update(self):
        if not self._discovering and self._discover_btn is not None:
            dpg.configure_item(self._discover_btn, enabled=True)
            dpg.configure_item(self._misi_btn, enabled=True)
            if self._async_msg:
                dpg.set_value(self._discover_status, self._async_msg)
                self._async_msg = None
                if getattr(self, "_misi_target", None) is not None:
                    self._show_misi_button()
                    self._misi_target = None
                elif self._discover_results:
                    self._rebuild_discover_results()

        if self._coord_text is not None:
            if self.state.needs_perturbation:
                re_s, im_s = self.state.center_re_hp, self.state.center_im_hp
                re_s = re_s[:42] + "..." if len(re_s) > 42 else re_s
                im_s = im_s[:42] + "..." if len(im_s) > 42 else im_s
                dpg.set_value(self._coord_text, f"Re: {re_s}\nIm: {im_s}")
            else:
                dpg.set_value(
                    self._coord_text,
                    f"Re: {self.state.center_re:.15g}\nIm: {self.state.center_im:.15g}")
