"""Tests for unbounded-depth zoom video (DZ-P3).

Verifies that the zoom path interpolation, dataclass, and save/load round-trip
all handle zoom levels beyond float64's 1e308 ceiling (carried as strings).
"""

import math

import mpmath
import pytest

from fractalforge.artist.zoompath import Keyframe, ZoomPath, _zoom_log10


def test_keyframe_accepts_string_zoom():
    kf = Keyframe(frame=0, zoom="1e500")
    assert kf.zoom_log10 == pytest.approx(500.0)


def test_keyframe_float_zoom_unchanged():
    kf = Keyframe(frame=0, zoom=1e6)
    assert kf.zoom_log10 == pytest.approx(6.0)


def test_zoom_log10_helper():
    assert _zoom_log10(1.0) == 0.0
    assert _zoom_log10(1e13) == pytest.approx(13.0)
    assert _zoom_log10("1e404") == pytest.approx(404.0)
    assert _zoom_log10("3.16e200") == pytest.approx(200.5, abs=1e-2)
    assert _zoom_log10(0.0) == 0.0  # guard


def _deep_path():
    """A 2-keyframe dive from zoom 1 to 1e400 (past float64 range)."""
    hp_re = "-0.7746724469356738080461171765322245435665"
    hp_im = "0.1374292923409168905915434640978695290073"
    return ZoomPath(
        name="deep_test", fps=60, width=320, height=180,
        keyframes=[
            Keyframe(frame=0, center_re=-0.5, center_im=0.0, zoom=1.0,
                     max_iter=500),
            Keyframe(frame=300, center_re=float(hp_re), center_im=float(hp_im),
                     zoom="1e400", max_iter=8000,
                     center_re_hp=hp_re, center_im_hp=hp_im),
        ],
    )


def test_interpolation_log10_monotonic_past_float_ceiling():
    path = _deep_path()
    prev = -1.0
    for frame in (0, 60, 150, 240, 299, 300):
        params = path.interpolate(frame)
        lz = params["log10_zoom"]
        assert math.isfinite(lz)
        assert lz >= prev - 1e-9, f"log10_zoom not monotonic at frame {frame}"
        prev = lz
    # Endpoints
    assert path.interpolate(0)["log10_zoom"] == pytest.approx(0.0)
    assert path.interpolate(300)["log10_zoom"] == pytest.approx(400.0)


def test_midframe_zoom_is_string_when_deep():
    path = _deep_path()
    # A late frame should exceed float64 range -> zoom carried as a string.
    params = path.interpolate(295)
    assert params["log10_zoom"] > 300.0
    assert isinstance(params["zoom"], str)
    # And it must be mpmath-parseable back to the right magnitude
    with mpmath.workdps(30):
        assert float(mpmath.log10(mpmath.mpf(params["zoom"]))) == pytest.approx(
            params["log10_zoom"], abs=1e-3)


def test_shallow_frame_zoom_is_float():
    path = _deep_path()
    params = path.interpolate(30)
    assert isinstance(params["zoom"], float)
    assert math.isfinite(params["zoom"])


def test_deep_coords_stay_locked_on_target():
    """Zoom-weighted interpolation must keep hp coords converging to target."""
    path = _deep_path()
    tgt_re = path.keyframes[-1].center_re_hp
    p_late = path.interpolate(298)
    # Deep frame provides hp coords that match the target to many digits.
    assert "center_re_hp" in p_late
    with mpmath.workdps(60):
        err = abs(mpmath.mpf(p_late["center_re_hp"]) - mpmath.mpf(tgt_re))
        assert err < mpmath.mpf("1e-50")


def test_save_load_roundtrip_string_zoom(tmp_path):
    path = _deep_path()
    f = tmp_path / "deep.json"
    path.save(f)
    loaded = ZoomPath.load(f)
    assert loaded.keyframes[-1].zoom == "1e400"
    assert loaded.keyframes[-1].zoom_log10 == pytest.approx(400.0)
    # Interpolation still works after round-trip
    assert loaded.interpolate(300)["log10_zoom"] == pytest.approx(400.0)
