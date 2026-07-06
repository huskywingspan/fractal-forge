"""Ground-truth regression: the fxp deep kernel must match plain float64.

At shallow zoom both engines are valid, but the fxp kernel exercises its full
deep machinery (floatexp arithmetic, BLA, and — critically — rebasing, which
fires even at shallow zoom because it is always-on in this kernel).

Guards against the rebase-count inflation bug: rebases were counted as
iterations, shifting patches of pixels by integer amounts (blocky seams at
depth). With correct accounting the smooth fields agree almost everywhere;
only chaotic filament-edge pixels may differ.
"""

import numpy as np
import pytest
from numba import cuda

from fractalforge.engine.mandelbrot import render_frame as render_std
from fractalforge.engine.perturbation import _render_deep_fxp

# A boundary-adjacent point with mixed interior/exterior structure
CRE = "-0.7746724469356738080461171765322245435665009634996757"
CIM = "0.13742929234091689059154346409786952900730470590693816"


@pytest.mark.skipif(not cuda.is_available(), reason="requires CUDA GPU")
@pytest.mark.parametrize("zoom", [2e3, 5e4])
def test_fxp_matches_float64_ground_truth(zoom):
    W, H = 160, 120
    std = render_std(float(CRE), float(CIM), zoom, W, H,
                     max_iter=2500, use_gpu=True)
    fxp = _render_deep_fxp(CRE, CIM, str(zoom), W, H, 2500, gpu=True)

    both = np.isfinite(std) & np.isfinite(fxp)
    assert both.all()

    # Interior/exterior classification agreement
    agree = (((std < 0) == (fxp < 0)) & both).mean()
    assert agree > 0.995, f"classification agreement {agree:.4f}"

    # Smooth escape values: median must be ~0 (the inflation bug showed
    # an exact integer median of 3.0); allow chaotic outliers at p90.
    eb = (std >= 0) & (fxp >= 0)
    diff = np.abs(std[eb] - fxp[eb])
    assert np.median(diff) < 0.05, f"median diff {np.median(diff):.4f}"
    assert np.percentile(diff, 90) < 0.5, f"p90 diff {np.percentile(diff, 90):.4f}"
