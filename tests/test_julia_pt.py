"""Julia perturbation engine tests.

The standard Julia kernel computes z_0 = center + offset in absolute float64
coordinates and dissolves into noise once pixel spacing nears the ulp
(~1e12 at preview sizes). The perturbation path must (a) agree with the
standard kernel at shallow zoom and (b) produce a smooth field at depths
where the standard kernel is pure static.
"""

import random

import mpmath
import numpy as np
import pytest
from numba import cuda

from fractalforge.engine.julia import render_frame_julia, _render_julia_pt

CJ_RE, CJ_IM = -0.7269, 0.1889


def _julia_set_point(digits=110, steps=300, seed=11):
    """A point exactly on J(c) via inverse iteration at high precision."""
    mpmath.mp.dps = digits + 10
    rng = random.Random(seed)
    c = mpmath.mpc(CJ_RE, CJ_IM)
    z = (1 + mpmath.sqrt(1 - 4 * c)) / 2  # repelling fixed point
    for _ in range(steps):
        z = mpmath.sqrt(z - c)
        if rng.random() < 0.5:
            z = -z
    return (mpmath.nstr(z.real, digits, strip_zeros=False),
            mpmath.nstr(z.imag, digits, strip_zeros=False))


@pytest.mark.skipif(not cuda.is_available(), reason="requires CUDA GPU")
def test_julia_pt_matches_standard_shallow():
    cre, cim = _julia_set_point()
    std = render_frame_julia(CJ_RE, CJ_IM, float(cre), float(cim), 1e6,
                             160, 120, max_iter=2000, use_gpu=True)
    pt = _render_julia_pt(CJ_RE, CJ_IM, cre, cim, 1e6,
                          160, 120, 2000, gpu=True)
    # Identical classification and median-zero smooth diff; chaotic filament
    # pixels (we are ON the Julia set) may differ in the tail.
    assert ((std < 0) == (pt < 0)).mean() > 0.995
    both = (std >= 0) & (pt >= 0)
    assert np.median(np.abs(std[both] - pt[both])) < 0.05


@pytest.mark.skipif(not cuda.is_available(), reason="requires CUDA GPU")
def test_julia_deep_zoom_is_structure_not_noise():
    cre, cim = _julia_set_point()
    img = render_frame_julia(CJ_RE, CJ_IM, cre, cim, 1e16,
                             160, 120, max_iter=6000, use_gpu=True)
    ext = img[img >= 0]
    assert ext.size > 1000
    # Noise fingerprint: huge gradients EVERYWHERE (median >> 1). Structure:
    # smooth field (tiny median gradient) with large jumps only at filaments.
    gx = np.abs(np.diff(img, axis=1))
    okx = (img[:, 1:] >= 0) & (img[:, :-1] >= 0)
    assert np.median(gx[okx]) < 0.5
    # Escape range should be wide (real depth variation, not a flat frame)
    assert ext.max() - ext.min() > 100
