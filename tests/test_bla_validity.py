"""BLA validity radii must collapse at near-zero orbit passages.

The single-step criterion is |d| <= eps*|Z_n|: the dropped d^2 term stays
negligible vs the kept 2*Z_n*d only while the delta is small RELATIVE TO Z_n.
An earlier formula divided by |A| = 2|Z_n|, making the radius explode exactly
at near-zero passages — pixels jumped through critical-point approaches and
landed with O(1) relative error (rendered as circular coherent-but-wrong
regions centered on the reference).

Uses a deterministic synthetic orbit with one near-zero dip and asserts no
stored table entry spanning the dip permits deltas anywhere near the broken
regime (the bug allowed 2^-33; correct is ~2^-83 for this orbit).
"""

import math

import numpy as np

from fractalforge.engine.bla import compute_bla_table, compute_bla_table_fxp
from fractalforge.engine.floatexp import EXP_ZERO

N = 64
NEAR_ZERO_IDX = 33
NEAR_ZERO_MAG = 1e-9
EPS = 1e-16


def _synthetic_orbit():
    rng = np.random.default_rng(7)
    mag = rng.uniform(0.3, 1.4, N + 1)
    ang = rng.uniform(0, 2 * math.pi, N + 1)
    z = mag * np.exp(1j * ang)
    z[0] = 0.0
    z[NEAR_ZERO_IDX] = NEAR_ZERO_MAG * np.exp(1j * 0.7)
    return z


def _spanning_max_rlog2(level_bits, level_offsets, level_sizes, r_log2):
    worst = -math.inf
    for li in range(len(level_bits)):
        skip = 1 << int(level_bits[li])
        off = int(level_offsets[li])
        size = int(level_sizes[li])
        for i in range(size):
            if i <= NEAR_ZERO_IDX < i + skip:
                worst = max(worst, r_log2[off + i])
    return worst


def test_fxp_table_radii_collapse_at_near_zero_passage():
    z = _synthetic_orbit()
    z_m_re = np.zeros(N + 1)
    z_m_im = np.zeros(N + 1)
    z_exp = np.full(N + 1, EXP_ZERO, dtype=np.int64)
    for i in range(N + 1):
        m = max(abs(z[i].real), abs(z[i].imag))
        if m == 0:
            continue
        _, k = np.frexp(m)
        z_m_re[i] = np.ldexp(z[i].real, -int(k))
        z_m_im[i] = np.ldexp(z[i].imag, -int(k))
        z_exp[i] = int(k)

    tbl = compute_bla_table_fxp(z_m_re, z_m_im, z_exp, N,
                                dc_max_m=0.5, dc_max_e=-233, eps=EPS)
    assert tbl.num_levels > 0

    worst = _spanning_max_rlog2(tbl.level_bits, tbl.level_offsets,
                                tbl.level_sizes, tbl.r_log2)
    ceiling_log2 = math.log2(EPS * NEAR_ZERO_MAG)  # ~ -83
    # Generous 8-bit slack; the inverted formula sat ~50 bits above.
    assert worst <= ceiling_log2 + 8.0, (
        f"table permits jumps through the near-zero passage: "
        f"max r_log2 {worst:.1f} vs ceiling {ceiling_log2:.1f}")


def test_float64_level0_radius_proportional_to_z():
    z = _synthetic_orbit()
    tbl = compute_bla_table(z.real.copy(), z.imag.copy(), N,
                            dc_max=1e-20, eps=EPS)
    assert tbl.num_levels > 0
    # Level 0 radius must scale with |Z_n| — tiny at the dip, zero at Z_0.
    r0 = tbl.validity_r[: int(tbl.level_sizes[0])]
    assert r0[0] == 0.0
    assert r0[NEAR_ZERO_IDX] <= EPS * NEAR_ZERO_MAG * 1.001
    # And a normal-magnitude step allows a normal radius.
    normal = np.abs(z[1])
    assert abs(r0[1] - EPS * normal) / (EPS * normal) < 1e-9
