"""Unit tests for extended-range (floatexp) complex arithmetic.

Validates the CPU-jitted primitives against mpmath at exponents far beyond
float64 range. The CUDA device versions share the same source bodies, so CPU
correctness plus the GPU/CPU cross-render test in test_deep_kernel.py covers
both compilations.
"""

import math

import mpmath
import pytest

from fractalforge.engine.floatexp import (
    EXP_ZERO,
    LOG2_ZERO,
    fx_add,
    fx_mag_lt,
    fx_mag_sq_log2,
    fx_mul,
    fx_norm,
)

mpmath.mp.dps = 60


def to_mp(m_re, m_im, e):
    """Convert a floatexp triple to an mpmath complex."""
    return mpmath.mpc(mpmath.ldexp(mpmath.mpf(m_re), int(e)),
                      mpmath.ldexp(mpmath.mpf(m_im), int(e)))


def from_mp(z):
    """Convert an mpmath complex to a normalized floatexp triple."""
    mag = max(abs(z.real), abs(z.imag))
    if mag == 0:
        return 0.0, 0.0, EXP_ZERO
    _, k = mpmath.frexp(mag)
    k = int(k)
    return (float(mpmath.ldexp(z.real, -k)),
            float(mpmath.ldexp(z.imag, -k)), k)


def assert_close(triple, expected_mp, rel=1e-13):
    got = to_mp(*triple)
    err = abs(got - expected_mp)
    scale = abs(expected_mp)
    if scale == 0:
        assert err == 0, f"expected exact zero, got {got}"
    else:
        assert err / scale < rel, f"rel err {err / scale} for {got} vs {expected_mp}"


CASES = [
    (mpmath.mpc("3.25", "-1.5"), mpmath.mpc("-0.75", "2.0")),
    (mpmath.mpc("1e-400", "2e-401"), mpmath.mpc("3e-400", "-1e-399")),
    (mpmath.mpc("1e-2500", "0"), mpmath.mpc("-7e-2501", "4e-2500")),
    (mpmath.mpc("5e+600", "-2e+601"), mpmath.mpc("1e+600", "3e+599")),
    (mpmath.mpc("1e-5000", "1e-5000"), mpmath.mpc("2", "-3")),
]


@pytest.mark.parametrize("za,zb", CASES)
def test_mul_matches_mpmath(za, zb):
    a = from_mp(za)
    b = from_mp(zb)
    assert_close(fx_mul(*a, *b), za * zb)


@pytest.mark.parametrize("za,zb", CASES)
def test_add_matches_mpmath(za, zb):
    a = from_mp(za)
    b = from_mp(zb)
    got = to_mp(*fx_add(*a, *b))
    expected = za + zb
    # When magnitudes differ by >108 bits the small term is dropped — the
    # result then equals the dominant term to float64 precision.
    err = abs(got - expected)
    assert err / abs(expected) < 1e-13


def test_add_drops_negligible_term():
    a = from_mp(mpmath.mpc("1e-100", "0"))
    b = from_mp(mpmath.mpc("1e-400", "0"))
    got = to_mp(*fx_add(*a, *b))
    assert_close(from_mp(got) and fx_add(*a, *b), mpmath.mpc("1e-100", "0"))


def test_norm_zero():
    m_re, m_im, e = fx_norm(0.0, 0.0, 1234)
    assert (m_re, m_im, e) == (0.0, 0.0, EXP_ZERO)


def test_norm_invariant():
    m_re, m_im, e = fx_norm(123456.789, -0.001, 7)
    assert 0.5 <= max(abs(m_re), abs(m_im)) < 1.0
    assert_close((m_re, m_im, e), mpmath.mpc(123456.789, -0.001) * mpmath.mpf(2) ** 7)


def test_mag_sq_log2():
    z = mpmath.mpc("3e-1000", "-4e-1000")
    got = fx_mag_sq_log2(*from_mp(z))
    expected = float(2 * mpmath.log(abs(z), 2))
    assert abs(got - expected) < 1e-9
    assert fx_mag_sq_log2(0.0, 0.0, EXP_ZERO) == LOG2_ZERO


def test_mag_lt():
    a = from_mp(mpmath.mpc("1e-500", "0"))
    b = from_mp(mpmath.mpc("2e-500", "0"))
    c = from_mp(mpmath.mpc("1e-200", "1e-200"))
    zero = (0.0, 0.0, EXP_ZERO)
    assert fx_mag_lt(*a, *b)
    assert not fx_mag_lt(*b, *a)
    assert fx_mag_lt(*a, *c)
    assert not fx_mag_lt(*c, *a)
    assert fx_mag_lt(*zero, *a)
    assert not fx_mag_lt(*a, *zero)
    # Equal magnitudes: strictly-less is False
    assert not fx_mag_lt(*a, *a)


def test_delta_iteration_step_extreme_depth():
    """One perturbation step d' = 2*Z*d + d^2 + dc at 1e-700 scale."""
    Z = mpmath.mpc("-0.7436438870371587", "0.1318259043091895")
    d = mpmath.mpc("3e-700", "-1e-700")
    dc = mpmath.mpc("1e-702", "2e-702")

    fZ = from_mp(Z)
    fd = from_mp(d)
    fdc = from_mp(dc)

    # 2*Z*d via mul then exponent bump
    t_re, t_im, t_e = fx_mul(*fZ, *fd)
    t = (t_re, t_im, t_e + 1)
    d_sq = fx_mul(*fd, *fd)
    s = fx_add(*t, *d_sq)
    result = fx_add(*s, *fdc)

    expected = 2 * Z * d + d * d + dc
    assert_close(result, expected)
