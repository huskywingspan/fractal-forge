"""Extended-range (floatexp) complex arithmetic for ultra-deep zoom.

Standard float64 underflows at ~1e-308, which caps perturbation deltas — and
therefore zoom depth — at roughly 1e290. This module represents a complex
value as a triple (m_re, m_im, exp): two float64 mantissas sharing a single
power-of-two exponent, normalized so max(|m_re|, |m_im|) is in [0.5, 1).
That keeps float64 mantissa precision (~15.9 digits) while extending the
exponent range to int64 — effectively unbounded for any practical zoom.

The same primitive bodies are compiled twice via a factory:
  - CPU:  fx_*   (@njit, used by the CPU deep kernel and the table builder)
  - CUDA: dfx_*  (@cuda.jit(device=True), used by the GPU deep kernel)

Magnitude comparisons use the log2 domain (2*exp + log2(m_re^2 + m_im^2))
so BLA validity radii far below 1e-308 stay representable.
"""

import math

from numba import cuda, njit

# Exponent sentinel for the value zero. Any real delta exponent is far above
# this, so zero always compares as smaller in magnitude.
EXP_ZERO = -(1 << 60)

# log2-magnitude sentinel for zero (mag_sq_log2 of zero returns this).
LOG2_ZERO = -1.0e30


def _build(jit):
    """Compile the floatexp primitives with the given decorator.

    Returns (norm, mul, add, mag_sq_log2, mag_lt) jitted functions.
    Functions take/return scalar triples (m_re, m_im, exp:int64).
    """

    @jit
    def fx_norm(m_re, m_im, e):
        """Normalize so max(|m_re|, |m_im|) is in [0.5, 1)."""
        mag = max(abs(m_re), abs(m_im))
        if mag == 0.0:
            return 0.0, 0.0, EXP_ZERO
        _, k = math.frexp(mag)
        return math.ldexp(m_re, -k), math.ldexp(m_im, -k), e + k

    @jit
    def fx_mul(a_re, a_im, a_e, b_re, b_im, b_e):
        """Complex multiply. Mantissa products stay in float64 range."""
        p_re = a_re * b_re - a_im * b_im
        p_im = a_re * b_im + a_im * b_re
        return fx_norm(p_re, p_im, a_e + b_e)

    @jit
    def fx_add(a_re, a_im, a_e, b_re, b_im, b_e):
        """Complex add with exponent alignment.

        When the exponents differ by more than ~108 bits the smaller term
        is below one ulp of the larger and is dropped — same behavior as
        float64 addition, just at an arbitrary exponent offset.
        """
        if a_re == 0.0 and a_im == 0.0:
            return b_re, b_im, b_e
        if b_re == 0.0 and b_im == 0.0:
            return a_re, a_im, a_e
        diff = b_e - a_e
        if diff > 108:
            return b_re, b_im, b_e
        if diff < -108:
            return a_re, a_im, a_e
        if diff >= 0:
            # b dominates alignment: shift a down to b's exponent
            s_re = math.ldexp(a_re, -diff)
            s_im = math.ldexp(a_im, -diff)
            return fx_norm(s_re + b_re, s_im + b_im, b_e)
        s_re = math.ldexp(b_re, diff)
        s_im = math.ldexp(b_im, diff)
        return fx_norm(a_re + s_re, a_im + s_im, a_e)

    @jit
    def fx_mag_sq_log2(m_re, m_im, e):
        """log2(|x|^2) — safe for magnitudes far outside float64 range."""
        msq = m_re * m_re + m_im * m_im
        if msq == 0.0:
            return LOG2_ZERO
        return 2.0 * e + math.log2(msq)

    @jit
    def fx_mag_lt(a_re, a_im, a_e, b_re, b_im, b_e):
        """True when |a| < |b|. Avoids log2 except in the ambiguous band."""
        a_zero = a_re == 0.0 and a_im == 0.0
        b_zero = b_re == 0.0 and b_im == 0.0
        if a_zero:
            return not b_zero
        if b_zero:
            return False
        diff = a_e - b_e
        # Mantissa mag_sq is in [0.25, 2): ratio < 8 = 2^3, so a 2-bit
        # exponent gap (4 bits in the squared domain) is decisive.
        if diff >= 2:
            return False
        if diff <= -2:
            return True
        msq_a = a_re * a_re + a_im * a_im
        msq_b = b_re * b_re + b_im * b_im
        return msq_a < math.ldexp(msq_b, 2 * (b_e - a_e))

    return fx_norm, fx_mul, fx_add, fx_mag_sq_log2, fx_mag_lt


# CPU versions (used by CPU kernels; also exercised by unit tests)
fx_norm, fx_mul, fx_add, fx_mag_sq_log2, fx_mag_lt = _build(njit)

# CUDA device versions
if cuda.is_available():
    dfx_norm, dfx_mul, dfx_add, dfx_mag_sq_log2, dfx_mag_lt = _build(
        cuda.jit(device=True)
    )
else:  # pragma: no cover - allows import on machines without CUDA
    dfx_norm = dfx_mul = dfx_add = dfx_mag_sq_log2 = dfx_mag_lt = None
