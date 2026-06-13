"""Newton-Raphson coordinate finder for the Mandelbrot set.

Finds precise boundary coordinates at arbitrary zoom depth by:
1. Detecting the period of the nearest hyperbolic component
2. Newton's method to refine the nucleus (center) to arbitrary precision
3. Finding boundary points at specified internal angles
4. Estimating component size for zoom level suggestions

These high-precision coordinates are ideal targets for deep zoom videos,
as they sit exactly on the Mandelbrot set boundary where the most intricate
detail exists at every scale.

Algorithms based on:
- Wolf Jung's "Mandel" software and publications
- Claude Heiland-Allen's mandelbrot-numerics library
- Robert Munafo's mu-unit size calculations
"""

import math
import time
from dataclasses import dataclass

import mpmath
import numpy as np

try:
    import gmpy2
    _HAS_GMPY2 = True
except ImportError:
    _HAS_GMPY2 = False


@dataclass
class MandelbrotNucleus:
    """A nucleus (center) of a hyperbolic component of the Mandelbrot set.

    Attributes:
        c_re: Real part of nucleus as high-precision string.
        c_im: Imaginary part of nucleus as high-precision string.
        period: Period of the component.
        size: Approximate radius of the component in the c-plane.
        precision: Decimal digits used for computation.
        converged: Whether Newton's method converged.
        newton_steps: Number of Newton iterations used.
    """
    c_re: str
    c_im: str
    period: int
    size: float
    precision: int
    converged: bool
    newton_steps: int


@dataclass
class BoundaryPoint:
    """A point on the boundary of a hyperbolic component.

    Attributes:
        c_re: Real part as high-precision string.
        c_im: Imaginary part as high-precision string.
        period: Period of the parent component.
        internal_angle: Internal angle in turns (0 to 1).
        suggested_zoom: Suggested zoom level for interesting detail.
        precision: Decimal digits used.
        converged: Whether Newton's method converged.
    """
    c_re: str
    c_im: str
    period: int
    internal_angle: float
    suggested_zoom: float
    precision: int
    converged: bool


@dataclass
class MisiurewiczPoint:
    """A Misiurewicz (pre-periodic) point of the Mandelbrot set.

    The critical orbit settles onto a repelling cycle after a short
    transient: f^{preperiod+period}(0) = f^{preperiod}(0). These points sit
    at the centers of spirals and dendrites, and are the ideal targets for
    *extreme* deep zoom: unlike minibrot nuclei (whose period -- and hence
    reference-orbit length -- grows with depth), a Misiurewicz reference
    orbit stays short while the embedded Julia structure repeats at every
    scale. Zoom depth is then limited only by coordinate precision.

    Attributes:
        c_re, c_im: High-precision coordinate strings.
        preperiod: Transient length before the orbit enters its cycle.
        period: Period of the repelling cycle.
        precision: Decimal digits used.
        converged: Whether Newton converged to the requested precision.
    """
    c_re: str
    c_im: str
    preperiod: int
    period: int
    precision: int
    converged: bool


def detect_period(c_re_str: str, c_im_str: str, max_period: int = 10000,
                  precision: int = 50) -> int:
    """Detect the period of the nearest hyperbolic component.

    Iterates z = z^2 + c from z=0 and finds the smallest p where |z_p|
    achieves a local minimum close to zero, indicating proximity to a
    period-p nucleus.

    Args:
        c_re_str: Real part of approximate location.
        c_im_str: Imaginary part of approximate location.
        max_period: Maximum period to search for.
        precision: Decimal digits for computation.

    Returns:
        Detected period, or -1 if no period found within max_period.
    """
    if _HAS_GMPY2:
        return _detect_period_gmpy2(c_re_str, c_im_str, max_period, precision)
    return _detect_period_mpmath(c_re_str, c_im_str, max_period, precision)


def _detect_period_gmpy2(c_re_str, c_im_str, max_period, precision):
    bits = int(precision * 3.3219) + 20
    gmpy2.get_context().precision = bits

    c_re = gmpy2.mpfr(c_re_str)
    c_im = gmpy2.mpfr(c_im_str)

    z_re = gmpy2.mpfr(0)
    z_im = gmpy2.mpfr(0)

    best_mag = float('inf')
    best_period = -1

    for n in range(1, max_period + 1):
        # z = z^2 + c
        new_re = z_re * z_re - z_im * z_im + c_re
        new_im = gmpy2.mpfr(2) * z_re * z_im + c_im
        z_re, z_im = new_re, new_im

        mag = float(z_re * z_re + z_im * z_im)

        # Escaped -- if we found a good minimum before escape, return it.
        # Points near the boundary escape eventually but pass close to
        # the attracting cycle first.
        if mag > 1e6:
            if best_mag < 1.0:
                return best_period
            return -1

        if mag < best_mag:
            best_mag = mag
            best_period = n

    return best_period


def _detect_period_mpmath(c_re_str, c_im_str, max_period, precision):
    mpmath.mp.dps = precision

    c = mpmath.mpc(c_re_str, c_im_str)
    z = mpmath.mpc(0, 0)

    best_mag = float('inf')
    best_period = -1

    for n in range(1, max_period + 1):
        z = z * z + c
        mag = float(abs(z) ** 2)

        if mag > 1e6:
            if best_mag < 1.0:
                return best_period
            return -1

        if mag < best_mag:
            best_mag = mag
            best_period = n

    return best_period


def _true_period(c_re_str: str, c_im_str: str, detected_period: int,
                 precision: int = 50) -> int:
    """Verify and correct detected period by checking all divisors.

    After Newton's method converges to a nucleus for period p, we need
    to verify that p is the TRUE (minimal) period, not a multiple.
    For example, a period-39 nucleus also satisfies f^78(0,c) = 0.

    Returns the smallest divisor d of detected_period such that
    |f^d(0, c)| is negligibly small.
    """
    if _HAS_GMPY2:
        return _true_period_gmpy2(c_re_str, c_im_str, detected_period, precision)
    return _true_period_mpmath(c_re_str, c_im_str, detected_period, precision)


def _true_period_gmpy2(c_re_str, c_im_str, detected_period, precision):
    bits = int(precision * 3.3219) + 20
    gmpy2.get_context().precision = bits

    c_re = gmpy2.mpfr(c_re_str)
    c_im = gmpy2.mpfr(c_im_str)
    two = gmpy2.mpfr(2)

    # Iterate and record |z_n| at all divisors of detected_period
    z_re = gmpy2.mpfr(0)
    z_im = gmpy2.mpfr(0)

    # Find all divisors
    divisors = sorted(d for d in range(1, detected_period + 1)
                      if detected_period % d == 0)

    threshold = gmpy2.mpfr(10) ** (-(precision // 3))

    for n in range(1, detected_period + 1):
        new_re = z_re * z_re - z_im * z_im + c_re
        new_im = two * z_re * z_im + c_im
        z_re, z_im = new_re, new_im

        if n in divisors:
            mag = gmpy2.sqrt(z_re * z_re + z_im * z_im)
            if mag < threshold:
                return n

    return detected_period


def _true_period_mpmath(c_re_str, c_im_str, detected_period, precision):
    mpmath.mp.dps = precision

    c = mpmath.mpc(c_re_str, c_im_str)
    z = mpmath.mpc(0, 0)

    divisors = sorted(d for d in range(1, detected_period + 1)
                      if detected_period % d == 0)

    threshold = mpmath.mpf(10) ** (-(precision // 3))

    for n in range(1, detected_period + 1):
        z = z * z + c
        if n in divisors:
            if abs(z) < threshold:
                return n

    return detected_period


def find_nucleus(c_re_str: str, c_im_str: str, period: int,
                 precision: int = 50, max_steps: int = 100,
                 tolerance: float = 0.0) -> MandelbrotNucleus:
    """Find the exact nucleus of a period-p hyperbolic component.

    Uses Newton's method to solve f^p(0, c) = 0, where f(z,c) = z^2 + c.
    The Newton step is: c_{n+1} = c_n - z_p / (dz_p/dc)

    The derivative dz/dc is computed alongside the orbit:
        dz_0/dc = 0
        dz_{n+1}/dc = 2 * z_n * dz_n/dc + 1

    Args:
        c_re_str: Real part of approximate nucleus location.
        c_im_str: Imaginary part of approximate nucleus location.
        period: Period of the component.
        precision: Decimal digits for computation.
        max_steps: Maximum Newton iterations.
        tolerance: Convergence threshold. If 0, uses 10^(-precision/2).

    Returns:
        MandelbrotNucleus with the refined nucleus location.
    """
    if _HAS_GMPY2:
        nuc = _find_nucleus_gmpy2(c_re_str, c_im_str, period, precision,
                                  max_steps, tolerance)
    else:
        nuc = _find_nucleus_mpmath(c_re_str, c_im_str, period, precision,
                                   max_steps, tolerance)

    # Verify true period — Newton may converge to a sub-period nucleus
    if nuc.converged:
        true_p = _true_period(nuc.c_re, nuc.c_im, period, precision)
        if true_p < period:
            # Re-run Newton at the correct period for proper nucleus
            if _HAS_GMPY2:
                nuc = _find_nucleus_gmpy2(nuc.c_re, nuc.c_im, true_p,
                                          precision, max_steps, tolerance)
            else:
                nuc = _find_nucleus_mpmath(nuc.c_re, nuc.c_im, true_p,
                                           precision, max_steps, tolerance)
    return nuc


def _find_nucleus_gmpy2(c_re_str, c_im_str, period, precision, max_steps,
                        tolerance):
    bits = int(precision * 3.3219) + 20
    gmpy2.get_context().precision = bits

    c_re = gmpy2.mpfr(c_re_str)
    c_im = gmpy2.mpfr(c_im_str)
    two = gmpy2.mpfr(2)

    if tolerance == 0.0:
        tol = gmpy2.mpfr(10) ** (-(precision // 2))
    else:
        tol = gmpy2.mpfr(tolerance)

    converged = False
    step = 0

    for step in range(1, max_steps + 1):
        # Iterate z = z^2 + c for `period` steps, tracking dz/dc
        z_re = gmpy2.mpfr(0)
        z_im = gmpy2.mpfr(0)
        dc_re = gmpy2.mpfr(0)  # dz/dc real
        dc_im = gmpy2.mpfr(0)  # dz/dc imag

        for _ in range(period):
            # dz/dc = 2*z*dz/dc + 1
            new_dc_re = two * (z_re * dc_re - z_im * dc_im) + gmpy2.mpfr(1)
            new_dc_im = two * (z_re * dc_im + z_im * dc_re)
            # z = z^2 + c
            new_z_re = z_re * z_re - z_im * z_im + c_re
            new_z_im = two * z_re * z_im + c_im

            z_re, z_im = new_z_re, new_z_im
            dc_re, dc_im = new_dc_re, new_dc_im

        # Newton step: c -= z / (dz/dc)
        # Division: (z_re + i*z_im) / (dc_re + i*dc_im)
        denom = dc_re * dc_re + dc_im * dc_im
        if float(denom) == 0.0:
            break

        delta_re = (z_re * dc_re + z_im * dc_im) / denom
        delta_im = (z_im * dc_re - z_re * dc_im) / denom

        c_re -= delta_re
        c_im -= delta_im

        delta_mag = gmpy2.sqrt(delta_re * delta_re + delta_im * delta_im)
        if delta_mag < tol:
            converged = True
            break

    # Compute component size estimate
    size = _estimate_size_gmpy2(c_re, c_im, period, bits)

    # Convert to string
    mpmath.mp.dps = precision
    c_re_mp = mpmath.mpf(gmpy2.digits(c_re, 10)[0] + "e" + str(gmpy2.digits(c_re, 10)[1] - len(gmpy2.digits(c_re, 10)[0])) if False else str(c_re))
    c_im_mp = mpmath.mpf(str(c_im))

    return MandelbrotNucleus(
        c_re=mpmath.nstr(c_re_mp, precision, strip_zeros=False),
        c_im=mpmath.nstr(c_im_mp, precision, strip_zeros=False),
        period=period,
        size=size,
        precision=precision,
        converged=converged,
        newton_steps=step,
    )


def _estimate_size_gmpy2(c_re, c_im, period, bits):
    """Estimate component size using the multiplier derivative at nucleus."""
    gmpy2.get_context().precision = bits
    two = gmpy2.mpfr(2)

    # At the nucleus, the multiplier lambda = product(2*z_k) for k=0..p-1 = 0
    # The size is related to 1/|b_p| where b_p = dz_p/dc at the nucleus
    # We also need d(lambda)/dc for the full size estimate
    # Size ~ |b_p| / |d²z_p/dz_0²| but simpler: size ~ 1/|b_p|

    z_re = gmpy2.mpfr(0)
    z_im = gmpy2.mpfr(0)
    dc_re = gmpy2.mpfr(0)
    dc_im = gmpy2.mpfr(0)

    for _ in range(period):
        new_dc_re = two * (z_re * dc_re - z_im * dc_im) + gmpy2.mpfr(1)
        new_dc_im = two * (z_re * dc_im + z_im * dc_re)
        new_z_re = z_re * z_re - z_im * z_im + c_re
        new_z_im = two * z_re * z_im + c_im
        z_re, z_im = new_z_re, new_z_im
        dc_re, dc_im = new_dc_re, new_dc_im

    dc_mag = float(gmpy2.sqrt(dc_re * dc_re + dc_im * dc_im))
    if dc_mag > 0:
        return 1.0 / dc_mag
    return 0.0


def _find_nucleus_mpmath(c_re_str, c_im_str, period, precision, max_steps,
                         tolerance):
    mpmath.mp.dps = precision

    c = mpmath.mpc(c_re_str, c_im_str)

    if tolerance == 0.0:
        tol = mpmath.mpf(10) ** (-(precision // 2))
    else:
        tol = mpmath.mpf(tolerance)

    converged = False
    step = 0

    for step in range(1, max_steps + 1):
        z = mpmath.mpc(0, 0)
        dc = mpmath.mpc(0, 0)

        for _ in range(period):
            dc = 2 * z * dc + 1
            z = z * z + c

        if abs(dc) == 0:
            break

        delta = z / dc
        c -= delta

        if abs(delta) < tol:
            converged = True
            break

    # Size estimate
    z = mpmath.mpc(0, 0)
    dc = mpmath.mpc(0, 0)
    for _ in range(period):
        dc = 2 * z * dc + 1
        z = z * z + c
    dc_mag = float(abs(dc))
    size = 1.0 / dc_mag if dc_mag > 0 else 0.0

    return MandelbrotNucleus(
        c_re=mpmath.nstr(c.real, precision, strip_zeros=False),
        c_im=mpmath.nstr(c.imag, precision, strip_zeros=False),
        period=period,
        size=size,
        precision=precision,
        converged=converged,
        newton_steps=step,
    )


def find_boundary_point(nucleus: MandelbrotNucleus, internal_angle: float = 0.0,
                        precision: int | None = None,
                        max_steps: int = 100) -> BoundaryPoint:
    """Find a point on the boundary of a hyperbolic component.

    Solves the system:
        f^p(z, c) = z        (fixed point condition)
        df^p/dz(z, c) = e^{2*pi*i*theta}  (multiplier = target)

    using Newton's method on (z, c) simultaneously.

    The internal angle determines which cusp/antenna of the component
    boundary is targeted:
        0.0 = cusp (most detail, deepest zoom potential)
        0.5 = 1/2 bulb junction
        1/3 = 1/3 bulb junction (period tripling)
        1/golden_ratio = Siegel disk boundary (dense spirals)

    Args:
        nucleus: The nucleus of the parent component.
        internal_angle: Internal angle in turns (0 to 1). Default 0 = cusp.
        precision: Override precision. If None, uses nucleus precision.
        max_steps: Maximum Newton iterations.

    Returns:
        BoundaryPoint on the component boundary.
    """
    if precision is None:
        precision = nucleus.precision

    # The exact cusp (angle=0) has a degenerate Jacobian because the
    # multiplier lambda = 1 makes (lambda - 1) = 0 in the Jacobian.
    # Standard workaround: approach the cusp at a tiny angle offset.
    # The resulting point is indistinguishable from the true cusp at
    # any practical zoom level.
    original_angle = internal_angle
    if internal_angle == 0.0:
        internal_angle = 1e-10

    if _HAS_GMPY2:
        bp = _find_boundary_gmpy2(nucleus, internal_angle, precision,
                                  max_steps)
    else:
        bp = _find_boundary_mpmath(nucleus, internal_angle, precision,
                                   max_steps)
    bp.internal_angle = original_angle
    return bp


def _find_boundary_gmpy2(nucleus, internal_angle, precision, max_steps):
    bits = int(precision * 3.3219) + 20
    gmpy2.get_context().precision = bits
    two = gmpy2.mpfr(2)
    one = gmpy2.mpfr(1)
    period = nucleus.period

    # Target multiplier: e^{2*pi*i*theta}
    theta = 2.0 * math.pi * internal_angle
    target_re = gmpy2.mpfr(math.cos(theta))
    target_im = gmpy2.mpfr(math.sin(theta))

    # Start from nucleus
    c_re = gmpy2.mpfr(nucleus.c_re)
    c_im = gmpy2.mpfr(nucleus.c_im)
    # Initial guess for fixed point: z = 0 (near nucleus, the fixed point is near 0)
    z0_re = gmpy2.mpfr(0)
    z0_im = gmpy2.mpfr(0)

    tol = gmpy2.mpfr(10) ** (-(precision // 2))
    converged = False

    for step in range(max_steps):
        # Iterate f^p from z0, tracking derivatives:
        # a = dz/dz0 (multiplier), b = dz/dc
        # aa = d²z/dz0², ab = d²z/(dz0 dc)
        z_re, z_im = z0_re, z0_im
        a_re, a_im = one, gmpy2.mpfr(0)       # dz/dz0
        b_re, b_im = gmpy2.mpfr(0), gmpy2.mpfr(0)  # dz/dc
        aa_re, aa_im = gmpy2.mpfr(0), gmpy2.mpfr(0)  # d²z/dz0²
        ab_re, ab_im = gmpy2.mpfr(0), gmpy2.mpfr(0)  # d²z/(dz0 dc)

        for _ in range(period):
            # Update second derivatives first (they depend on current a, b, z)
            # aa' = 2*(a^2 + z*aa)
            # a^2
            a2_re = a_re * a_re - a_im * a_im
            a2_im = two * a_re * a_im
            # z*aa
            zaa_re = z_re * aa_re - z_im * aa_im
            zaa_im = z_re * aa_im + z_im * aa_re
            new_aa_re = two * (a2_re + zaa_re)
            new_aa_im = two * (a2_im + zaa_im)

            # ab' = 2*(a*b + z*ab)
            # a*b
            ab_prod_re = a_re * b_re - a_im * b_im
            ab_prod_im = a_re * b_im + a_im * b_re
            # z*ab
            zab_re = z_re * ab_re - z_im * ab_im
            zab_im = z_re * ab_im + z_im * ab_re
            new_ab_re = two * (ab_prod_re + zab_re)
            new_ab_im = two * (ab_prod_im + zab_im)

            # a' = 2*z*a
            new_a_re = two * (z_re * a_re - z_im * a_im)
            new_a_im = two * (z_re * a_im + z_im * a_re)

            # b' = 2*z*b + 1
            new_b_re = two * (z_re * b_re - z_im * b_im) + one
            new_b_im = two * (z_re * b_im + z_im * b_re)

            # z' = z^2 + c
            new_z_re = z_re * z_re - z_im * z_im + c_re
            new_z_im = two * z_re * z_im + c_im

            z_re, z_im = new_z_re, new_z_im
            a_re, a_im = new_a_re, new_a_im
            b_re, b_im = new_b_re, new_b_im
            aa_re, aa_im = new_aa_re, new_aa_im
            ab_re, ab_im = new_ab_re, new_ab_im

        # Residuals: G1 = z_p - z0, G2 = a_p - target
        g1_re = z_re - z0_re
        g1_im = z_im - z0_im
        g2_re = a_re - target_re
        g2_im = a_im - target_im

        # Jacobian:
        # J = [[a-1, b], [aa, ab]]  (complex 2x2)
        j11_re = a_re - one
        j11_im = a_im
        j12_re, j12_im = b_re, b_im
        j21_re, j21_im = aa_re, aa_im
        j22_re, j22_im = ab_re, ab_im

        # det = j11*j22 - j12*j21
        det_re = (j11_re * j22_re - j11_im * j22_im) - (j12_re * j21_re - j12_im * j21_im)
        det_im = (j11_re * j22_im + j11_im * j22_re) - (j12_re * j21_im + j12_im * j21_re)

        det_mag_sq = det_re * det_re + det_im * det_im
        if float(det_mag_sq) == 0.0:
            break

        # Inverse of det
        inv_re = det_re / det_mag_sq
        inv_im = -det_im / det_mag_sq

        # [dz0, dc] = J^{-1} @ [g1, g2]
        # J^{-1} = (1/det) * [[j22, -j12], [-j21, j11]]
        # dz0 = (j22*g1 - j12*g2) / det
        t1_re = (j22_re * g1_re - j22_im * g1_im) - (j12_re * g2_re - j12_im * g2_im)
        t1_im = (j22_re * g1_im + j22_im * g1_re) - (j12_re * g2_im + j12_im * g2_re)
        dz0_re = t1_re * inv_re - t1_im * inv_im
        dz0_im = t1_re * inv_im + t1_im * inv_re

        # dc = (-j21*g1 + j11*g2) / det
        t2_re = -(j21_re * g1_re - j21_im * g1_im) + (j11_re * g2_re - j11_im * g2_im)
        t2_im = -(j21_re * g1_im + j21_im * g1_re) + (j11_re * g2_im + j11_im * g2_re)
        dc_re = t2_re * inv_re - t2_im * inv_im
        dc_im = t2_re * inv_im + t2_im * inv_re

        z0_re -= dz0_re
        z0_im -= dz0_im
        c_re -= dc_re
        c_im -= dc_im

        delta_mag = float(gmpy2.sqrt(dc_re * dc_re + dc_im * dc_im))
        if delta_mag < float(tol):
            converged = True
            break

    # Suggested zoom based on component size
    size = nucleus.size
    if size > 0:
        suggested_zoom = 1.0 / (size * 0.01)  # Zoom to ~100x the component size
    else:
        suggested_zoom = 1e10

    mpmath.mp.dps = precision
    return BoundaryPoint(
        c_re=mpmath.nstr(mpmath.mpf(str(c_re)), precision, strip_zeros=False),
        c_im=mpmath.nstr(mpmath.mpf(str(c_im)), precision, strip_zeros=False),
        period=nucleus.period,
        internal_angle=internal_angle,
        suggested_zoom=suggested_zoom,
        precision=precision,
        converged=converged,
    )


def _find_boundary_mpmath(nucleus, internal_angle, precision, max_steps):
    mpmath.mp.dps = precision
    period = nucleus.period

    theta = 2.0 * math.pi * internal_angle
    target = mpmath.mpc(math.cos(theta), math.sin(theta))

    c = mpmath.mpc(nucleus.c_re, nucleus.c_im)
    z0 = mpmath.mpc(0, 0)

    tol = mpmath.mpf(10) ** (-(precision // 2))
    converged = False

    for step in range(max_steps):
        z = z0
        a = mpmath.mpc(1, 0)    # dz/dz0
        b = mpmath.mpc(0, 0)    # dz/dc
        aa = mpmath.mpc(0, 0)   # d²z/dz0²
        ab = mpmath.mpc(0, 0)   # d²z/(dz0 dc)

        for _ in range(period):
            new_aa = 2 * (a * a + z * aa)
            new_ab = 2 * (a * b + z * ab)
            new_a = 2 * z * a
            new_b = 2 * z * b + 1
            z = z * z + c

            a, b, aa, ab = new_a, new_b, new_aa, new_ab

        # Residuals
        g1 = z - z0
        g2 = a - target

        # Jacobian
        j11 = a - 1
        j12 = b
        j21 = aa
        j22 = ab

        det = j11 * j22 - j12 * j21
        if abs(det) == 0:
            break

        dz0 = (j22 * g1 - j12 * g2) / det
        dc = (-j21 * g1 + j11 * g2) / det

        z0 -= dz0
        c -= dc

        if abs(dc) < tol:
            converged = True
            break

    size = nucleus.size
    suggested_zoom = 1.0 / (size * 0.01) if size > 0 else 1e10

    return BoundaryPoint(
        c_re=mpmath.nstr(c.real, precision, strip_zeros=False),
        c_im=mpmath.nstr(c.imag, precision, strip_zeros=False),
        period=nucleus.period,
        internal_angle=internal_angle,
        suggested_zoom=suggested_zoom,
        precision=precision,
        converged=converged,
    )


def discover_coordinates(c_re_str: str, c_im_str: str,
                         precision: int = 100,
                         angles: list[float] | None = None,
                         max_period: int = 10000,
                         verbose: bool = True) -> list[BoundaryPoint]:
    """Full discovery pipeline: detect period, find nucleus, find boundary points.

    This is the main entry point for the coordinate finder. Given an
    approximate location near the Mandelbrot set boundary, it:
    1. Detects the period of the nearest hyperbolic component
    2. Refines the nucleus to full precision
    3. Finds boundary points at the specified internal angles

    Args:
        c_re_str: Approximate real coordinate.
        c_im_str: Approximate imaginary coordinate.
        precision: Decimal digits for output coordinates.
        angles: Internal angles to sample. Default covers the most
            visually interesting points (cusp, bulb junctions, spirals).
        max_period: Maximum period to search for.
        verbose: Print progress information.

    Returns:
        List of BoundaryPoint objects, sorted by suggested zoom depth.
    """
    if angles is None:
        angles = [
            0.0,      # Cusp -- deepest detail, most complex spirals
            0.5,      # 1/2 bulb -- period doubling cascade
            1/3,      # 1/3 bulb -- period tripling, trefoil patterns
            2/3,      # 2/3 bulb -- same component, opposite side
            1/4,      # 1/4 bulb -- quaternary branching
            3/4,      # 3/4 bulb
            1/5,      # 1/5 bulb -- fivefold symmetry
            2/5,      # Golden ratio adjacent
        ]

    t0 = time.perf_counter()

    if verbose:
        print(f"Detecting period near ({c_re_str}, {c_im_str})...")

    period = detect_period(c_re_str, c_im_str, max_period, precision)
    if period < 0:
        if verbose:
            print("  Point appears to escape (not in/near the Mandelbrot set).")
        return []

    if verbose:
        print(f"  Detected period: {period}")
        print(f"Finding nucleus at {precision}-digit precision...")

    nucleus = find_nucleus(c_re_str, c_im_str, period, precision)

    if verbose:
        status = "CONVERGED" if nucleus.converged else "DID NOT CONVERGE"
        print(f"  Nucleus: ({nucleus.c_re}, {nucleus.c_im})")
        print(f"  Status: {status} in {nucleus.newton_steps} steps")
        print(f"  Component size: {nucleus.size:.3e}")
        if nucleus.size > 0:
            print(f"  Suggested min zoom: {1.0/nucleus.size:.3e}")

    if verbose:
        print(f"Finding {len(angles)} boundary points...")

    results = []
    for angle in angles:
        bp = find_boundary_point(nucleus, angle, precision)
        results.append(bp)
        if verbose:
            status = "OK" if bp.converged else "FAILED"
            angle_str = _format_angle(angle)
            print(f"  angle={angle_str:>10s}  zoom={bp.suggested_zoom:.1e}  [{status}]")

    elapsed = time.perf_counter() - t0
    if verbose:
        ok = sum(1 for bp in results if bp.converged)
        print(f"\nDone in {elapsed:.1f}s -- {ok}/{len(results)} converged")

    return sorted(results, key=lambda bp: bp.suggested_zoom, reverse=True)


def _format_angle(angle: float) -> str:
    """Format an angle as a readable fraction if possible."""
    # Check common fractions
    for denom in range(1, 13):
        for numer in range(denom):
            if abs(angle - numer / denom) < 1e-10:
                if numer == 0:
                    return "0 (cusp)"
                return f"{numer}/{denom}"
    return f"{angle:.4f}"


def scan_region(re_min: float, re_max: float, im_min: float, im_max: float,
                grid_size: int = 20, max_period: int = 500,
                precision: int = 50, verbose: bool = True) -> list[MandelbrotNucleus]:
    """Scan a region for hyperbolic component nuclei.

    Samples points on a grid, detects periods, and finds unique nuclei.
    Useful for discovering interesting locations in a region of the c-plane.

    Args:
        re_min, re_max: Real coordinate range.
        im_min, im_max: Imaginary coordinate range.
        grid_size: Number of points per axis (total = grid_size^2).
        max_period: Maximum period to search at each point.
        precision: Digits for nucleus refinement.
        verbose: Print progress.

    Returns:
        List of unique MandelbrotNucleus objects found, sorted by period.
    """
    t0 = time.perf_counter()
    if verbose:
        total = grid_size * grid_size
        print(f"Scanning {total} points in [{re_min:.6f}, {re_max:.6f}] x "
              f"[{im_min:.6f}, {im_max:.6f}]...")

    re_steps = np.linspace(re_min, re_max, grid_size)
    im_steps = np.linspace(im_min, im_max, grid_size)

    found = {}  # period -> nucleus (keep best-converged per period)
    scanned = 0

    for re_val in re_steps:
        for im_val in im_steps:
            scanned += 1
            re_str = f"{re_val:.15f}"
            im_str = f"{im_val:.15f}"

            period = detect_period(re_str, im_str, max_period, min(precision, 30))
            if period < 0:
                continue

            # Only refine if we haven't found this period yet
            if period not in found:
                try:
                    nuc = find_nucleus(re_str, im_str, period, precision)
                    if nuc.converged:
                        found[period] = nuc
                        if verbose:
                            print(f"  [{scanned}/{grid_size**2}] Period {period} "
                                  f"nucleus at ({nuc.c_re[:20]}..., "
                                  f"{nuc.c_im[:20]}...) size={nuc.size:.3e}")
                except Exception:
                    pass

    elapsed = time.perf_counter() - t0
    nuclei = sorted(found.values(), key=lambda n: n.period)

    if verbose:
        print(f"\nFound {len(nuclei)} unique nuclei in {elapsed:.1f}s")

    return nuclei


def find_deep_target(c_re_str: str, c_im_str: str,
                     target_zoom: float = 1e30,
                     precision: int | None = None,
                     internal_angle: float = 0.5,
                     verbose: bool = True) -> BoundaryPoint | None:
    """Find a coordinate with fractal detail at extreme zoom depth.

    Recursively discovers nested minibrots by:
    1. Finding the nearest hyperbolic component nucleus
    2. Finding the boundary point at the given internal angle
    3. Using that boundary point as seed for the next iteration
    4. Repeating until the feature scale is smaller than 1/target_zoom

    The internal angle determines the character of the nested structure:
        0.5 = period-doubling cascade (most common, self-similar spirals)
        1/3 = period-tripling (trefoil patterns)

    Args:
        c_re_str: Starting coordinate (real).
        c_im_str: Starting coordinate (imaginary).
        target_zoom: Desired minimum zoom depth for interesting detail.
        precision: Digits (auto-computed from target_zoom if None).
        internal_angle: Angle for nesting (0.5 = doubling, 1/3 = tripling).
        verbose: Print progress.

    Returns:
        BoundaryPoint valid at the target zoom depth, or None on failure.
    """
    if precision is None:
        precision = max(100, int(1.5 * math.log10(max(target_zoom, 10))) + 50)

    if verbose:
        print(f"Finding deep target at zoom >= {target_zoom:.0e} "
              f"({precision} digits)...")

    current_re = c_re_str
    current_im = c_im_str
    depth = 0
    accumulated_zoom = 1.0

    while accumulated_zoom < target_zoom:
        depth += 1
        if verbose:
            print(f"\n  Depth {depth}: accumulated zoom = {accumulated_zoom:.1e}")

        period = detect_period(current_re, current_im, 10000, precision)
        if period < 0:
            if verbose:
                print(f"    Point escapes, cannot go deeper.")
            break

        nuc = find_nucleus(current_re, current_im, period, precision)
        if not nuc.converged:
            if verbose:
                print(f"    Nucleus did not converge at period {period}.")
            break

        if verbose:
            print(f"    Period {nuc.period}, size = {nuc.size:.3e}")

        if nuc.size <= 0:
            if verbose:
                print(f"    Zero-size component, stopping.")
            break

        # Update accumulated zoom
        accumulated_zoom = 1.0 / nuc.size

        # Find boundary point for the next level
        bp = find_boundary_point(nuc, internal_angle, precision)
        if not bp.converged:
            if verbose:
                print(f"    Boundary point did not converge.")
            break

        current_re = bp.c_re
        current_im = bp.c_im

        if verbose:
            print(f"    Boundary at angle {_format_angle(internal_angle)}: "
                  f"zoom = {accumulated_zoom:.1e}")

        # Safety limit on recursion depth
        if depth >= 20:
            if verbose:
                print(f"    Max recursion depth reached.")
            break

    # Return the last valid boundary point with updated zoom info
    if accumulated_zoom >= target_zoom or depth > 0:
        if verbose:
            print(f"\n  Final: depth={depth}, zoom={accumulated_zoom:.1e}, "
                  f"precision={precision} digits")
            print(f"  Re: {current_re[:80]}...")
            print(f"  Im: {current_im[:80]}...")
        return BoundaryPoint(
            c_re=current_re,
            c_im=current_im,
            period=nuc.period if 'nuc' in dir() else -1,
            internal_angle=internal_angle,
            suggested_zoom=accumulated_zoom,
            precision=precision,
            converged=True,
        )
    return None


def _orbit_and_deriv_mpmath(c, n):
    """Return (f_c^n(0), d/dc f_c^n(0)) tracking z and dz/dc at precision."""
    z = mpmath.mpf(0)
    dz = mpmath.mpf(0)
    for _ in range(n):
        dz = 2 * z * dz + 1
        z = z * z + c
    return z, dz


def find_misiurewicz(
    c_re_str: str,
    c_im_str: str,
    preperiod: int | None = None,
    period: int | None = None,
    precision: int = 120,
    max_preperiod: int = 40,
    max_period: int = 40,
    newton_steps: int = 80,
    verbose: bool = False,
) -> MisiurewiczPoint | None:
    """Find a Misiurewicz (pre-periodic) point near a seed coordinate.

    Solves g(c) = f_c^{m+p}(0) - f_c^{m}(0) = 0 by Newton's method, where m
    is the pre-period and p the period of the repelling cycle. When m and p
    are not given, searches small (m, p) pairs and returns the lowest-order
    pre-periodic point reachable from the seed.

    Misiurewicz points are the preferred targets for extreme deep zoom: the
    reference orbit stays short at any depth (see MisiurewiczPoint), so the
    floatexp deep kernel can dive to 1e500+ without an intractable orbit.

    Args:
        c_re_str, c_im_str: Seed coordinate (strings preserve precision).
        preperiod, period: Force a specific (m, p); auto-searched if None.
        precision: Decimal digits for the refined coordinate.
        max_preperiod, max_period: Search bounds when auto-detecting.
        newton_steps: Maximum Newton iterations.
        verbose: Print progress.

    Returns:
        A converged MisiurewiczPoint, or None if none found.
    """
    mpmath.mp.dps = precision
    seed = mpmath.mpc(c_re_str, c_im_str)
    tol = mpmath.mpf(10) ** (-precision + 12)

    if preperiod is not None and period is not None:
        pairs = [(preperiod, period)]
    else:
        # Search by total order m+p so the simplest (most robust, widest-basin)
        # point wins.
        pairs = sorted(
            ((m, p) for m in range(1, max_preperiod + 1)
             for p in range(1, max_period + 1)),
            key=lambda mp: (mp[0] + mp[1], mp[1]),
        )

    for m, p in pairs:
        c = seed
        converged = False
        for _ in range(newton_steps):
            zm, dzm = _orbit_and_deriv_mpmath(c, m)
            zmp, dzmp = _orbit_and_deriv_mpmath(c, m + p)
            g = zmp - zm
            dg = dzmp - dzm
            if abs(dg) == 0:
                break
            dc = g / dg
            c = c - dc
            if abs(dc) < tol:
                converged = True
                break
        if not converged:
            continue

        # Validate: g ~ 0 (Misiurewicz) and NOT a plain period-p nucleus
        # (a nucleus has f^p(0)=0, which the equation also admits).
        zm, _ = _orbit_and_deriv_mpmath(c, m)
        zmp, _ = _orbit_and_deriv_mpmath(c, m + p)
        z_p, _ = _orbit_and_deriv_mpmath(c, p)
        resid = abs(zmp - zm)
        if resid > mpmath.mpf(10) ** (-precision + 20):
            continue
        if abs(z_p) < tol:
            continue  # periodic nucleus, not pre-periodic
        # Reject if a smaller pre-period already satisfies (true m is minimal)
        zm1, _ = _orbit_and_deriv_mpmath(c, m - 1) if m > 1 else (None, None)
        if m > 1:
            zmp1, _ = _orbit_and_deriv_mpmath(c, m - 1 + p)
            if abs(zmp1 - zm1) < tol:
                continue  # m-1 also works, so this m isn't minimal

        if verbose:
            print(f"Misiurewicz M({m},{p}) found, residual {mpmath.nstr(resid, 3)}")
        return MisiurewiczPoint(
            c_re=mpmath.nstr(c.real, precision, strip_zeros=False),
            c_im=mpmath.nstr(c.imag, precision, strip_zeros=False),
            preperiod=m,
            period=p,
            precision=precision,
            converged=True,
        )

    if verbose:
        print("No Misiurewicz point found near seed")
    return None
