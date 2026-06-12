"""Precision management -- arbitrary precision reference orbits for deep zoom.

At zoom depths beyond ~1e13, standard float64 loses the precision needed to
distinguish adjacent pixels. This module computes a single reference orbit
at arbitrary precision using mpmath (or gmpy2 when available for 5-10x speed).
The perturbation kernel (perturbation.py) then uses this orbit to compute all
other pixels as float64 deltas.

Reference orbit pipeline:
  1. Determine required precision from zoom level
  2. Iterate Z_{n+1} = Z_n^2 + C at that precision (gmpy2/mpmath, CPU)
  3. Store orbit as float64 arrays for GPU upload
  4. Track escape iteration and |Z_n|^2 for glitch detection
"""

import time
from dataclasses import dataclass

import mpmath
import numpy as np

# gmpy2 provides 5-10x faster arbitrary precision arithmetic than mpmath.
# Fall back to mpmath when gmpy2 is not installed.
try:
    import gmpy2
    _HAS_GMPY2 = True
except ImportError:
    _HAS_GMPY2 = False


def required_precision(zoom: float, center_re: str | None = None,
                       center_im: str | None = None) -> int:
    """Compute the number of decimal digits needed for a given zoom level.

    Uses the research-validated formula P = 1.5 * log10(zoom) + 30, which
    accounts for non-linear error propagation during chaotic orbit segments.
    Additionally ensures precision exceeds the significant digits in the
    center coordinate strings (+ 20 digit buffer).

    Args:
        zoom: The zoom level (e.g. 1e50).
        center_re: Optional center real coordinate string for digit counting.
        center_im: Optional center imaginary coordinate string for digit counting.

    Returns:
        Number of decimal digits for mpmath.mp.dps.
    """
    import math

    if zoom <= 1.0:
        return 16  # float64 is fine

    # Research formula: 1.5 * log10(zoom) + 30
    digits = int(1.5 * math.log10(zoom)) + 30

    # Ensure precision exceeds coordinate string digits + buffer
    for coord_str in (center_re, center_im):
        if coord_str:
            # Count significant digits (strip sign, leading zeros, decimal point)
            s = coord_str.lstrip("-").lstrip("0").replace(".", "")
            sig_digits = len(s.rstrip("0")) if s else 0
            digits = max(digits, sig_digits + 20)

    return max(16, digits)


@dataclass
class ReferenceOrbit:
    """Stores a reference orbit computed at arbitrary precision.

    The orbit arrays (z_re, z_im, z_mag_sq) are stored as float64 for
    efficient GPU upload. The full-precision center is retained for
    computing per-pixel dc values.

    Attributes:
        z_re: Real parts of Z_n, shape (num_iters+1,). z_re[0] = 0.
        z_im: Imaginary parts of Z_n, shape (num_iters+1,). z_im[0] = 0.
        z_mag_sq: |Z_n|^2 at each iteration, shape (num_iters+1,).
            Used by glitch detection to compare delta magnitude.
        num_iters: Number of iterations computed (length of orbit - 1).
        escape_iter: Iteration at which |Z_n|^2 > escape_radius^2,
            or -1 if the reference point is interior (never escaped).
        center_re_str: Full-precision real center as decimal string.
        center_im_str: Full-precision imaginary center as decimal string.
        precision: Decimal digits used for computation.
        compute_time: Wall-clock seconds to compute the orbit.
    """

    z_re: np.ndarray
    z_im: np.ndarray
    z_mag_sq: np.ndarray
    num_iters: int
    escape_iter: int
    center_re_str: str
    center_im_str: str
    precision: int
    compute_time: float


def compute_reference_orbit(
    center_re: str | float,
    center_im: str | float,
    max_iter: int,
    precision: int | None = None,
    zoom: float | None = None,
    escape_radius_sq: float = 256.0,
) -> ReferenceOrbit:
    """Compute a reference orbit at arbitrary precision.

    Iterates Z_{n+1} = Z_n^2 + C starting from Z_0 = 0, storing every
    iterate as float64 for GPU consumption. Stops at escape or max_iter.

    Args:
        center_re: Real part of reference point C. Pass as string for
            full precision (e.g. "-0.7436438870371587").
        center_im: Imaginary part of reference point C.
        max_iter: Maximum number of iterations.
        precision: Decimal digits for mpmath. If None, auto-computed from zoom.
        zoom: Zoom level, used to auto-compute precision if precision is None.
        escape_radius_sq: Squared escape radius (default 256 = radius 16).

    Returns:
        ReferenceOrbit with the full orbit data.

    Raises:
        ValueError: If neither precision nor zoom is provided.
    """
    if precision is None:
        if zoom is None:
            raise ValueError("Must provide either precision or zoom")
        precision = required_precision(zoom, str(center_re), str(center_im))

    if _HAS_GMPY2:
        return _compute_reference_orbit_gmpy2(
            str(center_re), str(center_im), max_iter, precision, escape_radius_sq
        )

    return _compute_reference_orbit_mpmath(
        str(center_re), str(center_im), max_iter, precision, escape_radius_sq
    )


def _compute_reference_orbit_gmpy2(
    center_re_str: str,
    center_im_str: str,
    max_iter: int,
    precision: int,
    escape_radius_sq: float,
) -> ReferenceOrbit:
    """Compute reference orbit using gmpy2 (5-10x faster than mpmath)."""
    # gmpy2 precision is in bits, not decimal digits
    bits = int(precision * 3.3219) + 20  # log2(10) ≈ 3.3219, plus margin
    gmpy2.get_context().precision = bits

    c_re = gmpy2.mpfr(center_re_str)
    c_im = gmpy2.mpfr(center_im_str)
    escape_sq = gmpy2.mpfr(escape_radius_sq)

    # Pre-allocate numpy arrays (faster than list append for large orbits)
    z_re_arr = np.empty(max_iter + 1, dtype=np.float64)
    z_im_arr = np.empty(max_iter + 1, dtype=np.float64)
    z_mag_arr = np.empty(max_iter + 1, dtype=np.float64)

    z_r = gmpy2.mpfr(0)
    z_i = gmpy2.mpfr(0)
    z_re_arr[0] = 0.0
    z_im_arr[0] = 0.0
    z_mag_arr[0] = 0.0

    escape_iter = -1
    two = gmpy2.mpfr(2)
    actual_len = max_iter + 1

    start_time = time.perf_counter()

    for n in range(1, max_iter + 1):
        z_r_new = z_r * z_r - z_i * z_i + c_re
        z_i_new = two * z_r * z_i + c_im
        z_r = z_r_new
        z_i = z_i_new

        mag_sq = z_r * z_r + z_i * z_i

        z_re_arr[n] = float(z_r)
        z_im_arr[n] = float(z_i)
        z_mag_arr[n] = float(mag_sq)

        if mag_sq > escape_sq:
            escape_iter = n
            actual_len = n + 1
            break

    elapsed = time.perf_counter() - start_time

    # Use mpmath for the final string representation (gmpy2 str format differs)
    mpmath.mp.dps = precision
    c_re_mp = mpmath.mpf(center_re_str)
    c_im_mp = mpmath.mpf(center_im_str)

    return ReferenceOrbit(
        z_re=z_re_arr[:actual_len],
        z_im=z_im_arr[:actual_len],
        z_mag_sq=z_mag_arr[:actual_len],
        num_iters=actual_len - 1,
        escape_iter=escape_iter,
        center_re_str=mpmath.nstr(c_re_mp, precision, strip_zeros=False),
        center_im_str=mpmath.nstr(c_im_mp, precision, strip_zeros=False),
        precision=precision,
        compute_time=elapsed,
    )


def _compute_reference_orbit_mpmath(
    center_re_str: str,
    center_im_str: str,
    max_iter: int,
    precision: int,
    escape_radius_sq: float,
) -> ReferenceOrbit:
    """Compute reference orbit using mpmath (fallback when gmpy2 unavailable)."""
    mpmath.mp.dps = precision

    c_re = mpmath.mpf(center_re_str)
    c_im = mpmath.mpf(center_im_str)

    z_re_list = []
    z_im_list = []
    z_mag_sq_list = []

    z_r = mpmath.mpf(0)
    z_i = mpmath.mpf(0)
    z_re_list.append(0.0)
    z_im_list.append(0.0)
    z_mag_sq_list.append(0.0)

    escape_iter = -1
    escape_radius_sq_mp = mpmath.mpf(escape_radius_sq)

    start_time = time.perf_counter()

    for n in range(1, max_iter + 1):
        z_r_new = z_r * z_r - z_i * z_i + c_re
        z_i_new = 2 * z_r * z_i + c_im
        z_r = z_r_new
        z_i = z_i_new

        mag_sq = z_r * z_r + z_i * z_i

        z_re_list.append(float(z_r))
        z_im_list.append(float(z_i))
        z_mag_sq_list.append(float(mag_sq))

        if mag_sq > escape_radius_sq_mp:
            escape_iter = n
            break

    elapsed = time.perf_counter() - start_time

    return ReferenceOrbit(
        z_re=np.array(z_re_list, dtype=np.float64),
        z_im=np.array(z_im_list, dtype=np.float64),
        z_mag_sq=np.array(z_mag_sq_list, dtype=np.float64),
        num_iters=len(z_re_list) - 1,
        escape_iter=escape_iter,
        center_re_str=mpmath.nstr(c_re, precision, strip_zeros=False),
        center_im_str=mpmath.nstr(c_im, precision, strip_zeros=False),
        precision=precision,
        compute_time=elapsed,
    )


def compute_series_approximation_hp(
    ref: ReferenceOrbit,
    dc_max_mag_sq: float,
    tolerance: float = 1e-6,
    probe_dc_re: np.ndarray | None = None,
    probe_dc_im: np.ndarray | None = None,
    pixel_spacing: float = 0.0,
) -> tuple:
    """Compute SA coefficients at arbitrary precision, then downcast to float64.

    The SA recurrence A_{n+1} = 2*Z_n*A_n + 1, etc. is computed in gmpy2/mpmath
    at the same precision used for the reference orbit. This prevents the
    truncation errors that occur when computing in float64 at extreme zoom.

    Probe-based validation: if probe_dc_re/im are provided, the SA polynomial
    is evaluated at each probe point and compared against actual perturbation
    iteration. SA stops when any probe deviates, catching divergent series
    that pass analytical checks (e.g., chaotic orbits near the antenna).

    Args:
        ref: The reference orbit (contains precision level and center strings).
        dc_max_mag_sq: Maximum |dc|^2 across the viewport (corner pixel).
        tolerance: SA validity threshold for |C * dc_max^3|.
        probe_dc_re: Float64 array of probe dc real parts (typically 8 probes).
        probe_dc_im: Float64 array of probe dc imaginary parts.
        pixel_spacing: Pixel spacing in complex plane (for probe tolerance floor).

    Returns:
        Tuple (skip_iters, A_re, A_im, B_re, B_im, C_re, C_im) as float64.
    """
    precision = ref.precision

    if _HAS_GMPY2:
        return _sa_hp_gmpy2(ref, dc_max_mag_sq, tolerance, precision,
                            probe_dc_re, probe_dc_im, pixel_spacing)
    return _sa_hp_mpmath(ref, dc_max_mag_sq, tolerance, precision,
                         probe_dc_re, probe_dc_im, pixel_spacing)


def _sa_hp_gmpy2(ref, dc_max_mag_sq, tolerance, precision,
                 probe_dc_re=None, probe_dc_im=None, pixel_spacing=0.0):
    """SA coefficients via gmpy2 arbitrary precision with probe validation."""
    import math as _math

    bits = int(precision * 3.3219) + 20
    gmpy2.get_context().precision = bits

    dc_max_mag = gmpy2.mpfr(_math.sqrt(dc_max_mag_sq))
    dc_max_cubed = dc_max_mag * dc_max_mag * dc_max_mag
    tol = gmpy2.mpfr(tolerance)
    zero = gmpy2.mpfr(0)
    one = gmpy2.mpfr(1)
    two = gmpy2.mpfr(2)

    a_re, a_im = zero, zero
    b_re, b_im = zero, zero
    c_re, c_im = zero, zero

    skip_iters = 0
    save = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Probe-based validation: track actual perturbation deltas at sample points
    # and compare against SA polynomial. This catches divergent series that
    # pass analytical checks (e.g., chaotic orbits near the antenna tip).
    use_probes = (probe_dc_re is not None and probe_dc_im is not None
                  and len(probe_dc_re) > 0)
    if use_probes:
        num_probes = len(probe_dc_re)
        # True perturbation deltas for each probe (float64)
        pd_re = np.array(probe_dc_re, dtype=np.float64)
        pd_im = np.array(probe_dc_im, dtype=np.float64)
        probe_tol_floor = max(pixel_spacing * 1e-6, 1e-30)

    for n in range(ref.num_iters):
        zn_re_f = float(ref.z_re[n])
        zn_im_f = float(ref.z_im[n])
        zn_re = gmpy2.mpfr(zn_re_f)
        zn_im = gmpy2.mpfr(zn_im_f)

        # A_{n+1} = 2*Z_n*A_n + 1
        new_a_re = two * (zn_re * a_re - zn_im * a_im) + one
        new_a_im = two * (zn_re * a_im + zn_im * a_re)

        # B_{n+1} = 2*Z_n*B_n + A_n^2
        a_sq_re = a_re * a_re - a_im * a_im
        a_sq_im = two * a_re * a_im
        new_b_re = two * (zn_re * b_re - zn_im * b_im) + a_sq_re
        new_b_im = two * (zn_re * b_im + zn_im * b_re) + a_sq_im

        # C_{n+1} = 2*Z_n*C_n + 2*A_n*B_n
        ab_re = a_re * b_re - a_im * b_im
        ab_im = a_re * b_im + a_im * b_re
        new_c_re = two * (zn_re * c_re - zn_im * c_im) + two * ab_re
        new_c_im = two * (zn_re * c_im + zn_im * c_re) + two * ab_im

        a_re, a_im = new_a_re, new_a_im
        b_re, b_im = new_b_re, new_b_im
        c_re, c_im = new_c_re, new_c_im

        # Analytical check: |C * dc_max^3| < tolerance
        c_mag = gmpy2.sqrt(c_re * c_re + c_im * c_im)
        if c_mag * dc_max_cubed > tol:
            return (skip_iters, *save)

        # Probe validation: advance probe deltas and compare against SA polynomial
        if use_probes:
            # Advance probes: d_{n+1} = 2*Z_n*d_n + d_n^2 + dc
            new_pd_re = (2.0 * (zn_re_f * pd_re - zn_im_f * pd_im)
                         + pd_re * pd_re - pd_im * pd_im + probe_dc_re)
            new_pd_im = (2.0 * (zn_re_f * pd_im + zn_im_f * pd_re)
                         + 2.0 * pd_re * pd_im + probe_dc_im)
            pd_re = new_pd_re
            pd_im = new_pd_im

            # Evaluate SA polynomial at each probe: d_SA = A*dc + B*dc^2 + C*dc^3
            fa_re = float(a_re)
            fa_im = float(a_im)
            fb_re = float(b_re)
            fb_im = float(b_im)
            fc_re = float(c_re)
            fc_im = float(c_im)

            for p in range(num_probes):
                pcr, pci = probe_dc_re[p], probe_dc_im[p]
                # dc^2
                dc2r = pcr * pcr - pci * pci
                dc2i = 2.0 * pcr * pci
                # dc^3
                dc3r = pcr * dc2r - pci * dc2i
                dc3i = pcr * dc2i + pci * dc2r
                # SA prediction
                sa_re = (fa_re * pcr - fa_im * pci
                         + fb_re * dc2r - fb_im * dc2i
                         + fc_re * dc3r - fc_im * dc3i)
                sa_im = (fa_re * pci + fa_im * pcr
                         + fb_re * dc2i + fb_im * dc2r
                         + fc_re * dc3i + fc_im * dc3r)
                # Error vs true delta
                err_re = sa_re - pd_re[p]
                err_im = sa_im - pd_im[p]
                err_sq = err_re * err_re + err_im * err_im
                true_sq = pd_re[p] * pd_re[p] + pd_im[p] * pd_im[p]
                # Stop when error exceeds 0.1% of true delta (or pixel floor)
                threshold = max(true_sq * 1e-6, probe_tol_floor * probe_tol_floor)
                if err_sq > threshold:
                    return (skip_iters, *save)

        save = (float(a_re), float(a_im), float(b_re), float(b_im),
                float(c_re), float(c_im))
        skip_iters = n + 1

    return (skip_iters, *save)


def _sa_hp_mpmath(ref, dc_max_mag_sq, tolerance, precision,
                  probe_dc_re=None, probe_dc_im=None, pixel_spacing=0.0):
    """SA coefficients via mpmath (fallback) with probe validation."""
    import math as _math

    mpmath.mp.dps = precision

    dc_max_mag = mpmath.mpf(_math.sqrt(dc_max_mag_sq))
    dc_max_cubed = dc_max_mag ** 3
    tol = mpmath.mpf(tolerance)

    a_re, a_im = mpmath.mpf(0), mpmath.mpf(0)
    b_re, b_im = mpmath.mpf(0), mpmath.mpf(0)
    c_re, c_im = mpmath.mpf(0), mpmath.mpf(0)

    skip_iters = 0
    save = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Probe validation (same logic as gmpy2 version)
    use_probes = (probe_dc_re is not None and probe_dc_im is not None
                  and len(probe_dc_re) > 0)
    if use_probes:
        num_probes = len(probe_dc_re)
        pd_re = np.array(probe_dc_re, dtype=np.float64)
        pd_im = np.array(probe_dc_im, dtype=np.float64)
        probe_tol_floor = max(pixel_spacing * 1e-6, 1e-30)

    for n in range(ref.num_iters):
        zn_re_f = float(ref.z_re[n])
        zn_im_f = float(ref.z_im[n])
        zn_re = mpmath.mpf(zn_re_f)
        zn_im = mpmath.mpf(zn_im_f)

        new_a_re = 2 * (zn_re * a_re - zn_im * a_im) + 1
        new_a_im = 2 * (zn_re * a_im + zn_im * a_re)

        a_sq_re = a_re * a_re - a_im * a_im
        a_sq_im = 2 * a_re * a_im
        new_b_re = 2 * (zn_re * b_re - zn_im * b_im) + a_sq_re
        new_b_im = 2 * (zn_re * b_im + zn_im * b_re) + a_sq_im

        ab_re = a_re * b_re - a_im * b_im
        ab_im = a_re * b_im + a_im * b_re
        new_c_re = 2 * (zn_re * c_re - zn_im * c_im) + 2 * ab_re
        new_c_im = 2 * (zn_re * c_im + zn_im * c_re) + 2 * ab_im

        a_re, a_im = new_a_re, new_a_im
        b_re, b_im = new_b_re, new_b_im
        c_re, c_im = new_c_re, new_c_im

        c_mag = mpmath.sqrt(c_re ** 2 + c_im ** 2)
        if c_mag * dc_max_cubed > tol:
            return (skip_iters, *save)

        # Probe validation
        if use_probes:
            new_pd_re = (2.0 * (zn_re_f * pd_re - zn_im_f * pd_im)
                         + pd_re * pd_re - pd_im * pd_im + probe_dc_re)
            new_pd_im = (2.0 * (zn_re_f * pd_im + zn_im_f * pd_re)
                         + 2.0 * pd_re * pd_im + probe_dc_im)
            pd_re = new_pd_re
            pd_im = new_pd_im

            fa_re, fa_im = float(a_re), float(a_im)
            fb_re, fb_im = float(b_re), float(b_im)
            fc_re, fc_im = float(c_re), float(c_im)

            for p in range(num_probes):
                pcr, pci = probe_dc_re[p], probe_dc_im[p]
                dc2r = pcr * pcr - pci * pci
                dc2i = 2.0 * pcr * pci
                dc3r = pcr * dc2r - pci * dc2i
                dc3i = pcr * dc2i + pci * dc2r
                sa_re = (fa_re * pcr - fa_im * pci + fb_re * dc2r - fb_im * dc2i
                         + fc_re * dc3r - fc_im * dc3i)
                sa_im = (fa_re * pci + fa_im * pcr + fb_re * dc2i + fb_im * dc2r
                         + fc_re * dc3i + fc_im * dc3r)
                err_sq = (sa_re - pd_re[p])**2 + (sa_im - pd_im[p])**2
                true_sq = pd_re[p]**2 + pd_im[p]**2
                threshold = max(true_sq * 1e-6, probe_tol_floor ** 2)
                if err_sq > threshold:
                    return (skip_iters, *save)

        save = (float(a_re), float(a_im), float(b_re), float(b_im),
                float(c_re), float(c_im))
        skip_iters = n + 1

    return (skip_iters, *save)


def validate_reference_orbit(
    ref: ReferenceOrbit,
    center_re: float,
    center_im: float,
    max_iter: int,
) -> dict:
    """Validate a reference orbit against standard float64 computation.

    Useful for testing: at shallow zoom levels where float64 is sufficient,
    the reference orbit and standard computation should match closely.

    Args:
        ref: The reference orbit to validate.
        center_re: Same center as float64.
        center_im: Same center as float64.
        max_iter: Same max_iter used for the orbit.

    Returns:
        Dict with max_error, mean_error, matching_escape, and per-iteration
        error at key points.
    """
    # Standard float64 iteration
    z_r = 0.0
    z_i = 0.0
    f64_re = [0.0]
    f64_im = [0.0]
    f64_escape = -1

    for n in range(1, max_iter + 1):
        z_r_new = z_r * z_r - z_i * z_i + center_re
        z_i_new = 2.0 * z_r * z_i + center_im
        z_r = z_r_new
        z_i = z_i_new
        f64_re.append(z_r)
        f64_im.append(z_i)

        if z_r * z_r + z_i * z_i > 256.0:
            f64_escape = n
            break

    # Compare up to the shorter orbit
    n_compare = min(len(f64_re), len(ref.z_re))
    f64_re_arr = np.array(f64_re[:n_compare])
    f64_im_arr = np.array(f64_im[:n_compare])

    err_re = np.abs(ref.z_re[:n_compare] - f64_re_arr)
    err_im = np.abs(ref.z_im[:n_compare] - f64_im_arr)
    err_mag = np.sqrt(err_re**2 + err_im**2)

    return {
        "max_error": float(np.max(err_mag)),
        "mean_error": float(np.mean(err_mag)),
        "matching_escape": ref.escape_iter == f64_escape,
        "ref_escape": ref.escape_iter,
        "f64_escape": f64_escape,
        "n_compared": n_compare,
        "error_at_10": float(err_mag[10]) if n_compare > 10 else None,
        "error_at_100": float(err_mag[100]) if n_compare > 100 else None,
        "error_at_1000": float(err_mag[1000]) if n_compare > 1000 else None,
    }
