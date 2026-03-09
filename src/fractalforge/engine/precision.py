"""Precision management -- arbitrary precision reference orbits for deep zoom.

At zoom depths beyond ~1e13, standard float64 loses the precision needed to
distinguish adjacent pixels. This module computes a single reference orbit
at arbitrary precision using mpmath. The perturbation kernel (perturbation.py)
then uses this orbit to compute all other pixels as float64 deltas.

Reference orbit pipeline:
  1. Determine required precision from zoom level
  2. Iterate Z_{n+1} = Z_n^2 + C at that precision (mpmath, CPU)
  3. Store orbit as float64 arrays for GPU upload
  4. Track escape iteration and |Z_n|^2 for glitch detection
"""

import time
from dataclasses import dataclass

import mpmath
import numpy as np


def required_precision(zoom: float, margin: int = 10) -> int:
    """Compute the number of decimal digits needed for a given zoom level.

    At zoom level Z, adjacent pixels differ by ~3/(Z * width). To resolve
    this, we need at least log10(Z) + log10(width) digits. We add a margin
    for intermediate computation stability.

    Args:
        zoom: The zoom level (e.g. 1e50).
        margin: Extra digits beyond the minimum for safety.

    Returns:
        Number of decimal digits for mpmath.mp.dps.
    """
    import math

    if zoom <= 1.0:
        return 16  # float64 is fine
    digits = int(math.log10(zoom)) + margin
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
        precision = required_precision(zoom)

    # Set mpmath precision
    mpmath.mp.dps = precision

    # Parse center as mpmath floats (strings preserve full precision)
    c_re = mpmath.mpf(str(center_re))
    c_im = mpmath.mpf(str(center_im))

    # Pre-allocate orbit storage (max_iter + 1 for Z_0 through Z_{max_iter})
    # We'll trim if it escapes early
    z_re_list = []
    z_im_list = []
    z_mag_sq_list = []

    # Z_0 = 0
    z_r = mpmath.mpf(0)
    z_i = mpmath.mpf(0)
    z_re_list.append(0.0)
    z_im_list.append(0.0)
    z_mag_sq_list.append(0.0)

    escape_iter = -1
    escape_radius_sq_mp = mpmath.mpf(escape_radius_sq)

    start_time = time.perf_counter()

    for n in range(1, max_iter + 1):
        # Z_{n} = Z_{n-1}^2 + C
        z_r_new = z_r * z_r - z_i * z_i + c_re
        z_i_new = 2 * z_r * z_i + c_im
        z_r = z_r_new
        z_i = z_i_new

        mag_sq = z_r * z_r + z_i * z_i

        # Store as float64 (sufficient for the delta kernel)
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
