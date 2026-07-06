"""Julia set computation kernels.

Provides both GPU (CUDA) and CPU (Numba JIT) kernels.
Same escape-time algorithm as Mandelbrot, but with fixed c and z_0 = pixel coord.

Deep zoom (perturbation): the standard kernel computes z_0 = center + offset
in absolute float64 coordinates, which collapses adjacent pixels into noise
once pixel spacing nears the float64 ulp (~1e12 zoom at preview sizes). The
perturbation path iterates a single reference orbit from Z_0 = center at
arbitrary precision, then tracks pixels as float64 deltas:

    z_0 = center + dz          ->  d_0 = dz
    d_{n+1} = 2*Z_n*d_n + d_n^2     (no +dc term: c is shared by all pixels)

Julia perturbation is simpler than Mandelbrot's: c needs no per-pixel
precision (it is exactly representable in float64), so when the reference
escapes early the standard-iteration continuation is numerically sound.
v1 has no rebasing / floatexp: practical ceiling is roughly 1e30, far past
the ~1e12 where the standard kernel dissolves.
"""

import math

import numpy as np
from numba import cuda, njit, prange

from fractalforge.engine.mandelbrot import CUDA_AVAILABLE
from fractalforge.engine.precision import required_precision

try:
    import gmpy2
    _HAS_GMPY2 = True
except ImportError:
    _HAS_GMPY2 = False


# --- GPU kernel ---

@cuda.jit
def _julia_cuda(c_re, c_im, min_re, min_im, step_re, step_im, max_iter, smooth_out):
    """CUDA kernel: one thread per pixel."""
    x, y = cuda.grid(2)
    height, width = smooth_out.shape

    if x >= height or y >= width:
        return

    z_re = min_re + y * step_re
    z_im = min_im + x * step_im

    iteration = 0
    escape_radius_sq = 256.0

    while iteration < max_iter:
        z_re_sq = z_re * z_re
        z_im_sq = z_im * z_im

        if z_re_sq + z_im_sq > escape_radius_sq:
            break

        z_im = 2.0 * z_re * z_im + c_im
        z_re = z_re_sq - z_im_sq + c_re
        iteration += 1

    if iteration < max_iter:
        log_zn = 0.5 * math.log(z_re * z_re + z_im * z_im)
        nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
        smooth_out[x, y] = float(iteration) + 1.0 - nu
    else:
        smooth_out[x, y] = -1.0


# --- CPU fallback kernel ---

@njit(parallel=True, cache=True)
def _julia_cpu(c_re, c_im, min_re, min_im, step_re, step_im, max_iter, height, width):
    """CPU kernel using Numba JIT with parallel loop."""
    smooth_out = np.empty((height, width), dtype=np.float64)

    for x in prange(height):
        for y in range(width):
            z_re = min_re + y * step_re
            z_im = min_im + x * step_im

            iteration = 0
            escape_radius_sq = 256.0

            while iteration < max_iter:
                z_re_sq = z_re * z_re
                z_im_sq = z_im * z_im

                if z_re_sq + z_im_sq > escape_radius_sq:
                    break

                z_im = 2.0 * z_re * z_im + c_im
                z_re = z_re_sq - z_im_sq + c_re
                iteration += 1

            if iteration < max_iter:
                log_zn = 0.5 * math.log(z_re * z_re + z_im * z_im)
                nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                smooth_out[x, y] = float(iteration) + 1.0 - nu
            else:
                smooth_out[x, y] = -1.0

    return smooth_out


# --- Perturbation kernels (deep zoom) ---

@cuda.jit
def _julia_pt_cuda(
    z_re, z_im, ref_num_iters,
    c_re, c_im,
    min_d_re, min_d_im, step_re, step_im,
    max_iter, smooth_out,
):
    """CUDA kernel: Julia perturbation, one thread per pixel.

    d is the pixel's offset from the reference orbit (starts as the pixel's
    offset from the view center). When the reference orbit ends (escaped or
    exhausted), reconstruct the full value and continue standard iteration —
    exact here because c is a plain float64 shared by all pixels.
    """
    x, y = cuda.grid(2)
    height, width = smooth_out.shape
    if x >= height or y >= width:
        return

    d_re = min_d_re + y * step_re
    d_im = min_d_im + x * step_im

    escape_radius_sq = 256.0
    log2 = math.log(2.0)
    iteration = 0

    while iteration < ref_num_iters and iteration < max_iter:
        zn_re = z_re[iteration]
        zn_im = z_im[iteration]
        full_re = zn_re + d_re
        full_im = zn_im + d_im
        full_mag_sq = full_re * full_re + full_im * full_im

        if full_mag_sq > escape_radius_sq:
            log_zn = 0.5 * math.log(full_mag_sq)
            nu = math.log(log_zn / log2) / log2
            smooth_out[x, y] = float(iteration) + 1.0 - nu
            return

        # d' = 2*Z_n*d + d^2   (no dc term)
        d_re_new = 2.0 * (zn_re * d_re - zn_im * d_im) + (d_re * d_re - d_im * d_im)
        d_im_new = 2.0 * (zn_re * d_im + zn_im * d_re) + 2.0 * d_re * d_im
        d_re = d_re_new
        d_im = d_im_new
        iteration += 1

    # Reference ended: continue standard iteration from the full value.
    z_r = z_re[ref_num_iters] + d_re
    z_i = z_im[ref_num_iters] + d_im

    while iteration < max_iter:
        z_r_sq = z_r * z_r
        z_i_sq = z_i * z_i
        if z_r_sq + z_i_sq > escape_radius_sq:
            log_zn = 0.5 * math.log(z_r_sq + z_i_sq)
            nu = math.log(log_zn / log2) / log2
            smooth_out[x, y] = float(iteration) + 1.0 - nu
            return
        z_i = 2.0 * z_r * z_i + c_im
        z_r = z_r_sq - z_i_sq + c_re
        iteration += 1

    smooth_out[x, y] = -1.0


@njit(parallel=True, cache=True)
def _julia_pt_cpu(
    z_re, z_im, ref_num_iters,
    c_re, c_im,
    min_d_re, min_d_im, step_re, step_im,
    max_iter, height, width,
):
    """CPU kernel: Julia perturbation with Numba parallel."""
    smooth_out = np.empty((height, width), dtype=np.float64)
    escape_radius_sq = 256.0
    log2 = math.log(2.0)

    for x in prange(height):
        for y in range(width):
            d_re = min_d_re + y * step_re
            d_im = min_d_im + x * step_im
            iteration = 0
            resolved = False

            while iteration < ref_num_iters and iteration < max_iter:
                zn_re = z_re[iteration]
                zn_im = z_im[iteration]
                full_re = zn_re + d_re
                full_im = zn_im + d_im
                full_mag_sq = full_re * full_re + full_im * full_im

                if full_mag_sq > escape_radius_sq:
                    log_zn = 0.5 * math.log(full_mag_sq)
                    nu = math.log(log_zn / log2) / log2
                    smooth_out[x, y] = float(iteration) + 1.0 - nu
                    resolved = True
                    break

                d_re_new = (2.0 * (zn_re * d_re - zn_im * d_im)
                            + (d_re * d_re - d_im * d_im))
                d_im_new = (2.0 * (zn_re * d_im + zn_im * d_re)
                            + 2.0 * d_re * d_im)
                d_re = d_re_new
                d_im = d_im_new
                iteration += 1

            if resolved:
                continue

            z_r = z_re[ref_num_iters] + d_re
            z_i = z_im[ref_num_iters] + d_im

            escaped = False
            while iteration < max_iter:
                z_r_sq = z_r * z_r
                z_i_sq = z_i * z_i
                if z_r_sq + z_i_sq > escape_radius_sq:
                    log_zn = 0.5 * math.log(z_r_sq + z_i_sq)
                    nu = math.log(log_zn / log2) / log2
                    smooth_out[x, y] = float(iteration) + 1.0 - nu
                    escaped = True
                    break
                z_i = 2.0 * z_r * z_i + c_im
                z_r = z_r_sq - z_i_sq + c_re
                iteration += 1

            if not escaped:
                smooth_out[x, y] = -1.0

    return smooth_out


# --- Julia reference orbit (arbitrary precision, cached) ---

# Keyed on (center_re, center_im, c_re, c_im); sufficiency-checked like the
# Mandelbrot orbit cache so click-zooming into a fixed point reuses it.
_JULIA_ORBIT_CACHE: dict = {}
_JULIA_ORBIT_CACHE_MAX = 4


def _julia_reference_orbit(
    center_re_str: str, center_im_str: str,
    c_re: float, c_im: float,
    max_iter: int, precision: int,
):
    """Iterate Z_{n+1} = Z_n^2 + c from Z_0 = center at high precision.

    Returns (z_re_arr, z_im_arr, num_iters, escaped, precision) with arrays
    of length num_iters + 1 (float64 downcast for the GPU).
    """
    key = (center_re_str, center_im_str, repr(c_re), repr(c_im))
    cached = _JULIA_ORBIT_CACHE.get(key)
    if cached is not None:
        c_prec, c_num, c_escaped = cached[4], cached[2], cached[3]
        if c_prec >= precision and (c_escaped or c_num >= max_iter):
            _JULIA_ORBIT_CACHE.pop(key, None)
            _JULIA_ORBIT_CACHE[key] = cached
            return cached

    z_re_arr = np.empty(max_iter + 1, dtype=np.float64)
    z_im_arr = np.empty(max_iter + 1, dtype=np.float64)

    if _HAS_GMPY2:
        bits = int(precision * 3.3219) + 20
        gmpy2.get_context().precision = bits
        z_r = gmpy2.mpfr(center_re_str)
        z_i = gmpy2.mpfr(center_im_str)
        cr = gmpy2.mpfr(c_re)
        ci = gmpy2.mpfr(c_im)
        esc = gmpy2.mpfr(256.0)
    else:
        import mpmath
        mpmath.mp.dps = precision
        z_r = mpmath.mpf(center_re_str)
        z_i = mpmath.mpf(center_im_str)
        cr = mpmath.mpf(c_re)
        ci = mpmath.mpf(c_im)
        esc = mpmath.mpf(256.0)

    z_re_arr[0] = float(z_r)
    z_im_arr[0] = float(z_i)
    escaped = False
    n = 0
    for n in range(1, max_iter + 1):
        z_r_new = z_r * z_r - z_i * z_i + cr
        z_i_new = 2 * z_r * z_i + ci
        z_r = z_r_new
        z_i = z_i_new
        z_re_arr[n] = float(z_r)
        z_im_arr[n] = float(z_i)
        if z_r * z_r + z_i * z_i > esc:
            escaped = True
            break

    num_iters = n
    result = (z_re_arr[: num_iters + 1], z_im_arr[: num_iters + 1],
              num_iters, escaped, precision)

    _JULIA_ORBIT_CACHE.pop(key, None)
    _JULIA_ORBIT_CACHE[key] = result
    while len(_JULIA_ORBIT_CACHE) > _JULIA_ORBIT_CACHE_MAX:
        _JULIA_ORBIT_CACHE.pop(next(iter(_JULIA_ORBIT_CACHE)))
    return result


def _render_julia_pt(
    c_re, c_im, center_re_str, center_im_str, zoom,
    width, height, max_iter, gpu,
):
    """Render a deep Julia frame via perturbation."""
    precision = required_precision(zoom, center_re_str, center_im_str)
    z_re, z_im, num_iters, escaped, _ = _julia_reference_orbit(
        center_re_str, center_im_str, c_re, c_im, max_iter, precision)

    aspect = width / height
    view_height = 3.0 / zoom
    view_width = view_height * aspect
    # Pixel deltas RELATIVE to the center: tiny values near zero, which
    # float64 resolves exactly (unlike absolute coordinates).
    min_d_re = -view_width / 2.0
    min_d_im = -view_height / 2.0
    step_re = view_width / width
    step_im = view_height / height

    if gpu:
        smooth_out = np.zeros((height, width), dtype=np.float64)
        d_smooth = cuda.to_device(smooth_out)
        d_z_re = cuda.to_device(z_re)
        d_z_im = cuda.to_device(z_im)
        threads = (16, 16)
        blocks = ((height + 15) // 16, (width + 15) // 16)
        _julia_pt_cuda[blocks, threads](
            d_z_re, d_z_im, num_iters, c_re, c_im,
            min_d_re, min_d_im, step_re, step_im, max_iter, d_smooth)
        d_smooth.copy_to_host(smooth_out)
        return smooth_out
    return _julia_pt_cpu(
        z_re, z_im, num_iters, c_re, c_im,
        min_d_re, min_d_im, step_re, step_im, max_iter, height, width)


# --- Public API ---

def render_frame_julia(
    c_re: float,
    c_im: float,
    center_re: float | str = 0.0,
    center_im: float | str = 0.0,
    zoom: float = 1.0,
    width: int = 1920,
    height: int = 1080,
    max_iter: int = 1000,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Render a single Julia set frame and return smooth iteration counts.

    Args:
        c_re: Real part of the Julia parameter (fixed c).
        c_im: Imaginary part of the Julia parameter (fixed c).
        center_re: Real part of the viewport center in the z-plane.
        center_im: Imaginary part of the viewport center in the z-plane.
        zoom: Zoom level (1.0 = full view, higher = more zoomed in).
        width: Frame width in pixels.
        height: Frame height in pixels.
        max_iter: Maximum iterations before considering a point interior.
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None).

    Returns:
        2D numpy array (height x width) of smooth iteration counts.
        Interior points have value -1.0.

    Center coordinates may be passed as strings to preserve precision; the
    perturbation path engages automatically when the standard float64 engine
    can no longer resolve pixel spacing at this resolution.
    """
    gpu = use_gpu if use_gpu is not None else CUDA_AVAILABLE

    from fractalforge.render.frame_renderer import needs_perturbation
    zoom = float(zoom)
    if needs_perturbation(zoom, height):
        return _render_julia_pt(
            float(c_re), float(c_im), str(center_re), str(center_im),
            zoom, width, height, max_iter, gpu,
        )

    center_re = float(center_re)
    center_im = float(center_im)

    aspect = width / height
    view_height = 3.0 / zoom
    view_width = view_height * aspect

    min_re = center_re - view_width / 2.0
    min_im = center_im - view_height / 2.0
    step_re = view_width / width
    step_im = view_height / height

    if gpu:
        return _render_gpu_julia(
            c_re, c_im, min_re, min_im, step_re, step_im, max_iter, height, width
        )
    else:
        return _julia_cpu(
            c_re, c_im, min_re, min_im, step_re, step_im, max_iter, height, width
        )


def _render_gpu_julia(c_re, c_im, min_re, min_im, step_re, step_im, max_iter, height, width):
    """Dispatch to CUDA kernel."""
    smooth_out = np.zeros((height, width), dtype=np.float64)
    d_smooth = cuda.to_device(smooth_out)

    threads_per_block = (16, 16)
    blocks_x = (height + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    _julia_cuda[blocks_per_grid, threads_per_block](
        c_re, c_im, min_re, min_im, step_re, step_im, max_iter, d_smooth
    )

    d_smooth.copy_to_host(smooth_out)
    return smooth_out
