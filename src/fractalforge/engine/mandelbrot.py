"""Mandelbrot set computation kernels.

Provides both GPU (CUDA) and CPU (Numba JIT) kernels.
Standard double-precision for zooms up to ~1e15.
For deeper zooms, see perturbation.py.

Supports optional distance estimation: tracks dz/dc alongside z to compute
the distance from each pixel to the Mandelbrot boundary. This gives
completely smooth coloring (no integer banding) and analytic surface normals
for high-quality slope shading.
"""

import math

import numpy as np
from numba import cuda, njit, prange


# --- GPU kernel (standard) ---

@cuda.jit
def _mandelbrot_cuda(min_re, min_im, step_re, step_im, max_iter, smooth_out):
    """CUDA kernel: one thread per pixel."""
    x, y = cuda.grid(2)
    height, width = smooth_out.shape

    if x >= height or y >= width:
        return

    c_re = min_re + y * step_re
    c_im = min_im + x * step_im

    z_re = 0.0
    z_im = 0.0
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


# --- GPU kernel (with distance estimation) ---

@cuda.jit
def _mandelbrot_de_cuda(
    min_re, min_im, step_re, step_im, max_iter, smooth_out, dist_out
):
    """CUDA kernel with distance estimation: tracks dz/dc per pixel."""
    x, y = cuda.grid(2)
    height, width = smooth_out.shape

    if x >= height or y >= width:
        return

    c_re = min_re + y * step_re
    c_im = min_im + x * step_im

    z_re = 0.0
    z_im = 0.0
    # Derivative dz/dc, initialized to 0
    dz_re = 0.0
    dz_im = 0.0
    iteration = 0
    escape_radius_sq = 256.0

    while iteration < max_iter:
        z_re_sq = z_re * z_re
        z_im_sq = z_im * z_im

        if z_re_sq + z_im_sq > escape_radius_sq:
            break

        # dz' = 2 * z * dz + 1
        new_dz_re = 2.0 * (z_re * dz_re - z_im * dz_im) + 1.0
        new_dz_im = 2.0 * (z_re * dz_im + z_im * dz_re)
        dz_re = new_dz_re
        dz_im = new_dz_im

        # z' = z^2 + c
        z_im = 2.0 * z_re * z_im + c_im
        z_re = z_re_sq - z_im_sq + c_re
        iteration += 1

    if iteration < max_iter:
        log_zn = 0.5 * math.log(z_re * z_re + z_im * z_im)
        nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
        smooth_out[x, y] = float(iteration) + 1.0 - nu

        # Distance estimate: d = |z| * log|z| / |z'|
        z_mag = math.sqrt(z_re * z_re + z_im * z_im)
        dz_mag = math.sqrt(dz_re * dz_re + dz_im * dz_im)
        if dz_mag > 0.0:
            dist_out[x, y] = z_mag * math.log(z_mag) / dz_mag
        else:
            dist_out[x, y] = 0.0
    else:
        smooth_out[x, y] = -1.0
        dist_out[x, y] = 0.0


# --- CPU fallback kernel ---

@njit(parallel=True, cache=True)
def _mandelbrot_cpu(min_re, min_im, step_re, step_im, max_iter, height, width):
    """CPU kernel using Numba JIT with parallel loop."""
    smooth_out = np.empty((height, width), dtype=np.float64)

    for x in prange(height):
        for y in range(width):
            c_re = min_re + y * step_re
            c_im = min_im + x * step_im

            z_re = 0.0
            z_im = 0.0
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


# --- CPU kernel (with distance estimation) ---

@njit(parallel=True, cache=True)
def _mandelbrot_de_cpu(min_re, min_im, step_re, step_im, max_iter, height, width):
    """CPU kernel with distance estimation."""
    smooth_out = np.empty((height, width), dtype=np.float64)
    dist_out = np.empty((height, width), dtype=np.float64)

    for x in prange(height):
        for y in range(width):
            c_re = min_re + y * step_re
            c_im = min_im + x * step_im

            z_re = 0.0
            z_im = 0.0
            dz_re = 0.0
            dz_im = 0.0
            iteration = 0
            escape_radius_sq = 256.0

            while iteration < max_iter:
                z_re_sq = z_re * z_re
                z_im_sq = z_im * z_im

                if z_re_sq + z_im_sq > escape_radius_sq:
                    break

                new_dz_re = 2.0 * (z_re * dz_re - z_im * dz_im) + 1.0
                new_dz_im = 2.0 * (z_re * dz_im + z_im * dz_re)
                dz_re = new_dz_re
                dz_im = new_dz_im

                z_im = 2.0 * z_re * z_im + c_im
                z_re = z_re_sq - z_im_sq + c_re
                iteration += 1

            if iteration < max_iter:
                log_zn = 0.5 * math.log(z_re * z_re + z_im * z_im)
                nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                smooth_out[x, y] = float(iteration) + 1.0 - nu

                z_mag = math.sqrt(z_re * z_re + z_im * z_im)
                dz_mag = math.sqrt(dz_re * dz_re + dz_im * dz_im)
                if dz_mag > 0.0:
                    dist_out[x, y] = z_mag * math.log(z_mag) / dz_mag
                else:
                    dist_out[x, y] = 0.0
            else:
                smooth_out[x, y] = -1.0
                dist_out[x, y] = 0.0

    return smooth_out, dist_out


# --- Auto-detect backend ---

def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        return cuda.is_available()
    except Exception:
        return False


CUDA_AVAILABLE = _has_cuda()


def render_frame(
    center_re: float,
    center_im: float,
    zoom: float,
    width: int,
    height: int,
    max_iter: int = 1000,
    use_gpu: bool | None = None,
    distance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Render a single Mandelbrot frame and return smooth iteration counts.

    Args:
        center_re: Real part of the center coordinate.
        center_im: Imaginary part of the center coordinate.
        zoom: Zoom level (1.0 = full view, higher = more zoomed in).
        width: Frame width in pixels.
        height: Frame height in pixels.
        max_iter: Maximum iterations before considering a point interior.
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None).
        distance: If True, also return distance estimation array.

    Returns:
        If distance=False: 2D array of smooth iteration counts (-1.0 for interior).
        If distance=True: tuple of (smooth_data, distance_data).
    """
    aspect = width / height
    view_height = 3.0 / zoom
    view_width = view_height * aspect

    min_re = center_re - view_width / 2.0
    min_im = center_im - view_height / 2.0
    step_re = view_width / width
    step_im = view_height / height

    gpu = use_gpu if use_gpu is not None else CUDA_AVAILABLE

    if distance:
        if gpu:
            return _render_gpu_de(
                min_re, min_im, step_re, step_im, max_iter, height, width
            )
        else:
            return _mandelbrot_de_cpu(
                min_re, min_im, step_re, step_im, max_iter, height, width
            )
    else:
        if gpu:
            return _render_gpu(
                min_re, min_im, step_re, step_im, max_iter, height, width
            )
        else:
            return _mandelbrot_cpu(
                min_re, min_im, step_re, step_im, max_iter, height, width
            )


def _render_gpu(min_re, min_im, step_re, step_im, max_iter, height, width):
    """Dispatch to CUDA kernel."""
    smooth_out = np.zeros((height, width), dtype=np.float64)
    d_smooth = cuda.to_device(smooth_out)

    threads_per_block = (16, 16)
    blocks_x = (height + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    _mandelbrot_cuda[blocks_per_grid, threads_per_block](
        min_re, min_im, step_re, step_im, max_iter, d_smooth
    )

    d_smooth.copy_to_host(smooth_out)
    return smooth_out


def _render_gpu_de(min_re, min_im, step_re, step_im, max_iter, height, width):
    """Dispatch to CUDA kernel with distance estimation."""
    smooth_out = np.zeros((height, width), dtype=np.float64)
    dist_out = np.zeros((height, width), dtype=np.float64)
    d_smooth = cuda.to_device(smooth_out)
    d_dist = cuda.to_device(dist_out)

    threads_per_block = (16, 16)
    blocks_x = (height + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    _mandelbrot_de_cuda[blocks_per_grid, threads_per_block](
        min_re, min_im, step_re, step_im, max_iter, d_smooth, d_dist
    )

    d_smooth.copy_to_host(smooth_out)
    d_dist.copy_to_host(dist_out)
    return smooth_out, dist_out
