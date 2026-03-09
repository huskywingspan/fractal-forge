"""Burning Ship fractal computation kernels.

Provides both GPU (CUDA) and CPU (Numba JIT) kernels.
The Burning Ship takes absolute values of z_re and z_im before squaring.
"""

import math

import numpy as np
from numba import cuda, njit, prange

from fractalforge.engine.mandelbrot import CUDA_AVAILABLE


# --- GPU kernel ---

@cuda.jit
def _burning_ship_cuda(min_re, min_im, step_re, step_im, max_iter, smooth_out):
    """CUDA kernel: one thread per pixel."""
    x, y = cuda.grid(2)
    height, width = smooth_out.shape

    if x >= height or y >= width:
        return

    c_re = min_re + y * step_re
    c_im = -(min_im + x * step_im)

    z_re = 0.0
    z_im = 0.0
    iteration = 0
    escape_radius_sq = 256.0

    while iteration < max_iter:
        z_re = abs(z_re)
        z_im = abs(z_im)

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
def _burning_ship_cpu(min_re, min_im, step_re, step_im, max_iter, height, width):
    """CPU kernel using Numba JIT with parallel loop."""
    smooth_out = np.empty((height, width), dtype=np.float64)

    for x in prange(height):
        for y in range(width):
            c_re = min_re + y * step_re
            c_im = -(min_im + x * step_im)

            z_re = 0.0
            z_im = 0.0
            iteration = 0
            escape_radius_sq = 256.0

            while iteration < max_iter:
                z_re = abs(z_re)
                z_im = abs(z_im)

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


# --- Public API ---

def render_frame_burning_ship(
    center_re: float = -0.5,
    center_im: float = -0.5,
    zoom: float = 1.0,
    width: int = 1920,
    height: int = 1080,
    max_iter: int = 1000,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Render a single Burning Ship frame and return smooth iteration counts.

    Args:
        center_re: Real part of the center coordinate.
        center_im: Imaginary part of the center coordinate.
        zoom: Zoom level (1.0 = full view, higher = more zoomed in).
        width: Frame width in pixels.
        height: Frame height in pixels.
        max_iter: Maximum iterations before considering a point interior.
        use_gpu: Force GPU (True), CPU (False), or auto-detect (None).

    Returns:
        2D numpy array (height x width) of smooth iteration counts.
        Interior points have value -1.0.
    """
    aspect = width / height
    view_height = 3.0 / zoom
    view_width = view_height * aspect

    min_re = center_re - view_width / 2.0
    min_im = center_im - view_height / 2.0
    step_re = view_width / width
    step_im = view_height / height

    gpu = use_gpu if use_gpu is not None else CUDA_AVAILABLE

    if gpu:
        return _render_gpu_burning_ship(
            min_re, min_im, step_re, step_im, max_iter, height, width
        )
    else:
        return _burning_ship_cpu(min_re, min_im, step_re, step_im, max_iter, height, width)


def _render_gpu_burning_ship(min_re, min_im, step_re, step_im, max_iter, height, width):
    """Dispatch to CUDA kernel."""
    smooth_out = np.zeros((height, width), dtype=np.float64)
    d_smooth = cuda.to_device(smooth_out)

    threads_per_block = (16, 16)
    blocks_x = (height + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    _burning_ship_cuda[blocks_per_grid, threads_per_block](
        min_re, min_im, step_re, step_im, max_iter, d_smooth
    )

    d_smooth.copy_to_host(smooth_out)
    return smooth_out
