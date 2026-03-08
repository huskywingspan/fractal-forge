"""Mandelbrot set computation kernels.

Standard double-precision kernel for zooms up to ~1e15.
For deeper zooms, see perturbation.py.
"""

from numba import cuda
import numpy as np


@cuda.jit
def mandelbrot_kernel(
    min_re: float,
    min_im: float,
    step_re: float,
    step_im: float,
    max_iter: int,
    smooth_out: np.ndarray,
):
    """Compute smooth iteration counts for a Mandelbrot frame.

    Each GPU thread computes one pixel. The output is a smooth (continuous)
    iteration count suitable for gradient-based coloring.

    Args:
        min_re: Real coordinate of the top-left corner.
        min_im: Imaginary coordinate of the top-left corner.
        step_re: Real step per pixel.
        step_im: Imaginary step per pixel.
        max_iter: Maximum iteration count (bailout).
        smooth_out: 2D output array (height × width) for smooth iteration values.
    """
    x, y = cuda.grid(2)
    height, width = smooth_out.shape

    if x >= height or y >= width:
        return

    # Map pixel to complex plane
    c_re = min_re + y * step_re
    c_im = min_im + x * step_im

    # Iterate z = z² + c
    z_re = 0.0
    z_im = 0.0
    iteration = 0
    escape_radius_sq = 256.0  # Large radius for smoother coloring

    while iteration < max_iter:
        z_re_sq = z_re * z_re
        z_im_sq = z_im * z_im

        if z_re_sq + z_im_sq > escape_radius_sq:
            break

        z_im = 2.0 * z_re * z_im + c_im
        z_re = z_re_sq - z_im_sq + c_re
        iteration += 1

    # Smooth iteration count (continuous / normalized)
    if iteration < max_iter:
        # Smooth escape time: subtract fractional part based on final |z|
        log_zn = 0.5 * np.log(z_re * z_re + z_im * z_im)
        nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
        smooth_out[x, y] = float(iteration) + 1.0 - nu
    else:
        # Interior point — mark as -1 for special coloring
        smooth_out[x, y] = -1.0


def render_frame(
    center_re: float,
    center_im: float,
    zoom: float,
    width: int,
    height: int,
    max_iter: int = 1000,
) -> np.ndarray:
    """Render a single Mandelbrot frame and return smooth iteration counts.

    Args:
        center_re: Real part of the center coordinate.
        center_im: Imaginary part of the center coordinate.
        zoom: Zoom level (1.0 = full view, higher = more zoomed in).
        width: Frame width in pixels.
        height: Frame height in pixels.
        max_iter: Maximum iterations before considering a point interior.

    Returns:
        2D numpy array (height × width) of smooth iteration counts.
        Interior points have value -1.0.
    """
    # Compute viewport bounds
    aspect = width / height
    view_height = 3.0 / zoom  # At zoom=1, we see roughly (-1.5, 1.5) in imaginary
    view_width = view_height * aspect

    min_re = center_re - view_width / 2.0
    min_im = center_im - view_height / 2.0
    step_re = view_width / width
    step_im = view_height / height

    # Allocate output on device
    smooth_out = np.zeros((height, width), dtype=np.float64)
    d_smooth = cuda.to_device(smooth_out)

    # Launch kernel
    threads_per_block = (16, 16)
    blocks_x = (height + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    mandelbrot_kernel[blocks_per_grid, threads_per_block](
        min_re, min_im, step_re, step_im, max_iter, d_smooth
    )

    # Copy result back
    d_smooth.copy_to_host(smooth_out)
    return smooth_out
