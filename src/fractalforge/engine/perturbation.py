"""Perturbation theory engine for deep Mandelbrot zooms.

At zoom depths beyond ~1e15, standard float64 arithmetic loses precision.
Perturbation theory solves this by:
1. Computing one reference orbit at arbitrary precision (CPU, mpmath) -- see precision.py
2. Expressing all other pixels as small deltas from that reference (GPU, float64)

The delta iteration formula:
    d_{n+1} = 2 * Z_n * d_n + d_n^2 + dc
where Z_n is the reference orbit and dc = c - C (pixel offset from reference).

Components:
  P3-02: Delta iteration kernel (CUDA + CPU)
  P3-03: Glitch detection and correction
  P3-04: Series approximation for iteration skipping
  P3-05: Rebasing logic for delta orbit divergence
"""

import math
import time

import numpy as np
from numba import cuda, njit, prange

from fractalforge.engine.precision import compute_reference_orbit, ReferenceOrbit


# ============================================================================
# P3-04: Series Approximation (CPU pre-computation)
# ============================================================================

@njit(cache=True)
def _compute_series_approximation(
    z_re, z_im, num_iters, dc_max_mag_sq, tolerance
):
    """Compute series approximation coefficients A, B, C.

    A_{n+1} = 2*Z_n*A_n + 1
    B_{n+1} = 2*Z_n*B_n + A_n^2
    C_{n+1} = 2*Z_n*C_n + 2*A_n*B_n

    Then d_n ~ A_n*dc + B_n*dc^2 + C_n*dc^3.
    We skip while |C_n * dc_max^3| < tolerance.

    Returns (skip_iters, A_re, A_im, B_re, B_im, C_re, C_im) at the skip point.
    """
    dc_max_mag = math.sqrt(dc_max_mag_sq)
    dc_max_cubed = dc_max_mag * dc_max_mag * dc_max_mag

    # A_0 = 0, B_0 = 0, C_0 = 0
    a_re = 0.0
    a_im = 0.0
    b_re = 0.0
    b_im = 0.0
    c_re = 0.0
    c_im = 0.0

    skip_iters = 0

    for n in range(num_iters):
        zn_re = z_re[n]
        zn_im = z_im[n]

        # A_{n+1} = 2*Z_n*A_n + 1
        # (2*Z_n*A_n) = 2*(zn_re*a_re - zn_im*a_im, zn_re*a_im + zn_im*a_re)
        new_a_re = 2.0 * (zn_re * a_re - zn_im * a_im) + 1.0
        new_a_im = 2.0 * (zn_re * a_im + zn_im * a_re)

        # B_{n+1} = 2*Z_n*B_n + A_n^2
        # A_n^2 = (a_re^2 - a_im^2, 2*a_re*a_im)
        a_sq_re = a_re * a_re - a_im * a_im
        a_sq_im = 2.0 * a_re * a_im
        new_b_re = 2.0 * (zn_re * b_re - zn_im * b_im) + a_sq_re
        new_b_im = 2.0 * (zn_re * b_im + zn_im * b_re) + a_sq_im

        # C_{n+1} = 2*Z_n*C_n + 2*A_n*B_n
        ab_re = a_re * b_re - a_im * b_im
        ab_im = a_re * b_im + a_im * b_re
        new_c_re = 2.0 * (zn_re * c_re - zn_im * c_im) + 2.0 * ab_re
        new_c_im = 2.0 * (zn_re * c_im + zn_im * c_re) + 2.0 * ab_im

        a_re, a_im = new_a_re, new_a_im
        b_re, b_im = new_b_re, new_b_im
        c_re, c_im = new_c_re, new_c_im

        # Check if series approximation is still valid:
        # |C_n * dc_max^3| < tolerance
        c_mag = math.sqrt(c_re * c_re + c_im * c_im)
        if c_mag * dc_max_cubed > tolerance:
            break

        skip_iters = n + 1

    return skip_iters, a_re, a_im, b_re, b_im, c_re, c_im


# ============================================================================
# P3-02: Delta Iteration Kernel (CUDA)
# ============================================================================

@cuda.jit
def _perturbation_cuda(
    z_re, z_im, z_mag_sq,
    ref_num_iters, ref_escape_iter,
    min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    glitch_tolerance,
    smooth_out, glitch_out,
):
    """CUDA kernel: perturbation iteration, one thread per pixel.

    Each pixel computes d_n (delta from reference orbit Z_n):
        d_{n+1} = 2*Z_n*d_n + d_n^2 + dc

    With series approximation: d_{sa_skip} = A*dc + B*dc^2 + C*dc^3,
    then iterate from sa_skip onward.

    Rebasing: if reference escapes but pixel hasn't, switch to standard iteration.
    Glitch detection: flag when |d_n|^2 > tolerance * |Z_n|^2.
    """
    x, y = cuda.grid(2)
    height, width = smooth_out.shape

    if x >= height or y >= width:
        return

    # Compute dc for this pixel
    dc_re = min_dc_re + y * step_dc_re
    dc_im = min_dc_im + x * step_dc_im

    escape_radius_sq = 256.0

    # Initialize delta from series approximation
    if sa_skip > 0:
        # d_n ~ A_n*dc + B_n*dc^2 + C_n*dc^3
        # dc^2 = (dc_re + i*dc_im)^2
        dc_sq_re = dc_re * dc_re - dc_im * dc_im
        dc_sq_im = 2.0 * dc_re * dc_im
        # dc^3 = dc * dc^2
        dc_cu_re = dc_re * dc_sq_re - dc_im * dc_sq_im
        dc_cu_im = dc_re * dc_sq_im + dc_im * dc_sq_re

        d_re = (sa_a_re * dc_re - sa_a_im * dc_im
                + sa_b_re * dc_sq_re - sa_b_im * dc_sq_im
                + sa_c_re * dc_cu_re - sa_c_im * dc_cu_im)
        d_im = (sa_a_re * dc_im + sa_a_im * dc_re
                + sa_b_re * dc_sq_im + sa_b_im * dc_sq_re
                + sa_c_re * dc_cu_im + sa_c_im * dc_cu_re)
        iteration = sa_skip
    else:
        # d_0 = dc (no SA skip)
        d_re = dc_re
        d_im = dc_im
        iteration = 1  # start from iteration 1 (Z_0=0, d_0=dc gives Z_1+d_1)

    glitch_out[x, y] = 0

    # Phase 1: Perturbation iteration (while reference orbit is valid)
    while iteration < max_iter and iteration <= ref_num_iters:
        zn_re = z_re[iteration]
        zn_im = z_im[iteration]

        # Full z = Z_n + d_n
        full_re = zn_re + d_re
        full_im = zn_im + d_im
        full_mag_sq = full_re * full_re + full_im * full_im

        # Escape check on full value
        if full_mag_sq > escape_radius_sq:
            # Smooth coloring
            log_zn = 0.5 * math.log(full_mag_sq)
            nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
            smooth_out[x, y] = float(iteration) + 1.0 - nu
            return

        # P3-03: Glitch detection
        d_mag_sq = d_re * d_re + d_im * d_im
        zn_mag_sq = z_mag_sq[iteration]
        if zn_mag_sq > 0.0 and d_mag_sq > glitch_tolerance * zn_mag_sq:
            glitch_out[x, y] = 1
            smooth_out[x, y] = -1.0
            return

        # Delta iteration: d_{n+1} = 2*Z_n*d_n + d_n^2 + dc
        d_re_new = 2.0 * (zn_re * d_re - zn_im * d_im) + (d_re * d_re - d_im * d_im) + dc_re
        d_im_new = 2.0 * (zn_re * d_im + zn_im * d_re) + 2.0 * d_re * d_im + dc_im
        d_re = d_re_new
        d_im = d_im_new

        iteration += 1

    # P3-05: Rebasing -- reference escaped but pixel hasn't
    # Compute full value and continue with standard iteration
    if iteration <= ref_num_iters:
        full_re = z_re[iteration] + d_re
        full_im = z_im[iteration] + d_im
    else:
        # Past reference orbit end, reconstruct full z
        full_re = z_re[ref_num_iters] + d_re
        full_im = z_im[ref_num_iters] + d_im

    # Compute actual c = C + dc (using last reference center + dc offset)
    # We approximate c from the full z value at the rebasing point
    # Actually: c_re = center_re + dc_re, c_im = center_im + dc_im
    # But we don't have center as float64 here with sufficient precision.
    # Instead, the pixel's c is: z[0]^2 + c should give z[1], but z[0]=0 so c=z[1].
    # For rebasing, we use: c_re = dc_re (since center is the reference,
    # and the reference Z_1 = C, so dc is the offset from C in complex plane).
    # Actually we need the actual c value. Since dc = c - C, and reference starts at Z_0=0,
    # we can recover c as a float64 approximation: c ~ Z_1 + dc (not correct).
    # The correct approach: c_re = (center float64) + dc_re, but we don't pass center.
    # Simplest correct: pass center coordinates to kernel too.
    # But for a CUDA kernel, we can use z_re[1] and z_im[1] since Z_1 = C for Mandelbrot.
    # Z_1 = Z_0^2 + C = 0 + C = C. So c_re = z_re[1] + dc_re (as float64 offset).
    c_re = z_re[1] + dc_re
    c_im = z_im[1] + dc_im

    z_r = full_re
    z_i = full_im

    while iteration < max_iter:
        z_r_sq = z_r * z_r
        z_i_sq = z_i * z_i

        if z_r_sq + z_i_sq > escape_radius_sq:
            log_zn = 0.5 * math.log(z_r_sq + z_i_sq)
            nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
            smooth_out[x, y] = float(iteration) + 1.0 - nu
            return

        z_i = 2.0 * z_r * z_i + c_im
        z_r = z_r_sq - z_i_sq + c_re
        iteration += 1

    # Interior point
    smooth_out[x, y] = -1.0


# ============================================================================
# P3-02: Delta Iteration Kernel (CPU)
# ============================================================================

@njit(parallel=True, cache=True)
def _perturbation_cpu(
    z_re, z_im, z_mag_sq,
    ref_num_iters, ref_escape_iter,
    min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter,
    height, width,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    glitch_tolerance,
):
    """CPU kernel: perturbation iteration with Numba parallel JIT.

    Same algorithm as CUDA kernel but uses prange for parallelism.
    Returns (smooth_out, glitch_out).
    """
    smooth_out = np.empty((height, width), dtype=np.float64)
    glitch_out = np.zeros((height, width), dtype=np.int32)
    escape_radius_sq = 256.0
    log2 = math.log(2.0)

    for x in prange(height):
        for y in range(width):
            dc_re = min_dc_re + y * step_dc_re
            dc_im = min_dc_im + x * step_dc_im

            # Initialize delta from series approximation
            if sa_skip > 0:
                dc_sq_re = dc_re * dc_re - dc_im * dc_im
                dc_sq_im = 2.0 * dc_re * dc_im
                dc_cu_re = dc_re * dc_sq_re - dc_im * dc_sq_im
                dc_cu_im = dc_re * dc_sq_im + dc_im * dc_sq_re

                d_re = (sa_a_re * dc_re - sa_a_im * dc_im
                        + sa_b_re * dc_sq_re - sa_b_im * dc_sq_im
                        + sa_c_re * dc_cu_re - sa_c_im * dc_cu_im)
                d_im = (sa_a_re * dc_im + sa_a_im * dc_re
                        + sa_b_re * dc_sq_im + sa_b_im * dc_sq_re
                        + sa_c_re * dc_cu_im + sa_c_im * dc_cu_re)
                iteration = sa_skip
            else:
                d_re = dc_re
                d_im = dc_im
                iteration = 1

            glitched = False
            escaped = False

            # Phase 1: Perturbation iteration
            while iteration < max_iter and iteration <= ref_num_iters:
                zn_re = z_re[iteration]
                zn_im = z_im[iteration]

                full_re = zn_re + d_re
                full_im = zn_im + d_im
                full_mag_sq = full_re * full_re + full_im * full_im

                if full_mag_sq > escape_radius_sq:
                    log_zn = 0.5 * math.log(full_mag_sq)
                    nu = math.log(log_zn / log2) / log2
                    smooth_out[x, y] = float(iteration) + 1.0 - nu
                    escaped = True
                    break

                # Glitch detection
                d_mag_sq = d_re * d_re + d_im * d_im
                zn_mag_sq = z_mag_sq[iteration]
                if zn_mag_sq > 0.0 and d_mag_sq > glitch_tolerance * zn_mag_sq:
                    glitch_out[x, y] = 1
                    smooth_out[x, y] = -1.0
                    glitched = True
                    break

                # Delta iteration
                d_re_new = 2.0 * (zn_re * d_re - zn_im * d_im) + (d_re * d_re - d_im * d_im) + dc_re
                d_im_new = 2.0 * (zn_re * d_im + zn_im * d_re) + 2.0 * d_re * d_im + dc_im
                d_re = d_re_new
                d_im = d_im_new

                iteration += 1

            if escaped or glitched:
                continue

            # P3-05: Rebasing
            if iteration <= ref_num_iters:
                full_re = z_re[iteration] + d_re
                full_im = z_im[iteration] + d_im
            else:
                full_re = z_re[ref_num_iters] + d_re
                full_im = z_im[ref_num_iters] + d_im

            c_re = z_re[1] + dc_re
            c_im = z_im[1] + dc_im

            z_r = full_re
            z_i = full_im

            standard_escaped = False
            while iteration < max_iter:
                z_r_sq = z_r * z_r
                z_i_sq = z_i * z_i

                if z_r_sq + z_i_sq > escape_radius_sq:
                    log_zn = 0.5 * math.log(z_r_sq + z_i_sq)
                    nu = math.log(log_zn / log2) / log2
                    smooth_out[x, y] = float(iteration) + 1.0 - nu
                    standard_escaped = True
                    break

                z_i = 2.0 * z_r * z_i + c_im
                z_r = z_r_sq - z_i_sq + c_re
                iteration += 1

            if not standard_escaped:
                smooth_out[x, y] = -1.0

    return smooth_out, glitch_out


# ============================================================================
# GPU dispatch
# ============================================================================

def _render_gpu_perturbation(
    z_re, z_im, z_mag_sq,
    ref_num_iters, ref_escape_iter,
    min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter, height, width,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    glitch_tolerance,
):
    """Dispatch perturbation iteration to CUDA kernel."""
    # Upload reference orbit to device
    d_z_re = cuda.to_device(z_re)
    d_z_im = cuda.to_device(z_im)
    d_z_mag_sq = cuda.to_device(z_mag_sq)

    # Allocate output arrays on device
    smooth_out = np.zeros((height, width), dtype=np.float64)
    glitch_out = np.zeros((height, width), dtype=np.int32)
    d_smooth = cuda.to_device(smooth_out)
    d_glitch = cuda.to_device(glitch_out)

    threads_per_block = (16, 16)
    blocks_x = (height + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    _perturbation_cuda[blocks_per_grid, threads_per_block](
        d_z_re, d_z_im, d_z_mag_sq,
        ref_num_iters, ref_escape_iter,
        min_dc_re, min_dc_im, step_dc_re, step_dc_im,
        max_iter,
        sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
        glitch_tolerance,
        d_smooth, d_glitch,
    )

    d_smooth.copy_to_host(smooth_out)
    d_glitch.copy_to_host(glitch_out)
    return smooth_out, glitch_out


# ============================================================================
# Auto-detect CUDA
# ============================================================================

def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        return cuda.is_available()
    except Exception:
        return False


CUDA_AVAILABLE = _has_cuda()


# ============================================================================
# P3-03: Glitch Correction
# ============================================================================

def _find_glitch_reference(
    smooth_out, glitch_out, min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    center_re_str, center_im_str, width,
):
    """Find a glitched pixel and compute its full-precision coordinate.

    Returns (re_str, im_str) for the new reference point, or None if no glitches.
    """
    import mpmath

    glitch_coords = np.argwhere(glitch_out > 0)
    if len(glitch_coords) == 0:
        return None

    # Pick the glitched pixel closest to center (most representative)
    center_y = glitch_out.shape[0] // 2
    center_x = glitch_out.shape[1] // 2
    distances = (glitch_coords[:, 0] - center_y) ** 2 + (glitch_coords[:, 1] - center_x) ** 2
    best_idx = np.argmin(distances)
    px_row, px_col = glitch_coords[best_idx]

    # Compute the dc offset for this pixel
    dc_re = min_dc_re + px_col * step_dc_re
    dc_im = min_dc_im + px_row * step_dc_im

    # Compute full-precision coordinate: C_new = C_ref + dc
    # Use mpmath for the addition to preserve precision
    prec = max(50, len(center_re_str))
    mpmath.mp.dps = prec
    new_re = mpmath.mpf(center_re_str) + mpmath.mpf(dc_re)
    new_im = mpmath.mpf(center_im_str) + mpmath.mpf(dc_im)

    return (
        mpmath.nstr(new_re, prec, strip_zeros=False),
        mpmath.nstr(new_im, prec, strip_zeros=False),
    )


# ============================================================================
# Main rendering function
# ============================================================================

def render_frame_perturbation(
    center_re: str | float,
    center_im: str | float,
    zoom: float,
    width: int,
    height: int,
    max_iter: int = 1000,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Render a Mandelbrot frame using perturbation theory.

    Computes a reference orbit at arbitrary precision, then iterates all
    pixels as float64 deltas from that reference. Handles glitch detection
    and correction with up to 3 re-reference passes.

    Args:
        center_re: Real part of the center coordinate. Pass as string for
            full precision at deep zoom (e.g. "-0.7436438870371587").
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
    center_re_str = str(center_re)
    center_im_str = str(center_im)

    gpu = use_gpu if use_gpu is not None else CUDA_AVAILABLE

    # Viewport geometry (same as mandelbrot.py)
    aspect = width / height
    view_height = 3.0 / zoom
    view_width = view_height * aspect

    # dc offsets: pixel coordinates relative to the reference center
    min_dc_re = -view_width / 2.0
    min_dc_im = -view_height / 2.0
    step_dc_re = view_width / width
    step_dc_im = view_height / height

    # Maximum dc magnitude (corner pixel) for series approximation
    max_dc_re = abs(min_dc_re) + step_dc_re * width / 2.0
    max_dc_im = abs(min_dc_im) + step_dc_im * height / 2.0
    dc_max_mag_sq = max_dc_re * max_dc_re + max_dc_im * max_dc_im

    # P3-01: Compute reference orbit
    ref = compute_reference_orbit(
        center_re=center_re_str,
        center_im=center_im_str,
        max_iter=max_iter,
        zoom=zoom,
    )

    # P3-04: Series approximation (skip early iterations)
    # SA is only beneficial at deep zoom where dc is tiny (dc ~ 1e-50).
    # At moderate zoom, the cubic approximation d_n ~ A*dc + B*dc^2 + C*dc^3
    # introduces significant error since dc^3 is not negligible.
    sa_tolerance = 1e-6
    if zoom >= 1e8 and ref.num_iters > 10:
        sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im = (
            _compute_series_approximation(
                ref.z_re, ref.z_im, ref.num_iters, dc_max_mag_sq, sa_tolerance
            )
        )
    else:
        sa_skip = 0
        sa_a_re = sa_a_im = sa_b_re = sa_b_im = sa_c_re = sa_c_im = 0.0

    # Glitch tolerance: flag when |d_n|^2 > tolerance * |Z_n|^2.
    # A pixel is "glitched" when the delta grows so large relative to the
    # reference that floating-point cancellation corrupts the result.
    # At deep zoom, deltas start tiny (1e-50) so any growth to ~|Z_n| is a
    # glitch. At moderate zoom, deltas can legitimately be comparable to |Z_n|.
    # Use 1e-3 at deep zoom; disable (1e300) at moderate zoom.
    if zoom >= 1e13:
        glitch_tolerance = 1e-3
    else:
        # At moderate zoom, perturbation is mathematically equivalent to
        # direct iteration -- no precision-loss glitches can occur.
        glitch_tolerance = 1e300  # effectively disabled

    # First render pass
    smooth_out, glitch_out = _dispatch_render(
        ref, min_dc_re, min_dc_im, step_dc_re, step_dc_im,
        max_iter, height, width,
        sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
        glitch_tolerance, gpu,
    )

    # P3-03: Glitch correction passes (up to 3)
    for correction_pass in range(3):
        glitch_count = np.sum(glitch_out > 0)
        if glitch_count == 0:
            break

        # Find a new reference point among glitched pixels
        new_ref_coords = _find_glitch_reference(
            smooth_out, glitch_out,
            min_dc_re, min_dc_im, step_dc_re, step_dc_im,
            center_re_str, center_im_str, width,
        )
        if new_ref_coords is None:
            break

        new_re_str, new_im_str = new_ref_coords

        # Compute new reference orbit at the glitched pixel's location
        ref2 = compute_reference_orbit(
            center_re=new_re_str,
            center_im=new_im_str,
            max_iter=max_iter,
            zoom=zoom,
        )

        # Compute dc offsets relative to the NEW reference center
        # The viewport hasn't changed, but the reference point has
        import mpmath
        mpmath.mp.dps = ref2.precision
        offset_re = float(mpmath.mpf(new_re_str) - mpmath.mpf(center_re_str))
        offset_im = float(mpmath.mpf(new_im_str) - mpmath.mpf(center_im_str))
        new_min_dc_re = min_dc_re - offset_re
        new_min_dc_im = min_dc_im - offset_im

        # Recompute SA for new reference
        new_max_dc_re = abs(new_min_dc_re) + step_dc_re * width / 2.0
        new_max_dc_im = abs(new_min_dc_im) + step_dc_im * height / 2.0
        new_dc_max_mag_sq = new_max_dc_re ** 2 + new_max_dc_im ** 2

        if ref2.num_iters > 10:
            sa2 = _compute_series_approximation(
                ref2.z_re, ref2.z_im, ref2.num_iters, new_dc_max_mag_sq, sa_tolerance,
            )
            sa2_skip, sa2_a_re, sa2_a_im, sa2_b_re, sa2_b_im, sa2_c_re, sa2_c_im = sa2
        else:
            sa2_skip = 0
            sa2_a_re = sa2_a_im = sa2_b_re = sa2_b_im = sa2_c_re = sa2_c_im = 0.0

        # Render the correction pass (full frame, then merge only glitched pixels)
        smooth2, glitch2 = _dispatch_render(
            ref2, new_min_dc_re, new_min_dc_im, step_dc_re, step_dc_im,
            max_iter, height, width,
            sa2_skip, sa2_a_re, sa2_a_im, sa2_b_re, sa2_b_im, sa2_c_re, sa2_c_im,
            glitch_tolerance, gpu,
        )

        # Merge: overwrite only previously glitched pixels
        glitch_mask = glitch_out > 0
        smooth_out[glitch_mask] = smooth2[glitch_mask]
        # Update glitch map: keep pixels that are STILL glitched
        glitch_out[glitch_mask] = glitch2[glitch_mask]

    return smooth_out


def _dispatch_render(
    ref, min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter, height, width,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    glitch_tolerance, gpu,
):
    """Dispatch to GPU or CPU kernel."""
    if gpu:
        return _render_gpu_perturbation(
            ref.z_re, ref.z_im, ref.z_mag_sq,
            ref.num_iters, ref.escape_iter,
            min_dc_re, min_dc_im, step_dc_re, step_dc_im,
            max_iter, height, width,
            sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
            glitch_tolerance,
        )
    else:
        return _perturbation_cpu(
            ref.z_re, ref.z_im, ref.z_mag_sq,
            ref.num_iters, ref.escape_iter,
            min_dc_re, min_dc_im, step_dc_re, step_dc_im,
            max_iter,
            height, width,
            sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
            glitch_tolerance,
        )
