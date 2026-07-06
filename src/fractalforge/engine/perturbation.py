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

from fractalforge.engine.precision import (
    compute_reference_orbit, compute_series_approximation_hp, ReferenceOrbit,
    zoom_to_log10,
)
from fractalforge.engine.bla import compute_bla_table, compute_bla_table_fxp, BLATable

# At/above this log10(zoom) frames route to the floatexp deep kernel.
#
# The float64 PT kernels run WITHOUT rebasing (float64 rebasing folds d to
# O(1) where the per-pixel dc underflows the mantissa — false-interior
# blocks), which means they quietly lose precision whenever the reference
# orbit passes near zero: user testing showed moire rings at 4.6e17 near the
# antenna. The fxp kernel handles those passages exactly via rebasing, so it
# owns everything from 1e13 up; float64 PT keeps only the narrow, safe window
# between the resolution-aware standard-engine cutoff (~1e11 at 1080p) and
# 1e13, where deltas are large enough that near-zero passages are harmless.
DEEP_FXP_LOG10 = 13.0


# ============================================================================
# P3-04: Series Approximation (CPU pre-computation)
# ============================================================================

@njit(cache=True)
def _compute_series_approximation(
    z_re, z_im, num_iters, dc_max_mag_sq, tolerance,
    probe_dc_re=None, probe_dc_im=None, pixel_spacing=0.0,
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

    a_re = 0.0
    a_im = 0.0
    b_re = 0.0
    b_im = 0.0
    c_re = 0.0
    c_im = 0.0

    skip_iters = 0
    save_a_re = a_re
    save_a_im = a_im
    save_b_re = b_re
    save_b_im = b_im
    save_c_re = c_re
    save_c_im = c_im

    # Probe validation: track true deltas at sample points
    use_probes = probe_dc_re is not None and len(probe_dc_re) > 0
    num_probes = len(probe_dc_re) if use_probes else 0
    # Numba needs fixed-size arrays — allocate for probes
    if use_probes:
        pd_re = probe_dc_re.copy()  # true deltas, start as dc
        pd_im = probe_dc_im.copy()
        probe_tol_floor = max(pixel_spacing * 1e-6, 1e-30)
    else:
        pd_re = np.empty(0)
        pd_im = np.empty(0)
        probe_tol_floor = 1e-30

    for n in range(num_iters):
        zn_re = z_re[n]
        zn_im = z_im[n]

        new_a_re = 2.0 * (zn_re * a_re - zn_im * a_im) + 1.0
        new_a_im = 2.0 * (zn_re * a_im + zn_im * a_re)

        a_sq_re = a_re * a_re - a_im * a_im
        a_sq_im = 2.0 * a_re * a_im
        new_b_re = 2.0 * (zn_re * b_re - zn_im * b_im) + a_sq_re
        new_b_im = 2.0 * (zn_re * b_im + zn_im * b_re) + a_sq_im

        ab_re = a_re * b_re - a_im * b_im
        ab_im = a_re * b_im + a_im * b_re
        new_c_re = 2.0 * (zn_re * c_re - zn_im * c_im) + 2.0 * ab_re
        new_c_im = 2.0 * (zn_re * c_im + zn_im * c_re) + 2.0 * ab_im

        a_re, a_im = new_a_re, new_a_im
        b_re, b_im = new_b_re, new_b_im
        c_re, c_im = new_c_re, new_c_im

        c_mag = math.sqrt(c_re * c_re + c_im * c_im)
        if c_mag * dc_max_cubed > tolerance:
            return skip_iters, save_a_re, save_a_im, save_b_re, save_b_im, save_c_re, save_c_im

        # Probe validation: advance probes and compare against SA polynomial
        if use_probes:
            probe_failed = False
            for p in range(num_probes):
                # Advance probe: d_{n+1} = 2*Z_n*d_n + d_n^2 + dc
                new_pr = (2.0 * (zn_re * pd_re[p] - zn_im * pd_im[p])
                          + pd_re[p] * pd_re[p] - pd_im[p] * pd_im[p]
                          + probe_dc_re[p])
                new_pi = (2.0 * (zn_re * pd_im[p] + zn_im * pd_re[p])
                          + 2.0 * pd_re[p] * pd_im[p] + probe_dc_im[p])
                pd_re[p] = new_pr
                pd_im[p] = new_pi

                # Evaluate SA polynomial at this probe
                pcr = probe_dc_re[p]
                pci = probe_dc_im[p]
                dc2r = pcr * pcr - pci * pci
                dc2i = 2.0 * pcr * pci
                dc3r = pcr * dc2r - pci * dc2i
                dc3i = pcr * dc2i + pci * dc2r
                sa_r = (a_re * pcr - a_im * pci + b_re * dc2r - b_im * dc2i
                        + c_re * dc3r - c_im * dc3i)
                sa_i = (a_re * pci + a_im * pcr + b_re * dc2i + b_im * dc2r
                        + c_re * dc3i + c_im * dc3r)
                err_sq = (sa_r - pd_re[p])**2 + (sa_i - pd_im[p])**2
                true_sq = pd_re[p]**2 + pd_im[p]**2
                threshold = max(true_sq * 1e-6, probe_tol_floor * probe_tol_floor)
                if err_sq > threshold:
                    probe_failed = True
                    break
            if probe_failed:
                return skip_iters, save_a_re, save_a_im, save_b_re, save_b_im, save_c_re, save_c_im

        save_a_re, save_a_im = a_re, a_im
        save_b_re, save_b_im = b_re, b_im
        save_c_re, save_c_im = c_re, c_im
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
    enable_rebasing,
    smooth_out, glitch_out,
):
    """CUDA kernel: perturbation iteration, one thread per pixel.

    Each pixel computes d_n (delta from reference orbit Z_n):
        d_{n+1} = 2*Z_n*d_n + d_n^2 + dc

    With series approximation: d_{sa_skip} = A*dc + B*dc^2 + C*dc^3,
    then iterate from sa_skip onward.

    Proactive rebasing (DZ-P1-03): when |Z+d| < |d|, fold delta back.
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
    total_iters = iteration  # tracks actual computation steps (survives rebasing)

    # Phase 1: Perturbation iteration (while reference orbit is valid)
    while iteration <= ref_num_iters and total_iters < max_iter:
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
            smooth_out[x, y] = float(total_iters) + 1.0 - nu
            return

        # DZ-P1-03: Proactive rebasing (only at deep zoom where deltas are tiny)
        # When |Z_n + d_n| < |d_n|, the orbit nears 0 and cancellation is imminent.
        # Fold the delta back: d = Z+d, restart reference from iteration 0.
        # Rebasing is bookkeeping, not an iteration: total_iters must NOT
        # advance or patches of pixels shift by their rebase count (blocky
        # seams). Z_0 = 0 prevents back-to-back rebases, so the loop still
        # terminates.
        d_mag_sq = d_re * d_re + d_im * d_im
        if enable_rebasing and full_mag_sq < d_mag_sq:
            d_re = full_re
            d_im = full_im
            iteration = 0
            continue

        # P3-03: Glitch detection (safety net, most issues caught by rebasing)
        # Skip when |Z_n|^2 < 1e-3: near-zero orbit points are expected in
        # periodic orbits (e.g., period-16 near the antenna), not glitches.
        zn_mag_sq = z_mag_sq[iteration]
        if zn_mag_sq > 1e-3 and d_mag_sq > glitch_tolerance * zn_mag_sq:
            glitch_out[x, y] = 1
            smooth_out[x, y] = -1.0
            return

        # Delta iteration: d_{n+1} = 2*Z_n*d_n + d_n^2 + dc
        d_re_new = 2.0 * (zn_re * d_re - zn_im * d_im) + (d_re * d_re - d_im * d_im) + dc_re
        d_im_new = 2.0 * (zn_re * d_im + zn_im * d_re) + 2.0 * d_re * d_im + dc_im
        d_re = d_re_new
        d_im = d_im_new

        iteration += 1
        total_iters += 1

    # Post-reference: reference escaped but pixel hasn't.
    # Reconstruct full z and continue with standard iteration.
    if iteration <= ref_num_iters:
        full_re = z_re[iteration] + d_re
        full_im = z_im[iteration] + d_im
    else:
        full_re = z_re[ref_num_iters] + d_re
        full_im = z_im[ref_num_iters] + d_im

    # c = C + dc. Z_1 = Z_0^2 + C = 0 + C = C, so c = z_re[1] + dc.
    c_re = z_re[1] + dc_re
    c_im = z_im[1] + dc_im

    z_r = full_re
    z_i = full_im

    while total_iters < max_iter:
        z_r_sq = z_r * z_r
        z_i_sq = z_i * z_i

        if z_r_sq + z_i_sq > escape_radius_sq:
            log_zn = 0.5 * math.log(z_r_sq + z_i_sq)
            nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
            smooth_out[x, y] = float(total_iters) + 1.0 - nu
            return

        z_i = 2.0 * z_r * z_i + c_im
        z_r = z_r_sq - z_i_sq + c_re
        total_iters += 1

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
    enable_rebasing,
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
            total_iters = iteration

            # Phase 1: Perturbation iteration
            while iteration <= ref_num_iters and total_iters < max_iter:
                zn_re = z_re[iteration]
                zn_im = z_im[iteration]

                full_re = zn_re + d_re
                full_im = zn_im + d_im
                full_mag_sq = full_re * full_re + full_im * full_im

                if full_mag_sq > escape_radius_sq:
                    log_zn = 0.5 * math.log(full_mag_sq)
                    nu = math.log(log_zn / log2) / log2
                    smooth_out[x, y] = float(total_iters) + 1.0 - nu
                    escaped = True
                    break

                # DZ-P1-03: Proactive rebasing (deep zoom only).
                # Not an iteration — do not advance total_iters (see CUDA).
                d_mag_sq = d_re * d_re + d_im * d_im
                if enable_rebasing and full_mag_sq < d_mag_sq:
                    d_re = full_re
                    d_im = full_im
                    iteration = 0
                    continue

                # Glitch detection (safety net, skip near-zero orbit points)
                zn_mag_sq = z_mag_sq[iteration]
                if zn_mag_sq > 1e-3 and d_mag_sq > glitch_tolerance * zn_mag_sq:
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
                total_iters += 1

            if escaped or glitched:
                continue

            # Post-reference: reference escaped but pixel hasn't
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
            while total_iters < max_iter:
                z_r_sq = z_r * z_r
                z_i_sq = z_i * z_i

                if z_r_sq + z_i_sq > escape_radius_sq:
                    log_zn = 0.5 * math.log(z_r_sq + z_i_sq)
                    nu = math.log(log_zn / log2) / log2
                    smooth_out[x, y] = float(total_iters) + 1.0 - nu
                    standard_escaped = True
                    break

                z_i = 2.0 * z_r * z_i + c_im
                z_r = z_r_sq - z_i_sq + c_re
                total_iters += 1

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
    enable_rebasing=True,
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
        enable_rebasing,
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
    zoom: float | str,
    width: int,
    height: int,
    max_iter: int = 1000,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Render a Mandelbrot frame using perturbation theory.

    Computes a reference orbit at arbitrary precision, then iterates all
    pixels as float64 deltas from that reference. Handles glitch detection
    and correction with up to 3 re-reference passes.

    Beyond ~1e150 zoom the float64 delta kernels approach underflow, so the
    frame is routed to the floatexp deep kernel (BLA + rebasing, unbounded
    exponent range). Zoom may be passed as a string (e.g. "1e500") for
    depths beyond float64 range entirely.

    Args:
        center_re: Real part of the center coordinate. Pass as string for
            full precision at deep zoom (e.g. "-0.7436438870371587").
        center_im: Imaginary part of the center coordinate.
        zoom: Zoom level (1.0 = full view, higher = more zoomed in).
            Accepts a string for arbitrary depth.
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

    log10_zoom = zoom_to_log10(zoom)
    if log10_zoom >= DEEP_FXP_LOG10:
        return _render_deep_fxp(
            center_re_str, center_im_str, zoom, width, height, max_iter, gpu,
        )
    zoom = float(zoom)

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
    #
    # Probe-based validation: compute SA polynomial at 8 sample points and
    # compare against actual perturbation iteration. This catches divergent
    # series that pass analytical checks (chaotic orbits near the antenna).
    sa_tolerance = 1e-6
    pixel_spacing = view_height / height
    if zoom >= 1e8 and ref.num_iters > 10:
        # Compute 8 probe dc values: 4 corners + 4 edge midpoints
        max_dc_re_val = abs(min_dc_re) + step_dc_re * (width - 1)
        max_dc_im_val = abs(min_dc_im) + step_dc_im * (height - 1)
        probe_dc_re = np.array([
            min_dc_re, max_dc_re_val, min_dc_re, max_dc_re_val,  # corners
            0.0, max_dc_re_val, 0.0, min_dc_re,                  # edge midpoints
        ], dtype=np.float64)
        probe_dc_im = np.array([
            min_dc_im, min_dc_im, max_dc_im_val, max_dc_im_val,  # corners
            min_dc_im, 0.0, max_dc_im_val, 0.0,                  # edge midpoints
        ], dtype=np.float64)

        if zoom >= 1e13:
            sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im = (
                compute_series_approximation_hp(
                    ref, dc_max_mag_sq, sa_tolerance,
                    probe_dc_re, probe_dc_im, pixel_spacing,
                )
            )
        else:
            sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im = (
                _compute_series_approximation(
                    ref.z_re, ref.z_im, ref.num_iters, dc_max_mag_sq,
                    sa_tolerance, probe_dc_re, probe_dc_im, pixel_spacing,
                )
            )
    else:
        sa_skip = 0
        sa_a_re = sa_a_im = sa_b_re = sa_b_im = sa_c_re = sa_c_im = 0.0

    # Glitch tolerance: flag when |d_n|^2 > tolerance * |Z_n|^2.
    # With proactive rebasing (DZ-P1-03) handling most precision issues inline,
    # the glitch detector is now a safety net rather than the primary defense.
    # We use a slower ramp and higher floor to reduce false positives.
    #
    # Scale the tolerance with zoom depth:
    #   zoom 1e13:  tolerance = 1e6   (|d| > 1000*|Z| to trigger)
    #   zoom 1e33:  tolerance = 1e0   (|d| > |Z|)
    #   zoom 1e40+: tolerance = 1e-2  (floor, generous with rebasing active)
    #   zoom < 1e13: disabled
    if zoom >= 1e13:
        log_zoom = math.log10(zoom)
        # Slower ramp: 6 - 0.3*(log_zoom - 13) → from 1e6 at 1e13 to 1e0 at 1e33
        exponent = 6.0 - 0.3 * (log_zoom - 13.0)
        glitch_tolerance = 10.0 ** max(exponent, -2.0)  # floor at 1e-2
    else:
        glitch_tolerance = 1e300  # effectively disabled

    # BLA table: pre-compute linear approximation coefficients for iteration skipping.
    # Only beneficial at very deep zoom where the reference orbit is very long
    # (10,000+ iterations). At moderate zoom with shorter orbits, BLA introduces
    # accuracy issues especially for orbits that pass near zero periodically.
    pixel_spacing = view_height / height
    bla_table = None
    if ref.num_iters > 10000:
        bla_table = compute_bla_table(
            ref.z_re, ref.z_im, ref.num_iters, dc_max=math.sqrt(dc_max_mag_sq),
        )

    # DZ-P1-03: Only enable proactive rebasing at deep zoom where deltas are
    # genuinely tiny. At moderate zoom (1e13-1e17), deltas start at ~1e-13
    # but grow to O(1) for chaotic orbits. Rebasing folds d=Z+d making d=O(1),
    # violating the perturbation assumption d << Z. The d^2 term then dominates
    # and the computation produces garbage (false interior pixels).
    # At zoom >= 1e18, initial deltas are ~1e-18 and rebasing is essential to
    # prevent catastrophic cancellation during near-zero orbit passages.
    enable_rebasing = zoom >= 1e18

    # First render pass
    smooth_out, glitch_out = _dispatch_render(
        ref, min_dc_re, min_dc_im, step_dc_re, step_dc_im,
        max_iter, height, width,
        sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
        glitch_tolerance, gpu, bla_table=bla_table,
        enable_rebasing=enable_rebasing,
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
            if zoom >= 1e13:
                sa2 = compute_series_approximation_hp(
                    ref2, new_dc_max_mag_sq, sa_tolerance,
                )
            else:
                sa2 = _compute_series_approximation(
                    ref2.z_re, ref2.z_im, ref2.num_iters, new_dc_max_mag_sq,
                    sa_tolerance,
                )
            sa2_skip, sa2_a_re, sa2_a_im, sa2_b_re, sa2_b_im, sa2_c_re, sa2_c_im = sa2
        else:
            sa2_skip = 0
            sa2_a_re = sa2_a_im = sa2_b_re = sa2_b_im = sa2_c_re = sa2_c_im = 0.0

        # BLA table for new reference
        bla_table2 = None
        if ref2.num_iters > 100:
            bla_table2 = compute_bla_table(
                ref2.z_re, ref2.z_im, ref2.num_iters,
                dc_max=math.sqrt(new_dc_max_mag_sq),
            )

        # Render the correction pass (full frame, then merge only glitched pixels)
        smooth2, glitch2 = _dispatch_render(
            ref2, new_min_dc_re, new_min_dc_im, step_dc_re, step_dc_im,
            max_iter, height, width,
            sa2_skip, sa2_a_re, sa2_a_im, sa2_b_re, sa2_b_im, sa2_c_re, sa2_c_im,
            glitch_tolerance, gpu, bla_table=bla_table2,
            enable_rebasing=enable_rebasing,
        )

        # Merge: overwrite only previously glitched pixels
        glitch_mask = glitch_out > 0
        smooth_out[glitch_mask] = smooth2[glitch_mask]
        # Update glitch map: keep pixels that are STILL glitched
        glitch_out[glitch_mask] = glitch2[glitch_mask]

    return smooth_out


# BLA tables are deterministic given an orbit and the viewport corner |dc|;
# progressive full-res passes and repeated tweaks at a fixed view rebuild the
# same table, so keep a small keyed cache.
_BLA_FXP_CACHE: dict[tuple, object] = {}
_BLA_FXP_CACHE_MAX = 4


def _render_deep_fxp(
    center_re_str: str,
    center_im_str: str,
    zoom: float | str,
    width: int,
    height: int,
    max_iter: int,
    gpu: bool,
) -> np.ndarray:
    """Render via the floatexp deep kernel (zoom beyond ~1e150).

    The dc pixel grid is expressed as float64 mantissas sharing one
    power-of-two frame exponent, so pixel offsets remain exact at depths
    where their absolute values underflow float64.
    """
    import mpmath

    from fractalforge.engine.deep_kernel import render_cpu_deep, render_gpu_deep

    aspect = width / height
    with mpmath.workdps(40):
        zoom_mp = mpmath.mpf(str(zoom))
        view_h = mpmath.mpf(3) / zoom_mp
        view_w = view_h * aspect

        _, frame_e = mpmath.frexp(max(view_w, view_h))
        frame_e = int(frame_e)
        min_dc_re_m = float(mpmath.ldexp(-view_w / 2, -frame_e))
        min_dc_im_m = float(mpmath.ldexp(-view_h / 2, -frame_e))
        step_re_m = float(mpmath.ldexp(view_w / width, -frame_e))
        step_im_m = float(mpmath.ldexp(view_h / height, -frame_e))

        corner = mpmath.sqrt((view_w / 2) ** 2 + (view_h / 2) ** 2)
        dc_max_m_mp, dc_max_e = mpmath.frexp(corner)
        dc_max_m = float(dc_max_m_mp)
        dc_max_e = int(dc_max_e)

    ref = compute_reference_orbit(
        center_re=center_re_str,
        center_im=center_im_str,
        max_iter=max_iter,
        zoom=zoom,
        extended=True,
    )

    bla_key = (center_re_str, center_im_str, ref.precision, ref.num_iters,
               dc_max_m, dc_max_e)
    bla_table = _BLA_FXP_CACHE.get(bla_key)
    if bla_table is None:
        bla_table = compute_bla_table_fxp(
            ref.z_m_re, ref.z_m_im, ref.z_exp, ref.num_iters,
            dc_max_m, dc_max_e,
        )
        _BLA_FXP_CACHE[bla_key] = bla_table
        while len(_BLA_FXP_CACHE) > _BLA_FXP_CACHE_MAX:
            _BLA_FXP_CACHE.pop(next(iter(_BLA_FXP_CACHE)))

    render_fn = render_gpu_deep if gpu else render_cpu_deep
    smooth_out, _ = render_fn(
        ref, bla_table,
        min_dc_re_m, min_dc_im_m, step_re_m, step_im_m, frame_e,
        max_iter, height, width,
    )
    return smooth_out


def _dispatch_render(
    ref, min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter, height, width,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    glitch_tolerance, gpu, bla_table=None, enable_rebasing=True,
):
    """Dispatch to GPU or CPU kernel, using BLA acceleration when available."""
    if bla_table is not None and bla_table.num_levels > 0:
        from fractalforge.engine.bla_kernel import render_gpu_bla, render_cpu_bla
        if gpu:
            return render_gpu_bla(
                ref.z_re, ref.z_im, ref.z_mag_sq,
                ref.num_iters,
                min_dc_re, min_dc_im, step_dc_re, step_dc_im,
                max_iter, height, width,
                sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
                bla_table, glitch_tolerance, enable_rebasing,
            )
        else:
            return render_cpu_bla(
                ref.z_re, ref.z_im, ref.z_mag_sq,
                ref.num_iters,
                min_dc_re, min_dc_im, step_dc_re, step_dc_im,
                max_iter, height, width,
                sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
                bla_table, glitch_tolerance, enable_rebasing,
            )

    if gpu:
        return _render_gpu_perturbation(
            ref.z_re, ref.z_im, ref.z_mag_sq,
            ref.num_iters, ref.escape_iter,
            min_dc_re, min_dc_im, step_dc_re, step_dc_im,
            max_iter, height, width,
            sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
            glitch_tolerance, enable_rebasing,
        )
    else:
        return _perturbation_cpu(
            ref.z_re, ref.z_im, ref.z_mag_sq,
            ref.num_iters, ref.escape_iter,
            min_dc_re, min_dc_im, step_dc_re, step_dc_im,
            max_iter,
            height, width,
            sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
            glitch_tolerance, enable_rebasing,
        )
