"""BLA-accelerated perturbation CUDA kernel.

Replaces the single-step perturbation kernel with one that uses the BLA
coefficient table to skip large blocks of iterations at once. Falls back
to single-step iteration near escape/glitch boundaries where BLA jumps
are not valid.

Typical speedup: 100-1000x at extreme deep zoom (1e50+), since the effective
iteration count drops from millions to thousands.
"""

import math

import numpy as np
from numba import cuda, njit, prange


# ============================================================================
# CUDA kernel with BLA acceleration
# ============================================================================

@cuda.jit
def _perturbation_bla_cuda(
    z_re, z_im, z_mag_sq,
    ref_num_iters,
    min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    bla_a_re, bla_a_im, bla_b_re, bla_b_im, bla_r,
    bla_offsets, bla_sizes, bla_num_levels,
    glitch_tolerance,
    enable_rebasing,
    smooth_out, glitch_out,
):
    """CUDA kernel: BLA-accelerated perturbation iteration.

    For each pixel, tries to use the largest valid BLA jump at each step.
    Falls back to single-step delta iteration when no BLA jump is valid
    (near escape or when delta is too large).
    """
    x, y = cuda.grid(2)
    height, width = smooth_out.shape

    if x >= height or y >= width:
        return

    dc_re = min_dc_re + y * step_dc_re
    dc_im = min_dc_im + x * step_dc_im

    escape_radius_sq = 256.0
    log2 = math.log(2.0)

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

    glitch_out[x, y] = 0
    total_iters = iteration  # tracks actual computation steps (survives rebasing)

    # Main iteration loop with BLA acceleration
    while iteration <= ref_num_iters and total_iters < max_iter:
        # Check escape on full value
        zn_re = z_re[iteration]
        zn_im = z_im[iteration]
        full_re = zn_re + d_re
        full_im = zn_im + d_im
        full_mag_sq = full_re * full_re + full_im * full_im

        if full_mag_sq > escape_radius_sq:
            log_zn = 0.5 * math.log(full_mag_sq)
            nu = math.log(log_zn / log2) / log2
            smooth_out[x, y] = float(total_iters) + 1.0 - nu
            return

        # DZ-P1-03: Proactive rebasing (deep zoom only).
        # Not an iteration — do not advance total_iters, or patches of
        # pixels shift by their rebase count (blocky seams).
        d_mag_sq = d_re * d_re + d_im * d_im
        if enable_rebasing and full_mag_sq < d_mag_sq:
            d_re = full_re
            d_im = full_im
            iteration = 0
            continue

        # Glitch detection (safety net)
        zn_mag_sq = z_mag_sq[iteration]
        if zn_mag_sq > 1e-3 and d_mag_sq > glitch_tolerance * zn_mag_sq:
            glitch_out[x, y] = 1
            smooth_out[x, y] = -1.0
            return

        # Try BLA jumps from highest level down
        d_mag = math.sqrt(d_mag_sq)
        jumped = False

        for level in range(bla_num_levels - 1, -1, -1):
            jump_size = 1 << level
            target = iteration + jump_size

            if target > ref_num_iters or target > max_iter:
                continue

            offset = bla_offsets[level]
            size = bla_sizes[level]
            if iteration >= size:
                continue

            idx = offset + iteration

            if d_mag >= bla_r[idx]:
                continue

            ba_re = bla_a_re[idx]
            ba_im = bla_a_im[idx]
            bb_re = bla_b_re[idx]
            bb_im = bla_b_im[idx]

            new_d_re = (ba_re * d_re - ba_im * d_im) + (bb_re * dc_re - bb_im * dc_im)
            new_d_im = (ba_re * d_im + ba_im * d_re) + (bb_re * dc_im + bb_im * dc_re)

            d_re = new_d_re
            d_im = new_d_im
            total_iters += (target - iteration)
            iteration = target
            jumped = True
            break

        if not jumped:
            d_re_new = 2.0 * (zn_re * d_re - zn_im * d_im) + (d_re * d_re - d_im * d_im) + dc_re
            d_im_new = 2.0 * (zn_re * d_im + zn_im * d_re) + 2.0 * d_re * d_im + dc_im
            d_re = d_re_new
            d_im = d_im_new
            iteration += 1
            total_iters += 1

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

    while total_iters < max_iter:
        z_r_sq = z_r * z_r
        z_i_sq = z_i * z_i

        if z_r_sq + z_i_sq > escape_radius_sq:
            log_zn = 0.5 * math.log(z_r_sq + z_i_sq)
            nu = math.log(log_zn / log2) / log2
            smooth_out[x, y] = float(total_iters) + 1.0 - nu
            return

        z_i = 2.0 * z_r * z_i + c_im
        z_r = z_r_sq - z_i_sq + c_re
        total_iters += 1

    smooth_out[x, y] = -1.0


# ============================================================================
# CPU kernel with BLA acceleration (fallback)
# ============================================================================

@njit(parallel=True, cache=True)
def _perturbation_bla_cpu(
    z_re, z_im, z_mag_sq,
    ref_num_iters,
    min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter, height, width,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    bla_a_re, bla_a_im, bla_b_re, bla_b_im, bla_r,
    bla_offsets, bla_sizes, bla_num_levels,
    glitch_tolerance,
    enable_rebasing,
):
    """CPU kernel: BLA-accelerated perturbation with Numba parallel."""
    smooth_out = np.empty((height, width), dtype=np.float64)
    glitch_out = np.zeros((height, width), dtype=np.int32)
    escape_radius_sq = 256.0
    log2 = math.log(2.0)

    for x in prange(height):
        for y in range(width):
            dc_re = min_dc_re + y * step_dc_re
            dc_im = min_dc_im + x * step_dc_im

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

            escaped = False
            glitched = False
            total_iters = iteration

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
                # Not an iteration — do not advance total_iters (see above).
                d_mag_sq = d_re * d_re + d_im * d_im
                if enable_rebasing and full_mag_sq < d_mag_sq:
                    d_re = full_re
                    d_im = full_im
                    iteration = 0
                    continue

                # Glitch detection (safety net)
                zn_mag_sq = z_mag_sq[iteration]
                if zn_mag_sq > 1e-3 and d_mag_sq > glitch_tolerance * zn_mag_sq:
                    glitch_out[x, y] = 1
                    smooth_out[x, y] = -1.0
                    glitched = True
                    break

                # Try BLA jumps
                d_mag = math.sqrt(d_mag_sq)
                jumped = False

                for level in range(bla_num_levels - 1, -1, -1):
                    jump_size = 1 << level
                    target = iteration + jump_size
                    if target > ref_num_iters or target > max_iter:
                        continue
                    offset = bla_offsets[level]
                    size = bla_sizes[level]
                    if iteration >= size:
                        continue
                    idx = offset + iteration
                    if d_mag >= bla_r[idx]:
                        continue

                    ba_re = bla_a_re[idx]
                    ba_im = bla_a_im[idx]
                    bb_re = bla_b_re[idx]
                    bb_im = bla_b_im[idx]

                    new_d_re = (ba_re * d_re - ba_im * d_im) + (bb_re * dc_re - bb_im * dc_im)
                    new_d_im = (ba_re * d_im + ba_im * d_re) + (bb_re * dc_im + bb_im * dc_re)

                    d_re = new_d_re
                    d_im = new_d_im
                    total_iters += (target - iteration)
                    iteration = target
                    jumped = True
                    break

                if not jumped:
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

def render_gpu_bla(
    z_re, z_im, z_mag_sq,
    ref_num_iters,
    min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter, height, width,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    bla_table,
    glitch_tolerance,
    enable_rebasing=True,
):
    """Dispatch BLA-accelerated perturbation to CUDA."""
    d_z_re = cuda.to_device(z_re)
    d_z_im = cuda.to_device(z_im)
    d_z_mag_sq = cuda.to_device(z_mag_sq)

    smooth_out = np.zeros((height, width), dtype=np.float64)
    glitch_out = np.zeros((height, width), dtype=np.int32)
    d_smooth = cuda.to_device(smooth_out)
    d_glitch = cuda.to_device(glitch_out)

    # Upload BLA table to device
    d_bla_a_re = cuda.to_device(bla_table.a_re)
    d_bla_a_im = cuda.to_device(bla_table.a_im)
    d_bla_b_re = cuda.to_device(bla_table.b_re)
    d_bla_b_im = cuda.to_device(bla_table.b_im)
    d_bla_r = cuda.to_device(bla_table.validity_r)
    d_bla_offsets = cuda.to_device(bla_table.level_offsets)
    d_bla_sizes = cuda.to_device(bla_table.level_sizes)

    threads_per_block = (16, 16)
    blocks_x = (height + 15) // 16
    blocks_y = (width + 15) // 16
    blocks_per_grid = (blocks_x, blocks_y)

    _perturbation_bla_cuda[blocks_per_grid, threads_per_block](
        d_z_re, d_z_im, d_z_mag_sq,
        ref_num_iters,
        min_dc_re, min_dc_im, step_dc_re, step_dc_im,
        max_iter,
        sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
        d_bla_a_re, d_bla_a_im, d_bla_b_re, d_bla_b_im, d_bla_r,
        d_bla_offsets, d_bla_sizes, bla_table.num_levels,
        glitch_tolerance,
        enable_rebasing,
        d_smooth, d_glitch,
    )

    d_smooth.copy_to_host(smooth_out)
    d_glitch.copy_to_host(glitch_out)
    return smooth_out, glitch_out


def render_cpu_bla(
    z_re, z_im, z_mag_sq,
    ref_num_iters,
    min_dc_re, min_dc_im, step_dc_re, step_dc_im,
    max_iter, height, width,
    sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
    bla_table,
    glitch_tolerance,
    enable_rebasing=True,
):
    """Dispatch BLA-accelerated perturbation to CPU."""
    return _perturbation_bla_cpu(
        z_re, z_im, z_mag_sq,
        ref_num_iters,
        min_dc_re, min_dc_im, step_dc_re, step_dc_im,
        max_iter, height, width,
        sa_skip, sa_a_re, sa_a_im, sa_b_re, sa_b_im, sa_c_re, sa_c_im,
        bla_table.a_re, bla_table.a_im, bla_table.b_re, bla_table.b_im, bla_table.validity_r,
        bla_table.level_offsets, bla_table.level_sizes, bla_table.num_levels,
        glitch_tolerance, enable_rebasing,
    )
