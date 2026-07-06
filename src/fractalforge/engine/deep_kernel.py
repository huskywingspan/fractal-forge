"""Deep perturbation kernel with floatexp (extended-range) arithmetic.

Float64 deltas underflow at ~1e-308, capping the standard perturbation
kernels at roughly 1e150-1e250 zoom (products like 2*Z*d die earlier than
the deltas themselves). This kernel represents every delta as a floatexp
triple (mantissa pair + shared int64 exponent), giving unbounded exponent
range — zoom depth is then limited only by reference orbit precision and
iteration budget, not by the floating point format.

Design points:
  - No series approximation: BLA jumps from iteration 0 subsume SA at these
    depths (the first jumps are huge while |d| is tiny), per modern practice
    (Zhuoran's BLA superseded SA/NanoMB2).
  - Proactive rebasing is always on: when |Z_n + d| < |d|, fold the delta
    into the absolute orbit and restart the reference index. The recurrence
    is algebraically exact under this substitution.
  - When the reference orbit is exhausted before the pixel resolves, the
    pixel is rebased to index 0 (exact) instead of falling back to standard
    float64 iteration, which would lose the deep coordinate entirely.
  - No glitch passes: with correct BLA validity radii and rebasing, the
    Pauldelbrot heuristic mostly produces false positives at extreme depth.
"""

import math

import numpy as np
from numba import cuda, njit, prange

from fractalforge.engine.floatexp import (
    fx_add,
    fx_mag_lt,
    fx_mag_sq_log2,
    fx_mul,
    fx_norm,
)

# log2 of the squared escape radius (256 = radius 16, matching other kernels)
ESCAPE_LOG2 = 8.0
LN2 = math.log(2.0)


# ============================================================================
# CUDA kernel
# ============================================================================

def _make_cuda_kernel():
    """Build the CUDA kernel lazily so import works without a GPU."""
    from fractalforge.engine.floatexp import (
        dfx_add, dfx_mag_lt, dfx_mag_sq_log2, dfx_mul, dfx_norm,
    )

    @cuda.jit
    def _deep_cuda(
        zm_re, zm_im, z_exp,
        ref_num_iters,
        dc_min_m_re, dc_min_m_im, dc_step_m_re, dc_step_m_im, dc_exp,
        max_iter,
        bla_a_m_re, bla_a_m_im, bla_a_e,
        bla_b_m_re, bla_b_m_im, bla_b_e,
        bla_r_log2, bla_offsets, bla_sizes, bla_bits, bla_num_levels,
        smooth_out,
    ):
        x, y = cuda.grid(2)
        height, width = smooth_out.shape
        if x >= height or y >= width:
            return

        # Degenerate orbit: nothing to iterate against, and the forced
        # end-of-orbit rebase would spin without advancing `total`.
        if ref_num_iters < 1:
            smooth_out[x, y] = -1.0
            return

        # Per-pixel dc: mantissas share the frame exponent
        dc_re, dc_im, dc_e = dfx_norm(
            dc_min_m_re + y * dc_step_m_re,
            dc_min_m_im + x * dc_step_m_im,
            dc_exp,
        )

        d_re, d_im, d_e = dc_re, dc_im, dc_e
        iteration = 1
        total = 1

        while total < max_iter:
            full_re, full_im, full_e = dfx_add(
                zm_re[iteration], zm_im[iteration], z_exp[iteration],
                d_re, d_im, d_e,
            )

            # Escape check (|full| > 16 needs exponent >= 4)
            if full_e >= 4:
                l2 = dfx_mag_sq_log2(full_re, full_im, full_e)
                if l2 > ESCAPE_LOG2:
                    log2_zabs = 0.5 * l2
                    nu = math.log(log2_zabs) / LN2
                    smooth_out[x, y] = float(total) + 1.0 - nu
                    return

            # Proactive rebasing: |Z_n + d| < |d| means cancellation is
            # imminent — fold to the absolute orbit (exact substitution).
            # Also forced when the reference orbit is exhausted.
            # A rebase re-expresses the SAME z_n against reference index 0 —
            # it is not an iteration, so `total` (the escape-time counter used
            # for coloring) must not advance. Counting it shifted patches of
            # pixels by their integer rebase count, producing blocky seams.
            # Termination is safe without it: Z_0 = 0 makes full == d at the
            # next pass, so a rebase can never fire twice in a row and every
            # other pass advances `total`.
            if iteration >= ref_num_iters or dfx_mag_lt(
                full_re, full_im, full_e, d_re, d_im, d_e
            ):
                d_re, d_im, d_e = full_re, full_im, full_e
                iteration = 0
                continue

            # Try BLA jumps from the largest stored level down
            d_l2 = 0.5 * dfx_mag_sq_log2(d_re, d_im, d_e)
            jumped = False
            for li in range(bla_num_levels - 1, -1, -1):
                skip = 1 << bla_bits[li]
                target = iteration + skip
                if target > ref_num_iters:
                    continue
                if total + skip > max_iter:
                    continue
                if iteration >= bla_sizes[li]:
                    continue
                idx = bla_offsets[li] + iteration
                if d_l2 >= bla_r_log2[idx]:
                    continue

                t1_re, t1_im, t1_e = dfx_mul(
                    bla_a_m_re[idx], bla_a_m_im[idx], bla_a_e[idx],
                    d_re, d_im, d_e,
                )
                t2_re, t2_im, t2_e = dfx_mul(
                    bla_b_m_re[idx], bla_b_m_im[idx], bla_b_e[idx],
                    dc_re, dc_im, dc_e,
                )
                d_re, d_im, d_e = dfx_add(t1_re, t1_im, t1_e,
                                          t2_re, t2_im, t2_e)
                iteration = target
                total += skip
                jumped = True
                break

            if not jumped:
                # Exact step: d' = 2*Z_n*d + d^2 + dc
                t1_re, t1_im, t1_e = dfx_mul(
                    zm_re[iteration], zm_im[iteration], z_exp[iteration],
                    d_re, d_im, d_e,
                )
                t2_re, t2_im, t2_e = dfx_mul(d_re, d_im, d_e,
                                             d_re, d_im, d_e)
                s_re, s_im, s_e = dfx_add(t1_re, t1_im, t1_e + 1,
                                          t2_re, t2_im, t2_e)
                d_re, d_im, d_e = dfx_add(s_re, s_im, s_e,
                                          dc_re, dc_im, dc_e)
                iteration += 1
                total += 1

        smooth_out[x, y] = -1.0  # interior

    return _deep_cuda


_DEEP_CUDA = None


def _get_cuda_kernel():
    global _DEEP_CUDA
    if _DEEP_CUDA is None:
        _DEEP_CUDA = _make_cuda_kernel()
    return _DEEP_CUDA


# ============================================================================
# CPU kernel
# ============================================================================

@njit(parallel=True)
def _deep_cpu(
    zm_re, zm_im, z_exp,
    ref_num_iters,
    dc_min_m_re, dc_min_m_im, dc_step_m_re, dc_step_m_im, dc_exp,
    max_iter, height, width,
    bla_a_m_re, bla_a_m_im, bla_a_e,
    bla_b_m_re, bla_b_m_im, bla_b_e,
    bla_r_log2, bla_offsets, bla_sizes, bla_bits, bla_num_levels,
):
    smooth_out = np.empty((height, width), dtype=np.float64)

    # Degenerate orbit: the forced end-of-orbit rebase would spin without
    # advancing `total`.
    if ref_num_iters < 1:
        smooth_out[:, :] = -1.0
        return smooth_out

    for x in prange(height):
        for y in range(width):
            dc_re, dc_im, dc_e = fx_norm(
                dc_min_m_re + y * dc_step_m_re,
                dc_min_m_im + x * dc_step_m_im,
                dc_exp,
            )

            d_re, d_im, d_e = dc_re, dc_im, dc_e
            iteration = 1
            total = 1
            resolved = False

            while total < max_iter:
                full_re, full_im, full_e = fx_add(
                    zm_re[iteration], zm_im[iteration], z_exp[iteration],
                    d_re, d_im, d_e,
                )

                if full_e >= 4:
                    l2 = fx_mag_sq_log2(full_re, full_im, full_e)
                    if l2 > ESCAPE_LOG2:
                        log2_zabs = 0.5 * l2
                        nu = math.log(log2_zabs) / LN2
                        smooth_out[x, y] = float(total) + 1.0 - nu
                        resolved = True
                        break

                # Rebase is bookkeeping, not an iteration — see CUDA kernel.
                if iteration >= ref_num_iters or fx_mag_lt(
                    full_re, full_im, full_e, d_re, d_im, d_e
                ):
                    d_re, d_im, d_e = full_re, full_im, full_e
                    iteration = 0
                    continue

                d_l2 = 0.5 * fx_mag_sq_log2(d_re, d_im, d_e)
                jumped = False
                for li in range(bla_num_levels - 1, -1, -1):
                    skip = 1 << bla_bits[li]
                    target = iteration + skip
                    if target > ref_num_iters:
                        continue
                    if total + skip > max_iter:
                        continue
                    if iteration >= bla_sizes[li]:
                        continue
                    idx = bla_offsets[li] + iteration
                    if d_l2 >= bla_r_log2[idx]:
                        continue

                    t1_re, t1_im, t1_e = fx_mul(
                        bla_a_m_re[idx], bla_a_m_im[idx], bla_a_e[idx],
                        d_re, d_im, d_e,
                    )
                    t2_re, t2_im, t2_e = fx_mul(
                        bla_b_m_re[idx], bla_b_m_im[idx], bla_b_e[idx],
                        dc_re, dc_im, dc_e,
                    )
                    d_re, d_im, d_e = fx_add(t1_re, t1_im, t1_e,
                                             t2_re, t2_im, t2_e)
                    iteration = target
                    total += skip
                    jumped = True
                    break

                if not jumped:
                    t1_re, t1_im, t1_e = fx_mul(
                        zm_re[iteration], zm_im[iteration], z_exp[iteration],
                        d_re, d_im, d_e,
                    )
                    t2_re, t2_im, t2_e = fx_mul(d_re, d_im, d_e,
                                                d_re, d_im, d_e)
                    s_re, s_im, s_e = fx_add(t1_re, t1_im, t1_e + 1,
                                             t2_re, t2_im, t2_e)
                    d_re, d_im, d_e = fx_add(s_re, s_im, s_e,
                                             dc_re, dc_im, dc_e)
                    iteration += 1
                    total += 1

            if not resolved:
                smooth_out[x, y] = -1.0

    return smooth_out


# ============================================================================
# Dispatch
# ============================================================================

def render_gpu_deep(
    ref, bla_table,
    dc_min_m_re, dc_min_m_im, dc_step_m_re, dc_step_m_im, dc_exp,
    max_iter, height, width,
):
    """Dispatch the deep floatexp kernel to CUDA.

    Args:
        ref: ReferenceOrbit with extended (floatexp) arrays populated.
        bla_table: BLATableFXP from compute_bla_table_fxp.
        dc_*: Per-frame dc grid — mantissas plus shared power-of-two exponent.
        max_iter, height, width: Render parameters.

    Returns:
        (smooth_out, glitch_out) — glitch_out is all zeros (deep kernel
        handles precision via rebasing, not glitch passes).
    """
    kernel = _get_cuda_kernel()

    d_zm_re = cuda.to_device(ref.z_m_re)
    d_zm_im = cuda.to_device(ref.z_m_im)
    d_z_exp = cuda.to_device(ref.z_exp)

    smooth_out = np.zeros((height, width), dtype=np.float64)
    d_smooth = cuda.to_device(smooth_out)

    d_a_m_re = cuda.to_device(bla_table.a_m_re)
    d_a_m_im = cuda.to_device(bla_table.a_m_im)
    d_a_e = cuda.to_device(bla_table.a_e)
    d_b_m_re = cuda.to_device(bla_table.b_m_re)
    d_b_m_im = cuda.to_device(bla_table.b_m_im)
    d_b_e = cuda.to_device(bla_table.b_e)
    d_r_log2 = cuda.to_device(bla_table.r_log2)
    d_offsets = cuda.to_device(bla_table.level_offsets)
    d_sizes = cuda.to_device(bla_table.level_sizes)
    d_bits = cuda.to_device(bla_table.level_bits)

    threads_per_block = (16, 16)
    blocks = ((height + 15) // 16, (width + 15) // 16)

    kernel[blocks, threads_per_block](
        d_zm_re, d_zm_im, d_z_exp,
        ref.num_iters,
        dc_min_m_re, dc_min_m_im, dc_step_m_re, dc_step_m_im, dc_exp,
        max_iter,
        d_a_m_re, d_a_m_im, d_a_e,
        d_b_m_re, d_b_m_im, d_b_e,
        d_r_log2, d_offsets, d_sizes, d_bits, bla_table.num_levels,
        d_smooth,
    )

    d_smooth.copy_to_host(smooth_out)
    glitch_out = np.zeros((height, width), dtype=np.int32)
    return smooth_out, glitch_out


def render_cpu_deep(
    ref, bla_table,
    dc_min_m_re, dc_min_m_im, dc_step_m_re, dc_step_m_im, dc_exp,
    max_iter, height, width,
):
    """Dispatch the deep floatexp kernel to the CPU (Numba parallel)."""
    smooth_out = _deep_cpu(
        ref.z_m_re, ref.z_m_im, ref.z_exp,
        ref.num_iters,
        dc_min_m_re, dc_min_m_im, dc_step_m_re, dc_step_m_im, dc_exp,
        max_iter, height, width,
        bla_table.a_m_re, bla_table.a_m_im, bla_table.a_e,
        bla_table.b_m_re, bla_table.b_m_im, bla_table.b_e,
        bla_table.r_log2, bla_table.level_offsets, bla_table.level_sizes,
        bla_table.level_bits, bla_table.num_levels,
    )
    glitch_out = np.zeros((height, width), dtype=np.int32)
    return smooth_out, glitch_out
