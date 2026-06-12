"""Bilinear Approximation (BLA) for ultra-deep Mandelbrot zoom.

At extreme zoom (1e50+), the perturbation delta iteration:
    d_{n+1} = 2*Z_n*d_n + d_n^2 + dc
needs millions of iterations per pixel. BLA pre-computes linear approximation
coefficients from the reference orbit so multiple iterations can be skipped:
    d_{n+k} ≈ A_k(n) * d_n + B_k(n) * dc

Coefficients are organized in a binary tree (levels = powers of 2):
  Level 0: single-step   A_1(n) = 2*Z_n,  B_1(n) = 1
  Level L: 2^L steps     composed from two level-(L-1) jumps

At runtime, the GPU kernel walks the tree from highest level down, choosing
the largest valid jump where |d_n| < validity_radius(level, n).

Reference: Claude Heiland-Allen, "Perturbation and BLA for the Mandelbrot set"
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class BLATable:
    """Pre-computed BLA coefficients for a reference orbit.

    All arrays are flat 1D, indexed by level_offsets[level] + n.

    Attributes:
        a_re, a_im: Complex A coefficients (derivative chain).
        b_re, b_im: Complex B coefficients (dc accumulator).
        validity_r: Maximum |d_n| for the approximation to be valid.
        level_offsets: Start index of each level in the flat arrays.
        level_sizes: Number of valid entries per level.
        num_levels: Total number of levels.
        total_entries: Total entries across all levels.
    """
    a_re: np.ndarray
    a_im: np.ndarray
    b_re: np.ndarray
    b_im: np.ndarray
    validity_r: np.ndarray
    level_offsets: np.ndarray  # int64
    level_sizes: np.ndarray   # int64
    num_levels: int
    total_entries: int


@njit(cache=True)
def _build_bla_level0(z_re, z_im, num_iters, pixel_spacing):
    """Build level-0 BLA coefficients (single-step linear approximation).

    The linearization of d_{n+1} = 2*Z_n*d_n + d_n^2 + dc is:
        d_{n+1} ≈ 2*Z_n * d_n + 1 * dc
    so A_1(n) = 2*Z_n, B_1(n) = 1.

    The validity radius is where the dropped d^2 term becomes significant
    relative to the pixel spacing: |d|^2 < epsilon * pixel_spacing.
    For level 0: r(n) = pixel_spacing (d^2 < pixel_spacing when |d| < sqrt(ps),
    but we use a relative criterion: r = epsilon / |A| where epsilon scales
    with pixel spacing).
    """
    n = num_iters  # number of single-step entries
    a_re = np.empty(n, dtype=np.float64)
    a_im = np.empty(n, dtype=np.float64)
    b_re = np.empty(n, dtype=np.float64)
    b_im = np.empty(n, dtype=np.float64)
    validity = np.empty(n, dtype=np.float64)

    epsilon = pixel_spacing * 1e-6  # tolerance for the dropped d^2 term

    for i in range(n):
        # A_1(n) = 2*Z_n
        a_re[i] = 2.0 * z_re[i]
        a_im[i] = 2.0 * z_im[i]
        # B_1(n) = 1
        b_re[i] = 1.0
        b_im[i] = 0.0
        # Validity: |d|^2 term is negligible when |d| < epsilon / |A|
        a_mag = math.sqrt(a_re[i] * a_re[i] + a_im[i] * a_im[i])
        if a_mag > 0:
            validity[i] = epsilon / a_mag
        else:
            validity[i] = 1e30  # Z_n = 0, any d is fine

    return a_re, a_im, b_re, b_im, validity


@njit(cache=True)
def _compose_bla_level(
    prev_a_re, prev_a_im, prev_b_re, prev_b_im, prev_r,
    prev_size, half_step,
):
    """Compose level L from level L-1 by combining pairs of jumps.

    For a 2k-step jump starting at n:
        d_{n+2k} = A_k(n+k) * (A_k(n) * d_n + B_k(n) * dc) + B_k(n+k) * dc
                 = [A_k(n+k) * A_k(n)] * d_n + [A_k(n+k) * B_k(n) + B_k(n+k)] * dc

    So: A_{2k}(n) = A_k(n+k) * A_k(n)
        B_{2k}(n) = A_k(n+k) * B_k(n) + B_k(n+k)

    Validity: r_{2k}(n) = min(r_k(n), r_k(n+k) / |A_k(n)|)
    The second term accounts for d growing by factor |A_k(n)| during the first half,
    which must still be within the validity radius of the second half.
    """
    # Number of valid entries: can compose pairs where n+half_step < prev_size
    new_size = prev_size - half_step
    if new_size <= 0:
        return (
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64), 0,
        )

    a_re = np.empty(new_size, dtype=np.float64)
    a_im = np.empty(new_size, dtype=np.float64)
    b_re = np.empty(new_size, dtype=np.float64)
    b_im = np.empty(new_size, dtype=np.float64)
    validity = np.empty(new_size, dtype=np.float64)

    for n in range(new_size):
        # First half: coefficients at position n
        a1_re = prev_a_re[n]
        a1_im = prev_a_im[n]
        b1_re = prev_b_re[n]
        b1_im = prev_b_im[n]
        r1 = prev_r[n]

        # Second half: coefficients at position n + half_step
        idx2 = n + half_step
        a2_re = prev_a_re[idx2]
        a2_im = prev_a_im[idx2]
        b2_re = prev_b_re[idx2]
        b2_im = prev_b_im[idx2]
        r2 = prev_r[idx2]

        # A_{2k}(n) = A_k(n+k) * A_k(n)   (complex multiply)
        a_re[n] = a2_re * a1_re - a2_im * a1_im
        a_im[n] = a2_re * a1_im + a2_im * a1_re

        # B_{2k}(n) = A_k(n+k) * B_k(n) + B_k(n+k)
        b_re[n] = (a2_re * b1_re - a2_im * b1_im) + b2_re
        b_im[n] = (a2_re * b1_im + a2_im * b1_re) + b2_im

        # Validity: min(r1, r2 / |A1|)
        a1_mag = math.sqrt(a1_re * a1_re + a1_im * a1_im)
        if a1_mag > 0:
            validity[n] = min(r1, r2 / a1_mag)
        else:
            validity[n] = r1

    return a_re, a_im, b_re, b_im, validity, new_size


def compute_bla_table(
    z_re: np.ndarray,
    z_im: np.ndarray,
    num_iters: int,
    pixel_spacing: float,
    max_memory_mb: float = 512.0,
) -> BLATable:
    """Compute BLA coefficient table from a reference orbit.

    Args:
        z_re, z_im: Reference orbit arrays (float64, length num_iters+1).
        num_iters: Number of iterations in the reference orbit.
        pixel_spacing: Viewport pixel spacing (view_height / height).
            Used to calibrate the validity radius.
        max_memory_mb: Maximum memory budget for the BLA table.

    Returns:
        BLATable with flat arrays ready for GPU upload.
    """
    if num_iters < 2:
        # Too short for BLA
        return BLATable(
            a_re=np.empty(0), a_im=np.empty(0),
            b_re=np.empty(0), b_im=np.empty(0),
            validity_r=np.empty(0),
            level_offsets=np.array([0], dtype=np.int64),
            level_sizes=np.array([0], dtype=np.int64),
            num_levels=0, total_entries=0,
        )

    # Maximum levels: log2(num_iters)
    max_levels = int(math.log2(num_iters)) + 1

    # Memory budget: each entry = 5 float64 = 40 bytes
    # Total entries across all levels ≈ 2 * num_iters (geometric sum)
    bytes_per_entry = 5 * 8
    max_entries = int(max_memory_mb * 1024 * 1024 / bytes_per_entry)

    # Build level 0
    l0_a_re, l0_a_im, l0_b_re, l0_b_im, l0_r = _build_bla_level0(
        z_re, z_im, num_iters, pixel_spacing
    )

    # Store levels in lists, then flatten
    levels_a_re = [l0_a_re]
    levels_a_im = [l0_a_im]
    levels_b_re = [l0_b_re]
    levels_b_im = [l0_b_im]
    levels_r = [l0_r]
    level_sizes = [num_iters]
    total = num_iters

    # Build higher levels by composing pairs
    for level in range(1, max_levels):
        half_step = 1 << (level - 1)  # 2^(level-1)
        prev_size = level_sizes[-1]

        if prev_size <= half_step:
            break  # Can't compose any more pairs

        a_re, a_im, b_re, b_im, r, new_size = _compose_bla_level(
            levels_a_re[-1], levels_a_im[-1],
            levels_b_re[-1], levels_b_im[-1],
            levels_r[-1], prev_size, half_step,
        )

        if new_size <= 0:
            break

        # Check memory budget
        if total + new_size > max_entries:
            break

        levels_a_re.append(a_re[:new_size])
        levels_a_im.append(a_im[:new_size])
        levels_b_re.append(b_re[:new_size])
        levels_b_im.append(b_im[:new_size])
        levels_r.append(r[:new_size])
        level_sizes.append(new_size)
        total += new_size

    num_levels = len(level_sizes)

    # Flatten all levels into contiguous arrays
    flat_a_re = np.concatenate(levels_a_re)
    flat_a_im = np.concatenate(levels_a_im)
    flat_b_re = np.concatenate(levels_b_re)
    flat_b_im = np.concatenate(levels_b_im)
    flat_r = np.concatenate(levels_r)

    # Compute level offsets (cumulative sum of sizes)
    offsets = np.zeros(num_levels, dtype=np.int64)
    for i in range(1, num_levels):
        offsets[i] = offsets[i - 1] + level_sizes[i - 1]

    sizes = np.array(level_sizes, dtype=np.int64)

    return BLATable(
        a_re=flat_a_re,
        a_im=flat_a_im,
        b_re=flat_b_re,
        b_im=flat_b_im,
        validity_r=flat_r,
        level_offsets=offsets,
        level_sizes=sizes,
        num_levels=num_levels,
        total_entries=total,
    )
