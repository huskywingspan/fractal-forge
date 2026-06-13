"""Bilinear Approximation (BLA) for ultra-deep Mandelbrot zoom.

At extreme zoom (1e50+), the perturbation delta iteration:
    d_{n+1} = 2*Z_n*d_n + d_n^2 + dc
needs millions of iterations per pixel. BLA pre-computes linear approximation
coefficients from the reference orbit so multiple iterations can be skipped:
    d_{n+k} ≈ A_k(n) * d_n + B_k(n) * dc

Coefficients are organized in a binary tree (levels = powers of 2):
  Level 0: single-step   A_1(n) = 2*Z_n,  B_1(n) = 1
  Level L: 2^L steps     composed from two level-(L-1) jumps

Validity radii follow the research-validated formulation (Zhuoran / Claude
Heiland-Allen) rather than the earlier epsilon/|A| heuristic:
    single step:  r_1(n)   = max(0, (eps*|Z_{n+1}| - |B_1|*|dc_max|) / |A_1|)
    merge x+y:    r_{x+y}  = min(r_x, max(0, (r_y - |B_x|*|dc_max|) / |A_x|))
The |B|*|dc_max| term accounts for the spatial divergence contributed by dc
during the skipped segment, which the old heuristic ignored — that omission
is what tears BLA apart beyond ~1e50.

Two table flavors:
  - BLATable     float64 coefficients, for the mid-depth kernel (<= ~1e150)
  - BLATableFXP  floatexp coefficients (mantissa pair + int64 exponent) with
                 log2-domain validity radii, for the deep kernel. Needed
                 because |A| grows multiplicatively and overflows float64 on
                 long orbits, and radii shrink below 1e-308.

Reference: Claude Heiland-Allen, "Perturbation and BLA for the Mandelbrot set"
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from fractalforge.engine.floatexp import EXP_ZERO, LOG2_ZERO

# Relative linearization error tolerance for float64 delta iteration.
# Research range for float64 implementations is 1e-26..1e-14; this default
# sits toward the conservative middle. Larger = more/longer jumps but risk
# of visible approximation artifacts; smaller = safer but slower.
DEFAULT_BLA_EPS = 1e-16

# Skips below this many iterations are barely faster than exact stepping but
# cost table memory and lookup bandwidth, so the deep table culls them.
DEFAULT_MIN_SKIP_LEVEL = 4  # 2^4 = 16 iterations


@dataclass
class BLATable:
    """Pre-computed float64 BLA coefficients for a reference orbit.

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


@dataclass
class BLATableFXP:
    """Extended-range BLA table for the deep (floatexp) kernel.

    Coefficients are stored as floatexp triples; validity radii in the log2
    domain so values far below 1e-308 remain representable. Levels with
    skips below 2^min_skip_level are culled — the kernel falls back to exact
    floatexp stepping there anyway.

    level_bits[li] gives the tree level (skip = 1 << level_bits[li]) of the
    li-th *stored* level, since low levels are culled.
    """
    a_m_re: np.ndarray
    a_m_im: np.ndarray
    a_e: np.ndarray       # int64
    b_m_re: np.ndarray
    b_m_im: np.ndarray
    b_e: np.ndarray       # int64
    r_log2: np.ndarray    # float64, log2 of validity radius
    level_offsets: np.ndarray  # int64
    level_sizes: np.ndarray    # int64
    level_bits: np.ndarray     # int64
    num_levels: int
    total_entries: int


# ============================================================================
# Float64 table (mid-depth kernel)
# ============================================================================

@njit(cache=True)
def _build_bla_level0(z_re, z_im, num_iters, dc_max, eps):
    """Build level-0 BLA coefficients (single-step linear approximation).

    The linearization of d_{n+1} = 2*Z_n*d_n + d_n^2 + dc is:
        d_{n+1} ≈ 2*Z_n * d_n + 1 * dc
    so A_1(n) = 2*Z_n, B_1(n) = 1.

    Validity radius (research formula):
        r_1(n) = max(0, (eps*|Z_{n+1}| - |B_1|*dc_max) / |A_1|)
    """
    n = num_iters  # number of single-step entries
    a_re = np.empty(n, dtype=np.float64)
    a_im = np.empty(n, dtype=np.float64)
    b_re = np.empty(n, dtype=np.float64)
    b_im = np.empty(n, dtype=np.float64)
    validity = np.empty(n, dtype=np.float64)

    for i in range(n):
        # A_1(n) = 2*Z_n
        a_re[i] = 2.0 * z_re[i]
        a_im[i] = 2.0 * z_im[i]
        # B_1(n) = 1
        b_re[i] = 1.0
        b_im[i] = 0.0

        a_mag = math.sqrt(a_re[i] * a_re[i] + a_im[i] * a_im[i])
        z1_mag = math.sqrt(z_re[i + 1] * z_re[i + 1] + z_im[i + 1] * z_im[i + 1])
        if a_mag > 0.0:
            r = (eps * z1_mag - dc_max) / a_mag
            validity[i] = r if r > 0.0 else 0.0
        else:
            # A = 0 (orbit at the critical point): the dropped d^2 term is
            # the whole step — no safe linear jump from here.
            validity[i] = 0.0

    return a_re, a_im, b_re, b_im, validity


@njit(cache=True)
def _compose_bla_level(
    prev_a_re, prev_a_im, prev_b_re, prev_b_im, prev_r,
    prev_size, half_step, dc_max,
):
    """Compose level L from level L-1 by combining pairs of jumps.

    For a 2k-step jump starting at n:
        d_{n+2k} = A_k(n+k) * (A_k(n) * d_n + B_k(n) * dc) + B_k(n+k) * dc
                 = [A_k(n+k) * A_k(n)] * d_n + [A_k(n+k) * B_k(n) + B_k(n+k)] * dc

    So: A_{2k}(n) = A_k(n+k) * A_k(n)
        B_{2k}(n) = A_k(n+k) * B_k(n) + B_k(n+k)

    Validity (research formula):
        r_{2k}(n) = min(r_k(n), max(0, (r_k(n+k) - |B_k(n)|*dc_max) / |A_k(n)|))
    The second term ensures the delta arriving at the second half — grown by
    A_k(n) and pushed by B_k(n)*dc — still fits that half's radius.
    """
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

        a1_mag = math.sqrt(a1_re * a1_re + a1_im * a1_im)
        b1_mag = math.sqrt(b1_re * b1_re + b1_im * b1_im)
        if a1_mag > 0.0:
            r_second = (r2 - b1_mag * dc_max) / a1_mag
            if r_second < 0.0:
                r_second = 0.0
            validity[n] = min(r1, r_second)
        else:
            validity[n] = 0.0

    return a_re, a_im, b_re, b_im, validity, new_size


def compute_bla_table(
    z_re: np.ndarray,
    z_im: np.ndarray,
    num_iters: int,
    dc_max: float,
    eps: float = DEFAULT_BLA_EPS,
    max_memory_mb: float = 512.0,
) -> BLATable:
    """Compute a float64 BLA coefficient table from a reference orbit.

    Args:
        z_re, z_im: Reference orbit arrays (float64, length num_iters+1).
        num_iters: Number of iterations in the reference orbit.
        dc_max: Maximum |dc| across the viewport (corner pixel offset from
            the reference). Used in the validity radius formulas.
        eps: Relative linearization error tolerance (see DEFAULT_BLA_EPS).
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
    bytes_per_entry = 5 * 8
    max_entries = int(max_memory_mb * 1024 * 1024 / bytes_per_entry)

    # Build level 0
    l0_a_re, l0_a_im, l0_b_re, l0_b_im, l0_r = _build_bla_level0(
        z_re, z_im, num_iters, dc_max, eps
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
            levels_r[-1], prev_size, half_step, dc_max,
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


# ============================================================================
# Floatexp table (deep kernel) — vectorized numpy build
# ============================================================================
#
# Complex values are (m_re, m_im, e) array triples; real magnitudes are
# (m, e) pairs with m >= 0. All exponents int64. Zero is (0, 0, EXP_ZERO).

def _vnorm(re, im, e):
    """Vectorized floatexp normalize: max(|re|,|im|) into [0.5, 1)."""
    mag = np.maximum(np.abs(re), np.abs(im))
    nz = mag > 0.0
    _, k = np.frexp(mag)
    k = k.astype(np.int64)
    out_re = np.where(nz, np.ldexp(re, -k.astype(np.int32)), 0.0)
    out_im = np.where(nz, np.ldexp(im, -k.astype(np.int32)), 0.0)
    out_e = np.where(nz, e + k, EXP_ZERO)
    return out_re, out_im, out_e


def _vmul(a_re, a_im, a_e, b_re, b_im, b_e):
    """Vectorized floatexp complex multiply."""
    p_re = a_re * b_re - a_im * b_im
    p_im = a_re * b_im + a_im * b_re
    return _vnorm(p_re, p_im, a_e + b_e)


def _vadd(a_re, a_im, a_e, b_re, b_im, b_e):
    """Vectorized floatexp complex add with exponent alignment."""
    base_e = np.maximum(a_e, b_e)
    sh_a = np.clip(a_e - base_e, -1100, 0).astype(np.int32)
    sh_b = np.clip(b_e - base_e, -1100, 0).astype(np.int32)
    s_re = np.ldexp(a_re, sh_a) + np.ldexp(b_re, sh_b)
    s_im = np.ldexp(a_im, sh_a) + np.ldexp(b_im, sh_b)
    return _vnorm(s_re, s_im, base_e)


def _vrnorm(m, e):
    """Vectorized real-magnitude normalize (m >= 0)."""
    nz = m > 0.0
    _, k = np.frexp(m)
    k = k.astype(np.int64)
    out_m = np.where(nz, np.ldexp(m, -k.astype(np.int32)), 0.0)
    out_e = np.where(nz, e + k, EXP_ZERO)
    return out_m, out_e


def _vrmag(re, im, e):
    """Magnitude of a complex floatexp triple as a real pair."""
    return _vrnorm(np.hypot(re, im), e)


def _vrsub_clamp0(a_m, a_e, b_m, b_e):
    """Real pair subtraction a - b, clamped at 0."""
    base_e = np.maximum(a_e, b_e)
    sh_a = np.clip(a_e - base_e, -1100, 0).astype(np.int32)
    sh_b = np.clip(b_e - base_e, -1100, 0).astype(np.int32)
    s = np.ldexp(a_m, sh_a) - np.ldexp(b_m, sh_b)
    s = np.maximum(s, 0.0)
    return _vrnorm(s, base_e)


def _vrdiv(a_m, a_e, b_m, b_e):
    """Real pair division a / b. b entries of 0 produce 0 (caller guards)."""
    safe_b = np.where(b_m > 0.0, b_m, 1.0)
    q = np.where(b_m > 0.0, a_m / safe_b, 0.0)
    return _vrnorm(q, a_e - b_e)


def _vrlog2(m, e):
    """Real pair -> log2 value (LOG2_ZERO for zero)."""
    nz = m > 0.0
    return np.where(nz, e.astype(np.float64) + np.log2(np.where(nz, m, 1.0)),
                    LOG2_ZERO)


def compute_bla_table_fxp(
    z_m_re: np.ndarray,
    z_m_im: np.ndarray,
    z_exp: np.ndarray,
    num_iters: int,
    dc_max_m: float,
    dc_max_e: int,
    eps: float = DEFAULT_BLA_EPS,
    min_skip_level: int = DEFAULT_MIN_SKIP_LEVEL,
    max_memory_mb: float = 1024.0,
) -> BLATableFXP:
    """Compute an extended-range BLA table from a floatexp reference orbit.

    Args:
        z_m_re, z_m_im, z_exp: Floatexp orbit triples (length num_iters+1).
        num_iters: Number of iterations in the reference orbit.
        dc_max_m, dc_max_e: Maximum |dc| across the viewport as a floatexp
            real pair (mantissa, exponent).
        eps: Relative linearization error tolerance.
        min_skip_level: Cull stored levels below this (skip < 2^level);
            composition still passes through them.
        max_memory_mb: Memory budget for the *stored* table.

    Returns:
        BLATableFXP ready for GPU upload.
    """
    empty = BLATableFXP(
        a_m_re=np.empty(0), a_m_im=np.empty(0), a_e=np.empty(0, dtype=np.int64),
        b_m_re=np.empty(0), b_m_im=np.empty(0), b_e=np.empty(0, dtype=np.int64),
        r_log2=np.empty(0),
        level_offsets=np.array([0], dtype=np.int64),
        level_sizes=np.array([0], dtype=np.int64),
        level_bits=np.array([0], dtype=np.int64),
        num_levels=0, total_entries=0,
    )
    if num_iters < (1 << min_skip_level) + 1:
        return empty

    n = num_iters
    max_levels = int(math.log2(n)) + 1
    bytes_per_entry = 7 * 8
    max_entries = int(max_memory_mb * 1024 * 1024 / bytes_per_entry)

    dc_m_arr = np.float64(dc_max_m)
    dc_e_arr = np.int64(dc_max_e)

    # ---- Level 0: A = 2*Z_n, B = 1 ----
    a_re = z_m_re[:n].copy()
    a_im = z_m_im[:n].copy()
    a_e = z_exp[:n].astype(np.int64) + 1  # x2
    # Z_0 = 0 gives A = 0 there; its EXP_ZERO exponent +1 stays ~EXP_ZERO.
    b_re = np.ones(n, dtype=np.float64) * 0.5
    b_im = np.zeros(n, dtype=np.float64)
    b_e = np.ones(n, dtype=np.int64)  # 0.5 * 2^1 = 1.0 (normalized form)

    # r_1 = max(0, (eps*|Z_{n+1}| - dc_max) / |A_1|)
    z1_m, z1_e = _vrmag(z_m_re[1:n + 1], z_m_im[1:n + 1],
                        z_exp[1:n + 1].astype(np.int64))
    ez_m, ez_e = _vrnorm(z1_m * eps, z1_e)
    num_m, num_e = _vrsub_clamp0(ez_m, ez_e, dc_m_arr, dc_e_arr)
    amag_m, amag_e = _vrmag(a_re, a_im, a_e)
    r_m, r_e = _vrdiv(num_m, num_e, amag_m, amag_e)
    # A = 0 entries: no valid jump
    r_m = np.where(amag_m > 0.0, r_m, 0.0)

    cur = dict(a_re=a_re, a_im=a_im, a_e=a_e,
               b_re=b_re, b_im=b_im, b_e=b_e,
               r_m=r_m, r_e=r_e, size=n)

    stored = []  # list of (level_bits, dict snapshot)
    total_stored = 0
    if min_skip_level == 0:
        stored.append((0, cur))
        total_stored += n

    for level in range(1, max_levels):
        half = 1 << (level - 1)
        size = cur["size"]
        new_size = size - half
        if new_size <= 0:
            break

        s = slice(0, new_size)
        s2 = slice(half, half + new_size)

        a1 = (cur["a_re"][s], cur["a_im"][s], cur["a_e"][s])
        a2 = (cur["a_re"][s2], cur["a_im"][s2], cur["a_e"][s2])
        b1 = (cur["b_re"][s], cur["b_im"][s], cur["b_e"][s])
        b2 = (cur["b_re"][s2], cur["b_im"][s2], cur["b_e"][s2])
        r1 = (cur["r_m"][s], cur["r_e"][s])
        r2 = (cur["r_m"][s2], cur["r_e"][s2])

        # A' = A2 * A1 ; B' = A2 * B1 + B2
        na = _vmul(*a2, *a1)
        nb = _vadd(*_vmul(*a2, *b1), *b2)

        # r' = min(r1, max(0, (r2 - |B1|*dc) / |A1|))
        b1mag_m, b1mag_e = _vrmag(*b1)
        push_m, push_e = _vrnorm(b1mag_m * dc_m_arr, b1mag_e + dc_e_arr)
        rem_m, rem_e = _vrsub_clamp0(r2[0], r2[1], push_m, push_e)
        a1mag_m, a1mag_e = _vrmag(*a1)
        rsec_m, rsec_e = _vrdiv(rem_m, rem_e, a1mag_m, a1mag_e)
        rsec_m = np.where(a1mag_m > 0.0, rsec_m, 0.0)

        # elementwise min of two real pairs via log2 compare
        l1 = _vrlog2(r1[0], r1[1])
        l2 = _vrlog2(rsec_m, rsec_e)
        take1 = l1 <= l2
        nr_m = np.where(take1, r1[0], rsec_m)
        nr_e = np.where(take1, r1[1], rsec_e)

        cur = dict(a_re=na[0], a_im=na[1], a_e=na[2],
                   b_re=nb[0], b_im=nb[1], b_e=nb[2],
                   r_m=nr_m, r_e=nr_e, size=new_size)

        if level >= min_skip_level:
            if total_stored + new_size > max_entries:
                break
            stored.append((level, cur))
            total_stored += new_size

    if not stored:
        return empty

    num_levels = len(stored)
    offsets = np.zeros(num_levels, dtype=np.int64)
    sizes = np.empty(num_levels, dtype=np.int64)
    bits = np.empty(num_levels, dtype=np.int64)
    for i, (lvl, d) in enumerate(stored):
        sizes[i] = d["size"]
        bits[i] = lvl
        if i > 0:
            offsets[i] = offsets[i - 1] + sizes[i - 1]

    return BLATableFXP(
        a_m_re=np.concatenate([d["a_re"] for _, d in stored]),
        a_m_im=np.concatenate([d["a_im"] for _, d in stored]),
        a_e=np.concatenate([d["a_e"] for _, d in stored]).astype(np.int64),
        b_m_re=np.concatenate([d["b_re"] for _, d in stored]),
        b_m_im=np.concatenate([d["b_im"] for _, d in stored]),
        b_e=np.concatenate([d["b_e"] for _, d in stored]).astype(np.int64),
        r_log2=np.concatenate([_vrlog2(d["r_m"], d["r_e"]) for _, d in stored]),
        level_offsets=offsets,
        level_sizes=sizes,
        level_bits=bits,
        num_levels=num_levels,
        total_entries=int(total_stored),
    )
