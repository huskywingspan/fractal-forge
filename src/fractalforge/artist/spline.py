"""Catmull-Rom spline interpolation for smooth camera paths.

Operates in zoom-scaled screen space: positions are transformed so that
distances correspond to visual distances on screen, preventing overshoots
at deep zoom where tiny complex-plane offsets fill the entire viewport.
"""

import math


def catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float,
                alpha: float = 0.5) -> float:
    """Centripetal Catmull-Rom interpolation between p1 and p2.

    Args:
        p0, p1, p2, p3: Control points (p1 and p2 are the segment endpoints).
        t: Parameter in [0, 1] within the p1-p2 segment.
        alpha: Parameterization (0=uniform, 0.5=centripetal, 1=chordal).

    Returns:
        Interpolated value at parameter t.
    """
    # For 1D points, use simple distance for knot parameterization
    # Fall back to uniform spacing (1.0) when points coincide
    def knot_interval(a, b):
        d = abs(b - a)
        if d < 1e-30:
            return 1.0  # uniform fallback for coincident points
        return d ** alpha

    t0 = 0.0
    t1 = t0 + knot_interval(p0, p1)
    t2 = t1 + knot_interval(p1, p2)
    t3 = t2 + knot_interval(p2, p3)

    # Map t from [0,1] to [t1, t2]
    u = t1 + t * (t2 - t1)

    # Guard against degenerate intervals
    def safe_lerp(ta, tb, va, vb, u_val):
        denom = tb - ta
        if abs(denom) < 1e-30:
            return 0.5 * (va + vb)
        return (tb - u_val) / denom * va + (u_val - ta) / denom * vb

    # De Boor-like evaluation
    a1 = safe_lerp(t0, t1, p0, p1, u)
    a2 = safe_lerp(t1, t2, p1, p2, u)
    a3 = safe_lerp(t2, t3, p2, p3, u)

    b1 = safe_lerp(t0, t2, a1, a2, u)
    b2 = safe_lerp(t1, t3, a3, a2, u)

    c = safe_lerp(t1, t2, b1, b2, u)
    return c


def catmull_rom_2d(
    points_re: list[float],
    points_im: list[float],
    t_global: float,
) -> tuple[float, float]:
    """Evaluate Catmull-Rom spline through a sequence of 2D points.

    For endpoints, phantom points are reflected to maintain natural behavior
    (the spline approaches the first/last point with the same tangent as the
    first/last segment).

    Args:
        points_re: Real components of the control points.
        points_im: Imaginary components of the control points.
        t_global: Global parameter in [0, n-1] where n is the number of points.
                  Integer values correspond to the control points.

    Returns:
        (re, im) tuple of the interpolated position.
    """
    n = len(points_re)
    if n < 2:
        return points_re[0], points_im[0]

    # Clamp to valid range
    t_global = max(0.0, min(float(n - 1), t_global))

    # Find segment index and local t
    seg = int(t_global)
    if seg >= n - 1:
        seg = n - 2
    t_local = t_global - seg

    # Get the 4 control points (p0, p1, p2, p3) with phantom endpoints
    def get_point(idx):
        if idx < 0:
            # Reflect: p[-1] = 2*p[0] - p[1]
            return (2.0 * points_re[0] - points_re[1],
                    2.0 * points_im[0] - points_im[1])
        elif idx >= n:
            # Reflect: p[n] = 2*p[n-1] - p[n-2]
            return (2.0 * points_re[-1] - points_re[-2],
                    2.0 * points_im[-1] - points_im[-2])
        return points_re[idx], points_im[idx]

    p0_re, p0_im = get_point(seg - 1)
    p1_re, p1_im = get_point(seg)
    p2_re, p2_im = get_point(seg + 1)
    p3_re, p3_im = get_point(seg + 2)

    re = catmull_rom(p0_re, p1_re, p2_re, p3_re, t_local)
    im = catmull_rom(p0_im, p1_im, p2_im, p3_im, t_local)
    return re, im


def smooth_zoom_path(
    keyframe_zooms: list[float],
    keyframe_frames: list[int],
    frame: int,
    easing_fn=None,
) -> float:
    """Interpolate zoom with optional easing for smooth acceleration.

    Zoom is interpolated exponentially (linear in log-space), with an optional
    easing function applied to the per-segment t parameter to smooth velocity
    changes at keyframe boundaries.

    Args:
        keyframe_zooms: Zoom levels at each keyframe.
        keyframe_frames: Frame numbers at each keyframe.
        frame: The frame to interpolate at.
        easing_fn: Optional easing function (float -> float) for the t parameter.

    Returns:
        Interpolated zoom level.
    """
    n = len(keyframe_zooms)
    if n < 2:
        return keyframe_zooms[0]

    # Clamp
    if frame <= keyframe_frames[0]:
        return keyframe_zooms[0]
    if frame >= keyframe_frames[-1]:
        return keyframe_zooms[-1]

    # Find segment
    for i in range(n - 1):
        if keyframe_frames[i] <= frame <= keyframe_frames[i + 1]:
            span = keyframe_frames[i + 1] - keyframe_frames[i]
            t = (frame - keyframe_frames[i]) / span if span > 0 else 0.0

            if easing_fn is not None:
                t = easing_fn(t)

            log_z0 = math.log(keyframe_zooms[i])
            log_z1 = math.log(keyframe_zooms[i + 1])
            return math.exp(log_z0 + t * (log_z1 - log_z0))

    return keyframe_zooms[-1]
