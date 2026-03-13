"""Easing functions for smooth interpolation.

Each function maps t in [0, 1] to t' in [0, 1], with f(0) = 0 and f(1) = 1.
Used by the cinematic interpolation mode for smooth acceleration/deceleration.
"""

import math


def linear(t: float) -> float:
    """Linear (no easing)."""
    return t


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in (accelerate)."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out (decelerate)."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out (Hermite smoothstep). C1 continuous."""
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_quint(t: float) -> float:
    """Quintic ease-in-out (smoother step). C2 continuous."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_out_sine(t: float) -> float:
    """Sinusoidal ease-in-out."""
    return 0.5 * (1.0 - math.cos(math.pi * t))


# Registry for JSON serialization
EASING_FUNCTIONS = {
    "linear": linear,
    "ease_in": ease_in_quad,
    "ease_out": ease_out_quad,
    "ease_in_out": ease_in_out_cubic,
    "smooth": ease_in_out_cubic,
    "smoother": ease_in_out_quint,
    "sine": ease_in_out_sine,
}


def get_easing(name: str):
    """Look up an easing function by name.

    Args:
        name: Easing function name (see EASING_FUNCTIONS).

    Returns:
        Callable (float) -> float.

    Raises:
        KeyError: If name is not a known easing function.
    """
    if name not in EASING_FUNCTIONS:
        raise KeyError(
            f"Unknown easing '{name}'. Available: {sorted(EASING_FUNCTIONS.keys())}"
        )
    return EASING_FUNCTIONS[name]
