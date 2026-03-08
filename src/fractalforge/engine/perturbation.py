"""Perturbation theory engine for deep Mandelbrot zooms.

At zoom depths beyond ~1e15, standard float64 arithmetic loses precision.
Perturbation theory solves this by:
1. Computing one reference orbit at arbitrary precision (CPU, mpmath)
2. Expressing all other pixels as small deltas from that reference (GPU, float64)

This module will be implemented in Phase 3.
"""

# TODO P3-01: Arbitrary precision reference orbit computation
# TODO P3-02: Perturbation iteration kernel (GPU)
# TODO P3-03: Glitch detection and correction
# TODO P3-04: Series approximation for iteration skipping
# TODO P3-05: Rebasing logic for delta orbit divergence
