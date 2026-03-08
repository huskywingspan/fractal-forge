"""Precision management — arbitrary precision arithmetic for reference orbits.

Uses mpmath for computing reference orbits at zoom depths where float64
is insufficient. The reference orbit is computed on CPU, then used by
the perturbation kernel on GPU.

This module will be implemented in Phase 3.
"""

# TODO P3-01: Reference orbit computation with mpmath
# TODO P3-06: Auto-detect required precision from zoom level
