# Sprint: Deep Zoom Phase 1 — Foundational Precision & Rebasing

**Target:** Stable rendering at 1e50 zoom (up from current 1e15 limit)
**Sprint ID:** DZ-P1
**Date:** 2026-03-14
**Source:** Deep research analysis (docs/DEEP_RESEARCH_RESULTS.md)

---

## Problem Statement

FractalForge's perturbation renderer breaks down beyond ~1e15 zoom. Most pixels
flag as "glitched" (precision exhausted) because:

1. **Insufficient reference orbit precision** — current formula `log10(zoom) + 10`
   digits provides inadequate margin for error propagation at extreme depth
2. **SA coefficients computed in float64** — truncation errors at the series
   approximation stage inject fatal inaccuracy into the starting delta state
3. **No proactive rebasing** — when a pixel's delta drifts far from the reference
   orbit, the only recovery is expensive reactive re-referencing (new CPU orbit +
   full re-render), which doesn't scale
4. **Glitch tolerance scaling** — the current linear ramp floors at 1e-6, which is
   too aggressive for zoom depths beyond 1e30

## Deliverables

### DZ-P1-01: Reference Orbit Precision Formula

**File:** `src/fractalforge/engine/precision.py` → `required_precision()`

**Current (broken):**
```python
digits = int(math.log10(zoom)) + margin  # margin=10
```

**New (research-validated):**
```python
digits = int(1.5 * math.log10(zoom)) + 30
```

**Rationale:** At zoom 1e200, the old formula gives 210 digits. The new formula
gives 330 digits. The 1.5x multiplier accounts for non-linear error propagation
during chaotic orbit segments where truncation compounds super-linearly. The +30
base provides headroom for intermediate arithmetic operations.

**Additionally:** When the center coordinate string has N significant digits, the
internal orbit computation should use at least N+20 digits to buffer against
accumulated rounding in the orbit generation loop. This means:
```python
digits = max(digits, len_significant_digits(center_str) + 20)
```

**Impact:** Immediate improvement for all zoom depths. No performance cost at
shallow zoom (auto-precision already gates on zoom level). At extreme zoom, orbit
computation takes longer due to higher-precision arithmetic, but this is a CPU-only
cost that scales linearly with digit count.

---

### DZ-P1-02: Arbitrary-Precision Series Approximation Coefficients

**Files:** `src/fractalforge/engine/precision.py` (new function),
`src/fractalforge/engine/perturbation.py` (caller update)

**Problem:** The SA coefficient recurrence (A, B, C) is currently computed by the
Numba `@njit` function `_compute_series_approximation()` in float64. At zoom 1e30+,
the coefficients A, B, C encode spatial derivatives that span many orders of
magnitude. Float64's 15-digit mantissa truncates these, injecting errors into every
pixel's starting delta value.

**Solution:** Compute SA coefficients in gmpy2/mpmath at the same precision as the
reference orbit, then downcast the final A, B, C values to float64 for GPU upload.

**New function:** `compute_series_approximation_hp()` in precision.py that:
1. Takes the reference orbit precision context (gmpy2 or mpmath)
2. Iterates the A/B/C recurrence at full arbitrary precision
3. Checks validity using high-precision |C * dc_max³| < tolerance
4. Returns final (skip_iters, A, B, C) as float64 after downcasting

**Caller change:** `render_frame_perturbation()` calls the new HP function instead
of the Numba `@njit` version when zoom >= 1e13.

**Impact:** Eliminates a class of silent precision bugs where SA injects bad
starting deltas. May increase SA skip count (more iterations deemed valid when
coefficients aren't truncated).

---

### DZ-P1-03: Proactive Rebasing in GPU/CPU Kernels

**Files:** `src/fractalforge/engine/perturbation.py` (both CUDA and CPU kernels),
`src/fractalforge/engine/bla_kernel.py` (both CUDA and CPU kernels)

**Problem:** Current kernels detect glitches reactively — when |d|² exceeds a
tolerance ratio vs |Z|², the pixel is flagged and the entire frame gets a costly
re-reference pass (new arbitrary-precision orbit on CPU + full re-render). This
approach:
- Caps at 3 correction passes (arbitrary limit)
- Each pass costs O(max_iter) CPU time for orbit computation
- Doesn't scale: at 1e50, the orbit itself takes minutes to compute

**Solution:** Add an inline rebasing check to the inner iteration loop. When the
perturbed orbit drifts too far from the reference (indicating impending cancellation),
fold the delta back into the reference orbit:

```
# Rebasing trigger: |Z_n + d_n| < |d_n|
# This means the combined orbit is closer to 0 than the delta alone,
# so Z_n and d_n are nearly cancelling — precision is about to be lost.
if |full_z|² < |d|²:
    d_re = full_re    # d becomes the absolute position
    d_im = full_im
    iteration_ref = 0  # restart reference index from beginning
```

**Why this works:** The Mandelbrot set has one critical point at 0+0i. When Z+d
approaches 0, the orbit is near the critical point and cancellation destroys
precision. By setting d = Z+d and restarting the reference index, we keep d small
relative to Z, maintaining float64's effective precision.

**Kernel changes (all 4 kernels):**
1. After computing `full_re, full_im` and before the escape check, add the
   rebasing condition
2. When triggered: `d = full_z`, reset the reference orbit index to 0
3. Continue the iteration loop (no early exit, no glitch flag)

**Impact:** Eliminates ~90% of reactive glitch correction passes. Transforms
glitch handling from O(passes × orbit_computation) CPU cost to O(1) GPU variable
swap. This is the single highest-impact change in the sprint.

---

### DZ-P1-04: Glitch Tolerance Scaling Update

**File:** `src/fractalforge/engine/perturbation.py` → `render_frame_perturbation()`

**Current:**
```python
exponent = 6.0 - 0.5 * (log_zoom - 13.0)
glitch_tolerance = 10.0 ** max(exponent, -6.0)  # floor at 1e-6
```

This floors at 1e-6 (reached at zoom 1e25), which is too tight — it triggers
false positives at extreme depth where orbits naturally pass close to minibrots.

**New:** With proactive rebasing handling most precision issues inline, the glitch
detector becomes a safety net rather than the primary defense. We can relax the
tolerance and extend the range:

```python
exponent = 6.0 - 0.3 * (log_zoom - 13.0)
glitch_tolerance = 10.0 ** max(exponent, -2.0)  # floor at 1e-2
```

Slower ramp (0.3 instead of 0.5) and higher floor (1e-2 instead of 1e-6) reduces
false positives while rebasing catches real precision issues proactively.

**Impact:** Fewer false glitch flags → fewer wasted re-reference passes → faster
renders at all zoom depths.

---

## Architecture Diagram

```
                    BEFORE (current)                          AFTER (this sprint)
                    ================                          ===================

Precision:          log10(zoom) + 10                          1.5 * log10(zoom) + 30
                    SA coeffs in float64                      SA coeffs in gmpy2/mpmath

Kernel loop:        iterate → escape? → glitch? → step       iterate → escape? → REBASE? → glitch? → step
                         ↓ (glitch)                                ↓ (glitch, rare)
                    flag pixel, exit                           flag pixel, exit
                         ↓                                         ↓
                    CPU: new orbit + re-render (×3)            CPU: new orbit + re-render (×3, rarely needed)

Glitch tolerance:   aggressive ramp, floor 1e-6               relaxed ramp, floor 1e-2
```

## Testing Strategy

1. **Regression test at 1e10:** Verify no visual change at shallow zoom (rebasing
   should never trigger, precision formula gives same or slightly higher precision)

2. **Improvement test at 1e15-1e20:** Render the same deep coordinate with old and
   new code. Compare:
   - Glitch pixel count (should drop dramatically)
   - Number of correction passes triggered
   - Visual quality (no block artifacts)

3. **Stretch test at 1e30-1e50:** Attempt renders at previously-impossible zoom
   depths. Success = any coherent fractal structure visible.

4. **Performance benchmark:** Time the reference orbit computation at various zoom
   depths to quantify the cost of higher precision.

## Test Coordinates

| Name | Re | Im | Target Zoom | Notes |
|------|----|----|-------------|-------|
| Seahorse Valley | -0.7436438870371587 | 0.1318259043091895 | 1e20 | Classic deep spiral |
| Elephant Valley | 0.2819296986239787 | 0.0100562697795524 | 1e25 | Dense detail |
| Deep Minibrot | -1.768778833 | -0.001738996 | 1e30 | Minibrot copy |
| Antenna Spike | -1.9999117501 | 0.0 | 1e40 | Real axis deep |

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Rebasing causes render artifacts | Medium | Compare against non-rebased renders at 1e15 where both work |
| HP SA coefficients too slow | Low | Only runs once per frame on CPU; orbit computation already dominates |
| Higher precision slows orbit computation | Expected | 1.5x more digits ≈ 2-3x slower orbit; acceptable tradeoff |
| Numba JIT recompilation needed | Certain | First render after code change will be slow (JIT warmup) |

## Dependencies

- gmpy2 (already installed, used by precision.py)
- mpmath (already installed, fallback)
- No new packages required

## Future Work (Phase 2: DZ-P2)

Items explicitly **not** in this sprint but enabled by it:
- BLA validity radius formula correction (needs the corrected precision foundation)
- BLA culling (skip <16-iter jumps for memory efficiency)
- Auto-epsilon pre-pass for BLA
- Floatexp / scaled delta arithmetic for underflow prevention
- Double-double arithmetic on GPU for 1e100+ zoom
