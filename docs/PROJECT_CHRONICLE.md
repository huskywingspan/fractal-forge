# FractalForge -- Project Chronicle

> **Purpose:** Institutional knowledge capture for AI-assisted development. This document records architecture decisions, failed approaches, performance findings, and lessons learned throughout the project's development.
>
> **For Copilot Agents:** Reference this document when working on FractalForge to avoid repeating past mistakes and understand why things are built the way they are.
>
> **Version:** 4.0 (March 9, 2026) -- alpha1.1.0: Julia sets, Burning Ship fractal, compilation pipeline with crossfade transitions.

---

## Project Status Dashboard

| Component | Status | Notes |
|-----------|--------|-------|
| **Mandelbrot Engine** | ✅ Done | Numba CUDA + CPU fallback, smooth iteration |
| **Coloring System** | ✅ Done | 5 palettes, gradient interpolation, SSAA |
| **Zoom Path Planner** | ✅ Done | Keyframes, zoom-weighted position interp |
| **Frame Renderer** | ✅ Done | Single-frame PNG, supersampling support |
| **Video Pipeline** | ✅ Done | FFmpeg, 4 encode presets, checkpoint/resume |
| **CLI** | ✅ Done | render, zoom, encode, info, palettes, resolutions, zoom-template, title, thumbnail, short, compile |
| **Configuration** | ✅ Done | Pydantic models, 7 resolution presets |
| **Perturbation Theory** | ✅ Done | Reference orbit, delta kernel, SA, glitch correction, auto-select |
| **Julia Set Engine** | ✅ Done | CUDA + CPU kernels, c-parameter interpolation in zoom paths |
| **Burning Ship Engine** | ✅ Done | CUDA + CPU kernels, y-axis flip for correct orientation |
| **Compilation Pipeline** | ✅ Done | Multi-clip assembly with crossfade transitions |
| **Post-Processing** | ✅ Done | Vignette, contrast/saturation/brightness, histogram EQ, 8x SSAA |
| **YouTube Pipeline** | ✅ Done | Title cards, thumbnails, Shorts, YouTube encode preset |
| **Live Preview** | 🔲 Planned | Interactive zoom preview window |
| **RunPod Integration** | 🔲 Future | Remote GPU rendering for 4K final output |

### Rendering Targets

| Format | Resolution | Aspect | Use Case |
|--------|-----------|--------|----------|
| 720p | 1280x720 | 16:9 | Fast test renders |
| 1080p | 1920x1080 | 16:9 | Preview / iteration |
| 2K Ultrawide | 5120x1440 | 32:9 | Personal monitor playback |
| 4K | 3840x2160 | 16:9 | YouTube / publish-ready |

### Hardware

| Environment | GPU | VRAM | Role |
|-------------|-----|------|------|
| Local | RTX 3070 | 8 GB | Development, preview, mid-quality renders |
| RunPod | A40/A100 (TBD) | 48/80 GB | 4K final renders, deep zoom production |

---

## Reference Documents

| Document | Purpose |
|----------|---------|
| [WORK_TRACKER.md](WORK_TRACKER.md) | Open work items and backlog |
| CLAUDE.md | Project conventions for AI agents |

---

## Table of Contents

1. [Architecture Decisions](#architecture-decisions)
2. [Performance Notes](#performance-notes)
3. [Bugs & Incidents](#bugs--incidents)
4. [Failed Approaches](#failed-approaches)
5. [Lessons Learned](#lessons-learned)

---

## Architecture Decisions

### AD-001: Python + Numba CUDA over C++/CUDA

**Date:** 2026-03-08
**Decision:** Use Python with Numba CUDA kernels as the primary rendering engine.
**Rationale:**

- Numba compiles Python to PTX (CUDA assembly) -- performance is near-native CUDA C++
- Dramatically faster iteration cycle for experimenting with coloring algorithms, palettes, zoom paths
- Easy integration with NumPy, Pillow, matplotlib for image handling
- PyTorch available as fallback for tensor-based operations or RunPod scaling
- C++ rewrite remains an option if we hit a Numba wall, but unlikely for this use case

**Trade-offs:**

- Numba's CUDA subset is limited (no dynamic allocation, limited stdlib)
- Debugging GPU kernels is harder than CPU Python
- Some advanced CUDA features (cooperative groups, shared memory tiling) require workarounds

### AD-002: Perturbation Theory for Deep Zooms

**Date:** 2026-03-08
**Decision:** Implement perturbation theory (PT) with series approximation (SA) for zooms beyond ~1e15.
**Rationale:**

- Standard `float64` loses precision at ~1e15 zoom depth
- True arbitrary precision (mpmath) on GPU is impractical -- too slow
- PT computes one reference orbit at full precision (CPU, mpmath), then all pixels as perturbations in float64 (GPU)
- This is the standard approach used by Kalles Fraktaler, Ultra Fractal, and Maths Town
- Series approximation (SA) can skip thousands of early iterations, further accelerating deep zooms

### AD-003: Keyframe-Based Zoom Path System

**Date:** 2026-03-08
**Decision:** Zoom videos defined as keyframe sequences with interpolation, stored as JSON presets.
**Rationale:**

- A zoom video is a camera path through fractal parameter space (center_re, center_im, zoom_level)
- Keyframes define artistic control points; smooth interpolation fills in frames
- Exponential interpolation for zoom level (linear looks wrong -- zoom is multiplicative)
- JSON presets allow sharing, version control, and reproducibility
- Future: GUI keyframe editor, but CLI-first for now

### AD-004: CLI-First Design

**Date:** 2026-03-08
**Decision:** Build a full CLI (`fractalforge`) as the primary interface before any GUI.
**Rationale:**

- CLI is scriptable, composable, and works over SSH (important for RunPod)
- Render jobs can be parallelized, resumed, and automated
- Preview window is a separate concern -- can be added without touching core logic
- Click chosen as CLI framework (with Rich for output formatting)

### AD-005: Zoom-Weighted Position Interpolation

**Date:** 2026-03-08
**Decision:** Position interpolation between keyframes scales offset as `1/zoom` rather than linear in time.
**Rationale:**

- Linear position interpolation causes the zoom target to drift off-screen at deep zoom levels
- Because viewport width is proportional to 1/zoom, a fixed-distance offset in complex plane coordinates becomes increasingly far in screen coordinates as zoom increases
- Formula: `center = target + (start - target) * (zoom_start / zoom_current)`
- This keeps the zoom target locked in the viewport throughout the dive
- The visual effect is a smooth, centered dive -- exactly what Maths Town style videos look like

**Previous approach (failed):** Linear interpolation for position caused visible thrashing/jitter starting around 200x zoom when the start and end centers differ significantly.

### AD-006: Supersampling for Anti-Aliasing

**Date:** 2026-03-08
**Decision:** Render at Nx resolution then downsample with Lanczos filter for anti-aliasing.
**Rationale:**

- Fractal boundaries create extreme high-frequency content -- adjacent pixels can have wildly different iteration counts
- Standard approach: render at `width*ss x height*ss`, color the full-res image, then downsample
- Coloring before downsample is correct (averaging iteration counts would give wrong colors due to palette cycling)
- 2x SSAA (4 samples/pixel) is the sweet spot: visible improvement, ~2.3x cost on stills, ~3.4x on video
- Available via `--ss 2` on both `render` and `zoom` CLI commands

### AD-007: Perturbation Theory Implementation Strategy

**Date:** 2026-03-08
**Decision:** Implement perturbation theory as a separate engine module with automatic selection based on zoom level.
**Rationale:**

- Standard float64 iteration gives wrong results at deep zoom (validated: 63 iterations off at just 50,000x zoom)
- Perturbation theory computes one reference orbit at arbitrary precision (mpmath, CPU), then all pixels as float64 deltas (GPU)
- Delta formula: `d_{n+1} = 2*Z_n*d_n + d_n^2 + dc` where dc = pixel offset from reference
- Auto-selection threshold: zoom >= 1e13 uses PT, below uses standard -- seamless to the user
- Series approximation (3rd-order Taylor) skips early iterations at deep zoom (enabled at zoom >= 1e8)
- Glitch detection flags pixels where `|d_n|^2 > tolerance * |Z_n|^2`, then re-renders with a new reference orbit
- Up to 3 glitch correction passes with automatic reference point selection
- Rebasing: when reference orbit escapes before a pixel, falls back to standard iteration from the full value

**Trade-offs:**

- Reference orbit computation adds overhead (~40-160ms depending on max_iter and precision)
- Series approximation only helps at deep zoom where dc is tiny; disabled at moderate zoom to avoid cubic approximation error
- Glitch correction re-renders the full frame per pass (could optimize to render only glitched pixels)
- Zoom path keyframe coordinates still stored as JSON floats (~15 digit limit), constraining zoom video depth to ~1e12

### AD-008: Fractal Type Routing Architecture

**Date:** 2026-03-09
**Decision:** All fractal types produce the same `smooth_data` float64 array, with dispatch by `fractal_type` parameter.
**Rationale:**

- Adding new fractal types (Julia, Burning Ship, future: Tricorn, Phoenix, etc.) should not require changes to the coloring, SSAA, post-processing, encoding, or YouTube pipeline
- Each engine module (`julia.py`, `burning_ship.py`) follows the same API pattern: `render_frame_*(center_re, center_im, zoom, width, height, max_iter, use_gpu) -> np.ndarray`
- The `Keyframe` dataclass carries `fractal_type`, `julia_re`, `julia_im` fields with backward-compatible defaults
- For Julia sets, the c-parameter is interpolated linearly between keyframes, enabling "Julia morphing" animations
- Routing happens in `frame_renderer.py` (single frames) and `sequence.py` (zoom videos)

### AD-009: Compilation Pipeline with Crossfade Transitions

**Date:** 2026-03-09
**Decision:** Multi-clip compilation assembler that copies pre-rendered frame ranges and inserts crossfade transitions.
**Rationale:**

- YouTube compilations ("Best Of", "Top 10") are a key content format for the Infinite Descent channel
- The pipeline takes a JSON spec with clip references, start times, durations, and transition config
- Crossfade: linear alpha blend `out = A * (1-t) + B * t` for smooth visual transitions
- Frames are renumbered sequentially, then encoded via the standard FFmpeg pipeline
- Extensible: future transition types (fade_black, wipe, zoom_match) can be added as new blend functions

### AD-010: Cinematic Camera Motion System

**Date:** 2026-03-12
**Decision:** Add a `"cinematic"` interpolation mode alongside the existing `"legacy"` mode for zoom paths.
**Problem:**

- Legacy zoom-weighted interpolation creates C0-continuous but not C1-continuous paths
- At keyframe boundaries where the target center changes, camera velocity jumps abruptly (visible as "jump cuts")
- Zoom rate also steps discontinuously between segments with different zoom ratios

**Solution:**

- **Catmull-Rom splines in zoom-scaled screen space** for position: `screen_pos = (center - reference) * zoom`. This ensures distances correspond to visual distances, preventing overshoots at deep zoom.
- **Easing functions** (Hermite smoothstep) on per-segment t for zoom interpolation: smooth acceleration/deceleration at keyframe boundaries
- New fields on `Keyframe`: `easing` (default "ease_in_out"), `tension` (default 0.5)
- New field on `ZoomPath`: `interpolation` (default "legacy")
- `camera-path` CLI command for visualizing path quality (position, velocity, zoom rate plots)
- For 2-keyframe paths, falls back to legacy (which is already optimal for single-target dives)

**Key insight:** The spline must operate in screen-space (`(center - ref) * zoom`), not raw complex-plane coordinates, because at deep zoom a tiny complex-plane offset can fill the entire screen.

---

## Performance Notes

### PERF-001: Baseline Render Times (RTX 3070, Phase 1)

**Date:** 2026-03-08
**Hardware:** RTX 3070, 8 GB VRAM, compute 8.6

| Resolution | Zoom | Max Iter | Time | File Size |
|-----------|------|----------|------|-----------|
| 1920x1080 (1080p) | 1x | 1000 | 0.59s | 201 KB |
| 1920x1080 (1080p) | 200x | 2000 | 0.60s | ~200 KB |
| 5120x1440 (32:9 superwide) | 500x | 2000 | 1.46s | 4.2 MB |
| 3840x2160 (4K) | 100x | 3000 | 1.39s | 1.1 MB |

**Observations:**

- Render time scales roughly with pixel count, not zoom depth (at standard precision)
- 4K is under 1.5s -- very comfortable for video frame sequences
- VRAM usage is minimal at these resolutions (well under 1 GB)

### PERF-002: Zoom Video Pipeline (720p, Phase 2)

**Date:** 2026-03-08
**Test:** Seahorse Valley dive, 301 frames (5s @ 60fps), 1x -> 50,000x zoom, 720p

| Stage | Time | Rate |
|-------|------|------|
| Frame render (301 frames) | 41.7s | 7.2 fps |
| Encode (preview/x264 crf23) | 2.2s | 17.8 MB output |
| Encode (quality/x264 crf18) | 4.2s | 24.6 MB output |
| Resume (skip existing) | 0.0s | instant |
| **Total (render + encode)** | **44.4s** | |

**Observations:**

- GPU render is the bottleneck, not encoding
- At 720p/7.2fps, a 1-minute video (3,600 frames) takes ~8.3 minutes
- At 1080p, expect ~5fps, so 1 min video ~12 min render
- Resume/checkpoint works reliably for interrupted renders

### PERF-003: Supersampling Cost (720p, 2x SSAA)

**Date:** 2026-03-08
**Test:** Same seahorse dive, 301 frames, 720p, with 2x supersampling

| Mode | Render Time | FPS | Slowdown |
|------|------------|-----|----------|
| No SSAA (1x) | 45.7s | 6.6 fps | baseline |
| 2x SSAA (4 spp) | 154.5s | 1.9 fps | 3.4x |

**Single frame (1080p, 5000x zoom):**

| Mode | Time | Slowdown |
|------|------|----------|
| No SSAA | 0.62s | baseline |
| 2x SSAA | 1.44s | 2.3x |

**Production estimates (RTX 3070, 2x SSAA):**

- 1-min 720p video: ~2.6 min render
- 1-min 1080p video: ~5-6 min render (estimated)
- 1-min 4K video: ~20-25 min render (estimated, better on RunPod)

### PERF-004: Perturbation Theory Render Times (RTX 3070, Phase 3)

**Date:** 2026-03-08
**Hardware:** RTX 3070, 8 GB VRAM, compute 8.6

| Resolution | Zoom | Max Iter | Engine | SSAA | Time |
|-----------|------|----------|--------|------|------|
| 1080p | 5e5 | 2500 | Standard | 1x | 0.48s |
| 1080p | 1e8 | 5000 | PT | 1x | 0.42s |
| 1080p | 1e8 | 5000 | PT | 2x | 1.68s |
| 1080p | 1x | 500 | Standard | 1x | 0.21s |
| 640x360 | 1e14 | 8000 | PT | 1x | 1.01s |

**Observations:**

- PT at 1e8 is actually *faster* than standard at 5e5 (0.42s vs 0.48s) -- series approximation skips early iterations
- Reference orbit overhead is small (~40ms for 5000 iters at 25-digit precision, ~160ms for 10000 iters at 60 digits)
- At 1e14 zoom, higher per-pixel cost but still under 1s at small resolution
- Validated: float64 gives wrong escape iteration at 50,000x zoom (2717 vs correct 2780 = 63 iterations off)

---

## Bugs & Incidents

### BUG-001: Unicode Encoding Error on Windows Console

**Date:** 2026-03-08
**Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'` when printing checkmark/arrow characters via Rich on Windows cp1252 terminal.
**Fix:** Used `Console(force_terminal=True)` and replaced Unicode symbols with ASCII equivalents.

---

## Failed Approaches

### FAIL-001: Linear Position Interpolation for Zoom Paths

**Date:** 2026-03-08
**What:** Interpolated center position linearly between keyframes while zoom grew exponentially.
**Why it failed:** At deep zoom levels (~200x+), the viewport is tiny but position is still far from the target. The zoom target appears to race off-screen, causing visible jitter/thrashing.
**Replaced with:** Zoom-weighted interpolation where offset scales as `zoom_start / zoom_current`. See AD-005.

---

## Lessons Learned

### LL-001: Zoom Target Coordinates Need High Precision

**Date:** 2026-03-08
At deep zoom levels, even small errors in the target coordinate matter enormously. Our first seahorse dive preset used coordinates with ~7 decimal places and landed in a featureless blue region at 50,000x zoom. Adding 4 more decimal places to target a boundary region fixed it. For very deep zooms (1e10+), coordinates need 15+ significant digits.

### LL-002: Color in Supersampled Space, Not Iteration Space

**Date:** 2026-03-08
When implementing supersampling, the correct order is: render iterations at high res -> color at high res -> downsample the RGB image. Averaging iteration counts before coloring gives wrong results because the palette wraps cyclically -- two iteration counts on opposite sides of the palette would average to a completely wrong middle color.

### LL-003: Chunked Palette Coloring Prevents OOM at High SSAA

**Date:** 2026-03-09
At 8x SSAA on 1080p (15360x8640 = 132M pixels), the palette coloring step tried to allocate multiple float64 arrays simultaneously (ext_vals, t, idx0, idx1, frac, color0, color1, interpolated), totaling ~13 GB for 132M exterior pixels. Fix: process the palette mapping in row chunks of ~8M pixels, keeping peak memory under 1 GB regardless of resolution or SSAA level.

### LL-004: Julia Set Zoom Targets Need Boundary Awareness

**Date:** 2026-03-09
Unlike Mandelbrot, where the main cardioid boundary is well-known, Julia set boundaries depend on the c-parameter and are not predictable from coordinates alone. Arbitrary deep zoom coordinates in Julia sets often land in regions with uniform iteration counts, producing boring wallpapers (3-8 unique colors). Moderate zooms (4-10x) near visually obvious boundary regions work much better for wallpapers. For zoom videos, start wide (zoom=1) and let the camera find the boundary naturally.
