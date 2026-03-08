# FractalForge -- Project Chronicle

> **Purpose:** Institutional knowledge capture for AI-assisted development. This document records architecture decisions, failed approaches, performance findings, and lessons learned throughout the project's development.
>
> **For Copilot Agents:** Reference this document when working on FractalForge to avoid repeating past mistakes and understand why things are built the way they are.
>
> **Version:** 2.0 (March 8, 2026) -- Phase 1+2 complete, polish pass done. Zoom-weighted interpolation, supersampling AA shipped.

---

## Project Status Dashboard

| Component | Status | Notes |
|-----------|--------|-------|
| **Mandelbrot Engine** | ✅ Done | Numba CUDA + CPU fallback, smooth iteration |
| **Coloring System** | ✅ Done | 5 palettes, gradient interpolation, SSAA |
| **Zoom Path Planner** | ✅ Done | Keyframes, zoom-weighted position interp |
| **Frame Renderer** | ✅ Done | Single-frame PNG, supersampling support |
| **Video Pipeline** | ✅ Done | FFmpeg, 4 encode presets, checkpoint/resume |
| **CLI** | ✅ Done | render, zoom, encode, info, palettes, resolutions, zoom-template |
| **Configuration** | ✅ Done | Pydantic models, 7 resolution presets |
| **Perturbation Theory** | 🔲 Planned | Deep zoom (1e50+) via reference orbit |
| **Live Preview** | 🔲 Planned | Interactive zoom preview window |
| **Post-Processing** | 🔲 Planned | Motion blur, vignette, color grading |
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
