# FractalForge — Project Chronicle

> **Purpose:** Institutional knowledge capture for AI-assisted development. This document records architecture decisions, failed approaches, performance findings, and lessons learned throughout the project's development.
>
> **For Copilot Agents:** Reference this document when working on FractalForge to avoid repeating past mistakes and understand why things are built the way they are.
>
> **Version:** 1.0 (March 8, 2026) — Project inception. Architecture designed, scaffolding laid.

---

## Project Status Dashboard

| Component | Status | Notes |
|-----------|--------|-------|
| **Mandelbrot Engine** | 🔲 Planned | Numba CUDA kernel, smooth iteration |
| **Perturbation Theory** | 🔲 Planned | Deep zoom (1e50+) via reference orbit |
| **Coloring System** | 🔲 Planned | Palette engine, smooth interpolation |
| **Zoom Path Planner** | 🔲 Planned | Keyframe-based zoom trajectory |
| **Frame Renderer** | 🔲 Planned | Single-frame PNG export |
| **Video Pipeline** | 🔲 Planned | FFmpeg sequence → MP4/ProRes |
| **CLI** | 🔲 Planned | Full CLI with render, preview, export commands |
| **Live Preview** | 🔲 Planned | Interactive zoom preview window |
| **Post-Processing** | 🔲 Planned | Motion blur, vignette, color grading |
| **RunPod Integration** | 🔲 Future | Remote GPU rendering for 4K final output |

### Rendering Targets

| Format | Resolution | Aspect | Use Case |
|--------|-----------|--------|----------|
| 1080p | 1920×1080 | 16:9 | Preview / fast iteration |
| 2K Ultrawide | 5120×1440 | 32:9 | Personal monitor playback |
| 4K | 3840×2160 | 16:9 | YouTube / publish-ready |

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
| README.md | Project overview and quickstart |

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
- Numba compiles Python to PTX (CUDA assembly) — performance is near-native CUDA C++
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
- True arbitrary precision (mpmath) on GPU is impractical — too slow
- PT computes one reference orbit at full precision (CPU, mpmath), then all pixels as perturbations in float64 (GPU)
- This is the standard approach used by Kalles Fraktaler, Ultra Fractal, and Maths Town
- Series approximation (SA) can skip thousands of early iterations, further accelerating deep zooms

### AD-003: Keyframe-Based Zoom Path System
**Date:** 2026-03-08
**Decision:** Zoom videos defined as keyframe sequences with interpolation, stored as JSON presets.
**Rationale:**
- A zoom video is a camera path through fractal parameter space (center_re, center_im, zoom_level)
- Keyframes define artistic control points; smooth interpolation fills in frames
- Exponential interpolation for zoom level (linear looks wrong — zoom is multiplicative)
- JSON presets allow sharing, version control, and reproducibility
- Future: GUI keyframe editor, but CLI-first for now

### AD-004: CLI-First Design
**Date:** 2026-03-08
**Decision:** Build a full CLI (`fractalforge`) as the primary interface before any GUI.
**Rationale:**
- CLI is scriptable, composable, and works over SSH (important for RunPod)
- Render jobs can be parallelized, resumed, and automated
- Preview window is a separate concern — can be added without touching core logic
- Click or Typer for CLI framework (TBD)

---

## Performance Notes

*No entries yet — will track GPU kernel timings, frame render benchmarks, memory usage patterns.*

---

## Bugs & Incidents

*No entries yet.*

---

## Failed Approaches

*No entries yet.*

---

## Lessons Learned

*No entries yet.*
