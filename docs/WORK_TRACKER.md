# FractalForge — Work Tracker

> **Purpose:** Living backlog of open work items, organized by phase. Items move from Backlog → In Progress → Done as work proceeds.

---

## Phase 1: Core Engine MVP

*Goal: Render a Mandelbrot frame on GPU with smooth coloring, export to PNG.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| P1-01 | Project scaffolding (repo, venv, pyproject.toml) | ✅ Done | |
| P1-02 | Numba CUDA Mandelbrot kernel (basic escape time) | ✅ Done | GPU + CPU fallback |
| P1-03 | Smooth iteration count (continuous coloring) | ✅ Done | Escape radius 256 for smooth gradients |
| P1-04 | Palette system (gradient interpolation, preset palettes) | ✅ Done | 5 palettes: ocean, fire, electric, monochrome, nebula |
| P1-05 | Frame renderer (kernel → colored image → PNG) | ✅ Done | Full pipeline working |
| P1-06 | CLI: `fractalforge render` command | ✅ Done | With --preset, --cpu flags |
| P1-07 | CLI: `fractalforge info` (GPU info, capabilities) | ✅ Done | |
| P1-08 | Configuration system (render params, output settings) | ✅ Done | Pydantic models, 7 resolution presets |

## Phase 2: Zoom Video Pipeline

*Goal: Define zoom paths, render frame sequences, encode to video.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| P2-01 | Keyframe data model (center, zoom, rotation, palette) | ✅ Done | Dataclass with JSON save/load |
| P2-02 | Zoom path interpolation (exponential zoom, smooth easing) | ✅ Done | Exp zoom, linear position, linear max_iter |
| P2-03 | Frame sequence renderer (parallel frame dispatch) | ✅ Done | Progress tracking, per-frame callback |
| P2-04 | FFmpeg video encoder integration | ✅ Done | 4 presets: preview, quality, lossless, prores |
| P2-05 | CLI: `fractalforge zoom` command | ✅ Done | Full pipeline: frames + encode, --resume flag |
| P2-06 | CLI: `fractalforge encode` (frames → video) | ✅ Done | Standalone re-encode from frames dir |
| P2-07 | Render resume/checkpoint (restart interrupted renders) | ✅ Done | skip_existing flag, instant skip on resume |
| P2-08 | Resolution presets (1080p, 2K UW 32:9, 4K) | ✅ Done | Moved to P1, 7 presets incl. superwide 32:9 |

## Phase 3: Deep Zoom (Perturbation Theory)

*Goal: Unlock zoom depths beyond 1e15 using perturbation theory.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| P3-01 | Arbitrary precision reference orbit (mpmath) | 🔲 Todo | |
| P3-02 | Perturbation kernel (delta orbit iteration on GPU) | 🔲 Todo | |
| P3-03 | Glitch detection and correction | 🔲 Todo | |
| P3-04 | Series approximation (skip early iterations) | 🔲 Todo | |
| P3-05 | Rebasing logic (handle delta orbit divergence) | 🔲 Todo | |
| P3-06 | Automatic precision selection (standard vs PT by zoom) | 🔲 Todo | |

## Phase 4: Artistic Layer

*Goal: Post-processing, advanced coloring, visual polish.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| P4-01 | Histogram equalization for iteration counts | 🔲 Todo | |
| P4-02 | Distance estimation coloring | 🔲 Todo | |
| P4-03 | Orbit trap coloring modes | 🔲 Todo | |
| P4-04 | Motion blur between frames | 🔲 Todo | |
| P4-05 | Vignette and color grading post-process | 🔲 Todo | |
| P4-06 | Anti-aliasing (supersampling) | 🔲 Todo | |
| P4-07 | Palette editor / designer tool | 🔲 Todo | |

## Phase 5: Production & Scaling

*Goal: RunPod integration, 4K renders, production workflow.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| P5-01 | RunPod deployment scripts | 🔲 Todo | |
| P5-02 | Distributed frame rendering (multi-GPU) | 🔲 Todo | |
| P5-03 | Render queue / job management | 🔲 Todo | |
| P5-04 | 4K render pipeline optimization | 🔲 Todo | |
| P5-05 | Audio sync / soundtrack alignment | 🔲 Todo | |

---

## Backlog (Unscheduled)

| ID | Item | Priority | Notes |
|----|------|----------|-------|
| BL-01 | Julia set support | Medium | Parameterized by c value |
| BL-02 | Burning Ship fractal | Low | Different iteration formula |
| BL-03 | Interactive preview window (pygame/OpenGL) | Medium | Real-time zoom navigation |
| BL-04 | Web viewer (WebGPU) | Low | Browser-based preview |
| BL-05 | Fractal location finder / scout tool | Medium | Auto-discover interesting zoom targets |
| BL-06 | Benchmark suite | Medium | Track perf across changes |
