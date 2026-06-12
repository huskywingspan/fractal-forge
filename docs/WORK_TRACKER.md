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
| P2-02 | Zoom path interpolation (exponential zoom, smooth easing) | ✅ Done | Exp zoom, zoom-weighted position interp (AD-005), linear max_iter |
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
| P3-01 | Arbitrary precision reference orbit (mpmath) | ✅ Done | precision.py, auto-precision, validation util |
| P3-02 | Perturbation kernel (delta orbit iteration on GPU) | ✅ Done | CUDA + CPU kernels, smooth coloring |
| P3-03 | Glitch detection and correction | ✅ Done | Per-pixel flag, up to 3 correction passes |
| P3-04 | Series approximation (skip early iterations) | ✅ Done | 3rd-order Taylor, auto-enabled at zoom >= 1e8 |
| P3-05 | Rebasing logic (handle delta orbit divergence) | ✅ Done | Falls back to standard iteration after ref escape |
| P3-06 | Automatic precision selection (standard vs PT by zoom) | ✅ Done | Auto-switch at zoom >= 1e13, string coord support |

## Phase 4: Artistic Layer & YouTube Pipeline

*Goal: Visual polish and tooling for the Infinite Descent YouTube channel.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| P4-01 | Histogram equalization for iteration counts | ✅ Done | CDF-based remapping, --histogram CLI flag |
| P4-02 | Distance estimation coloring | 🔲 Todo | |
| P4-03 | Orbit trap coloring modes | 🔲 Todo | |
| P4-04 | Motion blur between frames | 🔲 Todo | |
| P4-05 | Vignette and color grading post-process | ✅ Done | --vignette, --contrast, --saturation, --brightness |
| P4-06 | Anti-aliasing (supersampling) | ✅ Done | 2x SSAA via --ss flag, Lanczos downsample |
| P4-07 | Palette editor / designer tool | 🔲 Todo | |
| P4-08 | Title card overlay renderer (RGBA PNG) | ✅ Done | `fractalforge title` CLI, gradient + brand fonts |
| P4-09 | Thumbnail auto-sampler with text overlay | ✅ Done | `fractalforge thumbnail`, 1280x720, gradient+text |
| P4-10 | YouTube Shorts crop mode (9:16 vertical) | ✅ Done | `fractalforge short`, center-crop + encode |
| P4-11 | Brand asset config (colors, fonts, watermark) | ✅ Done | BRAND dict, font fallback chain, scale_size util |
| P4-12 | YouTube-optimized encode preset | ✅ Done | CRF 16, H.264 High Profile, faststart, B-frames |

## Sprint alpha1.1.0: New Fractal Types & Compilation Pipeline

*Goal: Add Julia set and Burning Ship fractals, plus automated short compilation pipeline.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| S1-01 | Julia CUDA kernel | ✅ Done | Same structure as Mandelbrot, c fixed, z_0 = pixel |
| S1-02 | Julia CPU kernel | ✅ Done | Numba njit parallel fallback |
| S1-03 | `render_frame_julia()` function | ✅ Done | `engine/julia.py` |
| S1-04 | Route in `frame_renderer.py` | ✅ Done | `fractal_type` param dispatches to correct kernel |
| S1-05 | CLI: `--fractal` flag | ✅ Done | On `render` command; zoom uses keyframe `fractal_type` |
| S1-06 | Julia zoom path support | ✅ Done | `julia_re`, `julia_im` in Keyframe, c-param interpolation |
| S2-01 | Burning Ship CUDA kernel | ✅ Done | abs(z_re), abs(z_im) before squaring, y-flip |
| S2-02 | Burning Ship CPU kernel | ✅ Done | Numba njit parallel fallback |
| S2-03 | `render_frame_burning_ship()` function | ✅ Done | `engine/burning_ship.py` |
| S2-04 | Route through `--fractal burning_ship` | ✅ Done | Same pipeline as Julia routing |
| S3-01 | `CompilationSpec` data model | ✅ Done | Pydantic model in `publish/compilation.py` |
| S3-02 | Frame extraction / trim | ✅ Done | Copy frame ranges from rendered preset dirs |
| S3-03 | Crossfade transition renderer | ✅ Done | Linear alpha blend between clip endpoints |
| S3-04 | Compilation assembler | ✅ Done | Full assembly with transitions, renumbering |
| S3-05 | `fractalforge compile` CLI command | ✅ Done | Takes spec JSON, assembles + encodes |
| S4-01 | Julia zoom presets (5) | ✅ Done | dendrite, spiral, siegel, dragon, rabbit |
| S4-02 | Julia wallpaper set (10) | ✅ Done | 5 c-params x 2 views (full + zoomed) |
| S4-03 | Burning Ship zoom presets (3) | ✅ Done | full, antenna, armada |
| S4-04 | Burning Ship wallpaper set (6) | ✅ Done | full, antenna, antenna_deep, armada, bow, smokestack |
| S5-01 | Easing functions module | ✅ Done | `artist/easing.py`, 7 functions |
| S5-02 | Catmull-Rom spline module | ✅ Done | `artist/spline.py`, zoom-scaled screen space |
| S5-03 | Cinematic interpolation mode | ✅ Done | `zoompath.py`, C1-continuous via spline + easing |
| S5-04 | Camera path preview CLI | ✅ Done | `fractalforge camera-path`, comparison plots |
| S5-05 | `--interpolation` flag on zoom command | ✅ Done | Override mode at render time |

## Sprint DZ-P1: Deep Zoom Phase 1 -- Precision & Rebasing

*Goal: Stable rendering at 1e50 zoom (up from 1e15 limit).*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| DZ-P1-01 | Precision formula: 1.5*log10(zoom)+30 | ✅ Done | Plus coordinate string digit floor |
| DZ-P1-02 | Arbitrary-precision SA coefficients | ✅ Done | gmpy2/mpmath HP, downcast to float64 |
| DZ-P1-03 | Proactive rebasing in all 4 kernels | ✅ Done | CUDA+CPU perturbation & BLA, total_iters tracking |
| DZ-P1-04 | Glitch tolerance relaxation | ✅ Done | 0.3 ramp, 1e-2 floor (was 0.5/1e-6) |

## Sprint DZ-P3: Newton-Raphson Coordinate Finder

*Goal: Automated discovery of precise boundary coordinates for deep zoom.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| DZ-P3-01 | Period detection | ✅ Done | Orbit minimum tracking, handles exterior points |
| DZ-P3-02 | Newton's method for nucleus finding | ✅ Done | Arbitrary precision, auto period verification |
| DZ-P3-03 | Boundary point finder (internal angles) | ✅ Done | Full Jacobian, cusp offset workaround |
| DZ-P3-04 | `fractalforge discover` CLI | ✅ Done | JSON output, render test previews, CLI-ready commands |
| DZ-P3-05 | `fractalforge scan-region` CLI | ✅ Done | Grid scan for nuclei in a region |
| DZ-P3-06 | Deep target finder (nested minibrots) | ✅ Done | Recursive period-doubling/tripling |
| DZ-P3-07 | Updated test coordinates | ✅ Done | Newton-exact boundary + deep stress tests |

## Sprint: Ultra Deep Zoom & Viewer Video Render

*Goal: 1e200+ deep zoom capability and in-viewer video render launch.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| DZ-01 | gmpy2 fast path for reference orbit | ✅ Done | 8x speedup over mpmath at 50-digit precision |
| DZ-02 | BLA coefficient computation (bla.py) | ✅ Done | Binary tree of linear approx coefficients |
| DZ-03 | BLA CUDA + CPU kernel (bla_kernel.py) | ✅ Done | Adaptive iteration skipping, falls back to single-step |
| DZ-04 | BLA integration into perturbation pipeline | ✅ Done | Auto-enabled when ref orbit > 100 iters |
| DZ-05 | ZoomPath hp string coordinates | ✅ Done | center_re_hp/center_im_hp fields, mpmath interpolation |
| DZ-06 | Sequence renderer hp coord support | ✅ Done | Uses hp strings for perturbation frames |
| DZ-07 | Video render panel in viewer | ✅ Done | Resolution, duration, FPS, preset, SSAA, progress |
| DZ-08 | Viewer video render integration | ✅ Done | Background thread, cancel, auto zoom path generation |

## Phase 5: Production & Scaling

*Goal: RunPod integration, 4K renders, production workflow.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| P5-01 | RunPod deployment scripts | 🔲 Todo | |
| P5-02 | Distributed frame rendering (multi-GPU) | 🔲 Todo | |
| P5-03 | Render queue / job management | 🔲 Todo | |
| P5-04 | 4K render pipeline optimization | 🔲 Todo | |
| P5-05 | Audio generation tool (ambient + binaural) | 🔲 Todo | `fractalforge audio`, custom per-video soundscapes |
| P5-06 | Project manifest (single source of truth per video) | 🔲 Todo | JSON: preset, title, description, tags, assets |

---

## Backlog (Unscheduled)

| ID | Item | Priority | Notes |
|----|------|----------|-------|
| BL-01 | ~~Julia set support~~ | ✅ Done | Moved to Sprint alpha1.1.0 |
| BL-02 | ~~Burning Ship fractal~~ | ✅ Done | Moved to Sprint alpha1.1.0 |
| BL-03 | ~~Interactive viewer (Dear PyGui)~~ | ✅ Done | V1 MVP: canvas, controls, coords, bookmarks. `fractalforge viewer` |
| BL-04 | Web viewer (WebGPU) | Low | Browser-based preview |
| BL-05 | ~~Fractal location finder / scout tool~~ | ✅ Done | `fractalforge discover` + `scan-region`, Newton-Raphson |
| BL-06 | Benchmark suite | Medium | Track perf across changes |
| BL-07 | Cinematic camera: palette crossfade | Medium | Blend palettes over transition window instead of hard cut |
| BL-08 | Cinematic camera: per-keyframe easing presets | Low | "linear", "ease_in", "ease_out" per keyframe |
