# FractalForge -- Project Chronicle

> **Purpose:** Institutional knowledge capture for AI-assisted development. This document records architecture decisions, failed approaches, performance findings, and lessons learned throughout the project's development.
>
> **For Copilot Agents:** Reference this document when working on FractalForge to avoid repeating past mistakes and understand why things are built the way they are.
>
> **Version:** 5.0 (June 13, 2026) -- DZ-P2: floatexp deep-zoom engine (unbounded magnification to 1e600+), Misiurewicz target finder, and a fully overhauled premium interactive viewer.

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
| **Live Preview** | ✅ Done | Dear PyGui viewer: canvas, controls, coords, video render panel |
| **BLA (Deep Zoom Accel)** | ✅ Done | Bilinear approximation for iteration skipping at 1e50+ |
| **gmpy2 Fast Path** | ✅ Done | 8x reference orbit speedup over mpmath |
| **Viewer Video Render** | ✅ Done | In-viewer render launch: explore -> find point -> render video |
| **Proactive Rebasing** | ✅ Done | Inline rebasing in all 4 kernels, total_iters tracking |
| **HP SA Coefficients** | ✅ Done | Series approx in gmpy2/mpmath, eliminates truncation at 1e13+ |
| **Newton-Raphson Finder** | ✅ Done | Period detect, nucleus, boundary points, scan-region, deep target |
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

---

## Architecture Decisions (cont.)

### AD-011: Interactive Viewer Architecture (Dear PyGui)

**Date:** 2026-03-12
**Decision:** Use Dear PyGui for the interactive fractal explorer, with a single-threaded GPU render bridge via ThreadPoolExecutor.

**Alternatives considered:**
- PyQt6/PySide6: Heavier, widget-based, slower texture updates for real-time display.
- pygame: No built-in widgets for sliders, panels, docking.
- tkinter: Poor GPU texture support, outdated.

**Key design choices:**
1. **Raw RGBA float32 textures** — DPG can update textures directly from flat float arrays, no PIL/QImage roundtrip.
2. **ThreadPoolExecutor(max_workers=1)** — Keeps GPU render off the UI thread. Only one render in-flight at a time; stale requests are replaced (not queued).
3. **ViewerState dataclass** — All mutable state in one object. Components read/write it freely; `render_pending` flag triggers re-render on next tick.
4. **Pixel-to-complex mapping** — `view_height = 3.0/zoom`, `view_width = view_height * aspect`. This matches the render pipeline's coordinate system exactly.
5. **Zoom-toward-cursor** — Compute complex coord under cursor before and after zoom change, adjust center by the difference to keep the point visually fixed.
6. **Module split** — `state.py` (data), `render_bridge.py` (async GPU), `canvas.py` (display + mouse), `controls.py` (parameter UI), `coordinate_panel.py` (HUD + bookmarks), `video_panel.py` (video render launch), `app.py` (lifecycle + layout).

---

### AD-013: Newton-Raphson Coordinate Finder

**Date:** 2026-03-14
**Decision:** Implement Newton's method for finding precise boundary coordinates at arbitrary precision.

**Problem:** Deep zoom rendering requires coordinates precisely on the Mandelbrot set boundary, where fractal detail exists at every scale. Approximate coordinates (hand-picked from viewers) run out of precision at extreme zoom and may not sit on the boundary at all.

**Architecture (`engine/newton.py`):**

1. **Period detection** -- iterate z=z^2+c and track |z_n| minimum. The period is the smallest n where the orbit passes closest to 0. Handles exterior points gracefully (returns best period found before escape).
2. **Nucleus finding** -- Newton's method on f^p(0,c) = 0. Derivative dz/dc tracked alongside the orbit. Includes period verification: checks all divisors of detected period to find the TRUE minimal period (prevents convergence to sub-period nuclei).
3. **Boundary point finding** -- 2D Newton on the system [f^p(z,c) = z, df^p/dz = e^{2*pi*i*theta}]. Requires second derivatives (d^2z/dz0^2, d^2z/(dz0*dc)). Cusp point (angle=0) uses tiny offset (1e-10) to avoid degenerate Jacobian.
4. **Deep target finder** -- recursive nested minibrot discovery by following period-doubling/tripling cascades.
5. **Region scanner** -- grid search for nuclei of different periods.

**CLI:** `fractalforge discover` (boundary point discovery), `fractalforge scan-region` (grid search). Both output JSON and CLI-ready render commands.

**Key lessons:**

- The cusp of a hyperbolic component has degenerate Jacobian (multiplier = 1 makes the first element of the Jacobian row zero). Standard workaround: use angle offset 1e-10.
- Period detection can find multiples of the true period. Always verify by checking divisors after Newton convergence.
- Boundary points are IN the set (they're on the Julia set). For rendering, use zoom levels matching the component scale (1/size) to see boundary structure.
- Points near but slightly outside the boundary give the best renders (mix of interior/exterior). Points exactly on the boundary show up as interior at extreme zoom.

---

### AD-012: Deep Zoom Phase 1 -- Precision & Rebasing

**Date:** 2026-03-14
**Decision:** Implement four changes to push rendering from 1e15 to 1e50 zoom.

**Changes:**

1. **Precision formula**: `1.5 * log10(zoom) + 30` (was `log10(zoom) + 10`). Plus floor at coordinate string digits + 20.
2. **HP SA coefficients**: Series approximation A/B/C computed in gmpy2/mpmath at zoom >= 1e13, preventing float64 truncation.
3. **Proactive rebasing**: When |Z+d| < |d|, fold delta back: d = Z+d, restart reference index. Added to all 4 kernels (CUDA/CPU perturbation, CUDA/CPU BLA). `total_iters` counter tracks actual work and NEVER resets on rebase.
4. **Glitch tolerance relaxation**: Ramp 0.3 (was 0.5), floor 1e-2 (was 1e-6). With rebasing handling precision issues, glitch detection becomes a safety net.

**Key bug:** Rebasing at moderate zoom (< 1e13) caused infinite loops because delta and reference magnitudes are similar. Fix: `enable_rebasing` flag gated on `zoom >= 1e13`.

**Key bug:** Interior reference orbits + rebasing = infinite loop. `iteration` resets to 0 on rebase but `while iteration < max_iter` never terminates because max_iter is large. Fix: `total_iters` counter that increments on every step and is used for the loop termination condition.

---

### AD-010: BLA (Bilinear Approximation) for Ultra-Deep Zoom

**Date:** 2026-03-13
**Decision:** Implement BLA to skip large blocks of perturbation iterations at deep zoom (1e50+).

**Architecture:**
1. **BLA table** (`engine/bla.py`): Pre-compute linear approximation coefficients from the reference orbit in a binary tree structure. Level 0 = single-step (A=2*Z_n, B=1), higher levels composed by pairing: A_{2k}(n) = A_k(n+k) * A_k(n). Validity radius r(n) = epsilon/|A| where epsilon scales with pixel spacing.
2. **BLA kernel** (`engine/bla_kernel.py`): At each iteration, try BLA jumps from highest level down. First valid jump (|d| < validity_radius) skips 2^level iterations. Falls back to single-step when near escape/glitch boundaries.
3. **Integration**: Auto-enabled in `perturbation.py` when reference orbit > 100 iterations. BLA table also computed for glitch correction passes.

**Key insight:** BLA speedup scales with zoom depth. At 1e14 with 5000 iterations, glitch detection terminates most pixels early, so BLA overhead dominates. At 1e50+ with millions of iterations, BLA should provide 100-1000x speedup.

**gmpy2 fast path:** Added `gmpy2` detection in `precision.py` — 8x faster reference orbit computation than mpmath at 50-digit precision. Falls back to mpmath when not installed.

---

### AD-011: Viewer Video Render Pipeline

**Date:** 2026-03-13
**Decision:** Enable "explore → find point → render video" workflow directly from the viewer UI.

**Architecture:**
1. **Video panel** (`viewer/video_panel.py`): Configurable render settings (resolution, duration, FPS, codec preset, SSAA, histogram EQ). Snapshots current viewer position as target.
2. **Auto zoom path**: Generates a two-keyframe ZoomPath from zoom=1.0 to the target zoom level. Uses hp string coordinates for deep zoom precision.
3. **Background thread**: Renders frames via `render_sequence()` with cancel support. Encodes to video via FFmpeg on completion.
4. **HP coordinate interpolation** (`zoompath.py`): New `_interp_coords_hp()` uses mpmath for zoom-weighted position interpolation when target zoom >= 1e13. Keyframe dataclass extended with `center_re_hp`/`center_im_hp` fields.
5. **Sequence renderer** (`render/sequence.py`): Updated to pass hp string coordinates to `render_frame_perturbation()` when available.

---

### AD-014: Floatexp Deep-Zoom Engine (DZ-P2)

**Date:** 2026-06-13
**Decision:** Add an extended-range ("floatexp") perturbation engine and route
all zooms `>= 1e18` to it, lifting the practical zoom ceiling from ~1e15 to
1e600+ (effectively unbounded).

**Problem:** The float64 perturbation kernels (DZ-P1) break down past ~1e18:

- Float64 deltas underflow near 1e-308; products like `2*Z*d` die even earlier.
- DZ-P1's proactive rebasing folds the delta to O(1) when `|Z+d| < |d|`, but at
  that point the per-pixel `dc` (~10^-zoom) is far below the float64 mantissa and
  is silently dropped. Pixels lose their identity, producing **false-interior
  blocks** — worst for bounded (Misiurewicz / dendrite) references whose orbits
  skirt the origin. Empirically, interior% for the `c=i` dendrite jumped from 0%
  to 12-70% exactly at the `zoom >= 1e18` rebasing gate.

**Architecture:**

1. **`engine/floatexp.py`** — a complex value is `(m_re, m_im, exp)`: two float64
   mantissas sharing one int64 power-of-two exponent, normalized so
   `max(|m_re|,|m_im|)` is in `[0.5, 1)`. Keeps float64's ~16-digit mantissa
   while extending range to the int64 exponent. The same primitive bodies are
   compiled twice via a factory: `fx_*` (`@njit`, CPU) and `dfx_*`
   (`@cuda.jit(device=True)`, GPU). Magnitude compares use the log2 domain so
   validity radii far below 1e-308 stay representable.
2. **`engine/precision.py`** — optional extended (floatexp) reference-orbit
   storage; `zoom` accepted as a string / `log10` so depth isn't capped at
   float64's 1e308 (`zoom_to_log10`, `log10_to_zoom_str`).
3. **`engine/bla.py`** — research-validated BLA validity radii that account for
   the `dc` divergence term (the old `epsilon/|A|` heuristic tears BLA apart past
   ~1e50), plus a vectorized floatexp BLA table (`compute_bla_table_fxp`) with
   sub-16-iteration culling.
4. **`engine/deep_kernel.py`** — floatexp perturbation kernel (CUDA + CPU) with
   floatexp BLA jumps and always-on proactive rebasing. No series approximation
   (BLA from iteration 0 subsumes it at depth). Reference-orbit exhaustion is
   handled by rebasing to index 0, not by a float64 fallback.
5. **`engine/perturbation.py`** — `render_frame_perturbation()` routes
   `zoom >= 1e18` to `_render_deep_fxp()`. Below that the float64 PT path runs
   without rebasing (correct in that regime).

**Validation:** floatexp arithmetic matches mpmath including a full delta step at
1e-700 scale; the deep kernel reproduces ground-truth float64 Mandelbrot
structure at shallow depth (99.9% pixel agreement); the `c=i` embedded-Julia
escape depth scales linearly with `log(zoom)` from 1e18 to 1e600 with no NaN; the
float64→fxp handoff at 1e18 is seamless. Visual renders at 1e100/1e200/1e300 show
coherent self-similar structure.

**Key insight (LL-005):** Proactive rebasing and float64 are fundamentally
incompatible at depth. Rebasing deliberately makes the delta O(1), which is
exactly when the tiny per-pixel `dc` underflows float64. Extended-range
arithmetic for the delta state isn't an optimization here — it's a correctness
requirement. Hand off to floatexp *at the rebasing threshold*, not later.

---

### AD-015: Misiurewicz Points as Deep-Zoom Targets

**Date:** 2026-06-13
**Decision:** Add `find_misiurewicz()` (`engine/newton.py`) as the primary tool
for discovering *tractable* extreme-zoom coordinates, exposed in both the CLI and
the viewer.

**Problem:** The existing `find_deep_target()` descends a period-doubling cascade,
but component period grows as `2^depth` while zoom grows only as `delta^depth`
(delta = 4.669). Reaching 1e120 would need period ~1e54 — an intractable
reference orbit. It stalls around 1e4 in practice.

**Solution:** Misiurewicz (pre-periodic) points. The critical orbit settles onto
a repelling cycle after a short transient, so `g(c) = f^{m+p}(0) - f^{m}(0) = 0`
(Newton) locks onto a point whose reference orbit stays *short at any depth*,
while the embedded Julia set (Tan Lei) repeats at every scale. Zoom depth is then
limited only by coordinate precision — perfect for the floatexp engine.
`find_misiurewicz()` auto-searches small `(preperiod, period)` pairs from a seed.
Example: seeding near the seahorse spiral yields M(22,1) at
`-0.77467...+0.13743...i`, which renders a clean self-similar spiral at 1e200.

**Lesson (LL-006):** For deep zoom, target points whose *reference orbit length*
stays bounded (Misiurewicz / boundary points that don't escape), not minibrot
nuclei whose period — and hence orbit length — explodes with depth.

---

### AD-016: Interactive Viewer Overhaul

**Date:** 2026-06-13
**Decision:** Rework the Dear PyGui viewer for unbounded depth, correct
navigation, and a premium feel.

**Changes:**

1. **`log10_zoom` is the authoritative scale** (`viewer/state.py`). A raw float
   zoom overflows at 1e308; `log10(1e500)` is just `500.0`. `zoom_str` feeds the
   string-zoom render path; `zoom` remains a (clamped) convenience float.
2. **Pan bug fix** (`viewer/canvas.py`). DPG reports drag delta as *cumulative*
   pixels since button-down. The old code re-applied that growing delta to the
   already-moved center every frame, causing runaway acceleration. Fix: snapshot
   the press-time center once and offset from it.
3. **Keyboard navigation**, **mpmath zoom-to-cursor at any depth**, uncapped
   go-to / discovery (was clamped to 1e20).
4. **Progressive rendering** (`viewer/app.py`): a fast reduced-scale pass on
   every change, replaced by a full-resolution pass after ~0.18 s of stillness.
   Keeps deep zooms responsive during exploration.
5. **Premium theme** (`viewer/theme.py`): Infinite Descent palette (navy/cyan/
   violet), unified collapsing sidebar, and a status bar showing zoom (10^x),
   engine badge (STD/PT/FXP), iterations, precision digits, and render time.

**Known follow-up:** the in-viewer *video* path still uses float zoom (capped at
~1e307). Deep-zoom *stills* are unbounded; deep-zoom *video* beyond 1e307 needs
string-zoom plumbing through `zoompath.py` / `sequence.py` (DZ-P3).
