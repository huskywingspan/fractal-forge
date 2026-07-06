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

## Sprint DZ-P2: Floatexp Deep-Zoom Engine & Viewer Overhaul

*Goal: Unbounded magnification (1e600+) and a premium, intuitive explorer.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| DZ-P2-01 | floatexp extended-range arithmetic (`engine/floatexp.py`) | ✅ Done | mantissa pair + int64 exp; CPU njit + CUDA device fns; 16 unit tests vs mpmath |
| DZ-P2-02 | Extended reference orbit + string/log10 zoom | ✅ Done | `precision.py`: floatexp orbit storage, `zoom_to_log10` |
| DZ-P2-03 | Corrected BLA validity radii + floatexp BLA table | ✅ Done | research formula w/ dc term; `compute_bla_table_fxp`, culling |
| DZ-P2-04 | Floatexp deep kernel (CUDA + CPU) | ✅ Done | `deep_kernel.py`; BLA jumps + always-on rebasing, no SA |
| DZ-P2-05 | Route zoom >= 1e18 to deep kernel | ✅ Done | Fixes float64 rebasing false-interior bug at depth |
| DZ-P2-06 | String-zoom through frame renderer | ✅ Done | `render_single`/`render_and_save` accept str zoom |
| DZ-P2-07 | Misiurewicz deep-target finder | ✅ Done | `find_misiurewicz()`; tractable orbit at any depth |
| DZ-P2-08 | Deep-zoom verification suite | ✅ Done | 99.9% vs ground truth, coherent structure to 1e600, no NaN |
| DZ-P2-09 | Viewer: log10 unbounded zoom | ✅ Done | `state.py` log10_zoom authoritative |
| DZ-P2-10 | Viewer: fix runaway pan | ✅ Done | snapshot press-time center vs cumulative delta |
| DZ-P2-11 | Viewer: keyboard nav + zoom-to-cursor | ✅ Done | arrows/+/-/R/B; mpmath at any depth |
| DZ-P2-12 | Viewer: progressive rendering | ✅ Done | reduced-scale during interaction, full-res on idle |
| DZ-P2-13 | Viewer: premium theme + status bar | ✅ Done | `theme.py` brand palette; engine/precision/time HUD |
| DZ-P2-14 | Viewer: in-app Misiurewicz finder | ✅ Done | "Misiurewicz" button -> dive to target |
| DZ-P2-15 | CLI: `misiurewicz` command | ✅ Done | Find pre-periodic deep targets from the CLI |

## Sprint DZ-P3: Deep-Zoom Video

*Goal: Render zoom animations beyond 1e307 (string-zoom through the video pipeline).*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| DZ-P3-01 | String/log10 zoom in `Keyframe`/`ZoomPath` | ✅ Done | `zoom: float \| str`, `zoom_log10` property |
| DZ-P3-02 | log10 exponential zoom interpolation past 1e307 | ✅ Done | legacy + cinematic interp in log10; result carries `log10_zoom`; zoom emitted as string when > 1e300 |
| DZ-P3-03 | Viewer video panel: unbounded target zoom | ✅ Done | uses `state.zoom_str`; depth-scaled end max_iter |
| DZ-P3-05 | Sequence dispatch + CLI/preview deep-safe | ✅ Done | engine dispatch on `log10_zoom`; zoom-cmd, thumbnail, camera-path labels deep-safe; `tests/test_deep_video.py` |
| DZ-P3-04 | Distance-estimation coloring for deep frames | 🔲 Optional | banding at depth FIXED by continuous EQ (CW-01); DE coloring remains a nice-to-have texture upgrade |

## Sprint CW: Creator Workflow (Explore -> Style -> Render)

*Goal: Flawless deep exploration + streamlined YouTube output. (2026-07-05)*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| CW-01 | Fix deep artifact banding: continuous (interpolated-CDF) histogram EQ | ✅ Done | root-caused via BLA A/B (engine exact, median diff 0.0000); AD-018 / LL-008 |
| CW-02 | `normalize_range` glow mapping + `color_mode` end to end | ✅ Done | default / histogram / normalized; real-time viewer combo |
| CW-03 | Reference-orbit LRU cache (per center, sufficiency checks) | ✅ Done | 42x same-view, 13x zoom-step at 1e300; `tests/test_orbit_cache.py` |
| CW-04 | Deep BLA table cache + quantized auto_max_iter | ✅ Done | quantization keeps orbit cache valid across zoom steps |
| CW-05 | Format x Quality video presets (Long/Shorts/Ultrawide x Draft/Std/Prod) | ✅ Done | Shorts native 1080x1920, <60s default; presets set, never lock |
| CW-06 | WYSIWYG effects passthrough to video renders | ✅ Done | full live look (color mode, HDR, grade) snapshotted per render |
| CW-07 | Live HDR in viewer (bloom/halation/tone map/exposure) | ✅ Done | scaled bloom radius on progressive pass |
| CW-08 | Shuffle Style randomizer + Reset | ✅ Done | look-only, safe mid-dive |
| CW-09 | Bilinear progressive upscale | ✅ Done | replaces blocky nearest during interaction |

## Sprint QA: Field-Testing Fixes (2026-07-05..06)

*Goal: Iron out artifacts found in user exploration testing.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| QA-01 | Rebase counted as iteration -> blocky seams | ✅ Done | all 6 kernels; ground-truth median 3.0 -> 0.0; AD-021/LL-009 |
| QA-02 | Inverted BLA level-0 validity radius -> circular disc | ✅ Done | r = eps*\|Z_n\| (research doc formula was garbled); AD-022/LL-010 |
| QA-03 | Resolution-aware STD->PT dispatch (streaks at 7e12@1080p) | ✅ Done | pixel spacing < 2e-14 rule; AD-023 |
| QA-04 | FXP handoff 1e18 -> 1e13 (PT moire rings at 4.6e17) | ✅ Done | fxp owns >=1e13; PT keeps narrow safe window |
| QA-05 | Auto max-iter too stingy at depth | ✅ Done | deep slope 60 -> 400/decade |
| QA-06 | Julia deep zoom = noise (no perturbation) | ✅ Done | Julia PT engine, ceiling ~1e30; AD-024; test_julia_pt.py |
| QA-07 | Julia rebasing/floatexp for 1e30+ | 🔲 Future | v1 ceiling fine for exploration; extend if needed |

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
