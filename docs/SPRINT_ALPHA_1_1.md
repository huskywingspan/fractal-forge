# Sprint: alpha1.1.0 — New Fractal Types & Compilation Pipeline

**Goal:** Add Julia set and Burning Ship fractal support, plus an automated short compilation pipeline that stitches multiple zoom clips with crossfade transitions.

**Base:** alpha1.0.0 (Mandelbrot engine complete, YouTube pipeline operational)

---

## S1: Julia Set Engine

**What:** Julia sets use the same `z = z^2 + c` iteration, but the roles are swapped — `c` is fixed (the "Julia parameter") and `z_0` varies per pixel. Every point on the Mandelbrot set boundary corresponds to a visually interesting Julia set.

**Implementation:**

| ID | Task | Notes |
|----|------|-------|
| S1-01 | Julia CUDA kernel | Same structure as Mandelbrot kernel, but `c` is a fixed parameter and initial `z = pixel_coord`. |
| S1-02 | Julia CPU kernel | Numba njit parallel fallback. |
| S1-03 | `render_frame_julia()` function | In `engine/julia.py`. Takes `c_re, c_im` (Julia parameter) + `center_re, center_im, zoom` (viewport). |
| S1-04 | Route in `frame_renderer.py` | Add `fractal_type` param to `render_single()` — "mandelbrot" (default), "julia", "burning_ship". |
| S1-05 | CLI: `--fractal` flag | Add `--fractal` to `render` and `zoom` commands. For Julia, `--julia-re` / `--julia-im` set the c parameter. |
| S1-06 | Julia zoom path support | Add `julia_re`, `julia_im` fields to Keyframe. When present, renders Julia instead of Mandelbrot. Enables animated c-parameter morphing. |

**Key design:** The Julia kernel is nearly identical to Mandelbrot — same escape test, same smooth coloring formula. The only differences are:
```
Mandelbrot: z_0 = 0,      c = pixel_coord
Julia:      z_0 = pixel_coord, c = fixed_param
```

**Interesting Julia parameters to ship as presets:**
- `c = -0.7269 + 0.1889i` — Dendrite (branching tree)
- `c = -0.8 + 0.156i` — Classic spiral Julia
- `c = 0.285 + 0.01i` — Siegel disk
- `c = -0.4 + 0.6i` — Dragon-like
- `c = -0.835 - 0.2321i` — Douady rabbit

---

## S2: Burning Ship Engine

**What:** The Burning Ship fractal uses `z = (|Re(z)| + i|Im(z)|)^2 + c`. Taking absolute values before squaring produces a jagged, asymmetric fractal with a distinctive "ship" shape.

**Implementation:**

| ID | Task | Notes |
|----|------|-------|
| S2-01 | Burning Ship CUDA kernel | Key change: `z_re = abs(z_re)`, `z_im = abs(z_im)` before the squaring step. |
| S2-02 | Burning Ship CPU kernel | Numba njit parallel fallback. |
| S2-03 | `render_frame_burning_ship()` function | In `engine/burning_ship.py`. Same interface as `render_frame()`. |
| S2-04 | Route through `--fractal burning_ship` | Connects to existing frame renderer pipeline. |

**Key difference from Mandelbrot:**
```python
# Mandelbrot
z_im = 2.0 * z_re * z_im + c_im
z_re = z_re_sq - z_im_sq + c_re

# Burning Ship
z_im = abs(2.0 * z_re * z_im) + c_im
z_re = z_re_sq - z_im_sq + c_re
# (with z_re, z_im taken as absolute values before squaring)
```

**Interesting coordinates:**
- Full view: `(-0.5, -0.5)` zoom 1 — shows the upside-down "ship"
- Antenna: `(-1.755, 0.02)` zoom 40 — intricate detail
- Armada: `(-1.862, -0.003)` zoom 200 — fleet of tiny ships

**Note:** Burning Ship renders upside-down by convention (the "ship" appears when im-axis is flipped). We'll render with flipped y so it looks right.

---

## S3: Short Compilation Pipeline

**What:** Automated pipeline that takes N zoom clips, trims to specified length, adds crossfade transitions, and outputs a single compilation video. For "Best Of" compilations, Shorts playlists, or quick showcase reels.

**Implementation:**

| ID | Task | Notes |
|----|------|-------|
| S3-01 | `CompilationSpec` data model | Pydantic model: list of `{preset_path, start_sec, duration_sec}` clips + transition config. |
| S3-02 | Frame extraction / trim | Read rendered frames from each clip's frames dir, select the range. |
| S3-03 | Crossfade transition renderer | Given frames A_end and B_start, generate `transition_frames` count of blended frames. Linear alpha blend: `out = A * (1-t) + B * t` for smooth crossfade. |
| S3-04 | Compilation assembler | Concatenate trimmed frame sequences with transition frames inserted between clips. Renumber sequentially. |
| S3-05 | `fractalforge compile` CLI command | Takes a JSON compilation spec or inline arguments. Outputs a single encoded video. |
| S3-06 | Quick mode: `--clips` shorthand | `fractalforge compile --clips "preset1.json:10-20s,preset2.json:5-15s" --transition 1s --output comp.mp4` for fast one-liners without a spec file. |

**Compilation spec JSON:**
```json
{
  "name": "best_of_march",
  "fps": 60,
  "transition_frames": 60,
  "transition_type": "crossfade",
  "encode_preset": "youtube",
  "clips": [
    {"preset": "presets/seahorse_fire.json", "start_sec": 10, "duration_sec": 8},
    {"preset": "presets/elephant_valley.json", "start_sec": 5, "duration_sec": 10},
    {"preset": "presets/triple_spiral.json", "start_sec": 8, "duration_sec": 8}
  ]
}
```

**Transition types (start with crossfade, extensible later):**
- `crossfade` — linear alpha blend (S3-03)
- Future: `fade_black`, `wipe`, `zoom_match`

---

## S4: Sample Content Library

**What:** Render sample content for each new fractal type to validate the pipeline and build the content library.

| ID | Task | Notes |
|----|------|-------|
| S4-01 | 5 Julia set zoom presets | One per interesting c-parameter. 1080p, 30s each. |
| S4-02 | Julia wallpaper set | 10 scenes x 4 resolutions = 40 images. |
| S4-03 | 3 Burning Ship zoom presets | Full view, antenna, armada. 1080p, 30s each. |
| S4-04 | Burning Ship wallpaper set | 6 scenes x 4 resolutions = 24 images. |
| S4-05 | Compilation test | 3-clip compilation with 1s crossfade transitions. |

---

## Implementation Order

```
S1-01 → S1-02 → S1-03 → S1-04 → S1-05   Julia engine (kernel → routing → CLI)
S2-01 → S2-02 → S2-03 → S2-04            Burning Ship engine (same pattern)
S1-06                                      Julia zoom path support
S3-01 → S3-02 → S3-03 → S3-04 → S3-05   Compilation pipeline
S3-06                                      Quick compile CLI
S4-01 → S4-05                              Sample content renders
```

**Parallelizable:** S1 and S2 are fully independent — can implement simultaneously.
S3 depends only on having rendered frame directories, so can start as soon as S1/S2 produce test renders.

---

## Architecture Notes

### Fractal type routing

Add a `fractal_type` parameter through the render pipeline:

```
CLI --fractal mandelbrot|julia|burning_ship
  → render_single(fractal_type=..., julia_re=..., julia_im=...)
    → dispatch to appropriate kernel module
      → same coloring/SSAA/postprocess pipeline
```

All fractal types produce the same output: a 2D `smooth_data` array of float64. Everything downstream (coloring, SSAA, post-processing, encoding) is fractal-agnostic.

### Keyframe extension for Julia

```python
@dataclass
class Keyframe:
    frame: int
    center_re: float
    center_im: float
    zoom: float
    max_iter: int
    palette: str
    rotation: float = 0.0
    fractal_type: str = "mandelbrot"  # NEW
    julia_re: float | None = None     # NEW
    julia_im: float | None = None     # NEW
```

This enables "Julia morphing" videos where the c-parameter smoothly changes between keyframes, producing mesmerizing shape-shifting animations.

---

## Deliverables

- [ ] `engine/julia.py` — Julia set CUDA + CPU kernels
- [ ] `engine/burning_ship.py` — Burning Ship CUDA + CPU kernels
- [ ] `publish/compilation.py` — Compilation assembler + crossfade
- [ ] `--fractal` CLI flag on render/zoom commands
- [ ] `--julia-re` / `--julia-im` CLI flags
- [ ] `fractalforge compile` CLI command
- [ ] 5 Julia zoom presets + 3 Burning Ship presets
- [ ] Julia + Burning Ship wallpaper packs
- [ ] 1 test compilation video
- [ ] Updated WORK_TRACKER.md, PROJECT_CHRONICLE.md
- [ ] Tag: alpha1.1.0
