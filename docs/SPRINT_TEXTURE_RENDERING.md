# Sprint: Texture Rendering Pipeline

> **Goal:** Transform FractalForge from escape-time coloring to a full texture rendering system capable of HDR, analytical lighting, orbit traps, and image-mapped textures. Unlock Maths Town-quality visuals.

## Context

Current FractalForge coloring uses smooth iteration count + palette mapping + finite-difference slope shading. This produces good results but lacks the depth, materiality, and visual richness of high-end renderers (Maths Town, UltraFractal, Kalles Fraktaler). The key insight from research: fractal "texturing" means extracting additional per-pixel data from the iteration loop (derivatives, orbit traps, phase angles) and using it in a sophisticated post-processing coloring pipeline.

## Architecture

```
CUDA Kernel (iteration loop)
  |
  v
Multi-Channel Pixel Buffer (9 x float32 per pixel)
  |  ch0: smooth_iteration
  |  ch1-2: z_final (re, im)
  |  ch3-4: dz/dc derivative (re, im)
  |  ch5: orbit_trap_distance
  |  ch6-7: trapped_z (re, im)
  |  ch8: stripe_average
  |
  v
Coloring Pipeline (post-processing, numpy/GPU)
  |-- Palette mapping (existing)
  |-- Analytical normals (from dz/dc)
  |-- Distance estimation (from z_final + dz/dc)
  |-- Orbit trap coloring
  |-- Image orbit trap mapping
  |-- Matcap / environment mapping
  |-- Lighting (Blinn-Phong, specular)
  |-- HDR bloom / halation / glow
  |-- Tone mapping (ACES filmic)
  |-- Vignette, color grading (existing)
  |
  v
Final Image (PNG / video frame)
```

---

## Tier 1: HDR Post-Processing (no kernel changes)

*Quick wins using existing smooth_data + slope normals.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| T1-01 | HDR bloom/glow post-processor | | Extract bright pixels, multi-pass Gaussian blur, additive blend |
| T1-02 | Halation effect (film light bleed) | | Per-channel blur radii (R wide, B narrow), warm tint |
| T1-03 | ACES filmic tone mapping | | Industry-standard HDR->SDR curve, replaces naive clamp |
| T1-04 | Reinhard tone mapping (alternative) | | Simpler, good for softer looks |
| T1-05 | Matcap shading from slope normals | | Load sphere photo, map (nx,ny) -> UV, sample matcap image |
| T1-06 | Integrate HDR pipeline into CLI | | `--bloom`, `--halation`, `--tone-map` flags on render/zoom |
| T1-07 | Integrate HDR pipeline into viewer | | Real-time preview of bloom/halation/tone mapping |

## Tier 2: Multi-Channel Kernel (iteration loop additions)

*Kernel outputs rich per-pixel data for advanced coloring.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| T2-01 | Multi-channel output buffer architecture | | 9-channel float32 buffer, GPU allocation, coloring pipeline reads |
| T2-02 | dz/dc derivative tracking in standard kernel | | `dz' = 2*z*dz + 1` per iteration, ~15-20% perf overhead |
| T2-03 | dz/dc derivative tracking in perturbation kernel | | Track dz/dc alongside delta iteration |
| T2-04 | z_final storage (re, im at escape) | | Store last z value for phase/angle coloring |
| T2-05 | Analytical distance estimation | | `d = 2|z|log|z| / |dz/dc|` from z_final + derivative |
| T2-06 | Analytical normal vectors | | `n = (z/dz) / |z/dz|`, per-pixel, resolution-independent |
| T2-07 | Phase/angle coloring channel | | `arg(z_final)` as additional color index ("chrome effect") |
| T2-08 | Stripe Average Coloring accumulator | | `sum += sin(s * arg(z_n))` during iteration, fine-grained stripes |
| T2-09 | Triangle Inequality Average (TIA) | | Accumulate per-iteration triangle inequality measure |
| T2-10 | Blinn-Phong lighting from analytical normals | | Replace finite-difference slope with analytical normals |
| T2-11 | Environment mapping | | Use normal vector to sample equirectangular environment map |
| T2-12 | DE-based bump mapping | | Treat distance estimate as height, compute perturbed normals |
| T2-13 | Combined coloring modes | | Blend smooth iteration + DE + phase + stripe with configurable weights |

## Tier 3: Orbit Trap Texturing

*Image and geometric orbit traps for full texture rendering.*

| ID | Item | Status | Notes |
|----|------|--------|-------|
| T3-01 | Point orbit trap | | Track min distance from z_n to a trap point, bubble patterns |
| T3-02 | Cross orbit trap | | `min(|Re(z_n)|, |Im(z_n)|)` with threshold, stalk patterns |
| T3-03 | Circle/ring orbit trap | | `||z_n| - radius|`, ring/web patterns |
| T3-04 | Line orbit trap | | Distance to line segment, creates fold patterns |
| T3-05 | Image orbit trap (the big one) | | When z_n falls in trap rectangle, map to image UV, sample texture |
| T3-06 | Multi-trap composition | | Combine multiple trap shapes (min/max/blend) |
| T3-07 | Dynamic traps (animation) | | Trap position/rotation varies with frame for zoom videos |
| T3-08 | Trap-based palette mapping | | Use trap distance as palette index instead of iteration |
| T3-09 | CLI integration for orbit traps | | `--trap point --trap-image photo.png` flags |
| T3-10 | Viewer trap preview | | Interactive trap position/size adjustment |

---

## Implementation Order

**Phase 1 — HDR Pipeline (Tier 1):** T1-01 through T1-07
- Pure post-processing, no kernel changes
- Bloom + ACES tone mapping alone is a dramatic visual upgrade
- Matcap shading gives instant material variety

**Phase 2 — Multi-Channel Kernel (Tier 2):** T2-01 through T2-06
- Architectural change: kernels output multi-channel buffer
- dz/dc derivative tracking unlocks analytical everything
- Phase coloring and stripe average for visual variety

**Phase 3 — Advanced Coloring (Tier 2 cont.):** T2-07 through T2-13
- Stripe average, TIA, environment mapping
- Combined coloring modes with blending weights
- Full Blinn-Phong with analytical normals

**Phase 4 — Orbit Traps (Tier 3):** T3-01 through T3-10
- Geometric traps first (simple, dramatic visual impact)
- Image traps last (complex but enables Maths Town-style textures)

## Key Files

| File | Role |
|------|------|
| `engine/mandelbrot.py` | Standard kernel — add dz/dc, z_final, trap outputs |
| `engine/perturbation.py` | Perturbation kernel — add dz/dc tracking alongside delta |
| `engine/coloring.py` | Coloring pipeline — analytical normals, phase, DE coloring |
| `engine/postprocess.py` | HDR bloom, halation, tone mapping, matcap |
| `render/frame_renderer.py` | Orchestrate multi-channel buffer through pipeline |
| `artist/palette.py` | Palette system — may need trap-based mapping mode |
| `cli/main.py` | CLI flags for new features |
| `viewer/render_bridge.py` | Viewer integration |

## Performance Budget

| Resolution | Buffer size (9ch) | Bloom overhead | dz/dc overhead |
|------------|-------------------|----------------|-----------------|
| 640x360 | 8 MB | ~5ms | ~15-20% iter time |
| 1920x1080 | 71 MB | ~20ms | ~15-20% iter time |
| 3840x2160 | 285 MB | ~80ms | ~15-20% iter time |

All within RTX 3070's 8GB VRAM budget with headroom.

## References

- Inigo Quilez: Geometric Orbit Traps
- UltraFractal: Slope Formulas, Direct Orbit Traps, Image Traps
- Kalles Fraktaler: KFB format, DE channels, phase channel
- Maths Town: KFMovieMaker layer sampling, multi-pass compositing
- Phil Thompson: Stripe Average Coloring, Smooth Colors + Slope Shading
- ACES filmic tone mapping (Narkowicz approximation)
