# FractalForge — Project Instructions

## Overview
GPU-accelerated fractal zoom renderer for creating long-form, artistic fractal zoom videos.
Inspired by the Maths Town YouTube channel.

## Tech Stack
- **Language:** Python 3.10+
- **GPU:** Numba CUDA kernels (RTX 3070 local, RunPod for production)
- **CLI:** Click + Rich
- **Config/Models:** Pydantic
- **Video:** FFmpeg (via ffmpeg-python)
- **Arbitrary Precision:** mpmath (for perturbation theory reference orbits)
- **Env:** venv (not conda)

## Project Structure
```
src/fractalforge/
├── engine/          # GPU kernels, iteration logic, precision
│   ├── mandelbrot.py    # Standard Mandelbrot kernel
│   ├── coloring.py      # Iteration count → RGB mapping
│   ├── perturbation.py  # Deep zoom perturbation theory (Phase 3)
│   └── precision.py     # Arbitrary precision management (Phase 3)
├── artist/          # Artistic tools — palettes, zoom paths, post-processing
│   ├── palette.py       # Color palette system
│   ├── zoompath.py      # Keyframe zoom path planner
│   └── postprocess.py   # Motion blur, vignette, etc. (Phase 4)
├── render/          # Rendering pipeline
│   ├── frame_renderer.py  # Single frame render + export
│   ├── sequence.py        # Multi-frame sequence (Phase 2)
│   └── video.py           # FFmpeg video encoding (Phase 2)
├── cli/             # CLI entry points
│   └── main.py          # fractalforge CLI commands
├── viewer/          # Interactive preview (future)
└── presets/         # Saved zoom paths, palette configs
```

## Conventions
- Use `ruff` for linting (line-length 100)
- Tests in `tests/` using pytest
- Render output goes to `output/` (gitignored)
- Palette presets defined in `artist/palette.py` as numpy arrays
- Zoom paths stored as JSON in `presets/`
- Commit messages: imperative mood, concise

## Key Architecture Decisions
- **Numba CUDA** over raw CUDA C++ — near-native perf with Python iteration speed
- **Perturbation theory** for deep zooms — computed reference orbit (CPU) + delta iteration (GPU)
- **Exponential interpolation** for zoom levels in zoom paths
- **CLI-first** design — GUI/viewer is a separate optional concern

## Development
```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"
fractalforge info              # Verify GPU
fractalforge render -o output/test.png
```

## Docs
- `docs/PROJECT_CHRONICLE.md` — Architecture decisions, bugs, lessons learned
- `docs/WORK_TRACKER.md` — Open work items by phase
