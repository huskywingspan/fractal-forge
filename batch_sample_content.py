"""Batch render sample content for Julia sets and Burning Ship fractals.

Renders:
- Wallpapers: 5 Julia scenes + 3 Burning Ship scenes at 1080p with 2x SSAA
- Smoke test: One quick frame per fractal type to validate pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

# Use the fractalforge CLI entry point (installed via pip install -e)
FRACTALFORGE = [sys.executable, "-c", "from fractalforge.cli.main import cli; cli()"]

# ---- Wallpaper scenes ----
JULIA_WALLPAPERS = [
    {
        "name": "julia_dendrite",
        "c_re": -0.7269, "c_im": 0.1889,
        "center_re": 0.0, "center_im": 0.0,
        "zoom": 1.0, "palette": "electric", "max_iter": 1000,
    },
    {
        "name": "julia_dendrite_zoom",
        "c_re": -0.7269, "c_im": 0.1889,
        "center_re": 0.5, "center_im": 0.5,
        "zoom": 5.0, "palette": "nebula", "max_iter": 1500,
    },
    {
        "name": "julia_spiral",
        "c_re": -0.8, "c_im": 0.156,
        "center_re": 0.0, "center_im": 0.0,
        "zoom": 1.0, "palette": "nebula", "max_iter": 1000,
    },
    {
        "name": "julia_spiral_zoom",
        "c_re": -0.8, "c_im": 0.156,
        "center_re": 0.0, "center_im": 1.0,
        "zoom": 4.0, "palette": "ocean", "max_iter": 1500,
    },
    {
        "name": "julia_siegel",
        "c_re": 0.285, "c_im": 0.01,
        "center_re": 0.0, "center_im": 0.0,
        "zoom": 1.0, "palette": "ocean", "max_iter": 1000,
    },
    {
        "name": "julia_dragon",
        "c_re": -0.4, "c_im": 0.6,
        "center_re": 0.0, "center_im": 0.0,
        "zoom": 1.0, "palette": "fire", "max_iter": 1000,
    },
    {
        "name": "julia_dragon_zoom",
        "c_re": -0.4, "c_im": 0.6,
        "center_re": -0.3, "center_im": 0.5,
        "zoom": 5.0, "palette": "electric", "max_iter": 1500,
    },
    {
        "name": "julia_rabbit",
        "c_re": -0.835, "c_im": -0.2321,
        "center_re": 0.0, "center_im": 0.0,
        "zoom": 1.0, "palette": "monochrome", "max_iter": 1000,
    },
    {
        "name": "julia_rabbit_zoom",
        "c_re": -0.835, "c_im": -0.2321,
        "center_re": 0.0, "center_im": -1.0,
        "zoom": 4.0, "palette": "nebula", "max_iter": 1500,
    },
    {
        "name": "julia_douady",
        "c_re": -0.123, "c_im": 0.745,
        "center_re": 0.0, "center_im": 0.0,
        "zoom": 1.0, "palette": "fire", "max_iter": 1000,
    },
]

BURNING_SHIP_WALLPAPERS = [
    {
        "name": "burning_ship_full",
        "center_re": -0.4, "center_im": -0.6,
        "zoom": 0.8, "palette": "fire", "max_iter": 800,
    },
    {
        "name": "burning_ship_antenna",
        "center_re": -1.755, "center_im": 0.02,
        "zoom": 40.0, "palette": "electric", "max_iter": 1500,
    },
    {
        "name": "burning_ship_antenna_deep",
        "center_re": -1.7578, "center_im": 0.0185,
        "zoom": 500.0, "palette": "ocean", "max_iter": 2500,
    },
    {
        "name": "burning_ship_armada",
        "center_re": -1.862, "center_im": -0.003,
        "zoom": 200.0, "palette": "nebula", "max_iter": 2000,
    },
    {
        "name": "burning_ship_bow",
        "center_re": -1.77, "center_im": 0.0,
        "zoom": 15.0, "palette": "fire", "max_iter": 1500,
    },
    {
        "name": "burning_ship_smokestack",
        "center_re": -1.755, "center_im": -0.02,
        "zoom": 8.0, "palette": "ocean", "max_iter": 1500,
    },
]


def run_render(name, fractal_type, output_path, extra_args):
    """Run a single fractalforge render command."""
    cmd = FRACTALFORGE + [
        "render",
        "--fractal", fractal_type,
        "-o", str(output_path),
        "--ss", "2",
        "--preset", "1080p",
    ] + extra_args

    print(f"  Rendering {name}...", flush=True)
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        size = output_path.stat().st_size / 1024
        print(f"    OK ({elapsed:.1f}s, {size:.0f} KB)", flush=True)
        return True
    else:
        print(f"    FAILED ({elapsed:.1f}s)", flush=True)
        print(f"    {result.stderr[:200]}", flush=True)
        return False


def render_julia_wallpapers():
    """Render all Julia set wallpapers."""
    print("\n=== Julia Set Wallpapers ===\n", flush=True)
    out_dir = Path("output/wallpapers/julia")
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for scene in JULIA_WALLPAPERS:
        name = scene["name"]
        output_path = out_dir / f"{name}.png"

        extra = [
            "-x", str(scene["center_re"]),
            "-y", str(scene["center_im"]),
            "-z", str(scene["zoom"]),
            "-p", scene["palette"],
            "-i", str(scene["max_iter"]),
            "--julia-re", str(scene["c_re"]),
            "--julia-im", str(scene["c_im"]),
        ]

        if run_render(name, "julia", output_path, extra):
            ok += 1
        else:
            fail += 1

    print(f"\nJulia wallpapers: {ok} OK, {fail} failed\n", flush=True)
    return ok, fail


def render_burning_ship_wallpapers():
    """Render all Burning Ship wallpapers."""
    print("\n=== Burning Ship Wallpapers ===\n", flush=True)
    out_dir = Path("output/wallpapers/burning_ship")
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for scene in BURNING_SHIP_WALLPAPERS:
        name = scene["name"]
        output_path = out_dir / f"{name}.png"

        extra = [
            "-x", str(scene["center_re"]),
            "-y", str(scene["center_im"]),
            "-z", str(scene["zoom"]),
            "-p", scene["palette"],
            "-i", str(scene["max_iter"]),
        ]

        if run_render(name, "burning_ship", output_path, extra):
            ok += 1
        else:
            fail += 1

    print(f"\nBurning Ship wallpapers: {ok} OK, {fail} failed\n", flush=True)
    return ok, fail


def main():
    total_start = time.perf_counter()
    print("=" * 60, flush=True)
    print("FractalForge Sample Content Renderer", flush=True)
    print("=" * 60, flush=True)

    j_ok, j_fail = render_julia_wallpapers()
    b_ok, b_fail = render_burning_ship_wallpapers()

    total_ok = j_ok + b_ok
    total_fail = j_fail + b_fail
    elapsed = time.perf_counter() - total_start

    print("=" * 60, flush=True)
    print(f"Complete: {total_ok} OK, {total_fail} failed ({elapsed:.1f}s total)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
