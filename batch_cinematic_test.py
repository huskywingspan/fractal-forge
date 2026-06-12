"""Render short test clips comparing legacy vs cinematic camera motion.

Renders 10-second clips (600 frames) at 1080p for the best multi-keyframe
presets, in both interpolation modes, for side-by-side comparison.
"""

import subprocess
import sys
import time
from pathlib import Path

FRACTALFORGE = [sys.executable, "-c", "from fractalforge.cli.main import cli; cli()"]

# Presets with 3+ keyframes and center changes (where cinematic matters most)
TEST_PRESETS = [
    "presets/elephant_valley.json",
    "presets/minibrot_hunt.json",
    "presets/julia_dendrite_zoom.json",
    "presets/burning_ship_full_zoom.json",
]

MODES = ["legacy", "cinematic"]


def run_zoom(preset_path, mode, encode_preset="youtube"):
    """Render a zoom video with the given interpolation mode."""
    name = Path(preset_path).stem
    frames_dir = f"output/{name}_{mode}_frames"
    video_out = f"output/{name}_{mode}.mp4"

    cmd = FRACTALFORGE + [
        "zoom", preset_path,
        "--interpolation", mode,
        "--frames-dir", frames_dir,
        "--output", video_out,
        "--encode-preset", encode_preset,
        "--ss", "4",
    ]

    print(f"\n{'='*60}", flush=True)
    print(f"  Preset: {name}", flush=True)
    print(f"  Mode:   {mode}", flush=True)
    print(f"  Output: {video_out}", flush=True)
    print(f"{'='*60}", flush=True)

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.perf_counter() - start

    video_path = Path(video_out)
    if result.returncode == 0 and video_path.exists():
        size_mb = video_path.stat().st_size / 1_048_576
        print(f"\n  OK: {elapsed:.0f}s, {size_mb:.1f} MB", flush=True)
        return True
    else:
        print(f"\n  FAILED after {elapsed:.0f}s", flush=True)
        return False


def main():
    total_start = time.perf_counter()

    print("=" * 60, flush=True)
    print("FractalForge - Cinematic Camera Test Renders", flush=True)
    print("1080p, youtube encode, full-length per preset", flush=True)
    print("=" * 60, flush=True)

    results = []
    for preset in TEST_PRESETS:
        for mode in MODES:
            ok = run_zoom(preset, mode)
            results.append((Path(preset).stem, mode, ok))

    elapsed = time.perf_counter() - total_start

    print(f"\n\n{'='*60}", flush=True)
    print(f"Results ({elapsed:.0f}s total):", flush=True)
    print(f"{'='*60}", flush=True)
    for name, mode, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"  {name:30s} {mode:10s} {status}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
