"""Deep zoom validation test -- verify DZ-P1 improvements.

Tests rendering at zoom depths from 1e5 to 1e50, measuring glitch rates
and visual coherence. Two test categories:

1. Boundary tests: Newton-exact boundary coordinates at appropriate zoom.
   These verify that renders correctly show the set boundary (mix of interior
   and exterior pixels). Zoom levels match the component scale.

2. Deep zoom stress tests: Approximate coordinates at extreme zoom depths.
   These verify that the perturbation engine, rebasing, and SA produce
   coherent results without crashing or glitching at 1e15-1e50.

Usage:
    python test_deep_zoom.py                # Quick test (320x180)
    python test_deep_zoom.py --full         # Full resolution (1280x720)
    python test_deep_zoom.py --save         # Save output PNGs for inspection
"""

import argparse
import math
import time
import sys

import numpy as np

from fractalforge.engine.perturbation import render_frame_perturbation
from fractalforge.engine.precision import required_precision


# Category 1: Newton-exact boundary coordinates.
# These are cusps/junctions of hyperbolic components, found via Newton-Raphson.
# They sit precisely on the Mandelbrot set boundary and should produce renders
# with BOTH interior and exterior pixels at the right zoom level.
BOUNDARY_TESTS = [
    {
        "name": "Seahorse Cusp (p78)",
        "re": "-0.74364370100564797457366520039421821802375908955924038408294634708269691803198978265871685137187160229219941560353660557642949682229350889240056341066534161142143957678657732181078713597137658880879434",
        "im": "0.13182567604026238010235223909156790593609644544108601724796664499668282256093790280438850021182089365760884988386035606608851269688319967005431908358133010755591629675114335852840029397806012182590754",
        "zooms": [1e5, 1e6, 5e6],
        "base_iter": 3000,
    },
    {
        "name": "Elephant Cusp (p50)",
        "re": "0.28183772439677078939902463652202895289900794593423391529805732284468778561192366152897821933742230042146623567000518942559253884173239474446805773212859706082842906411300503422107089128490214249169048",
        "im": "0.010058171744873996291128693366641356356374876871038834137879588330060442142757816480846413742111623638590895765031955697013067206027223597863870007904706761412761536631281373914587673953405198300369924",
        "zooms": [1e5, 1e6, 5e6],
        "base_iter": 3000,
    },
    {
        "name": "Minibrot Nucleus (p113)",
        "re": "-1.76877881882505244070204328921906568399856273481965709587200755304435521001682796491704283020662362167122981178033708257547888052867151827584139486413",
        "im": "-0.00173899059813790006444534742722313224745528942564519959026376600929212426675640297779738950164161248291905232628495345240125143239780042581054298429249",
        "zooms": [5e5, 1e6, 3e6],
        "base_iter": 5000,
    },
]

# Category 2: Deep zoom stress tests.
# Approximate coordinates near the Mandelbrot set boundary. These test the
# perturbation engine at extreme zoom depths. The coordinates don't need to be
# exactly on the boundary -- we care about coherent rendering (no glitches,
# smooth iteration values, reasonable performance).
DEEP_STRESS_TESTS = [
    {
        "name": "Seahorse Deep",
        "re": "-0.7436438870371587326159143588427",
        "im": "0.1318259043091895211526879720784",
        "zooms": [1e15, 1e20, 1e25],
        "base_iter": 5000,
    },
    {
        "name": "Minibrot Deep",
        "re": "-1.768778833000012498376892342344",
        "im": "-0.001738996042704908868459476484",
        "zooms": [1e15, 1e20, 1e25],
        "base_iter": 10000,
    },
    {
        "name": "Antenna Deep",
        "re": "-1.999911750100399052164701049985",
        "im": "0.0",
        "zooms": [1e15, 1e20, 1e30],
        "base_iter": 8000,
    },
]


def compute_max_iter(base_iter, zoom):
    """Scale max_iter with zoom depth."""
    return max(base_iter, int(500 + 300 * math.log10(zoom)))


def test_render(name, re_str, im_str, zoom, max_iter, width, height,
                save=False, expect_boundary=False):
    """Render a single test frame and report quality metrics."""
    prec = required_precision(zoom, re_str, im_str)
    print(f"  Zoom {zoom:.0e}, max_iter={max_iter}, precision={prec} digits")
    sys.stdout.flush()

    t0 = time.perf_counter()
    smooth = render_frame_perturbation(
        center_re=re_str,
        center_im=im_str,
        zoom=zoom,
        width=width,
        height=height,
        max_iter=max_iter,
    )
    elapsed = time.perf_counter() - t0

    total_pixels = width * height
    interior = np.sum(smooth < 0)
    exterior = np.sum(smooth >= 0)
    interior_rate = interior / total_pixels

    # Quality assessment
    if expect_boundary:
        # Boundary test: should have both interior and exterior pixels
        # 5-95% interior is ideal; pure interior or pure exterior is wrong
        if 0.05 <= interior_rate <= 0.95:
            quality = "GOOD"
        elif 0.01 <= interior_rate <= 0.99:
            quality = "OK"
        else:
            quality = "SUSPECT"
    else:
        # Stress test: we just want coherent output (no 100% glitch)
        # 0% interior (all escape) is fine -- it's consistent rendering
        # 100% interior is only bad if the reference orbit escapes
        if interior_rate < 0.95:
            quality = "GOOD"
        elif interior_rate < 0.99:
            quality = "SUSPECT"
        else:
            quality = "FAILED"

    # Check for variation in exterior values (should have smooth gradients)
    if exterior > 0:
        ext_vals = smooth[smooth >= 0]
        ext_range = np.ptp(ext_vals)
    else:
        ext_range = 0

    print(f"    Time: {elapsed:.1f}s | Interior: {interior}/{total_pixels} ({interior_rate:.1%}) | "
          f"Exterior range: {ext_range:.1f} | Quality: {quality}")

    if save:
        from fractalforge.engine.coloring import smooth_to_image
        from fractalforge.artist.palette import get_palette
        palette = get_palette("deep_blue")
        img = smooth_to_image(smooth, palette, histogram=True, slope_shading=True)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fname = f"output/deep_zoom_test/{safe_name}_z{zoom:.0e}.png"
        from pathlib import Path
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        img.save(fname)
        print(f"    Saved: {fname}")

    return {
        "name": name,
        "zoom": zoom,
        "time": elapsed,
        "interior_rate": interior_rate,
        "quality": quality,
        "ext_range": ext_range,
    }


def main():
    parser = argparse.ArgumentParser(description="Deep zoom validation tests")
    parser.add_argument("--full", action="store_true", help="Use 1280x720 (default: 320x180)")
    parser.add_argument("--save", action="store_true", help="Save output PNGs")
    parser.add_argument("--boundary-only", action="store_true", help="Run only boundary tests")
    parser.add_argument("--stress-only", action="store_true", help="Run only stress tests")
    args = parser.parse_args()

    width, height = (1280, 720) if args.full else (320, 180)
    print(f"Deep Zoom Validation Test -- {width}x{height}")
    print(f"{'='*60}\n")

    # Show precision formula
    for z in [1e5, 1e15, 1e20, 1e30, 1e50]:
        print(f"  Zoom {z:.0e} -> {required_precision(z)} digits")
    print()

    results = []

    # Category 1: Boundary tests
    if not args.stress_only:
        print(f"{'='*60}")
        print("BOUNDARY TESTS (Newton-exact coordinates)")
        print(f"{'='*60}\n")

        for point in BOUNDARY_TESTS:
            print(f"[{point['name']}]")
            print(f"  Re: {point['re'][:50]}...")
            print(f"  Im: {point['im'][:50]}...")

            for zoom in point["zooms"]:
                max_iter = compute_max_iter(point["base_iter"], zoom)
                try:
                    r = test_render(
                        point["name"], point["re"], point["im"],
                        zoom, max_iter, width, height,
                        save=args.save, expect_boundary=True,
                    )
                    results.append(r)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({
                        "name": point["name"], "zoom": zoom,
                        "quality": "ERROR", "interior_rate": 1.0,
                    })
            print()

    # Category 2: Deep zoom stress tests
    if not args.boundary_only:
        print(f"{'='*60}")
        print("DEEP ZOOM STRESS TESTS (perturbation + rebasing)")
        print(f"{'='*60}\n")

        for point in DEEP_STRESS_TESTS:
            print(f"[{point['name']}]")
            print(f"  Re: {point['re']}")
            print(f"  Im: {point['im']}")

            for zoom in point["zooms"]:
                max_iter = compute_max_iter(point["base_iter"], zoom)
                try:
                    r = test_render(
                        point["name"], point["re"], point["im"],
                        zoom, max_iter, width, height,
                        save=args.save, expect_boundary=False,
                    )
                    results.append(r)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({
                        "name": point["name"], "zoom": zoom,
                        "quality": "ERROR", "interior_rate": 1.0,
                    })
            print()

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    good = sum(1 for r in results if r["quality"] in ("GOOD", "OK"))
    suspect = sum(1 for r in results if r["quality"] == "SUSPECT")
    failed = sum(1 for r in results if r["quality"] in ("FAILED", "ERROR"))
    print(f"  GOOD/OK: {good}  |  SUSPECT: {suspect}  |  FAILED: {failed}")
    print()

    for r in results:
        markers = {"GOOD": "+", "OK": "~", "SUSPECT": "?", "FAILED": "X", "ERROR": "!"}
        marker = markers.get(r["quality"], "?")
        print(f"  [{marker}] {r['name']:25s} @ {r['zoom']:.0e}  "
              f"int={r['interior_rate']:.1%}  "
              f"time={r.get('time', 0):.1f}s")


if __name__ == "__main__":
    main()
