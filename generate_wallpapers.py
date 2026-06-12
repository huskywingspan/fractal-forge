"""Generate a curated set of fractal wallpapers.

Usage:
    # Preview set (720p, 30 wallpapers)
    python generate_wallpapers.py --preview

    # Render specific favorites at all resolutions
    python generate_wallpapers.py --favorites 1,5,12,22 --full

    # Render all at a specific resolution
    python generate_wallpapers.py --width 3840 --height 2160

    # Package favorites into a zip archive
    python generate_wallpapers.py --favorites 1,5,12,22 --full --package
"""

import argparse
import shutil
import time
from pathlib import Path

from fractalforge.render.frame_renderer import render_single

# --- Curated wallpaper locations ---
# Each entry: (name, center_re, center_im, zoom, max_iter, palette, slope_shading)

WALLPAPERS = [
    # Classic views with new rendering
    ("seahorse_valley_deep_blue", -0.745428, 0.113009788, 800, 3000,
     "deep_blue", True),
    ("seahorse_spiral_inferno", -0.7463, 0.1102, 5000, 4000,
     "inferno", True),
    ("elephant_valley_arctic", 0.2821, 0.0100, 500, 2000,
     "arctic", True),
    ("main_cardioid_edge_prism", -0.75, 0.01, 50, 1000,
     "prism", True),
    ("double_spiral_twilight", -0.7436439, 0.1318259, 200000, 8000,
     "twilight", True),

    # Antenna / needle region
    ("antenna_deep_blue", -1.478, 0.0, 300, 2000,
     "deep_blue", True),
    ("antenna_tip_inferno", -1.63, 0.0195, 1000, 3000,
     "inferno", True),
    ("antenna_bulb_arctic", -1.7685, 0.00178, 5000, 4000,
     "arctic", True),

    # Seahorse valley deep
    ("seahorse_zoom_electric", -0.74543, 0.11301, 50000, 6000,
     "electric", True),
    ("seahorse_tendril_fire", -0.7454294, 0.1130104, 1e6, 8000,
     "fire", True),

    # Spiral features
    ("triple_spiral_prism", -0.1011, 0.9563, 500, 2000,
     "prism", True),
    ("golden_spiral_inferno", -0.3905407, 0.5898879, 2000, 3000,
     "inferno", True),
    ("spiral_arm_twilight", -0.10109636384562, 0.95628651080914, 50000, 6000,
     "twilight", True),

    # Mini Mandelbrot copies
    ("minibrot_deep_blue", -1.768778833, -0.001738996, 1e6, 10000,
     "deep_blue", True),
    ("minibrot_arctic", -0.156520166, 1.032247108, 10000, 6000,
     "arctic", True),
    ("minibrot_needle_inferno", -1.7497667, 0.0000006, 200000, 10000,
     "inferno", True),

    # Period bulbs and cusps
    ("period3_cusp_electric", -0.1226, 0.7449, 2000, 3000,
     "electric", True),
    ("period4_bulb_nebula", 0.282, 0.530, 1000, 2000,
     "nebula", True),
    ("san_marco_twilight", -0.75, 0.0, 8, 800,
     "twilight", True),

    # Dramatic boundary detail
    ("boundary_fire_1", -0.748, 0.065, 2000, 3000,
     "fire", True),
    ("boundary_deep_blue_1", -1.256, 0.382, 2000, 3000,
     "deep_blue", True),
    ("filament_arctic", -0.55, 0.6264, 10000, 5000,
     "arctic", True),

    # Julia-like features in Mandelbrot
    ("dendrite_prism", -1.401155, 0.0, 20000, 6000,
     "prism", True),
    ("star_burst_inferno", 0.37001085, -0.1017168, 2000, 3000,
     "inferno", True),

    # Wide angle views with slope shading
    ("overview_deep_blue", -0.75, 0.0, 1.0, 500,
     "deep_blue", True),
    ("overview_inferno", -0.75, 0.0, 1.0, 500,
     "inferno", True),
    ("overview_arctic", -0.75, 0.0, 1.0, 500,
     "arctic", True),

    # Deep detail shots
    ("feather_twilight", -0.7453, 0.1127, 100000, 8000,
     "twilight", True),
    ("lightning_electric", -0.170337, -1.06506, 2000, 3000,
     "electric", True),
    ("vortex_deep_blue", -0.7258890085, 0.2500305545, 100000, 8000,
     "deep_blue", True),

    # --- Banded alternates of the 9 favorites (31-39) ---
    # Same coordinates, sandwich palettes for dramatic banded look
    ("seahorse_spiral_volcanic", -0.7463, 0.1102, 5000, 4000,
     "volcanic", True),
    ("double_spiral_aurora", -0.7436439, 0.1318259, 200000, 8000,
     "aurora", True),
    ("antenna_bulb_frozen", -1.7685, 0.00178, 5000, 4000,
     "frozen", True),
    ("seahorse_tendril_solar_flare", -0.7454294, 0.1130104, 1e6, 8000,
     "solar_flare", True),
    ("triple_spiral_stained_glass", -0.1011, 0.9563, 500, 2000,
     "stained_glass", True),
    ("boundary_neon_city", -0.748, 0.065, 2000, 3000,
     "neon_city", True),
    ("boundary_ocean_waves", -1.256, 0.382, 2000, 3000,
     "ocean_waves", True),
    ("dendrite_supernova", -1.401155, 0.0, 20000, 6000,
     "supernova", True),
    ("feather_midnight_rose", -0.7453, 0.1127, 100000, 8000,
     "midnight_rose", True),
]

# Output resolutions for full renders
FULL_RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4K": (3840, 2160),
    "ultrawide": (5120, 1440),
}


def render_wallpaper(entry, width, height, output_path, histogram=True,
                     log_scaling=False, ssaa=1):
    """Render a single wallpaper."""
    name, re, im, zoom, max_iter, palette, slope = entry
    img = render_single(
        center_re=re,
        center_im=im,
        zoom=zoom,
        width=width,
        height=height,
        max_iter=max_iter,
        palette_name=palette,
        slope_shading=slope,
        histogram=histogram,
        log_scaling=log_scaling,
        supersampling=ssaa,
        contrast=1.3,
        saturation=1.4,
        brightness=1.2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate fractal wallpapers")
    parser.add_argument("--preview", action="store_true",
                        help="Render all 30 at 720p (preview set)")
    parser.add_argument("--full", action="store_true",
                        help="Render at all standard resolutions")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--favorites", type=str, default="",
                        help="Comma-separated 1-based indices (e.g. 1,5,12)")
    parser.add_argument("--package", action="store_true",
                        help="Create zip archive after rendering")
    parser.add_argument("--output", type=str, default="output/wallpapers",
                        help="Output directory")
    parser.add_argument("--ssaa", type=int, default=1,
                        help="Supersampling factor (1=off, 2=4x)")
    parser.add_argument("--log-scaling", action="store_true",
                        help="Enable log scaling for smoother gradients")
    parser.add_argument("--list", action="store_true",
                        help="List all wallpaper entries and exit")
    args = parser.parse_args()

    if args.list:
        for i, entry in enumerate(WALLPAPERS, 1):
            name, re, im, zoom, mi, pal, slope = entry
            print(f"  {i:2d}. {name:<32s} zoom={zoom:<10.0e} palette={pal}")
        return

    # Determine which wallpapers to render
    if args.favorites:
        indices = [int(x.strip()) - 1 for x in args.favorites.split(",")]
        entries = [(i, WALLPAPERS[i]) for i in indices if 0 <= i < len(WALLPAPERS)]
    else:
        entries = list(enumerate(WALLPAPERS))

    # Determine resolutions
    if args.full:
        resolutions = FULL_RESOLUTIONS
    elif args.preview:
        resolutions = {"720p": (1280, 720)}
    else:
        label = f"{args.width}x{args.height}"
        resolutions = {label: (args.width, args.height)}

    base_dir = Path(args.output)
    total = len(entries) * len(resolutions)
    rendered = 0
    start = time.perf_counter()

    print(f"Rendering {len(entries)} wallpapers at {len(resolutions)} resolution(s)")
    print(f"Output: {base_dir.resolve()}\n")

    for res_label, (w, h) in resolutions.items():
        res_dir = base_dir / res_label
        print(f"--- {res_label} ({w}x{h}) ---")

        for idx, entry in entries:
            name = entry[0]
            filename = f"{idx+1:02d}_{name}.png"
            out_path = res_dir / filename

            if out_path.exists():
                print(f"  [{rendered+1}/{total}] {filename} (exists, skipping)")
                rendered += 1
                continue

            t0 = time.perf_counter()
            render_wallpaper(entry, w, h, out_path,
                             log_scaling=args.log_scaling, ssaa=args.ssaa)
            dt = time.perf_counter() - t0

            rendered += 1
            elapsed = time.perf_counter() - start
            eta = (elapsed / rendered) * (total - rendered) if rendered > 0 else 0
            print(f"  [{rendered}/{total}] {filename} ({dt:.1f}s, ETA {eta:.0f}s)")

    elapsed = time.perf_counter() - start
    print(f"\nDone! {rendered} wallpapers in {elapsed:.0f}s")

    if args.package:
        archive_name = base_dir / "fractalforge_wallpapers"
        print(f"Packaging to {archive_name}.zip ...")
        shutil.make_archive(str(archive_name), "zip", str(base_dir))
        print(f"Archive created: {archive_name}.zip")


if __name__ == "__main__":
    main()
