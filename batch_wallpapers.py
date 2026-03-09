"""Desktop wallpaper pack generator -- 25 scenes x 4 resolutions = 100 images.

Run with: python batch_wallpapers.py

Generates a zipped package of fractal desktop backgrounds at:
  - 1920x1080  (1080p 16:9)
  - 2560x1440  (2K 16:9)
  - 3840x2160  (4K 16:9)
  - 5120x1440  (Ultrawide 32:9)

All rendered with 2x SSAA and histogram equalization for wallpaper quality.
Estimated time: ~10-15 minutes on RTX 3070.
"""

import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fractalforge.render.frame_renderer import render_single

# ──────────────────────────────────────────────────────────────
# Resolutions
# ──────────────────────────────────────────────────────────────

RESOLUTIONS = {
    "1080p":     (1920, 1080),
    "2K":        (2560, 1440),
    "4K":        (3840, 2160),
    "ultrawide": (5120, 1440),
}

# ──────────────────────────────────────────────────────────────
# 25 unique scenes — diverse locations, palettes, zoom levels
# ──────────────────────────────────────────────────────────────

SCENES = [
    # --- Seahorse Valley ---
    {
        "name": "01_seahorse_overview_ocean",
        "center_re": -0.74529, "center_im": 0.11307,
        "zoom": 50, "max_iter": 800, "palette": "ocean",
        "vignette": 0.3, "contrast": 1.05, "saturation": 1.1,
    },
    {
        "name": "02_seahorse_spiral_fire",
        "center_re": -0.74529, "center_im": 0.11307,
        "zoom": 5000, "max_iter": 1500, "palette": "fire",
        "vignette": 0.35, "contrast": 1.1, "saturation": 1.0,
    },
    {
        "name": "03_seahorse_deep_electric",
        "center_re": -0.74529, "center_im": 0.11307,
        "zoom": 50000, "max_iter": 2000, "palette": "electric",
        "vignette": 0.4, "contrast": 1.0, "saturation": 1.15,
    },
    {
        "name": "04_seahorse_abyss_nebula",
        "center_re": -0.74529, "center_im": 0.11307,
        "zoom": 500000, "max_iter": 3000, "palette": "nebula",
        "vignette": 0.4, "contrast": 1.05, "saturation": 1.1,
    },
    {
        "name": "05_seahorse_tendril_ocean",
        "center_re": -0.7453, "center_im": 0.1127,
        "zoom": 2000, "max_iter": 1200, "palette": "ocean",
        "vignette": 0.3, "contrast": 1.0, "saturation": 1.2,
    },

    # --- Elephant Valley ---
    {
        "name": "06_elephant_valley_ocean",
        "center_re": 0.2826, "center_im": 0.0100,
        "zoom": 80, "max_iter": 600, "palette": "ocean",
        "vignette": 0.3, "contrast": 1.05, "saturation": 1.0,
    },
    {
        "name": "07_elephant_trunks_electric",
        "center_re": 0.2826, "center_im": 0.0100,
        "zoom": 2000, "max_iter": 1000, "palette": "electric",
        "vignette": 0.35, "contrast": 1.0, "saturation": 1.1,
    },
    {
        "name": "08_elephant_deep_fire",
        "center_re": 0.2826, "center_im": 0.0100,
        "zoom": 50000, "max_iter": 1800, "palette": "fire",
        "vignette": 0.4, "contrast": 1.1, "saturation": 1.0,
    },

    # --- Triple Spiral ---
    {
        "name": "09_triple_spiral_nebula",
        "center_re": -0.0452, "center_im": 0.9868,
        "zoom": 200, "max_iter": 800, "palette": "nebula",
        "vignette": 0.35, "contrast": 1.0, "saturation": 1.1,
    },
    {
        "name": "10_triple_spiral_deep_ocean",
        "center_re": -0.0452, "center_im": 0.9868,
        "zoom": 10000, "max_iter": 1500, "palette": "ocean",
        "vignette": 0.4, "contrast": 1.05, "saturation": 1.15,
    },
    {
        "name": "11_triple_vortex_fire",
        "center_re": -0.0452, "center_im": 0.9868,
        "zoom": 100000, "max_iter": 2500, "palette": "fire",
        "vignette": 0.45, "contrast": 1.05, "saturation": 1.0,
    },

    # --- Antenna / Needle ---
    {
        "name": "12_antenna_electric",
        "center_re": -1.7685, "center_im": 0.0014,
        "zoom": 100, "max_iter": 1500, "palette": "electric",
        "vignette": 0.3, "contrast": 1.0, "saturation": 1.1,
    },
    {
        "name": "13_antenna_deep_nebula",
        "center_re": -1.7685, "center_im": 0.0014,
        "zoom": 20000, "max_iter": 3000, "palette": "nebula",
        "vignette": 0.4, "contrast": 1.0, "saturation": 1.0,
    },

    # --- Spiral Arm (optimized coords) ---
    {
        "name": "14_spiral_arm_ocean",
        "center_re": -0.7746806106269039, "center_im": -0.1374168856037867,
        "zoom": 200, "max_iter": 800, "palette": "ocean",
        "vignette": 0.35, "contrast": 1.0, "saturation": 1.2,
    },
    {
        "name": "15_spiral_deep_nebula",
        "center_re": -0.7746806106269039, "center_im": -0.1374168856037867,
        "zoom": 10000, "max_iter": 1500, "palette": "nebula",
        "vignette": 0.4, "contrast": 1.1, "saturation": 1.1,
    },
    {
        "name": "16_spiral_abyss_electric",
        "center_re": -0.7746806106269039, "center_im": -0.1374168856037867,
        "zoom": 200000, "max_iter": 2500, "palette": "electric",
        "vignette": 0.45, "contrast": 1.0, "saturation": 1.15,
    },

    # --- Showcase coords (seahorse deep) ---
    {
        "name": "17_descent_nebula",
        "center_re": -0.7436438870371587, "center_im": 0.1318259042053119,
        "zoom": 1000, "max_iter": 1200, "palette": "nebula",
        "vignette": 0.3, "contrast": 1.05, "saturation": 1.1,
    },
    {
        "name": "18_descent_deep_ocean",
        "center_re": -0.7436438870371587, "center_im": 0.1318259042053119,
        "zoom": 100000, "max_iter": 2000, "palette": "ocean",
        "vignette": 0.4, "contrast": 1.0, "saturation": 1.15,
    },
    {
        "name": "19_descent_abyss_fire",
        "center_re": -0.7436438870371587, "center_im": 0.1318259042053119,
        "zoom": 5000000, "max_iter": 3500, "palette": "fire",
        "vignette": 0.45, "contrast": 1.1, "saturation": 1.0,
    },

    # --- Classic views ---
    {
        "name": "20_full_set_ocean",
        "center_re": -0.5, "center_im": 0.0,
        "zoom": 1.0, "max_iter": 500, "palette": "ocean",
        "vignette": 0.25, "contrast": 1.0, "saturation": 1.1,
    },
    {
        "name": "21_full_set_nebula",
        "center_re": -0.5, "center_im": 0.0,
        "zoom": 1.0, "max_iter": 500, "palette": "nebula",
        "vignette": 0.25, "contrast": 1.0, "saturation": 1.1,
    },
    {
        "name": "22_cardioid_cusp_fire",
        "center_re": -0.75, "center_im": 0.01,
        "zoom": 150, "max_iter": 800, "palette": "fire",
        "vignette": 0.35, "contrast": 1.05, "saturation": 1.0,
    },

    # --- Exotic locations ---
    {
        "name": "23_lightning_electric",
        "center_re": -0.16, "center_im": 1.0405,
        "zoom": 150, "max_iter": 800, "palette": "electric",
        "vignette": 0.3, "contrast": 1.0, "saturation": 1.2,
    },
    {
        "name": "24_period2_bulb_ocean",
        "center_re": -1.0, "center_im": 0.0,
        "zoom": 30, "max_iter": 600, "palette": "ocean",
        "vignette": 0.3, "contrast": 1.0, "saturation": 1.1,
    },
    {
        "name": "25_scepter_nebula",
        "center_re": -0.1011, "center_im": 0.9563,
        "zoom": 300, "max_iter": 1000, "palette": "nebula",
        "vignette": 0.35, "contrast": 1.05, "saturation": 1.1,
    },
]


def main():
    output_root = Path("output/wallpapers")
    total = len(SCENES) * len(RESOLUTIONS)

    print(f"FractalForge Wallpaper Pack Generator")
    print(f"  Scenes:      {len(SCENES)}")
    print(f"  Resolutions: {', '.join(RESOLUTIONS.keys())}")
    print(f"  Total:       {total} images")
    print(f"  SSAA:        8x (1080p/2K), 4x (4K/ultrawide)")
    print(f"  Histogram:   enabled")
    print()

    # Adaptive SSAA: 8x for smaller resolutions, 4x for large (avoids RAM OOM in coloring)
    ssaa_map = {"1080p": 8, "2K": 8, "4K": 4, "ultrawide": 4}

    batch_start = time.perf_counter()
    count = 0
    all_paths = []

    for scene in SCENES:
        for res_name, (w, h) in RESOLUTIONS.items():
            count += 1
            res_dir = output_root / res_name
            res_dir.mkdir(parents=True, exist_ok=True)

            ss = ssaa_map[res_name]
            filename = f"{scene['name']}_{res_name}.png"
            out_path = res_dir / filename

            if out_path.exists():
                print(f"  [{count:3d}/{total}] {filename} -- exists, skipping")
                all_paths.append(out_path)
                continue

            start = time.perf_counter()
            img = render_single(
                center_re=scene["center_re"],
                center_im=scene["center_im"],
                zoom=scene["zoom"],
                width=w,
                height=h,
                max_iter=scene["max_iter"],
                palette_name=scene["palette"],
                use_gpu=None,
                supersampling=ss,
                histogram=True,
                vignette=scene.get("vignette", 0.0),
                contrast=scene.get("contrast", 1.0),
                saturation=scene.get("saturation", 1.0),
                brightness=scene.get("brightness", 1.0),
            )
            img.save(out_path, format="PNG", optimize=True)
            elapsed = time.perf_counter() - start
            size_kb = out_path.stat().st_size / 1024
            all_paths.append(out_path)

            pct = count / total * 100
            total_elapsed = time.perf_counter() - batch_start
            avg = total_elapsed / count
            remaining = avg * (total - count)
            print(f"  [{count:3d}/{total}] {filename:<55s} "
                  f"{elapsed:5.1f}s  {size_kb:7.0f} KB  "
                  f"(~{remaining/60:.0f}m left)")

    total_elapsed = time.perf_counter() - batch_start
    total_size_mb = sum(p.stat().st_size for p in all_paths) / 1_048_576
    print(f"\nRendering complete: {total} images in {total_elapsed/60:.1f} minutes")
    print(f"Total size: {total_size_mb:.0f} MB")

    # ── Create ZIP ──
    zip_path = Path("output/FractalForge_Wallpapers.zip")
    print(f"\nCreating {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(all_paths):
            # Archive path: resolution/filename.png
            arcname = f"{p.parent.name}/{p.name}"
            zf.write(p, arcname)

    zip_size_mb = zip_path.stat().st_size / 1_048_576
    print(f"Done: {zip_path} ({zip_size_mb:.0f} MB)")
    print(f"\nContents:")
    for res_name in RESOLUTIONS:
        res_count = sum(1 for p in all_paths if p.parent.name == res_name)
        print(f"  {res_name}/:  {res_count} images")


if __name__ == "__main__":
    main()
