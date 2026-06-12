"""Generate 10 test wallpapers showcasing the HDR bloom pipeline.

Tests bloom, halation, ACES/Reinhard tone mapping, vignette, and various
palettes at multiple resolutions.

Usage:
    python test_hdr_wallpapers.py
"""

import time
from pathlib import Path

from fractalforge.render.frame_renderer import render_single

# 10 wallpapers: (name, center_re, center_im, zoom, max_iter, palette, resolution, hdr_opts)
WALLPAPERS = [
    # 1. Seahorse valley — volcanic bloom, 4K
    (
        "01_seahorse_volcanic_bloom_4k",
        -0.7463, 0.1102, 5000, 4000, "volcanic",
        (3840, 2160),
        dict(bloom=0.4, tone_map="aces", exposure=1.2, vignette=0.3),
    ),
    # 2. Elephant valley — aurora halation, ultrawide 5120x1440
    (
        "02_elephant_aurora_halation_ultrawide",
        0.2821, 0.0100, 500, 2000, "aurora",
        (5120, 1440),
        dict(bloom=0.25, halation=0.2, tone_map="aces", exposure=1.1),
    ),
    # 3. Deep spiral — supernova full HDR, 1440p
    (
        "03_deep_spiral_supernova_1440p",
        -0.7436439, 0.1318259, 200000, 8000, "supernova",
        (2560, 1440),
        dict(bloom=0.5, halation=0.15, tone_map="aces", exposure=1.3, vignette=0.4),
    ),
    # 4. Antenna region — deep_blue Reinhard, 1080p
    (
        "04_antenna_deepblue_reinhard_1080p",
        -1.7685, 0.00178, 5000, 4000, "deep_blue",
        (1920, 1080),
        dict(bloom=0.35, tone_map="reinhard", exposure=1.4),
    ),
    # 5. Mini Mandelbrot — midnight_rose heavy bloom, 4K
    (
        "05_minibrot_midnight_rose_4k",
        -1.768778833, -0.001738996, 1e6, 10000, "midnight_rose",
        (3840, 2160),
        dict(bloom=0.6, halation=0.2, tone_map="aces", exposure=1.1, vignette=0.35),
    ),
    # 6. Triple spiral — stained_glass warm halation, 1080p
    (
        "06_triple_spiral_stained_glass_1080p",
        -0.1011, 0.9563, 500, 2000, "stained_glass",
        (1920, 1080),
        dict(bloom=0.3, halation=0.25, tone_map="aces", exposure=1.0),
    ),
    # 7. Feather detail — twilight subtle bloom, ultrawide
    (
        "07_feather_twilight_ultrawide",
        -0.7453, 0.1127, 100000, 8000, "twilight",
        (5120, 1440),
        dict(bloom=0.2, halation=0.1, tone_map="aces", exposure=1.15, vignette=0.25),
    ),
    # 8. Overview — neon_city max bloom, 1440p
    (
        "08_overview_neon_city_1440p",
        -0.75, 0.0, 1.0, 500, "neon_city",
        (2560, 1440),
        dict(bloom=0.7, halation=0.3, tone_map="aces", exposure=0.9, vignette=0.5),
    ),
    # 9. Star burst — fire Reinhard warm, 4K
    (
        "09_starburst_fire_reinhard_4k",
        0.37001085, -0.1017168, 2000, 3000, "fire",
        (3840, 2160),
        dict(bloom=0.45, tone_map="reinhard", exposure=1.5, vignette=0.3),
    ),
    # 10. Boundary — frozen bloom no tone map (A/B baseline), 1080p
    (
        "10_boundary_frozen_bloom_raw_1080p",
        -0.748, 0.065, 2000, 3000, "frozen",
        (1920, 1080),
        dict(bloom=0.4, halation=0.15),
    ),
]


def main():
    out_dir = Path("output/hdr_wallpaper_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(WALLPAPERS)
    start = time.perf_counter()

    print(f"Rendering {total} HDR test wallpapers")
    print(f"Output: {out_dir.resolve()}\n")

    for i, (name, re, im, zoom, max_iter, palette, (w, h), hdr) in enumerate(WALLPAPERS, 1):
        t0 = time.perf_counter()
        img = render_single(
            center_re=re,
            center_im=im,
            zoom=zoom,
            width=w,
            height=h,
            max_iter=max_iter,
            palette_name=palette,
            slope_shading=True,
            histogram=True,
            supersampling=1,
            **hdr,
        )
        out_path = out_dir / f"{name}.png"
        img.save(out_path, format="PNG")
        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - start
        eta = (elapsed / i) * (total - i)

        size_mb = out_path.stat().st_size / 1_048_576
        effects = []
        if hdr.get("bloom", 0) > 0:
            effects.append(f"bloom={hdr['bloom']}")
        if hdr.get("halation", 0) > 0:
            effects.append(f"halation={hdr['halation']}")
        if hdr.get("tone_map", "none") != "none":
            effects.append(f"{hdr['tone_map']}(exp={hdr.get('exposure', 1.0)})")
        if hdr.get("vignette", 0) > 0:
            effects.append(f"vig={hdr['vignette']}")

        print(f"  [{i}/{total}] {name}")
        print(f"          {w}x{h} | {palette} | {' + '.join(effects)}")
        print(f"          {dt:.1f}s | {size_mb:.1f} MB | ETA {eta:.0f}s")
        print()

    elapsed = time.perf_counter() - start
    print(f"Done! {total} wallpapers in {elapsed:.0f}s")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
