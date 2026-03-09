"""Overnight batch render -- ~6 hours of rendering to test all Phase 4 features.

Run with: python batch_overnight.py

Renders 8 zoom sequences with various feature combinations:
- Multiple locations (seahorse, elephant, triple spiral, antenna, minibrot)
- All 5 palettes (ocean, fire, electric, monochrome, nebula)
- Post-processing (histogram, vignette, color grading)
- YouTube encode preset
- Shorts generation
- Title cards and thumbnails

Estimated total: ~5.5-6 hours on RTX 3070.
"""

import sys
import time
from pathlib import Path

# Add project source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fractalforge.artist.zoompath import ZoomPath
from fractalforge.render.sequence import render_sequence
from fractalforge.render.video import encode_video, check_ffmpeg
from fractalforge.publish.titlecard import render_title_card
from fractalforge.publish.thumbnail import generate_thumbnail_samples, format_zoom
from fractalforge.publish.shorts import generate_short_frames

# ──────────────────────────────────────────────────────────────
# Render jobs: (preset_path, render_options, metadata)
# ──────────────────────────────────────────────────────────────

JOBS = [
    # ── Job 1: Seahorse Valley (Fire) ── ~30 min
    # Tests: fire palette, histogram EQ, moderate vignette
    {
        "preset": "presets/seahorse_fire.json",
        "ss": 2,
        "histogram": True,
        "vignette": 0.4,
        "contrast": 1.0,
        "saturation": 1.0,
        "brightness": 1.0,
        "encode": "youtube",
        "title": "Seahorse Valley",
        "subtitle": "5 Million x Zoom",
    },

    # ── Job 2: Seahorse Valley (Electric) ── ~30 min
    # Tests: electric palette, slight contrast boost, no histogram
    {
        "preset": "presets/seahorse_electric.json",
        "ss": 2,
        "histogram": False,
        "vignette": 0.3,
        "contrast": 1.1,
        "saturation": 1.0,
        "brightness": 1.0,
        "encode": "youtube",
        "title": "Seahorse Valley",
        "subtitle": "Electric Palette | 5 Million x",
    },

    # ── Job 3: Elephant Valley ── ~30 min
    # Tests: ocean->electric palette transition, histogram, saturation boost
    {
        "preset": "presets/elephant_valley.json",
        "ss": 2,
        "histogram": True,
        "vignette": 0.35,
        "contrast": 1.0,
        "saturation": 1.15,
        "brightness": 1.0,
        "encode": "youtube",
        "title": "Elephant Valley",
        "subtitle": "2 Million x Zoom",
    },

    # ── Job 4: Triple Spiral ── ~30 min
    # Tests: nebula palette, strong vignette for dramatic feel
    {
        "preset": "presets/triple_spiral.json",
        "ss": 2,
        "histogram": True,
        "vignette": 0.5,
        "contrast": 1.05,
        "saturation": 1.1,
        "brightness": 1.0,
        "encode": "youtube",
        "title": "Triple Spiral",
        "subtitle": "3 Million x Zoom",
    },

    # ── Job 5: Antenna Deep Dive ── ~60 min (3600 frames, no SSAA)
    # Tests: long-form 60s video, electric palette, light post-processing
    {
        "preset": "presets/antenna_deep.json",
        "ss": 1,
        "histogram": True,
        "vignette": 0.3,
        "contrast": 1.0,
        "saturation": 1.0,
        "brightness": 1.0,
        "encode": "youtube",
        "title": "Antenna Deep Dive",
        "subtitle": "50 Million x Zoom",
    },

    # ── Job 6: Mini-Mandelbrot Hunt ── ~40 min
    # Tests: monochrome->electric transition, high max_iter, deep zoom
    {
        "preset": "presets/minibrot_hunt.json",
        "ss": 2,
        "histogram": True,
        "vignette": 0.4,
        "contrast": 1.0,
        "saturation": 1.0,
        "brightness": 1.0,
        "encode": "youtube",
        "title": "Mini-Mandelbrot Hunt",
        "subtitle": "100 Million x Zoom",
    },

    # ── Job 7: Spiral Arm (Nebula) ── ~30 min
    # Tests: nebula palette at optimized coords, saturation + contrast
    {
        "preset": "presets/spiral_nebula.json",
        "ss": 2,
        "histogram": False,
        "vignette": 0.45,
        "contrast": 1.1,
        "saturation": 1.2,
        "brightness": 1.0,
        "encode": "youtube",
        "title": "Spiral Arm",
        "subtitle": "Nebula Palette | 500,000 x",
    },

    # ── Job 8: Deep Showcase (reference, no post-processing) ── ~43 min
    # Tests: baseline comparison -- same as original showcase, no effects
    {
        "preset": "presets/deep_showcase.json",
        "ss": 2,
        "histogram": False,
        "vignette": 0.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "brightness": 1.0,
        "encode": "youtube",
        "title": "Seahorse Descent",
        "subtitle": "10 Billion x Zoom",
    },
]


def print_header(msg: str):
    """Print a prominent section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {msg}")
    print(f"{bar}\n")


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def run_job(job: dict, job_idx: int, total_jobs: int):
    """Run a single render job: frames -> encode -> title card -> thumbnails -> short."""
    preset_path = Path(job["preset"])
    zoom_path = ZoomPath.load(preset_path)
    name = zoom_path.name

    frames_dir = Path(f"output/{name}_frames")
    video_path = Path(f"output/{name}.mp4")

    print_header(f"Job {job_idx}/{total_jobs}: {name}")
    print(f"  Preset:     {preset_path}")
    print(f"  Resolution: {zoom_path.width}x{zoom_path.height}")
    print(f"  Frames:     {zoom_path.total_frames} ({zoom_path.duration_seconds:.1f}s)")
    print(f"  SSAA:       {job['ss']}x")
    print(f"  Histogram:  {job['histogram']}")
    if job["vignette"] > 0:
        print(f"  Vignette:   {job['vignette']}")
    if job["contrast"] != 1.0 or job["saturation"] != 1.0 or job["brightness"] != 1.0:
        print(f"  Grade:      c={job['contrast']} s={job['saturation']} b={job['brightness']}")
    print()

    # ── Step 1: Render frames ──
    print(f"  [1/5] Rendering frames...")
    render_start = time.perf_counter()
    last_print = [render_start]

    def on_progress(frame_idx, total, elapsed, fps, skipped=False):
        now = time.perf_counter()
        if now - last_print[0] >= 10.0 or frame_idx == total - 1:
            pct = (frame_idx + 1) / total * 100
            remaining = (total - frame_idx - 1) / fps if fps > 0 else 0
            status = "skip" if skipped else f"{fps:.1f}fps"
            print(f"        {pct:5.1f}% | frame {frame_idx+1}/{total} | {status} | "
                  f"{format_time(elapsed)} elapsed | ~{format_time(remaining)} left")
            last_print[0] = now

    render_sequence(
        zoom_path=zoom_path,
        output_dir=frames_dir,
        use_gpu=None,
        skip_existing=True,
        supersampling=job["ss"],
        on_progress=on_progress,
        histogram=job["histogram"],
        vignette=job["vignette"],
        contrast=job["contrast"],
        saturation=job["saturation"],
        brightness=job["brightness"],
    )
    render_elapsed = time.perf_counter() - render_start
    avg_fps = zoom_path.total_frames / render_elapsed if render_elapsed > 0 else 0
    print(f"        Done: {zoom_path.total_frames} frames in {format_time(render_elapsed)} ({avg_fps:.1f} fps)")

    # ── Step 2: Encode video ──
    if check_ffmpeg():
        print(f"  [2/5] Encoding video ({job['encode']})...")
        encode_start = time.perf_counter()
        video_path = encode_video(
            frames_dir=frames_dir,
            output_path=video_path,
            fps=zoom_path.fps,
            preset=job["encode"],
            overwrite=True,
        )
        encode_elapsed = time.perf_counter() - encode_start
        size_mb = video_path.stat().st_size / 1_048_576
        print(f"        Done: {video_path} ({size_mb:.1f} MB, {encode_elapsed:.1f}s)")
    else:
        print(f"  [2/5] Skipping encode (FFmpeg not found)")

    # ── Step 3: Title card ──
    print(f"  [3/5] Generating title card...")
    tc_path = Path(f"output/{name}_title_card.png")
    render_title_card(
        title=job["title"],
        subtitle=job["subtitle"],
        width=zoom_path.width,
        height=zoom_path.height,
        output_path=tc_path,
    )
    print(f"        Done: {tc_path}")

    # ── Step 4: Thumbnails ──
    print(f"  [4/5] Generating thumbnails...")
    final_zoom = zoom_path.keyframes[-1].zoom if zoom_path.keyframes else 1.0
    zoom_text = format_zoom(final_zoom)
    thumbs = generate_thumbnail_samples(
        frames_dir=frames_dir,
        total_frames=zoom_path.total_frames,
        num_samples=5,
        output_dir=Path(f"output/{name}_thumbs"),
        title_text=job["title"],
        zoom_text=zoom_text,
    )
    print(f"        Done: {len(thumbs)} thumbnails in output/{name}_thumbs/")

    # ── Step 5: YouTube Short ──
    if check_ffmpeg():
        print(f"  [5/5] Generating YouTube Short (30s)...")
        short_start = int(zoom_path.total_frames * 0.4)
        short_end = min(short_start + 30 * zoom_path.fps, zoom_path.total_frames)
        short_frames_dir = Path(f"output/{name}_short_frames")

        short_paths = generate_short_frames(
            frames_dir=frames_dir,
            output_dir=short_frames_dir,
            start_frame=short_start,
            end_frame=short_end,
        )

        if short_paths:
            short_video = encode_video(
                frames_dir=short_frames_dir,
                output_path=Path(f"output/{name}_short.mp4"),
                fps=zoom_path.fps,
                preset="youtube",
                overwrite=True,
            )
            size_mb = short_video.stat().st_size / 1_048_576
            print(f"        Done: {short_video} ({size_mb:.1f} MB, {len(short_paths)} frames)")
        else:
            print(f"        Skipped (no frames found)")
    else:
        print(f"  [5/5] Skipping Short (FFmpeg not found)")

    total_elapsed = time.perf_counter() - render_start
    print(f"\n  Job complete: {format_time(total_elapsed)} total")
    return total_elapsed


def main():
    print_header("FractalForge Overnight Batch Render")
    print(f"  Jobs:     {len(JOBS)}")
    print(f"  FFmpeg:   {'available' if check_ffmpeg() else 'NOT FOUND'}")
    print(f"  Started:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    batch_start = time.perf_counter()
    job_times = []

    for i, job in enumerate(JOBS, 1):
        try:
            elapsed = run_job(job, i, len(JOBS))
            job_times.append((Path(job["preset"]).stem, elapsed))
        except Exception as e:
            print(f"\n  ERROR in job {i}: {e}")
            import traceback
            traceback.print_exc()
            job_times.append((Path(job["preset"]).stem, -1))

    # ── Summary ──
    total_elapsed = time.perf_counter() - batch_start
    print_header("Batch Complete")
    print(f"  Total time: {format_time(total_elapsed)}")
    print(f"  Finished:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  Job Summary:")
    print(f"  {'Name':<25s} {'Time':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    for name, t in job_times:
        if t >= 0:
            print(f"  {name:<25s} {format_time(t):>10s}")
        else:
            print(f"  {name:<25s} {'FAILED':>10s}")

    # List all output files
    output_dir = Path("output")
    videos = sorted(output_dir.glob("*.mp4"))
    print(f"\n  Videos generated: {len(videos)}")
    for v in videos:
        size_mb = v.stat().st_size / 1_048_576
        print(f"    {v.name:<40s} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
