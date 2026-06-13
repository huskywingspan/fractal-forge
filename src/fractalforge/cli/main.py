"""FractalForge CLI -- main entry point.

Usage:
    fractalforge render [OPTIONS]    Render a single frame
    fractalforge info                Show GPU and system info
    fractalforge palettes            List available palettes
    fractalforge resolutions         List resolution presets
    fractalforge zoom [OPTIONS]      Render a zoom video (Phase 2)
    fractalforge encode [OPTIONS]    Encode frames to video (Phase 2)
    fractalforge title TITLE         Generate a title card overlay (Phase 4)
    fractalforge thumbnail PATH      Generate thumbnail candidates (Phase 4)
    fractalforge short PATH          Generate a YouTube Short (Phase 4)
    fractalforge viewer              Launch interactive fractal explorer
    fractalforge discover [OPTIONS]  Find deep zoom boundary coordinates
    fractalforge scan-region [OPTS]  Scan a region for interesting nuclei
"""

import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from fractalforge import __version__
from fractalforge.config import RESOLUTION_PRESETS, RenderConfig

console = Console(force_terminal=True)


@click.group()
@click.version_option(version=__version__, prog_name="fractalforge")
def cli():
    """FractalForge -- GPU-accelerated fractal zoom renderer."""
    pass


@cli.command()
@click.option("--center-re", "-x", default="-0.75", type=str, help="Real part of center (string for deep zoom precision).")
@click.option("--center-im", "-y", default="0.0", type=str, help="Imaginary part of center (string for deep zoom precision).")
@click.option("--zoom", "-z", default=1.0, type=float, help="Zoom level.")
@click.option("--width", "-w", default=None, type=int, help="Frame width (overrides preset).")
@click.option("--height", "-h", default=None, type=int, help="Frame height (overrides preset).")
@click.option("--max-iter", "-i", default=1000, type=int, help="Maximum iterations.")
@click.option("--palette", "-p", default="ocean", type=str, help="Color palette name.")
@click.option(
    "--preset",
    "-r",
    default=None,
    type=click.Choice(sorted(RESOLUTION_PRESETS.keys()), case_sensitive=False),
    help="Resolution preset (e.g. 1080p, 4k, superwide).",
)
@click.option("--cpu", is_flag=True, default=False, help="Force CPU rendering (no GPU).")
@click.option("--ss", default=1, type=click.IntRange(1, 8), help="Supersampling (1=off, 2=4x, 4=16x, 8=64x AA).")
@click.option("--histogram", is_flag=True, default=False, help="Apply histogram equalization for even color distribution.")
@click.option("--vignette", default=0.0, type=click.FloatRange(0.0, 1.0), help="Vignette strength (0=off, 0.5=moderate, 1=strong).")
@click.option("--contrast", default=1.0, type=float, help="Contrast multiplier (1.0=unchanged).")
@click.option("--saturation", default=1.0, type=float, help="Saturation multiplier (1.0=unchanged).")
@click.option("--brightness", default=1.0, type=float, help="Brightness multiplier (1.0=unchanged).")
@click.option(
    "--fractal", "-f", default="mandelbrot",
    type=click.Choice(["mandelbrot", "julia", "burning_ship"], case_sensitive=False),
    help="Fractal type.",
)
@click.option("--julia-re", default=None, type=float, help="Julia c parameter (real part).")
@click.option("--julia-im", default=None, type=float, help="Julia c parameter (imaginary part).")
@click.option("--bloom", default=0.0, type=click.FloatRange(0.0, 2.0), help="HDR bloom intensity (0=off, 0.3=subtle, 1.0=heavy glow).")
@click.option("--halation", default=0.0, type=click.FloatRange(0.0, 1.0), help="Film halation intensity (warm light bleed, 0=off, 0.15=subtle).")
@click.option(
    "--tone-map", default="none",
    type=click.Choice(["none", "aces", "reinhard"], case_sensitive=False),
    help="HDR tone mapping curve (none, aces=filmic, reinhard=soft).",
)
@click.option("--exposure", default=1.0, type=float, help="Exposure multiplier for tone mapping (1.0=unchanged).")
@click.option(
    "--output", "-o", default="output/frame.png", type=click.Path(), help="Output file path."
)
def render(
    center_re: str,
    center_im: str,
    zoom: float,
    width: int | None,
    height: int | None,
    max_iter: int,
    palette: str,
    preset: str | None,
    cpu: bool,
    ss: int,
    histogram: bool,
    vignette: float,
    contrast: float,
    saturation: float,
    brightness: float,
    fractal: str,
    julia_re: float | None,
    julia_im: float | None,
    bloom: float,
    halation: float,
    tone_map: str,
    exposure: float,
    output: str,
):
    """Render a single Mandelbrot frame to PNG.

    For deep zooms (>= 1e13), pass coordinates as full-precision strings:
      fractalforge render -x "-0.7436438870371587" -y "0.1318259043091895" -z 1e15
    The engine automatically selects perturbation theory when needed.
    """
    from fractalforge.render.frame_renderer import render_and_save
    from fractalforge.engine.mandelbrot import CUDA_AVAILABLE

    # Preserve string coordinates for deep zoom; also parse as float for config display
    center_re_str = center_re
    center_im_str = center_im
    center_re_float = float(center_re)
    center_im_float = float(center_im)

    # Resolve resolution: preset -> explicit w/h -> default
    config = RenderConfig(
        center_re=center_re_float,
        center_im=center_im_float,
        zoom=zoom,
        max_iter=max_iter,
        palette=palette,
    )

    if preset:
        config = config.apply_preset(preset)

    if width is not None:
        config = config.model_copy(update={"width": width})
    if height is not None:
        config = config.model_copy(update={"height": height})

    use_gpu = not cpu and CUDA_AVAILABLE
    backend = "GPU (CUDA)" if use_gpu else "CPU"
    deep_zoom = config.zoom >= 1e13 and fractal == "mandelbrot"
    engine = "perturbation theory" if deep_zoom else "standard float64"

    console.print(f"[bold cyan]FractalForge[/] v{__version__}")
    if fractal != "mandelbrot":
        console.print(f"  Fractal:   {fractal}")
    if fractal == "julia":
        jre = julia_re if julia_re is not None else -0.7269
        jim = julia_im if julia_im is not None else 0.1889
        console.print(f"  Julia c:   ({jre}, {jim})")
    console.print(f"  Center:    ({center_re_str}, {center_im_str})")
    console.print(f"  Zoom:      {config.zoom:.2e}")
    console.print(f"  Size:      {config.width}x{config.height}")
    if preset:
        console.print(f"  Preset:    {preset}")
    console.print(f"  Max iter:  {config.max_iter}")
    console.print(f"  Palette:   {config.palette}")
    if ss > 1:
        console.print(f"  SSAA:      {ss}x ({ss*ss} samples/pixel)")
    if histogram:
        console.print(f"  Histogram: enabled")
    if vignette > 0:
        console.print(f"  Vignette:  {vignette:.1f}")
    if contrast != 1.0 or saturation != 1.0 or brightness != 1.0:
        console.print(f"  Grade:     contrast={contrast:.2f} sat={saturation:.2f} bright={brightness:.2f}")
    if bloom > 0:
        console.print(f"  Bloom:     {bloom:.2f}")
    if halation > 0:
        console.print(f"  Halation:  {halation:.2f}")
    if tone_map != "none":
        console.print(f"  Tone map:  {tone_map} (exposure={exposure:.1f})")
    console.print(f"  Backend:   {backend}")
    console.print(f"  Engine:    {engine}")
    console.print(f"  Output:    {output}")
    console.print()

    start = time.perf_counter()
    # Pass string coordinates for deep zoom precision preservation
    output_path = render_and_save(
        output_path=Path(output),
        center_re=center_re_str if deep_zoom else config.center_re,
        center_im=center_im_str if deep_zoom else config.center_im,
        zoom=config.zoom,
        width=config.width,
        height=config.height,
        max_iter=config.max_iter,
        palette_name=config.palette,
        use_gpu=use_gpu,
        supersampling=ss,
        histogram=histogram,
        vignette=vignette,
        contrast=contrast,
        saturation=saturation,
        brightness=brightness,
        bloom=bloom,
        halation=halation,
        tone_map=tone_map,
        exposure=exposure,
        fractal_type=fractal,
        julia_re=julia_re,
        julia_im=julia_im,
    )
    elapsed = time.perf_counter() - start

    file_size = output_path.stat().st_size
    size_str = f"{file_size / 1024:.0f} KB" if file_size < 1_048_576 else f"{file_size / 1_048_576:.1f} MB"

    console.print(f"[green]Done:[/] Rendered in {elapsed:.2f}s -> {output_path} ({size_str})")


@cli.command()
def info():
    """Show GPU and system information."""
    console.print(f"[bold cyan]FractalForge[/] v{__version__}")
    console.print()

    try:
        from numba import cuda

        if cuda.is_available():
            gpu = cuda.get_current_device()
            console.print("[green]CUDA available[/]")
            console.print(f"  Device:        {gpu.name.decode()}")
            console.print(f"  Compute:       {gpu.compute_capability}")

            free, total = cuda.current_context().get_memory_info()
            console.print(f"  VRAM:          {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
            console.print(f"  Max threads:   {gpu.MAX_THREADS_PER_BLOCK}")
        else:
            console.print("[yellow]CUDA not available[/] -- will use CPU fallback")
    except ImportError:
        console.print("[yellow]numba CUDA not available[/] -- will use CPU fallback")

    import sys

    console.print(f"\n  Python:        {sys.version.split()[0]}")
    console.print(f"  Platform:      {sys.platform}")


@cli.command()
def palettes():
    """List available color palettes."""
    from fractalforge.artist.palette import BUILTIN_PALETTES

    table = Table(title="Available Palettes")
    table.add_column("Name", style="cyan")
    table.add_column("Colors", style="white")

    for name, pal in sorted(BUILTIN_PALETTES.items()):
        first = tuple(int(c) for c in pal[0])
        last = tuple(int(c) for c in pal[-1])
        table.add_row(name, f"{len(pal)} steps  {first} -> {last}")

    console.print(table)


@cli.command()
def resolutions():
    """List available resolution presets."""
    table = Table(title="Resolution Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Resolution", style="white")
    table.add_column("Aspect", style="yellow")
    table.add_column("Label", style="dim")

    for name, preset in sorted(RESOLUTION_PRESETS.items()):
        table.add_row(
            name,
            f"{preset.width}x{preset.height}",
            preset.aspect_ratio,
            preset.label,
        )

    console.print(table)
    console.print("\nUsage: [dim]fractalforge render --preset superwide -o output/uw.png[/]")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--frames-dir", "-f", default=None, type=click.Path(),
    help="Output directory for frames (default: output/<name>_frames).",
)
@click.option(
    "--output", "-o", default=None, type=click.Path(),
    help="Output video file (default: output/<name>.mp4).",
)
@click.option(
    "--encode-preset", "-e", default="quality",
    type=click.Choice(["preview", "quality", "lossless", "prores", "youtube"], case_sensitive=False),
    help="Video encoding preset.",
)
@click.option("--cpu", is_flag=True, default=False, help="Force CPU rendering.")
@click.option("--ss", default=1, type=click.IntRange(1, 8), help="Supersampling (1=off, 2=4x, 4=16x, 8=64x AA).")
@click.option("--histogram", is_flag=True, default=False, help="Apply histogram equalization for even color distribution.")
@click.option("--vignette", default=0.0, type=click.FloatRange(0.0, 1.0), help="Vignette strength (0=off, 0.5=moderate, 1=strong).")
@click.option("--contrast", default=1.0, type=float, help="Contrast multiplier (1.0=unchanged).")
@click.option("--saturation", default=1.0, type=float, help="Saturation multiplier (1.0=unchanged).")
@click.option("--brightness", default=1.0, type=float, help="Brightness multiplier (1.0=unchanged).")
@click.option("--frames-only", is_flag=True, default=False, help="Render frames only, skip encoding.")
@click.option("--resume", is_flag=True, default=False, help="Resume an interrupted render (skip existing frames).")
@click.option(
    "--interpolation", "-I", default=None,
    type=click.Choice(["legacy", "cinematic"], case_sensitive=False),
    help="Override interpolation mode (default: from zoom path JSON).",
)
def zoom(
    path: str,
    frames_dir: str | None,
    output: str | None,
    encode_preset: str,
    cpu: bool,
    ss: int,
    histogram: bool,
    vignette: float,
    contrast: float,
    saturation: float,
    brightness: float,
    frames_only: bool,
    resume: bool,
    interpolation: str | None,
):
    """Render a zoom video from a zoom path JSON file.

    PATH is the zoom path JSON file (see fractalforge zoom-template for format).
    """
    from fractalforge.artist.zoompath import ZoomPath
    from fractalforge.engine.mandelbrot import CUDA_AVAILABLE
    from fractalforge.render.sequence import render_sequence
    from fractalforge.render.video import encode_video, check_ffmpeg, get_video_info

    zoom_path = ZoomPath.load(Path(path))
    if interpolation is not None:
        zoom_path.interpolation = interpolation
    use_gpu = not cpu and CUDA_AVAILABLE
    backend = "GPU (CUDA)" if use_gpu else "CPU"

    # Default output paths
    if frames_dir is None:
        frames_dir = f"output/{zoom_path.name}_frames"
    if output is None:
        output = f"output/{zoom_path.name}.mp4"

    frames_path = Path(frames_dir)
    video_path = Path(output)

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Zoom Render")
    console.print(f"  Path:      {path}")
    console.print(f"  Name:      {zoom_path.name}")
    console.print(f"  Size:      {zoom_path.width}x{zoom_path.height}")
    console.print(f"  Frames:    {zoom_path.total_frames} ({zoom_path.duration_seconds:.1f}s at {zoom_path.fps}fps)")
    console.print(f"  Keyframes: {len(zoom_path.keyframes)}")

    if zoom_path.keyframes:
        kf_first = zoom_path.keyframes[0]
        kf_last = zoom_path.keyframes[-1]
        console.print(f"  Zoom:      {kf_first.zoom:.2e} -> {kf_last.zoom:.2e}")
    if ss > 1:
        console.print(f"  SSAA:      {ss}x ({ss*ss} samples/pixel)")
    if histogram:
        console.print(f"  Histogram: enabled")
    if vignette > 0:
        console.print(f"  Vignette:  {vignette:.1f}")
    if contrast != 1.0 or saturation != 1.0 or brightness != 1.0:
        console.print(f"  Grade:     contrast={contrast:.2f} sat={saturation:.2f} bright={brightness:.2f}")
    if zoom_path.interpolation != "legacy":
        console.print(f"  Interp:    {zoom_path.interpolation}")
    console.print(f"  Backend:   {backend}")
    console.print(f"  Frames ->  {frames_path}")
    if not frames_only:
        console.print(f"  Video ->   {video_path} ({encode_preset})")
    console.print()

    # Render frames
    start = time.perf_counter()
    last_print_time = [start]

    def on_progress(frame_idx, total, elapsed, fps, skipped=False):
        now = time.perf_counter()
        # Print every 2 seconds or on last frame
        if now - last_print_time[0] >= 2.0 or frame_idx == total - 1:
            pct = (frame_idx + 1) / total * 100
            status = "skipped" if skipped else f"{fps:.1f} fps"
            remaining = (total - frame_idx - 1) / fps if fps > 0 else 0
            console.print(
                f"  [{pct:5.1f}%] Frame {frame_idx + 1}/{total}  "
                f"({status}, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                highlight=False,
            )
            last_print_time[0] = now

    console.print("[bold]Rendering frames...[/]")
    render_sequence(
        zoom_path=zoom_path,
        output_dir=frames_path,
        use_gpu=use_gpu,
        skip_existing=resume,
        supersampling=ss,
        on_progress=on_progress,
        histogram=histogram,
        vignette=vignette,
        contrast=contrast,
        saturation=saturation,
        brightness=brightness,
    )

    render_elapsed = time.perf_counter() - start
    avg_fps = zoom_path.total_frames / render_elapsed if render_elapsed > 0 else 0
    console.print(f"\n[green]Done:[/] {zoom_path.total_frames} frames in {render_elapsed:.1f}s ({avg_fps:.1f} avg fps)")

    # Encode video
    if not frames_only:
        if not check_ffmpeg():
            console.print("[red]Error:[/] FFmpeg not found -- cannot encode video.")
            console.print("  Install FFmpeg or use --frames-only to skip encoding.")
            raise SystemExit(1)

        console.print(f"\n[bold]Encoding video ({encode_preset})...[/]")
        encode_start = time.perf_counter()
        video_path = encode_video(
            frames_dir=frames_path,
            output_path=video_path,
            fps=zoom_path.fps,
            preset=encode_preset,
            overwrite=True,
        )
        encode_elapsed = time.perf_counter() - encode_start

        info = get_video_info(video_path)
        file_size = video_path.stat().st_size
        size_str = f"{file_size / 1_048_576:.1f} MB" if file_size >= 1_048_576 else f"{file_size / 1024:.0f} KB"

        console.print(f"[green]Done:[/] Encoded in {encode_elapsed:.1f}s -> {video_path} ({size_str})")

        total_elapsed = time.perf_counter() - start
        console.print(f"\n[bold green]Total time:[/] {total_elapsed:.1f}s")


@cli.command()
@click.argument("frames_dir", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path(), help="Output video file path.")
@click.option("--fps", default=60, type=int, help="Frames per second.")
@click.option(
    "--preset", "-e", default="quality",
    type=click.Choice(["preview", "quality", "lossless", "prores", "youtube"], case_sensitive=False),
    help="Encoding preset.",
)
def encode(frames_dir: str, output: str, fps: int, preset: str):
    """Encode a directory of frame PNGs into a video file.

    FRAMES_DIR is the directory containing frame_000000.png, frame_000001.png, etc.
    """
    from fractalforge.render.video import encode_video, check_ffmpeg, get_video_info

    if not check_ffmpeg():
        console.print("[red]Error:[/] FFmpeg not found.")
        raise SystemExit(1)

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Encode")
    console.print(f"  Frames:  {frames_dir}")
    console.print(f"  Output:  {output}")
    console.print(f"  FPS:     {fps}")
    console.print(f"  Preset:  {preset}")
    console.print()

    start = time.perf_counter()
    video_path = encode_video(
        frames_dir=Path(frames_dir),
        output_path=Path(output),
        fps=fps,
        preset=preset,
        overwrite=True,
    )
    elapsed = time.perf_counter() - start

    file_size = video_path.stat().st_size
    size_str = f"{file_size / 1_048_576:.1f} MB" if file_size >= 1_048_576 else f"{file_size / 1024:.0f} KB"

    console.print(f"[green]Done:[/] Encoded in {elapsed:.1f}s -> {video_path} ({size_str})")


@cli.command(name="zoom-template")
@click.option("--output", "-o", default="zoom_path.json", type=click.Path(), help="Output JSON path.")
def zoom_template(output: str):
    """Generate a sample zoom path JSON template."""
    from fractalforge.artist.zoompath import ZoomPath, Keyframe

    sample = ZoomPath(
        name="sample_zoom",
        fps=60,
        width=1920,
        height=1080,
        keyframes=[
            Keyframe(frame=0, center_re=-0.75, center_im=0.0, zoom=1.0, max_iter=500, palette="ocean"),
            Keyframe(frame=300, center_re=-0.7435669, center_im=0.1314023, zoom=1000.0, max_iter=2000, palette="ocean"),
            Keyframe(frame=600, center_re=-0.7435669, center_im=0.1314023, zoom=1e6, max_iter=5000, palette="nebula"),
        ],
    )

    output_path = Path(output)
    sample.save(output_path)
    console.print(f"[green]Done:[/] Template saved to {output_path}")
    console.print(f"  {len(sample.keyframes)} keyframes, {sample.total_frames} frames, {sample.duration_seconds:.1f}s")
    console.print("\nEdit the JSON, then run:")
    console.print(f"  [dim]fractalforge zoom {output_path}[/]")


@cli.command()
@click.argument("title")
@click.option("--subtitle", "-s", default="", help="Subtitle text (e.g. zoom depth).")
@click.option("--width", "-w", default=1920, type=int, help="Output width.")
@click.option("--height", default=1080, type=int, help="Output height.")
@click.option("--output", "-o", default="output/title_card.png", type=click.Path(), help="Output PNG path.")
def title(title: str, subtitle: str, width: int, height: int, output: str):
    """Generate a title card overlay (RGBA PNG) for video compositing.

    TITLE is the main video title text.  The overlay has a semi-transparent
    gradient, channel name, title, and optional subtitle -- ready to drop
    onto a DaVinci Resolve timeline above the fractal footage.
    """
    from fractalforge.publish.titlecard import render_title_card

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Title Card")
    console.print(f"  Title:     {title}")
    if subtitle:
        console.print(f"  Subtitle:  {subtitle}")
    console.print(f"  Size:      {width}x{height}")
    console.print(f"  Output:    {output}")
    console.print()

    output_path = Path(output)
    img = render_title_card(
        title=title,
        subtitle=subtitle,
        width=width,
        height=height,
        output_path=output_path,
    )

    file_size = output_path.stat().st_size
    size_str = f"{file_size / 1024:.0f} KB" if file_size < 1_048_576 else f"{file_size / 1_048_576:.1f} MB"
    console.print(f"[green]Done:[/] {img.size[0]}x{img.size[1]} RGBA -> {output_path} ({size_str})")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--samples", "-n", default=5, type=int, help="Number of thumbnail candidates.")
@click.option("--output-dir", "-o", default=None, type=click.Path(), help="Output directory for thumbnails.")
@click.option("--title", "-t", default=None, type=str, help="Title text on thumbnail.")
def thumbnail(path: str, samples: int, output_dir: str | None, title: str | None):
    """Generate YouTube thumbnail candidates from a zoom path render.

    PATH is the zoom path JSON file.  The command reads the zoom path to
    determine the frames directory, frame count, and final zoom level, then
    samples frames biased toward the deep end of the sequence.
    """
    from fractalforge.artist.zoompath import ZoomPath
    from fractalforge.publish.thumbnail import generate_thumbnail_samples, format_zoom

    zoom_path = ZoomPath.load(Path(path))

    # Determine frames directory (same convention as zoom command)
    frames_dir = Path(f"output/{zoom_path.name}_frames")

    # Get final zoom level for the zoom text
    if zoom_path.keyframes:
        final_zoom = zoom_path.keyframes[-1].zoom
        zoom_text = format_zoom(final_zoom)
    else:
        final_zoom = 1.0
        zoom_text = None

    out_dir = Path(output_dir) if output_dir else None

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Thumbnail Sampler")
    console.print(f"  Path:      {path}")
    console.print(f"  Name:      {zoom_path.name}")
    console.print(f"  Frames:    {zoom_path.total_frames} in {frames_dir}")
    console.print(f"  Zoom:      {final_zoom:.2e} ({zoom_text})")
    console.print(f"  Samples:   {samples}")
    if title:
        console.print(f"  Title:     {title}")
    console.print()

    if not frames_dir.exists():
        console.print(f"[red]Error:[/] Frames directory not found: {frames_dir}")
        console.print("  Render the zoom path first with: fractalforge zoom " + path)
        raise SystemExit(1)

    paths = generate_thumbnail_samples(
        frames_dir=frames_dir,
        total_frames=zoom_path.total_frames,
        num_samples=samples,
        output_dir=out_dir,
        title_text=title,
        zoom_text=zoom_text,
    )

    if not paths:
        console.print("[yellow]Warning:[/] No thumbnails generated (no matching frames found).")
        return

    console.print(f"[green]Done:[/] Generated {len(paths)} thumbnail(s):")
    for p in paths:
        file_size = p.stat().st_size
        size_str = f"{file_size / 1024:.0f} KB"
        console.print(f"  {p} ({size_str})")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--start-frame", "-s", default=None, type=int, help="Start frame index (default: 40% of sequence).")
@click.option("--duration", "-d", default=30, type=click.IntRange(5, 60), help="Duration in seconds (5-60, default 30).")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output video file path.")
@click.option(
    "--encode-preset", "-e", default="youtube",
    type=click.Choice(["preview", "quality", "youtube"], case_sensitive=False),
    help="Encoding preset.",
)
def short(path: str, start_frame: int | None, duration: int, output: str | None, encode_preset: str):
    """Generate a YouTube Short from a zoom path render.

    PATH is the zoom path JSON file. Crops landscape frames to 9:16 portrait,
    selects a segment, and encodes to a Shorts-ready MP4.
    """
    from fractalforge.artist.zoompath import ZoomPath
    from fractalforge.publish.shorts import generate_short_frames
    from fractalforge.render.video import encode_video, check_ffmpeg

    zoom_path = ZoomPath.load(Path(path))
    frames_dir = Path(f"output/{zoom_path.name}_frames")

    if not frames_dir.exists():
        console.print(f"[red]Error:[/] Frames directory not found: {frames_dir}")
        console.print("  Render the zoom path first with: fractalforge zoom " + path)
        raise SystemExit(1)

    # Calculate frame range
    num_short_frames = duration * zoom_path.fps
    if start_frame is None:
        # Default: start at 40% of sequence for interesting content
        start_frame = int(zoom_path.total_frames * 0.4)

    end_frame = min(start_frame + num_short_frames, zoom_path.total_frames)
    actual_duration = (end_frame - start_frame) / zoom_path.fps

    if output is None:
        output = f"output/{zoom_path.name}_short.mp4"

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- YouTube Short")
    console.print(f"  Source:    {path}")
    console.print(f"  Frames:    {start_frame} -> {end_frame} ({end_frame - start_frame} frames)")
    console.print(f"  Duration:  {actual_duration:.1f}s at {zoom_path.fps}fps")
    console.print(f"  Output:    1080x1920 (9:16 portrait)")
    console.print()

    # Crop frames to portrait
    short_frames_dir = Path(f"output/{zoom_path.name}_short_frames")
    console.print("[bold]Cropping frames to 9:16...[/]")
    short_paths = generate_short_frames(
        frames_dir=frames_dir,
        output_dir=short_frames_dir,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    if not short_paths:
        console.print("[red]Error:[/] No frames generated.")
        raise SystemExit(1)

    console.print(f"  Cropped {len(short_paths)} frames")

    # Encode
    if not check_ffmpeg():
        console.print("[red]Error:[/] FFmpeg not found -- cannot encode.")
        raise SystemExit(1)

    console.print(f"[bold]Encoding ({encode_preset})...[/]")
    video_path = encode_video(
        frames_dir=short_frames_dir,
        output_path=Path(output),
        fps=zoom_path.fps,
        preset=encode_preset,
        overwrite=True,
    )

    file_size = video_path.stat().st_size
    size_str = f"{file_size / 1_048_576:.1f} MB" if file_size >= 1_048_576 else f"{file_size / 1024:.0f} KB"
    console.print(f"\n[green]Done:[/] {video_path} ({size_str})")


@cli.command(name="camera-path")
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, type=click.Path(), help="Output PNG path.")
@click.option(
    "--compare", is_flag=True, default=False,
    help="Compare legacy vs cinematic interpolation side by side.",
)
@click.option(
    "--mode", "-m", default=None,
    type=click.Choice(["legacy", "cinematic"], case_sensitive=False),
    help="Override interpolation mode for preview.",
)
def camera_path(path: str, output: str | None, compare: bool, mode: str | None):
    """Visualize the camera path for a zoom path preset.

    Generates a 4-panel plot showing position trajectory, zoom level,
    screen-space velocity, and zoom rate. Useful for tuning keyframes
    and verifying smooth motion.

    Use --compare to see legacy vs cinematic side by side.
    """
    from fractalforge.artist.zoompath import ZoomPath
    from fractalforge.artist.path_preview import render_path_preview

    zoom_path = ZoomPath.load(Path(path))

    if mode is not None:
        zoom_path.interpolation = mode

    if output is None:
        suffix = "_compare" if compare else f"_{zoom_path.interpolation}"
        output = f"output/{zoom_path.name}_camera_path{suffix}.png"

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Camera Path Preview")
    console.print(f"  Path:          {path}")
    console.print(f"  Name:          {zoom_path.name}")
    console.print(f"  Keyframes:     {len(zoom_path.keyframes)}")
    console.print(f"  Total frames:  {zoom_path.total_frames}")
    console.print(f"  Mode:          {'comparison' if compare else zoom_path.interpolation}")
    console.print(f"  Output:        {output}")
    console.print()

    out = render_path_preview(zoom_path, Path(output), compare=compare)
    console.print(f"[green]Done:[/] Camera path preview saved to {out}")


@cli.command()
def viewer():
    """Launch the interactive fractal explorer.

    Opens a Dear PyGui window with real-time fractal rendering, parameter
    controls, click-to-zoom navigation, and coordinate bookmarking.

    Requires: pip install dearpygui
    """
    from fractalforge.viewer import launch_viewer

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Interactive Viewer")
    console.print("  Left-click:   re-center + zoom in")
    console.print("  Scroll:       zoom toward cursor")
    console.print("  Right-drag:   pan viewport")
    console.print()
    launch_viewer()


@cli.command(name="compile")
@click.argument("spec_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, type=click.Path(), help="Output video file.")
def compile_cmd(spec_path: str, output: str | None):
    """Compile multiple zoom clips into a single video with crossfade transitions.

    SPEC_PATH is a compilation spec JSON file defining clips, timing, and transitions.
    """
    from fractalforge.publish.compilation import CompilationSpec, assemble_compilation
    from fractalforge.render.video import encode_video, check_ffmpeg

    spec = CompilationSpec.load(Path(spec_path))

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Compile")
    console.print(f"  Spec:        {spec_path}")
    console.print(f"  Name:        {spec.name}")
    console.print(f"  Clips:       {len(spec.clips)}")
    console.print(f"  Transition:  {spec.transition_frames} frames ({spec.transition_frames / spec.fps:.1f}s crossfade)")
    console.print(f"  Encode:      {spec.encode_preset}")
    console.print()

    # Assemble frames
    assembly_dir = Path(f"output/{spec.name}_assembly")
    console.print("[bold]Assembling frames...[/]")
    assemble_compilation(spec, assembly_dir)

    frame_count = len(list(assembly_dir.glob("frame_*.png")))
    duration = frame_count / spec.fps
    console.print(f"  Assembled {frame_count} frames ({duration:.1f}s)")

    # Encode
    if not check_ffmpeg():
        console.print("[red]Error:[/] FFmpeg not found.")
        raise SystemExit(1)

    video_output = Path(output) if output else Path(f"output/{spec.name}.mp4")
    console.print(f"[bold]Encoding ({spec.encode_preset})...[/]")
    video_path = encode_video(
        frames_dir=assembly_dir,
        output_path=video_output,
        fps=spec.fps,
        preset=spec.encode_preset,
        overwrite=True,
    )

    file_size = video_path.stat().st_size
    size_str = f"{file_size / 1_048_576:.1f} MB" if file_size >= 1_048_576 else f"{file_size / 1024:.0f} KB"
    console.print(f"\n[green]Done:[/] {video_path} ({size_str})")


@cli.command()
@click.option("--center-re", "-x", default="-0.75", type=str,
              help="Approximate real coordinate near target.")
@click.option("--center-im", "-y", default="0.1", type=str,
              help="Approximate imaginary coordinate near target.")
@click.option("--precision", "-p", default=100, type=int,
              help="Decimal digits for output coordinates (default 100).")
@click.option("--max-period", default=10000, type=int,
              help="Maximum period to detect (default 10000).")
@click.option("--angles", "-a", default=None, type=str,
              help="Comma-separated internal angles (default: 0,0.5,1/3,2/3,1/4,3/4,1/5,2/5).")
@click.option("--render-test", is_flag=True, default=False,
              help="Render a quick preview of each discovered coordinate.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Save results to JSON file.")
def discover(center_re: str, center_im: str, precision: int, max_period: int,
             angles: str | None, render_test: bool, output: str | None):
    """Discover high-precision boundary coordinates for deep zoom videos.

    Uses Newton-Raphson to find the exact nucleus of the nearest hyperbolic
    component, then locates boundary points at various internal angles.
    These are ideal targets for deep zoom rendering -- they sit precisely
    on the Mandelbrot set boundary where fractal detail is densest.

    Examples:
      fractalforge discover -x "-0.75" -y "0.1"
      fractalforge discover -x "-1.768778833" -y "-0.001738996" -p 200
      fractalforge discover -x "-0.75" -y "0.1" --render-test
    """
    import json
    from fractalforge.engine.newton import discover_coordinates, detect_period

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Coordinate Discovery")
    console.print(f"  Approximate: ({center_re}, {center_im})")
    console.print(f"  Precision:   {precision} digits")
    console.print(f"  Max period:  {max_period}")
    console.print()

    # Parse angles
    parsed_angles = None
    if angles:
        parsed_angles = []
        for a in angles.split(","):
            a = a.strip()
            if "/" in a:
                num, den = a.split("/")
                parsed_angles.append(int(num) / int(den))
            else:
                parsed_angles.append(float(a))

    results = discover_coordinates(
        c_re_str=center_re,
        c_im_str=center_im,
        precision=precision,
        angles=parsed_angles,
        max_period=max_period,
        verbose=True,
    )

    if not results:
        console.print("[yellow]No boundary points found.[/]")
        return

    # Display results table
    console.print()
    table = Table(title="Discovered Boundary Points")
    table.add_column("Angle", style="cyan")
    table.add_column("Real", style="white", max_width=60)
    table.add_column("Imaginary", style="white", max_width=60)
    table.add_column("Zoom", style="yellow")
    table.add_column("Status", style="green")

    for bp in results:
        from fractalforge.engine.newton import _format_angle
        table.add_row(
            _format_angle(bp.internal_angle),
            bp.c_re[:50] + ("..." if len(bp.c_re) > 50 else ""),
            bp.c_im[:50] + ("..." if len(bp.c_im) > 50 else ""),
            f"{bp.suggested_zoom:.1e}",
            "OK" if bp.converged else "FAILED",
        )
    console.print(table)

    # Save to JSON
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "source": {"center_re": center_re, "center_im": center_im},
            "precision": precision,
            "points": [
                {
                    "c_re": bp.c_re,
                    "c_im": bp.c_im,
                    "period": bp.period,
                    "internal_angle": bp.internal_angle,
                    "suggested_zoom": bp.suggested_zoom,
                    "converged": bp.converged,
                }
                for bp in results
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        console.print(f"\n[green]Saved:[/] {out_path}")

    # Render test previews
    if render_test:
        console.print("\n[bold]Rendering test previews...[/]")
        from fractalforge.engine.perturbation import render_frame_perturbation
        from fractalforge.engine.coloring import smooth_to_image
        from fractalforge.artist.palette import get_palette
        import math

        palette = get_palette("deep_blue")
        test_dir = Path("output/discover_previews")
        test_dir.mkdir(parents=True, exist_ok=True)

        for i, bp in enumerate(results[:4]):  # Render top 4
            if not bp.converged:
                continue
            zoom_level = min(bp.suggested_zoom, 1e30)
            max_iter = max(2000, int(500 + 300 * math.log10(zoom_level)))
            console.print(f"  [{i+1}] angle={_format_angle(bp.internal_angle)} "
                          f"zoom={zoom_level:.1e} iter={max_iter}")

            t0 = time.perf_counter()
            smooth = render_frame_perturbation(
                center_re=bp.c_re,
                center_im=bp.c_im,
                zoom=zoom_level,
                width=640, height=360,
                max_iter=max_iter,
            )
            elapsed = time.perf_counter() - t0

            img = smooth_to_image(smooth, palette, histogram=True, slope_shading=True)
            fname = test_dir / f"discover_{i:02d}_angle{bp.internal_angle:.3f}.png"
            img.save(str(fname))
            console.print(f"      {elapsed:.1f}s -> {fname}")

        console.print(f"\n[green]Done:[/] Previews saved to {test_dir}")

    # Print CLI-ready commands for the best result
    best = next((bp for bp in results if bp.converged), None)
    if best:
        console.print("\n[bold]Quick render command:[/]")
        zoom_cmd = min(best.suggested_zoom, 1e20)
        console.print(f'  fractalforge render -x "{best.c_re}" '
                      f'-y "{best.c_im}" -z {zoom_cmd:.0e} '
                      f'-i 5000 --histogram -o output/discover_test.png')


@cli.command(name="scan-region")
@click.option("--re-min", default=-2.0, type=float, help="Real min (default -2.0).")
@click.option("--re-max", default=0.5, type=float, help="Real max (default 0.5).")
@click.option("--im-min", default=-1.25, type=float, help="Imaginary min (default -1.25).")
@click.option("--im-max", default=1.25, type=float, help="Imaginary max (default 1.25).")
@click.option("--grid", "-g", default=20, type=int, help="Grid resolution per axis (default 20).")
@click.option("--max-period", default=500, type=int, help="Max period to detect (default 500).")
@click.option("--precision", "-p", default=50, type=int, help="Precision digits (default 50).")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save results to JSON.")
def scan_region_cmd(re_min: float, re_max: float, im_min: float, im_max: float,
                    grid: int, max_period: int, precision: int, output: str | None):
    """Scan a region for hyperbolic component nuclei.

    Useful for finding interesting zoom targets in an area.

    Example:
      fractalforge scan-region --re-min -0.8 --re-max -0.7 --im-min 0.1 --im-max 0.2
    """
    import json
    from fractalforge.engine.newton import scan_region

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Region Scanner")
    console.print(f"  Region:    [{re_min}, {re_max}] x [{im_min}, {im_max}]")
    console.print(f"  Grid:      {grid}x{grid} = {grid*grid} sample points")
    console.print(f"  Max period: {max_period}")
    console.print()

    nuclei = scan_region(
        re_min=re_min, re_max=re_max,
        im_min=im_min, im_max=im_max,
        grid_size=grid, max_period=max_period,
        precision=precision, verbose=True,
    )

    if nuclei:
        console.print()
        table = Table(title="Discovered Nuclei")
        table.add_column("Period", style="cyan", justify="right")
        table.add_column("Real", style="white", max_width=40)
        table.add_column("Imaginary", style="white", max_width=40)
        table.add_column("Size", style="yellow")
        table.add_column("Status", style="green")

        for nuc in nuclei:
            table.add_row(
                str(nuc.period),
                nuc.c_re[:35] + ("..." if len(nuc.c_re) > 35 else ""),
                nuc.c_im[:35] + ("..." if len(nuc.c_im) > 35 else ""),
                f"{nuc.size:.3e}",
                "OK" if nuc.converged else "FAIL",
            )
        console.print(table)

    if output and nuclei:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "region": {"re_min": re_min, "re_max": re_max,
                        "im_min": im_min, "im_max": im_max},
            "nuclei": [
                {
                    "c_re": n.c_re, "c_im": n.c_im,
                    "period": n.period, "size": n.size,
                    "converged": n.converged,
                }
                for n in nuclei
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        console.print(f"\n[green]Saved:[/] {out_path}")


@cli.command()
@click.option("--center-re", "-x", default="-0.77568377", type=str,
              help="Approximate real coordinate near a spiral/dendrite.")
@click.option("--center-im", "-y", default="0.13646737", type=str,
              help="Approximate imaginary coordinate near a spiral/dendrite.")
@click.option("--precision", "-p", default=320, type=int,
              help="Decimal digits for the output coordinate (default 320).")
@click.option("--preperiod", default=None, type=int,
              help="Force a specific pre-period (auto-searched if omitted).")
@click.option("--period", default=None, type=int,
              help="Force a specific cycle period (auto-searched if omitted).")
@click.option("--zoom", "-z", default="1e60", type=str,
              help="Zoom for the test preview (default 1e60).")
@click.option("--render-test", is_flag=True, default=False,
              help="Render a quick preview at the found point.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Save the coordinate to a JSON file.")
def misiurewicz(center_re: str, center_im: str, precision: int,
                preperiod: int | None, period: int | None, zoom: str,
                render_test: bool, output: str | None):
    """Find a Misiurewicz (pre-periodic) point for tractable extreme zoom.

    Unlike minibrot nuclei -- whose period, and thus reference-orbit length,
    explodes with depth -- a Misiurewicz point keeps a short reference orbit at
    ANY zoom while its embedded Julia structure repeats at every scale. These
    are the best targets for 1e100+ deep zooms.

    Examples:
      fractalforge misiurewicz -x "-0.77568377" -y "0.13646737"
      fractalforge misiurewicz -x "-0.1" -y "0.95" -p 500 --render-test -z 1e120
    """
    import json
    from fractalforge.engine.newton import find_misiurewicz

    console.print(f"[bold cyan]FractalForge[/] v{__version__} -- Misiurewicz Finder")
    console.print(f"  Seed:      ({center_re}, {center_im})")
    console.print(f"  Precision: {precision} digits")
    console.print()

    mp = find_misiurewicz(
        center_re, center_im, preperiod=preperiod, period=period,
        precision=precision, verbose=True,
    )
    if mp is None:
        console.print("[yellow]No Misiurewicz point found near the seed.[/]")
        console.print("Try a coordinate nearer a visible spiral or dendrite tip.")
        return

    console.print()
    table = Table(title="Misiurewicz Point")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Pre-period", str(mp.preperiod))
    table.add_row("Period", str(mp.period))
    table.add_row("Real", mp.c_re[:60] + ("..." if len(mp.c_re) > 60 else ""))
    table.add_row("Imag", mp.c_im[:60] + ("..." if len(mp.c_im) > 60 else ""))
    console.print(table)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "c_re": mp.c_re, "c_im": mp.c_im,
            "preperiod": mp.preperiod, "period": mp.period,
            "precision": mp.precision,
        }, indent=2))
        console.print(f"\n[green]Saved:[/] {out_path}")

    if render_test:
        from fractalforge.render.frame_renderer import render_and_save
        console.print(f"\n[bold]Rendering preview at zoom {zoom}...[/]")
        t0 = time.perf_counter()
        out = render_and_save(
            output_path=Path("output/misiurewicz_test.png"),
            center_re=mp.c_re, center_im=mp.c_im, zoom=zoom,
            width=960, height=540, max_iter=8000,
            palette_name="nebula", histogram=True,
        )
        console.print(f"  {time.perf_counter() - t0:.1f}s -> {out}")

    console.print("\n[bold]Quick render command:[/]")
    console.print(f'  fractalforge render -x "{mp.c_re}" '
                  f'-y "{mp.c_im}" -z {zoom} -i 8000 --histogram '
                  f'-o output/misiurewicz.png')


if __name__ == "__main__":
    cli()
