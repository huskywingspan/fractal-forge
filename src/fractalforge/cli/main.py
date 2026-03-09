"""FractalForge CLI -- main entry point.

Usage:
    fractalforge render [OPTIONS]    Render a single frame
    fractalforge info                Show GPU and system info
    fractalforge palettes            List available palettes
    fractalforge resolutions         List resolution presets
    fractalforge zoom [OPTIONS]      Render a zoom video (Phase 2)
    fractalforge encode [OPTIONS]    Encode frames to video (Phase 2)
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
@click.option("--ss", default=1, type=click.IntRange(1, 4), help="Supersampling (1=off, 2=4x AA, 3=9x).")
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
    deep_zoom = config.zoom >= 1e13
    engine = "perturbation theory" if deep_zoom else "standard float64"

    console.print(f"[bold cyan]FractalForge[/] v{__version__}")
    console.print(f"  Center:    ({center_re_str}, {center_im_str})")
    console.print(f"  Zoom:      {config.zoom:.2e}")
    console.print(f"  Size:      {config.width}x{config.height}")
    if preset:
        console.print(f"  Preset:    {preset}")
    console.print(f"  Max iter:  {config.max_iter}")
    console.print(f"  Palette:   {config.palette}")
    if ss > 1:
        console.print(f"  SSAA:      {ss}x ({ss*ss} samples/pixel)")
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
    type=click.Choice(["preview", "quality", "lossless", "prores"], case_sensitive=False),
    help="Video encoding preset.",
)
@click.option("--cpu", is_flag=True, default=False, help="Force CPU rendering.")
@click.option("--ss", default=1, type=click.IntRange(1, 4), help="Supersampling (1=off, 2=4x AA, 3=9x).")
@click.option("--frames-only", is_flag=True, default=False, help="Render frames only, skip encoding.")
@click.option("--resume", is_flag=True, default=False, help="Resume an interrupted render (skip existing frames).")
def zoom(
    path: str,
    frames_dir: str | None,
    output: str | None,
    encode_preset: str,
    cpu: bool,
    ss: int,
    frames_only: bool,
    resume: bool,
):
    """Render a zoom video from a zoom path JSON file.

    PATH is the zoom path JSON file (see fractalforge zoom-template for format).
    """
    from fractalforge.artist.zoompath import ZoomPath
    from fractalforge.engine.mandelbrot import CUDA_AVAILABLE
    from fractalforge.render.sequence import render_sequence
    from fractalforge.render.video import encode_video, check_ffmpeg, get_video_info

    zoom_path = ZoomPath.load(Path(path))
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
    type=click.Choice(["preview", "quality", "lossless", "prores"], case_sensitive=False),
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


if __name__ == "__main__":
    cli()
