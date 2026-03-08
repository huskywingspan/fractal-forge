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
@click.option("--center-re", "-x", default=-0.75, type=float, help="Real part of center.")
@click.option("--center-im", "-y", default=0.0, type=float, help="Imaginary part of center.")
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
@click.option(
    "--output", "-o", default="output/frame.png", type=click.Path(), help="Output file path."
)
def render(
    center_re: float,
    center_im: float,
    zoom: float,
    width: int | None,
    height: int | None,
    max_iter: int,
    palette: str,
    preset: str | None,
    cpu: bool,
    output: str,
):
    """Render a single Mandelbrot frame to PNG."""
    from fractalforge.render.frame_renderer import render_and_save
    from fractalforge.engine.mandelbrot import CUDA_AVAILABLE

    # Resolve resolution: preset -> explicit w/h -> default
    config = RenderConfig(
        center_re=center_re,
        center_im=center_im,
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

    console.print(f"[bold cyan]FractalForge[/] v{__version__}")
    console.print(f"  Center:    ({config.center_re}, {config.center_im})")
    console.print(f"  Zoom:      {config.zoom:.2e}")
    console.print(f"  Size:      {config.width}x{config.height}")
    if preset:
        console.print(f"  Preset:    {preset}")
    console.print(f"  Max iter:  {config.max_iter}")
    console.print(f"  Palette:   {config.palette}")
    console.print(f"  Backend:   {backend}")
    console.print(f"  Output:    {output}")
    console.print()

    start = time.perf_counter()
    output_path = render_and_save(
        output_path=Path(output),
        center_re=config.center_re,
        center_im=config.center_im,
        zoom=config.zoom,
        width=config.width,
        height=config.height,
        max_iter=config.max_iter,
        palette_name=config.palette,
        use_gpu=use_gpu,
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


if __name__ == "__main__":
    cli()
