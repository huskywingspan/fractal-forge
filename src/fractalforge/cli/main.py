"""FractalForge CLI — main entry point.

Usage:
    fractalforge render [OPTIONS]    Render a single frame
    fractalforge info                Show GPU and system info
    fractalforge zoom [OPTIONS]      Render a zoom video (Phase 2)
    fractalforge encode [OPTIONS]    Encode frames to video (Phase 2)
    fractalforge palettes            List available palettes
"""

import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from fractalforge import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="fractalforge")
def cli():
    """FractalForge — GPU-accelerated fractal zoom renderer."""
    pass


@cli.command()
@click.option("--center-re", "-x", default=-0.75, type=float, help="Real part of center.")
@click.option("--center-im", "-y", default=0.0, type=float, help="Imaginary part of center.")
@click.option("--zoom", "-z", default=1.0, type=float, help="Zoom level.")
@click.option("--width", "-w", default=1920, type=int, help="Frame width in pixels.")
@click.option("--height", "-h", default=1080, type=int, help="Frame height in pixels.")
@click.option("--max-iter", "-i", default=1000, type=int, help="Maximum iterations.")
@click.option("--palette", "-p", default="ocean", type=str, help="Color palette name.")
@click.option(
    "--output", "-o", default="output/frame.png", type=click.Path(), help="Output file path."
)
def render(
    center_re: float,
    center_im: float,
    zoom: float,
    width: int,
    height: int,
    max_iter: int,
    palette: str,
    output: str,
):
    """Render a single Mandelbrot frame to PNG."""
    from fractalforge.render.frame_renderer import render_and_save

    console.print(f"[bold cyan]FractalForge[/] v{__version__}")
    console.print(f"  Center:    ({center_re}, {center_im})")
    console.print(f"  Zoom:      {zoom:.2e}")
    console.print(f"  Size:      {width}×{height}")
    console.print(f"  Max iter:  {max_iter}")
    console.print(f"  Palette:   {palette}")
    console.print(f"  Output:    {output}")
    console.print()

    start = time.perf_counter()
    output_path = render_and_save(
        output_path=Path(output),
        center_re=center_re,
        center_im=center_im,
        zoom=zoom,
        width=width,
        height=height,
        max_iter=max_iter,
        palette_name=palette,
    )
    elapsed = time.perf_counter() - start

    console.print(f"[green]✓[/] Rendered in {elapsed:.2f}s → {output_path}")


@cli.command()
def info():
    """Show GPU and system information."""
    console.print(f"[bold cyan]FractalForge[/] v{__version__}")
    console.print()

    # GPU info via numba CUDA
    try:
        from numba import cuda

        if cuda.is_available():
            gpu = cuda.get_current_device()
            console.print("[green]CUDA available[/]")
            console.print(f"  Device:        {gpu.name.decode()}")
            console.print(f"  Compute:       {gpu.compute_capability}")

            # Memory info
            free, total = cuda.current_context().get_memory_info()
            console.print(f"  VRAM:          {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
            console.print(f"  Max threads:   {gpu.MAX_THREADS_PER_BLOCK}")
        else:
            console.print("[red]CUDA not available[/] — GPU rendering disabled")
    except ImportError:
        console.print("[red]numba not installed[/] — GPU rendering unavailable")

    # Python info
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
        # Show first and last color as a hint
        first = tuple(pal[0])
        last = tuple(pal[-1])
        table.add_row(name, f"{len(pal)} steps  ({first} → {last})")

    console.print(table)


if __name__ == "__main__":
    cli()
