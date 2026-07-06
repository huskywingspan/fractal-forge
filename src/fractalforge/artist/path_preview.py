"""Camera path visualization for comparing interpolation modes.

Generates a multi-panel plot showing position trajectory, zoom level,
camera velocity, and zoom rate over time. Useful for tuning zoom paths
and verifying that cinematic mode eliminates velocity discontinuities.
"""

import math
from pathlib import Path

import numpy as np


def compute_path_data(zoom_path, mode: str | None = None) -> dict:
    """Compute interpolated data for every frame of a zoom path.

    Args:
        zoom_path: A ZoomPath instance.
        mode: Override interpolation mode ("legacy" or "cinematic").
              If None, uses the zoom path's own setting.

    Returns:
        Dict with arrays: frames, center_re, center_im, zoom, velocity_re,
        velocity_im, velocity_mag, zoom_rate.
    """
    original_mode = zoom_path.interpolation
    if mode is not None:
        zoom_path.interpolation = mode

    n = zoom_path.total_frames
    frames = np.arange(n)
    center_re = np.zeros(n)
    center_im = np.zeros(n)
    zoom = np.zeros(n)

    for i in range(n):
        params = zoom_path.interpolate(i)
        center_re[i] = params["center_re"]
        center_im[i] = params["center_im"]
        # zoom may be a string at extreme depth; plot from the finite log10
        # value, clamped to float range so the diagnostic never overflows.
        zoom[i] = 10.0 ** min(params.get("log10_zoom", 0.0), 307.0)

    # Restore original mode
    zoom_path.interpolation = original_mode

    # Compute velocity (derivative of position in screen-space)
    # Screen-space velocity = d(center) / d(frame) * zoom
    # This measures how fast the image appears to move on screen
    vel_re = np.zeros(n)
    vel_im = np.zeros(n)
    for i in range(1, n):
        vel_re[i] = (center_re[i] - center_re[i - 1]) * zoom[i]
        vel_im[i] = (center_im[i] - center_im[i - 1]) * zoom[i]
    vel_re[0] = vel_re[1] if n > 1 else 0.0
    vel_im[0] = vel_im[1] if n > 1 else 0.0

    velocity_mag = np.sqrt(vel_re**2 + vel_im**2)

    # Zoom rate: d(log(zoom))/d(frame)
    log_zoom = np.log(zoom)
    zoom_rate = np.zeros(n)
    for i in range(1, n):
        zoom_rate[i] = log_zoom[i] - log_zoom[i - 1]
    zoom_rate[0] = zoom_rate[1] if n > 1 else 0.0

    return {
        "frames": frames,
        "center_re": center_re,
        "center_im": center_im,
        "zoom": zoom,
        "velocity_re": vel_re,
        "velocity_im": vel_im,
        "velocity_mag": velocity_mag,
        "zoom_rate": zoom_rate,
    }


def render_path_preview(
    zoom_path,
    output_path: Path,
    compare: bool = False,
    width: int = 14,
    height: int = 10,
) -> Path:
    """Render a camera path visualization to PNG.

    Args:
        zoom_path: A ZoomPath instance.
        output_path: Where to save the PNG.
        compare: If True, show both legacy and cinematic side by side.
        width: Figure width in inches.
        height: Figure height in inches.

    Returns:
        The output path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for path preview: pip install matplotlib")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kf_frames = [kf.frame for kf in zoom_path.keyframes]

    if compare:
        _render_comparison(zoom_path, output_path, kf_frames, width, height)
    else:
        _render_single(zoom_path, output_path, kf_frames, width, height)

    return output_path


def _render_single(zoom_path, output_path, kf_frames, width, height):
    """Render a single-mode path preview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = compute_path_data(zoom_path)
    mode = zoom_path.interpolation

    fig, axes = plt.subplots(2, 2, figsize=(width, height))
    fig.suptitle(f"Camera Path: {zoom_path.name} ({mode})", fontsize=14, fontweight="bold")

    _plot_position(axes[0, 0], data, kf_frames, zoom_path)
    _plot_zoom(axes[0, 1], data, kf_frames)
    _plot_velocity(axes[1, 0], data, kf_frames)
    _plot_zoom_rate(axes[1, 1], data, kf_frames)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_comparison(zoom_path, output_path, kf_frames, width, height):
    """Render legacy vs cinematic comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    legacy = compute_path_data(zoom_path, mode="legacy")
    cinematic = compute_path_data(zoom_path, mode="cinematic")

    fig, axes = plt.subplots(2, 2, figsize=(width, height))
    fig.suptitle(f"Camera Path Comparison: {zoom_path.name}", fontsize=14, fontweight="bold")

    # Position trajectory
    ax = axes[0, 0]
    ax.plot(legacy["center_re"], legacy["center_im"], "r-", alpha=0.6, label="legacy", linewidth=1)
    ax.plot(cinematic["center_re"], cinematic["center_im"], "b-", alpha=0.6, label="cinematic", linewidth=1)
    for kf in zoom_path.keyframes:
        ax.plot(kf.center_re, kf.center_im, "ko", markersize=6)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Position Trajectory")
    ax.legend()
    ax.set_aspect("equal")

    # Zoom
    ax = axes[0, 1]
    ax.semilogy(legacy["frames"], legacy["zoom"], "r-", alpha=0.6, label="legacy")
    ax.semilogy(cinematic["frames"], cinematic["zoom"], "b-", alpha=0.6, label="cinematic")
    for kf_frame in kf_frames:
        ax.axvline(kf_frame, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Zoom (log)")
    ax.set_title("Zoom Level")
    ax.legend()

    # Screen-space velocity
    ax = axes[1, 0]
    ax.plot(legacy["frames"], legacy["velocity_mag"], "r-", alpha=0.6, label="legacy")
    ax.plot(cinematic["frames"], cinematic["velocity_mag"], "b-", alpha=0.6, label="cinematic")
    for kf_frame in kf_frames:
        ax.axvline(kf_frame, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Screen-space velocity")
    ax.set_title("Camera Velocity (lower & smoother = better)")
    ax.legend()

    # Zoom rate
    ax = axes[1, 1]
    ax.plot(legacy["frames"], legacy["zoom_rate"], "r-", alpha=0.6, label="legacy")
    ax.plot(cinematic["frames"], cinematic["zoom_rate"], "b-", alpha=0.6, label="cinematic")
    for kf_frame in kf_frames:
        ax.axvline(kf_frame, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("d(log zoom)/d(frame)")
    ax.set_title("Zoom Rate (smooth = no judder)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_position(ax, data, kf_frames, zoom_path):
    ax.plot(data["center_re"], data["center_im"], "b-", linewidth=0.5, alpha=0.7)
    ax.plot(data["center_re"][0], data["center_im"][0], "go", markersize=8, label="start")
    ax.plot(data["center_re"][-1], data["center_im"][-1], "rs", markersize=8, label="end")
    for kf in zoom_path.keyframes:
        ax.plot(kf.center_re, kf.center_im, "ko", markersize=5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Position Trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")


def _plot_zoom(ax, data, kf_frames):
    ax.semilogy(data["frames"], data["zoom"], "b-")
    for kf_frame in kf_frames:
        ax.axvline(kf_frame, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Zoom (log)")
    ax.set_title("Zoom Level")


def _plot_velocity(ax, data, kf_frames):
    ax.plot(data["frames"], data["velocity_mag"], "b-", linewidth=0.5)
    for kf_frame in kf_frames:
        ax.axvline(kf_frame, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Screen-space velocity")
    ax.set_title("Camera Velocity (spikes = jump cuts)")


def _plot_zoom_rate(ax, data, kf_frames):
    ax.plot(data["frames"], data["zoom_rate"], "b-", linewidth=0.5)
    for kf_frame in kf_frames:
        ax.axvline(kf_frame, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("d(log zoom)/d(frame)")
    ax.set_title("Zoom Rate")
