"""Configuration models — render parameters, resolution presets, output settings.

All render parameters are validated via Pydantic models. Presets provide
convenient shorthand for common resolutions and aspect ratios.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --- Resolution presets ---

class ResolutionPreset(BaseModel):
    """A named resolution preset."""

    name: str
    width: int
    height: int
    label: str  # Human-readable label

    @property
    def aspect_ratio(self) -> str:
        from math import gcd
        g = gcd(self.width, self.height)
        return f"{self.width // g}:{self.height // g}"


RESOLUTION_PRESETS: dict[str, ResolutionPreset] = {
    "720p": ResolutionPreset(name="720p", width=1280, height=720, label="720p (16:9)"),
    "1080p": ResolutionPreset(name="1080p", width=1920, height=1080, label="1080p (16:9)"),
    "1440p": ResolutionPreset(name="1440p", width=2560, height=1440, label="1440p (16:9)"),
    "4k": ResolutionPreset(name="4k", width=3840, height=2160, label="4K UHD (16:9)"),
    "uw-1080p": ResolutionPreset(
        name="uw-1080p", width=2560, height=1080, label="Ultrawide 1080p (21:9)"
    ),
    "uw-1440p": ResolutionPreset(
        name="uw-1440p", width=3440, height=1440, label="Ultrawide 1440p (21:9)"
    ),
    "superwide": ResolutionPreset(
        name="superwide", width=5120, height=1440, label="Super Ultrawide (32:9)"
    ),
}


# --- Render configuration ---

class RenderConfig(BaseModel):
    """Configuration for a single frame render."""

    center_re: float = Field(default=-0.75, description="Real part of center coordinate")
    center_im: float = Field(default=0.0, description="Imaginary part of center coordinate")
    zoom: float = Field(default=1.0, gt=0, description="Zoom level (>0)")
    width: int = Field(default=1920, gt=0, le=15360, description="Frame width in pixels")
    height: int = Field(default=1080, gt=0, le=8640, description="Frame height in pixels")
    max_iter: int = Field(default=1000, gt=0, le=1_000_000, description="Maximum iterations")
    palette: str = Field(default="ocean", description="Color palette name")
    interior_color: tuple[int, int, int] = Field(
        default=(0, 0, 0), description="RGB color for interior points"
    )
    supersampling: int = Field(
        default=1, ge=1, le=4, description="Supersampling factor (1=off, 2=4x, 4=16x)"
    )

    @field_validator("palette")
    @classmethod
    def validate_palette(cls, v: str) -> str:
        from fractalforge.artist.palette import BUILTIN_PALETTES
        if v not in BUILTIN_PALETTES:
            available = ", ".join(sorted(BUILTIN_PALETTES.keys()))
            raise ValueError(f"Unknown palette '{v}'. Available: {available}")
        return v

    def apply_preset(self, preset_name: str) -> "RenderConfig":
        """Return a copy with resolution from a named preset."""
        preset = RESOLUTION_PRESETS.get(preset_name)
        if preset is None:
            available = ", ".join(sorted(RESOLUTION_PRESETS.keys()))
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        return self.model_copy(update={"width": preset.width, "height": preset.height})


class OutputConfig(BaseModel):
    """Configuration for render output."""

    output_dir: Path = Field(default=Path("output"), description="Output directory")
    filename: str = Field(default="frame.png", description="Output filename")
    format: str = Field(default="PNG", description="Image format (PNG, TIFF, BMP)")
    overwrite: bool = Field(default=True, description="Overwrite existing files")

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.filename

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"PNG", "TIFF", "BMP"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"Unsupported format '{v}'. Allowed: {', '.join(sorted(allowed))}")
        return v


class ProjectConfig(BaseModel):
    """Top-level project configuration, loadable from JSON."""

    render: RenderConfig = Field(default_factory=RenderConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        return cls.model_validate_json(path.read_text())
