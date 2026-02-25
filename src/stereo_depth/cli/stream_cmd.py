from __future__ import annotations
from pathlib import Path
import typer

from stereo_depth.app.stream import run_stream


def stream(
    calib: Path = typer.Option(..., help="Calibration YAML path"),
    device: int = typer.Option(0, help="Camera device index"),
    preset: str = typer.Option(
        "indoor",
        help="SGBM preset (indoor|outdoor|high_quality) or Retinify mode (fast|balanced|accurate)",
    ),
    matcher: str = typer.Option(
        "sgbm",
        help="Disparity matcher backend: sgbm | retinify",
    ),
    width: int = typer.Option(2560, help="Capture frame width in pixels"),
    height: int = typer.Option(720, help="Capture frame height in pixels"),
):
    """Live stereo depth stream: left-rect | disparity | depth colourmap.

    Press q or ESC inside the window to quit.
    """
    run_stream(
        calib,
        device=device,
        preset=preset,
        matcher_name=matcher,
        width=width,
        height=height,
    )
