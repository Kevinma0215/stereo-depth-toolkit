"""CLI: live guided collection of mono calibration images."""
from __future__ import annotations

from pathlib import Path

import typer

from stereo_depth.app.calibrate_mono import run_capture_mono


def capture_mono(
    out_dir: Path = typer.Option(..., help="Folder to write captured images to"),
    path: str = typer.Option("/dev/video0", help="V4L2 device path or index"),
    width: int = typer.Option(0, help="0 = camera default"),
    height: int = typer.Option(0, help="0 = camera default"),
    fps: int = typer.Option(30),
    squares_x: int = typer.Option(7),
    squares_y: int = typer.Option(5),
    square_length: float = typer.Option(0.03, help="metres"),
    marker_length: float = typer.Option(0.022, help="metres"),
    dict_name: str = typer.Option("DICT_5X5_100"),
    min_markers: int = typer.Option(4),
    min_charuco: int = typer.Option(10),
    target_views: int = typer.Option(40, help="Views to collect before done"),
    blur_min: float = typer.Option(60.0, help="Minimum Laplacian variance"),
    grid: int = typer.Option(3, help="Coverage grid is grid x grid"),
    auto: bool = typer.Option(True, help="Auto-save views that pass every gate"),
):
    """Stream a camera and auto-collect calibration views with live guidance."""
    n = run_capture_mono(
        out_dir=out_dir,
        path=path,
        width=width,
        height=height,
        fps=fps,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        dict_name=dict_name,
        min_markers=min_markers,
        min_charuco=min_charuco,
        target_views=target_views,
        blur_min=blur_min,
        grid=grid,
        auto=auto,
    )
    if n:
        typer.echo(
            f"\nNext: stereo-depth calibrate-mono --data {out_dir} "
            f"--out outputs/calib/mono.yaml"
        )
