"""CLI: undistort images or preview undistortion live."""
from __future__ import annotations

from pathlib import Path

import typer

from stereo_depth.app.undistort_mono import run_undistort


def undistort(
    calib: Path = typer.Option(..., help="Mono calibration YAML"),
    images: Path = typer.Option(None, help="Folder of raw images to undistort"),
    out: Path = typer.Option(None, help="Folder to write undistorted images to"),
    live: bool = typer.Option(False, "--live", help="Live raw|undistorted preview"),
    path: str = typer.Option("/dev/video0", help="V4L2 device path or index (--live)"),
    width: int = typer.Option(0),
    height: int = typer.Option(0),
    fps: int = typer.Option(30),
    alpha: float = typer.Option(
        0.0, help="pinhole/rational: 0 crops to valid pixels, 1 keeps all pixels"
    ),
    balance: float = typer.Option(
        0.0, help="fisheye: 0 crops to valid pixels, 1 keeps all pixels"
    ),
):
    """Remove lens distortion; writes the matching K_new alongside the output."""
    run_undistort(
        calib_yaml=calib,
        images=images,
        out=out,
        live=live,
        path=path,
        width=width,
        height=height,
        fps=fps,
        alpha=alpha,
        balance=balance,
    )
