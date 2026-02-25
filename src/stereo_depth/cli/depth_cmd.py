from __future__ import annotations
from pathlib import Path
import typer

from stereo_depth.app.depth import run_depth_once

def depth(
    calib: Path = typer.Option(..., help="Calibration yaml"),
    left: Path = typer.Option(..., help="Left image path"),
    right: Path = typer.Option(..., help="Right image path"),
    out: Path = typer.Option(..., help="Output folder"),
    preset: str = typer.Option(
        "indoor",
        help="SGBM preset (indoor|outdoor|high_quality) or Retinify mode (fast|balanced|accurate)",
    ),
    matcher: str = typer.Option(
        "sgbm",
        help="Disparity matcher backend: sgbm | retinify",
    ),
    save_npy: bool = typer.Option(True, help="Save disparity.npy and depth_m.npy"),
):
    run_depth_once(
        calib, left, right, out,
        preset_name=preset,
        save_npy=save_npy,
        matcher_name=matcher,
    )
    typer.echo(f"Saved depth outputs -> {out}")
