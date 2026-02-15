from __future__ import annotations
from pathlib import Path
import typer

from stereo_depth.app.rectify import run_rectify_dataset

def rectify(
    calib: Path = typer.Option(..., help="Calibration yaml"),
    data: Path = typer.Option(..., help="Folder containing left/ and right/ pairs"),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output folder for rectified pairs"),
    limit: int = typer.Option(0, help="Process only first N pairs (0: all)"),
    preview: bool = typer.Option(False, help="Save a rectification preview image with epipolar lines"),
    pairing: str = typer.Option("auto", help="Pairing mode: auto|name|index"),
):
    out_dir, preview_path, n, mode_used = run_rectify_dataset(
        calib, data, out_dir, limit=limit, preview=preview, pairing=pairing
    )
    typer.echo(f"Rectified {n} pairs -> {out_dir} (pairing={mode_used})")
    if preview_path:
        typer.echo(f"Saved preview: {preview_path}")
