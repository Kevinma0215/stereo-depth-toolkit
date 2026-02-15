from __future__ import annotations
from pathlib import Path
import typer

def rectify(
    calib: Path = typer.Option(..., help="Calibration yaml"),
    data: Path = typer.Option(..., help="Folder containing pairs to rectify"),
    out_dir: Path = typer.Option(..., help="Output folder for rectified pairs"),
):
    # TODO: implement rectify
    typer.echo(f"[TODO] rectify {data} using {calib} -> {out_dir}")
