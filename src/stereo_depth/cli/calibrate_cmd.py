from __future__ import annotations
from pathlib import Path
import typer

def calibrate(
    data: Path = typer.Option(..., help="Folder containing calibration pairs"),
    out: Path = typer.Option(Path("calib.yaml"), help="Output calibration yaml"),
):
    # TODO: implement calibration
    typer.echo(f"[TODO] calibrate from {data} -> {out}")
