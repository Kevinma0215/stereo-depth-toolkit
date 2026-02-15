""" 把 CLI 的 calibrate 指令接上 app layer """

from __future__ import annotations
from pathlib import Path
import typer
from stereo_depth.app.calibrate import run_calibrate_charuco_stereo

def calibrate(
    data: Path = typer.Option(..., help="Folder with left/ and right/"),
    out: Path = typer.Option(Path("data/calib.yaml"), help="Output calib yaml"),
    squares_x: int = typer.Option(7),
    squares_y: int = typer.Option(5),
    square_length: float = typer.Option(0.03, help="meters"),
    marker_length: float = typer.Option(0.022, help="meters"),
    dict_name: str = typer.Option("DICT_5X5_100"),
    min_views: int = typer.Option(15),
    min_markers: int = typer.Option(4),
    min_charuco: int = typer.Option(10),
    min_common_ids: int = typer.Option(10),
):
    out_yaml, report_json = run_calibrate_charuco_stereo(
        data_dir=data,
        out_yaml=out,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        dict_name=dict_name,
        min_views=min_views,
        min_markers=min_markers,
        min_charuco=min_charuco,
        min_common_ids=min_common_ids,
    )
    typer.echo(f"Saved calibration: {out_yaml}")
    typer.echo(f"Saved report:      {report_json}")
