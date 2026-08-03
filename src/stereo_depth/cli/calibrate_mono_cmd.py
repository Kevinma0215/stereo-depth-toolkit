"""CLI: calibrate a single camera's intrinsics from ChArUco images."""
from __future__ import annotations

from pathlib import Path

import typer

from stereo_depth.app.calibrate_mono import run_calibrate_mono
from stereo_depth.adapters.calibration.isaac_export import DEFAULT_HORIZONTAL_APERTURE_MM


def calibrate_mono(
    data: Path = typer.Option(..., help="Folder of images from ONE camera"),
    out: Path = typer.Option(Path("outputs/calib/mono.yaml"), help="Output YAML"),
    model: str = typer.Option(
        "auto",
        help="Distortion model: auto | pinhole | rational | fisheye. "
             "'auto' fits all three and picks the best by holdout error.",
    ),
    squares_x: int = typer.Option(7),
    squares_y: int = typer.Option(5),
    square_length: float = typer.Option(0.03, help="metres"),
    marker_length: float = typer.Option(0.022, help="metres"),
    dict_name: str = typer.Option("DICT_5X5_100"),
    min_views: int = typer.Option(15),
    min_markers: int = typer.Option(4),
    min_charuco: int = typer.Option(10),
    holdout_frac: float = typer.Option(
        0.25, help="Fraction of views held out to compare models fairly."
    ),
    seed: int = typer.Option(0, help="Seed for the train/holdout split."),
    refine: bool = typer.Option(True, help="Sub-pixel refine detected corners."),
    sensor_width_mm: float = typer.Option(
        DEFAULT_HORIZONTAL_APERTURE_MM,
        help="Assumed sensor width for the Isaac Sim focal length (mm).",
    ),
):
    """Fit intrinsics + distortion, and export Isaac Sim camera parameters."""
    run_calibrate_mono(
        data_dir=data,
        out_yaml=out,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        dict_name=dict_name,
        min_markers=min_markers,
        min_charuco=min_charuco,
        min_views=min_views,
        model=model,
        holdout_frac=holdout_frac,
        seed=seed,
        refine=refine,
        sensor_width_mm=sensor_width_mm,
    )
