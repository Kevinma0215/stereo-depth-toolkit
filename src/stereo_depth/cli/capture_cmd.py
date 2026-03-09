from __future__ import annotations
from pathlib import Path
import typer

from stereo_depth.app.capture import run_capture


def capture(
    out_dir: Path = typer.Option(..., help="Output folder for saved pairs"),
    num_pairs: int = typer.Option(30, help="Stop after this many pairs"),
    path: str = typer.Option("/dev/video0", help="Camera device path or index"),
    width: int = typer.Option(2560, help="Capture frame width (full SBS width)"),
    height: int = typer.Option(720, help="Capture frame height"),
    fps: int = typer.Option(30, help="Requested camera frame rate"),
):
    """Stream the stereo camera and save left/right pairs for calibration.

    Press SPACE to save a pair, R to reject the last one, Q to quit.
    Images are written to <out_dir>/left/ and <out_dir>/right/.
    """
    run_capture(
        out_dir,
        path=path,
        width=width,
        height=height,
        fps=fps,
        num_pairs=num_pairs,
    )
