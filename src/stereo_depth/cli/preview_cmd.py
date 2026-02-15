from __future__ import annotations
from pathlib import Path
import typer

from stereo_depth.io.sources import open_source
from stereo_depth.io.sbs_capture import SBSSplitter
from stereo_depth.io.sinks import VideoRecorder
from stereo_depth.viz.preview import preview_sbs

def preview(
    device:  int = typer.Option(0, help="Camera device index"),
    path:    str | None = typer.Option(None, help="V4L2 device path, e.g. /dev/video1"),
    video:   str | None = typer.Option(None, help="Path to SBS video file"),
    width:   int = typer.Option(0, help="Capture width (0: default)"),
    height:  int = typer.Option(0, help="Capture height (0: default)"),
    fps:     int = typer.Option(0, help="Capture FPS (0: default)"),
    swap_lr: bool = typer.Option(False, "--swap-lr", help="Swap left/right after split"),
    record:  Path | None = typer.Option(None, help="Record preview stream to AVI (MJPG)"),
):
    cap = open_source(device=device, path=path, video=video, width=width, height=height, fps=fps)
    splitter = SBSSplitter(swap_lr=swap_lr)
    recorder = VideoRecorder(record) if record else None
    preview_sbs(cap=cap, splitter=splitter, recorder=recorder)
