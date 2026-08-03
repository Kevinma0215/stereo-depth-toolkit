"""List V4L2 capture devices so the user can find their camera."""
from __future__ import annotations

import typer

from stereo_depth.adapters.camera.v4l2_devices import (
    format_device_table,
    list_video_devices,
)


def devices(
    probe: bool = typer.Option(
        False, "--probe",
        help="Open each node and read a frame to see which ones really capture.",
    ),
):
    """Show /dev/video* devices with their supported resolutions and rates."""
    found = list_video_devices(probe=probe)
    typer.echo(format_device_table(found))
