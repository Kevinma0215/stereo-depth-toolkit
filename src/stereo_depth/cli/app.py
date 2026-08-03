import typer

from .preview_cmd import preview
from .capture_cmd import capture
from .calibrate_cmd import calibrate
from .rectify_cmd import rectify
from .depth_cmd import depth
from .stream_cmd import stream
from .devices_cmd import devices
from .capture_mono_cmd import capture_mono
from .calibrate_mono_cmd import calibrate_mono
from .undistort_cmd import undistort

app = typer.Typer(no_args_is_help=True, help="Stereo depth toolkit")

app.command("preview")(preview)
app.command("capture")(capture)
app.command("calibrate")(calibrate)
app.command("rectify")(rectify)
app.command("depth")(depth)
app.command("stream")(stream)

# Single-camera (mono) intrinsic calibration
app.command("devices")(devices)
app.command("capture-mono")(capture_mono)
app.command("calibrate-mono")(calibrate_mono)
app.command("undistort")(undistort)

def main():
    app()

if __name__ == "__main__":
    main()
