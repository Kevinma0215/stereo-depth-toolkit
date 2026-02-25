import typer

from .preview_cmd import preview
from .capture_cmd import capture
from .calibrate_cmd import calibrate
from .rectify_cmd import rectify
from .depth_cmd import depth
from .stream_cmd import stream

app = typer.Typer(no_args_is_help=True, help="Stereo depth toolkit")

app.command("preview")(preview)
app.command("capture")(capture)
app.command("calibrate")(calibrate)
app.command("rectify")(rectify)
app.command("depth")(depth)
app.command("stream")(stream)

def main():
    app()

if __name__ == "__main__":
    main()
