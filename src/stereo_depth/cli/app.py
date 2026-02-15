import typer

from .preview_cmd import preview
from .capture_cmd import capture
from .calibrate_cmd import calibrate
from .rectify_cmd import rectify

app = typer.Typer(no_args_is_help=True, help="Stereo depth toolkit")

# 四個一級命令
app.command("preview")(preview)
app.command("capture")(capture)
app.command("calibrate")(calibrate)
app.command("rectify")(rectify)

def main():
    app()

if __name__ == "__main__":
    main()
