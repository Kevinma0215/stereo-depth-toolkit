from __future__ import annotations
from pathlib import Path
import typer

def capture(
    out_dir: Path = typer.Option(..., help="Output folder for saved pairs"),
    # 先把常用參數留著，後面接 io/sources + sinks
    num_pairs: int = typer.Option(30, help="Number of pairs to capture"),
):
    # TODO: implement capture pipeline
    typer.echo(f"[TODO] capture {num_pairs} pairs to {out_dir}")
