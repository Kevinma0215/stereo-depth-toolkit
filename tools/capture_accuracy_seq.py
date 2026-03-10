"""tools/capture_accuracy_seq.py — record a stereo sequence at one known distance.

Saves frames in the format FileSource (directory mode) already expects:

    <output-dir>/dist_XXXcm/
        left/   0001.png  0002.png  ...
        right/  0001.png  0002.png  ...
        metadata.yaml

Usage:
    python tools/capture_accuracy_seq.py --distance 0.30 --frames 30
    python tools/capture_accuracy_seq.py --distance 0.55 --frames 30 \\
        --calib outputs/calib/0308_try4/calib.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from stereo_depth.infrastructure.config.io import save_yaml
from stereo_depth.infrastructure.io.sbs_capture import SBSSplitter, open_camera


_WIN              = "capture_accuracy_seq  |  Q = abort"
_DEFAULT_OUT      = Path("data/accuracy")
_DEFAULT_CALIB    = "outputs/calib/calib.yaml"
_COUNTDOWN_SEC    = 3
_PREVIEW_WIDTH    = 960
_PREVIEW_HEIGHT   = 240


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq_dir(output_dir: Path, distance_m: float) -> Path:
    """Return e.g. data/accuracy/dist_030cm/ for distance_m=0.30."""
    cm = round(distance_m * 100)
    return output_dir / f"dist_{cm:03d}cm"


def _open_camera(path: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open camera by integer index or V4L2 path string."""
    try:
        return open_camera(device=int(path), width=width, height=height, fps=fps)
    except ValueError:
        return open_camera(path=path, width=width, height=height, fps=fps)


def _make_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Side-by-side left|right scaled to _PREVIEW_WIDTH × _PREVIEW_HEIGHT."""
    combined = np.hstack([left, right])
    return cv2.resize(combined, (_PREVIEW_WIDTH, _PREVIEW_HEIGHT),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture a stereo image sequence at a known distance "
                    "for offline depth accuracy evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--distance",    required=True, type=float, metavar="M",
                   help="Ground-truth distance to the target in metres.")
    p.add_argument("--frames",      type=int, default=30,
                   help="Number of stereo pairs to capture.")
    p.add_argument("--output-dir",  type=Path, default=_DEFAULT_OUT, metavar="DIR",
                   help="Root output directory.")
    p.add_argument("--calib",       default=_DEFAULT_CALIB, metavar="PATH",
                   help="Calibration YAML path stored in metadata "
                        "(not loaded during capture).")
    p.add_argument("--path",        default="/dev/video0",
                   help="Camera device path or integer index.")
    p.add_argument("--width",       type=int, default=2560,
                   help="Full SBS frame width in pixels.")
    p.add_argument("--height",      type=int, default=720,
                   help="Frame height in pixels.")
    p.add_argument("--fps",         type=int, default=30,
                   help="Requested capture frame rate.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.distance <= 0:
        print("ERROR: --distance must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.frames < 1:
        print("ERROR: --frames must be >= 1.", file=sys.stderr)
        sys.exit(1)

    seq_dir   = _seq_dir(args.output_dir, args.distance)
    left_dir  = seq_dir / "left"
    right_dir = seq_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_camera(args.path, args.width, args.height, args.fps)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera: {args.path}", file=sys.stderr)
        sys.exit(1)

    splitter = SBSSplitter()
    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

    print("=== Accuracy Sequence Capture ===")
    print(f"  distance : {args.distance:.3f} m")
    print(f"  frames   : {args.frames}")
    print(f"  output   : {seq_dir}")
    print(f"  calib    : {args.calib}")
    print()
    print(f"  Point the camera at a target {args.distance:.2f} m away and hold still.")
    print(f"  Capture starts in {_COUNTDOWN_SEC} seconds.  Press Q to abort.")
    print()

    # ------------------------------------------------------------------ #
    # Countdown with live preview                                          #
    # ------------------------------------------------------------------ #
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed during countdown.", file=sys.stderr)
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)

        left, right = splitter.split(frame)
        display     = _make_preview(left, right)
        remaining   = max(0.0, _COUNTDOWN_SEC - (time.monotonic() - t0))

        cv2.putText(
            display,
            f"Starting in {remaining:.1f}s  —  hold target at {args.distance:.2f} m  |  Q abort",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA,
        )
        cv2.imshow(_WIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("Aborted during countdown.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        if time.monotonic() - t0 >= _COUNTDOWN_SEC:
            break

    # ------------------------------------------------------------------ #
    # Capture loop                                                         #
    # ------------------------------------------------------------------ #
    n_saved = 0
    try:
        while n_saved < args.frames:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed — stopping early.", file=sys.stderr)
                break

            left, right = splitter.split(frame)

            n_saved += 1
            stem = f"{n_saved:04d}.png"
            cv2.imwrite(str(left_dir  / stem), left)
            cv2.imwrite(str(right_dir / stem), right)

            display  = _make_preview(left, right)
            progress = int(30 * n_saved / args.frames)
            bar      = "[" + "=" * progress + " " * (30 - progress) + "]"
            cv2.putText(
                display,
                f"Capturing {bar} {n_saved}/{args.frames}  |  Q abort",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA,
            )
            cv2.imshow(_WIN, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print(f"\nAborted after {n_saved} frame(s).")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if n_saved == 0:
        print("No frames captured.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # metadata.yaml                                                        #
    # ------------------------------------------------------------------ #
    save_yaml(seq_dir / "metadata.yaml", {
        "distance_m": args.distance,
        "num_frames":  n_saved,
        "timestamp":   datetime.now().isoformat(),
        "calib_path":  args.calib,
    })

    print(f"\nCapture complete: {n_saved} pairs saved to {seq_dir}/")
    print(f"  left/  : {n_saved} images")
    print(f"  right/ : {n_saved} images")
    print(f"  metadata.yaml written")


if __name__ == "__main__":
    main()
