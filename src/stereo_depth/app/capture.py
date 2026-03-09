"""app/capture.py — live SBS stream with SPACE-to-save for calibration pairs.

Shows the raw side-by-side feed while the user collects image pairs.
Saved layout on disk:
    <out_dir>/
        left/  0001.png  0002.png  ...
        right/ 0001.png  0002.png  ...
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from stereo_depth.infrastructure.io.sbs_capture import SBSSplitter, open_camera

_WIN = "stereo-depth capture  |  SPACE=save  R=reject last  Q=quit"
_FLASH_FRAMES = 12   # frames the green-flash overlay is shown after a save


def run_capture(
    out_dir: Path,
    *,
    path: str = "/dev/video0",
    width: int = 2560,
    height: int = 720,
    fps: int = 30,
    num_pairs: int = 30,
) -> None:
    """Stream the SBS camera and save left/right pairs on SPACE press.

    Args:
        out_dir:   Root output folder; ``left/`` and ``right/`` sub-dirs are
                   created automatically.
        path:      V4L2 device path (e.g. ``/dev/video0``) or device index.
        width:     Capture frame width in pixels (full SBS width).
        height:    Capture frame height in pixels.
        fps:       Requested capture frame rate.
        num_pairs: Stop automatically after this many pairs are saved.
    """
    left_dir  = out_dir / "left"
    right_dir = out_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    # Try path as int (device index) first, fall back to string (V4L2 path)
    try:
        device = int(path)
        cap = open_camera(device=device, width=width, height=height, fps=fps)
    except ValueError:
        cap = open_camera(path=path, width=width, height=height, fps=fps)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {path}")

    splitter   = SBSSplitter()
    n_saved    = 0
    flash_left = 0      # countdown for the green save-flash

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed — stopping.")
                break

            left, right = splitter.split(frame)

            # --- Build display: left | right side-by-side at reduced height ---
            display = _make_display(left, right, target_h=360)

            # --- Green flash overlay on successful save ---
            if flash_left > 0:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]),
                              (0, 220, 0), -1)
                alpha = 0.25 * (flash_left / _FLASH_FRAMES)
                display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)
                flash_left -= 1

            # --- HUD ---
            _draw_hud(display, n_saved, num_pairs)

            cv2.imshow(_WIN, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key == ord(" "):
                n_saved += 1
                stem = f"{n_saved:04d}.png"
                cv2.imwrite(str(left_dir  / stem), left)
                cv2.imwrite(str(right_dir / stem), right)
                flash_left = _FLASH_FRAMES
                print(f"  Saved pair {n_saved:>3d}/{num_pairs}: {stem}")
                if n_saved >= num_pairs:
                    print(f"\nTarget of {num_pairs} pairs reached — done.")
                    break

            if key in (ord("r"), ord("R")) and n_saved > 0:
                stem = f"{n_saved:04d}.png"
                (left_dir  / stem).unlink(missing_ok=True)
                (right_dir / stem).unlink(missing_ok=True)
                print(f"  Rejected pair {n_saved:>3d}: {stem} deleted.")
                n_saved -= 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nCapture complete: {n_saved} pairs saved to {out_dir}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _make_display(
    left: np.ndarray, right: np.ndarray, target_h: int = 360
) -> np.ndarray:
    """Stack left and right side-by-side, scaled to target_h."""
    scale = target_h / left.shape[0]
    new_w = int(left.shape[1] * scale)
    left_s  = cv2.resize(left,  (new_w, target_h), interpolation=cv2.INTER_AREA)
    right_s = cv2.resize(right, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return np.hstack([left_s, right_s])


def _draw_hud(img: np.ndarray, n_saved: int, num_pairs: int) -> None:
    """Burn status text into img in-place."""
    h, w = img.shape[:2]

    # Progress bar background
    bar_h = 6
    bar_w = int(w * n_saved / max(num_pairs, 1))
    cv2.rectangle(img, (0, h - bar_h), (w, h), (60, 60, 60), -1)
    cv2.rectangle(img, (0, h - bar_h), (bar_w, h), (0, 200, 0), -1)

    # Status line
    status = f"Captured: {n_saved} / {num_pairs}   |   SPACE save   R reject last   Q quit"
    cv2.putText(img, status, (10, h - bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Panel labels
    mid = w // 2
    for x, label in ((10, "LEFT"), (mid + 10, "RIGHT")):
        cv2.putText(img, label, (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
