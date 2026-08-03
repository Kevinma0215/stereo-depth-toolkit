"""Undistort images (or a live stream) using a mono calibration.

The output of this step is a set of images that behave like an ideal pinhole
camera described by ``K_new`` with ZERO distortion. That is a different matrix
from the ``K`` in the calibration file, which describes the raw frames — using
the wrong one is the classic way to get a subtly wrong reconstruction, so the
pairing is restated in every artefact this module writes.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stereo_depth.adapters.calibration.isaac_export import (
    DEFAULT_HORIZONTAL_APERTURE_MM,
    usd_camera_params,
)
from stereo_depth.adapters.calibration.mono_calibrator import (
    build_undistort_maps,
    new_camera_matrix,
)
from stereo_depth.adapters.calibration.mono_yaml_repo import (
    SCHEMA,
    MonoYamlCalibrationRepo,
)
from stereo_depth.adapters.camera.uvc_source import open_source
from stereo_depth.infrastructure.config.io import save_yaml
from stereo_depth.infrastructure.io.pairs import list_images
from stereo_depth.infrastructure.viz.overlay import draw_text

_WIN = "stereo-depth undistort  |  a/z alpha  s snapshot  q quit"


def _load(calib_yaml: Path) -> tuple[str, np.ndarray, np.ndarray, tuple[int, int]]:
    repo = MonoYamlCalibrationRepo()
    data = repo.load_raw(str(calib_yaml))
    model = str(data["model"])
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64)
    size = (int(data["image_size"]["width"]), int(data["image_size"]["height"]))
    return model, K, D, size


def run_undistort(
    calib_yaml: Path,
    *,
    images: Path | None = None,
    out: Path | None = None,
    live: bool = False,
    path: str = "/dev/video0",
    width: int = 0,
    height: int = 0,
    fps: int = 30,
    alpha: float = 0.0,
    balance: float = 0.0,
    sensor_width_mm: float = DEFAULT_HORIZONTAL_APERTURE_MM,
) -> None:
    model, K, D, size = _load(calib_yaml)

    if live:
        _run_live(model, K, D, size, path=path, width=width, height=height,
                  fps=fps, alpha=alpha, balance=balance)
        return

    if images is None or out is None:
        raise ValueError("batch mode needs --images and --out (or use --live)")

    _run_batch(model, K, D, size, calib_yaml, images, out,
               alpha=alpha, balance=balance, sensor_width_mm=sensor_width_mm)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _run_batch(
    model, K, D, size, calib_yaml: Path, images: Path, out: Path,
    *, alpha: float, balance: float, sensor_width_mm: float,
) -> None:
    paths = list_images(images)
    if not paths:
        raise RuntimeError(f"No images found in {images}")

    K_new, roi = new_camera_matrix(K, D, size, model, alpha=alpha, balance=balance)
    map_x, map_y = build_undistort_maps(K, D, K_new, size, model)

    out.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_skipped = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            n_skipped += 1
            continue
        h, w = img.shape[:2]
        if (w, h) != size:
            print(f"  skipping {p.name}: {w}x{h} does not match calibration "
                  f"{size[0]}x{size[1]}")
            n_skipped += 1
            continue
        cv2.imwrite(str(out / p.name), cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR))
        n_done += 1

    sidecar = out / "undistorted_intrinsics.yaml"
    save_yaml(sidecar, {
        "schema": SCHEMA,
        "model": "pinhole",
        "image_size": {"width": size[0], "height": size[1]},
        "K": np.asarray(K_new, dtype=float).tolist(),
        "D": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rpe_px": 0.0,
        "views_used": 0,
        "note": (
            "These images are already undistorted: use this K with ZERO "
            "distortion. Do not reuse the K/D from the source calibration here."
        ),
        "source": {
            "calibration": str(calib_yaml),
            "model": model,
            ("balance" if model == "fisheye" else "alpha"): (
                balance if model == "fisheye" else alpha
            ),
            "roi": list(roi) if roi else None,
        },
        "isaac_sim": {
            "usd_camera": usd_camera_params(
                K_new, size, horizontal_aperture_mm=sensor_width_mm
            ),
        },
    })

    print(f"Undistorted {n_done} image(s) -> {out}" +
          (f"  ({n_skipped} skipped)" if n_skipped else ""))
    if roi:
        print(f"  valid ROI at alpha={alpha}: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
    print(f"  intrinsics for these images: {sidecar}")
    print(f"  fx={K_new[0, 0]:.2f} fy={K_new[1, 1]:.2f} "
          f"cx={K_new[0, 2]:.2f} cy={K_new[1, 2]:.2f}, distortion = 0")
    print("\nThese images pair with K_new above - NOT with the K in "
          f"{calib_yaml.name}, which describes the raw frames.")


# ---------------------------------------------------------------------------
# Live preview
# ---------------------------------------------------------------------------

def _run_live(
    model, K, D, size, *, path: str, width: int, height: int, fps: int,
    alpha: float, balance: float,
) -> None:
    try:
        device = int(path)
        cap = open_source(device=device, width=width, height=height, fps=fps)
    except ValueError:
        cap = open_source(path=path, width=width, height=height, fps=fps)

    knob = balance if model == "fisheye" else alpha
    knob_name = "balance" if model == "fisheye" else "alpha"
    maps: tuple[np.ndarray, np.ndarray] | None = None
    K_new = None
    roi = None
    cur_size = None

    print(f"Live undistort ({model}). a/z adjust {knob_name}, s snapshot, q quit.")
    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    n_snap = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed - stopping.")
                break

            h, w = frame.shape[:2]
            if maps is None or cur_size != (w, h):
                cur_size = (w, h)
                if (w, h) != size:
                    print(f"WARNING: stream is {w}x{h} but the calibration is "
                          f"{size[0]}x{size[1]}; the undistortion will be wrong.")
                K_new, roi = new_camera_matrix(
                    K, D, cur_size, model, alpha=knob, balance=knob
                )
                maps = build_undistort_maps(K, D, K_new, cur_size, model)

            und = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            shown = und.copy()
            if roi and roi[2] > 0 and roi[3] > 0:
                cv2.rectangle(shown, (roi[0], roi[1]),
                              (roi[0] + roi[2], roi[1] + roi[3]), (0, 220, 220), 2)

            display = _side_by_side(frame, shown)
            draw_text(display, "RAW  (pairs with K, D)", (10, 24), scale=0.6)
            draw_text(display, f"UNDISTORTED  (pairs with K_new, D=0)",
                      (display.shape[1] // 2 + 10, 24), scale=0.6)
            draw_text(
                display,
                f"{knob_name}={knob:.2f}   fx={K_new[0, 0]:.1f} cx={K_new[0, 2]:.1f}"
                f"   |  a/z {knob_name}   s snapshot   q quit",
                (10, display.shape[0] - 12), scale=0.5,
            )
            cv2.imshow(_WIN, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("a"), ord("z")):
                knob = float(np.clip(knob + (0.1 if key == ord("a") else -0.1), 0.0, 1.0))
                maps = None                      # force a rebuild
                print(f"  {knob_name} = {knob:.2f}")
            if key in (ord("s"), ord("S")):
                n_snap += 1
                name = f"undistort_snapshot_{n_snap:03d}.png"
                cv2.imwrite(name, und)
                print(f"  wrote {name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _side_by_side(left: np.ndarray, right: np.ndarray, target_h: int = 480) -> np.ndarray:
    scale = target_h / left.shape[0]
    w = int(left.shape[1] * scale)
    a = cv2.resize(left, (w, target_h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(right, (w, target_h), interpolation=cv2.INTER_AREA)
    return np.hstack([a, b])
