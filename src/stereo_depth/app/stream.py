"""app/stream.py — live stereo depth streaming loop.

Orchestrates camera → pipeline → display.  Called by cli/stream_cmd.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from stereo_depth.adapters.calibration.yaml_repo import YamlCalibrationRepo
from stereo_depth.adapters.camera.uvc_source import UVCSource, open_source
from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.infrastructure.io.sbs_capture import SBSSplitter
from stereo_depth.use_cases.pipeline import StereoPipeline
from stereo_depth.use_cases.ports import IDisparityMatcher

_DEPTH_NEAR_M = 0.1
_DEPTH_FAR_M  = 3.0
_WINDOW       = "stereo-depth stream  |  left-rect | disparity | depth  |  q/ESC=quit"


# ---------------------------------------------------------------------------
# Colourmap helpers
# ---------------------------------------------------------------------------

def _depth_colormap(depth: np.ndarray) -> np.ndarray:
    """float32 depth (H, W) metres → BGR colourmap.  NaN pixels → black."""
    clipped = np.clip(depth, _DEPTH_NEAR_M, _DEPTH_FAR_M)
    norm = (clipped - _DEPTH_NEAR_M) / (_DEPTH_FAR_M - _DEPTH_NEAR_M)
    grey = (norm * 255).astype(np.uint8)
    grey[~np.isfinite(depth)] = 0
    return cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)


def _disp_colormap(disp: np.ndarray) -> np.ndarray:
    """float32 disparity (H, W) → BGR colourmap.  Invalid (<=0) → black."""
    valid = disp[disp > 0]
    if valid.size == 0:
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    span = hi - lo if hi > lo else 1.0
    norm = np.clip((disp - lo) / span, 0.0, 1.0)
    grey = (norm * 255).astype(np.uint8)
    grey[disp <= 0] = 0
    return cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _central_median(depth: np.ndarray, fraction: float = 0.10) -> float | None:
    """Median of finite depth values inside the central *fraction* of the image."""
    H, W = depth.shape
    rh = max(1, int(H * fraction))
    rw = max(1, int(W * fraction))
    y  = (H - rh) // 2
    x  = (W - rw) // 2
    roi = depth[y : y + rh, x : x + rw].ravel()
    finite = roi[np.isfinite(roi)]
    return float(np.median(finite)) if finite.size > 0 else None


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def _overlay(img: np.ndarray, fps: float, median_m: float | None) -> np.ndarray:
    out = img.copy()
    depth_str = f"{median_m:.2f} m" if median_m is not None else "n/a"
    text = f"FPS {fps:.1f}  |  center {depth_str}"
    cv2.putText(out, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Matcher factory (mirrors app/depth.py)
# ---------------------------------------------------------------------------

def _build_matcher(
    matcher_name: str, preset: str, image_size: tuple[int, int]
) -> IDisparityMatcher:
    if matcher_name == "sgbm":
        return SgbmMatcher(preset=preset)
    if matcher_name == "retinify":
        from stereo_depth.adapters.matcher.retinify_matcher import RetinifyMatcher  # noqa: PLC0415
        w, h = image_size
        return RetinifyMatcher(width=w, height=h, mode=preset)
    raise ValueError(f"Unknown matcher '{matcher_name}'. Valid options: sgbm | retinify")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stream(
    calib_yaml: Path,
    *,
    device: int = 0,
    preset: str = "indoor",
    matcher_name: str = "sgbm",
    width: int = 2560,
    height: int = 720,
) -> None:
    """Open the UVC camera and run the depth pipeline on every frame.

    Displays a side-by-side window:
        left-rect | disparity colourmap | depth colourmap

    Press **q** or **ESC** to exit.
    """
    calib   = YamlCalibrationRepo().load(str(calib_yaml))
    cap     = open_source(device=device, width=width, height=height)
    source  = UVCSource(cap, SBSSplitter())
    matcher = _build_matcher(matcher_name, preset, calib.image_size)

    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=matcher,
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
    )

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

    fps    = 0.0
    t_prev = time.perf_counter()

    try:
        while True:
            pair      = source.grab()
            depth_map = pipeline.process(pair)

            # FPS: exponential moving average (α = 0.1)
            t_now  = time.perf_counter()
            fps    = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # Panels
            left_vis  = (
                depth_map.left_rect
                if depth_map.left_rect is not None
                else pair.left
            )
            if left_vis.ndim == 2:
                left_vis = cv2.cvtColor(left_vis, cv2.COLOR_GRAY2BGR)

            disp_vis  = _disp_colormap(depth_map.disparity)
            depth_vis = _depth_colormap(depth_map.data)

            combined  = cv2.hconcat([left_vis, disp_vis, depth_vis])
            combined  = _overlay(combined, fps, _central_median(depth_map.data))

            cv2.imshow(_WINDOW, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
