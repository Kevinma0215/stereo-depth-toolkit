"""app/stream.py — live stereo depth streaming loop.

Orchestrates camera → pipeline → DepthAggregator → display.
Called by cli/stream_cmd.py.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from stereo_depth.adapters.calibration.yaml_repo import YamlCalibrationRepo
from stereo_depth.adapters.camera.uvc_source import UVCSource, open_source
from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.post_processor.hole_fill_post_processor import HoleFillPostProcessor
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.infrastructure.depth_aggregator import DepthAggregator
from stereo_depth.infrastructure.io.sbs_capture import SBSSplitter
from stereo_depth.use_cases.pipeline import StereoPipeline
from stereo_depth.use_cases.ports import IDisparityMatcher

_WIN         = "Stereo Depth Stream"
_CROSSHAIR   = (320, 240)   # pixel queried by the aggregator
_BUFFER_SIZE = 15


# ---------------------------------------------------------------------------
# Colourmap helpers
# ---------------------------------------------------------------------------

def _disp_colormap(disp: np.ndarray) -> np.ndarray:
    """JET disparity colourmap (matches tune_sgbm.py rendering).

    Invalid pixels (<=0) map to black.
    """
    valid = disp[disp > 0]
    if valid.size == 0:
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    d_max = float(np.percentile(valid, 95))
    clipped = np.clip(disp, 0.0, max(d_max, 1e-6))
    normed = (clipped / max(d_max, 1e-6) * 255).astype(np.uint8)
    normed[disp <= 0] = 0
    return cv2.applyColorMap(normed, cv2.COLORMAP_JET)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _conf_color(conf: float) -> tuple[int, int, int]:
    """BGR colour encoding for confidence level."""
    if conf >= 0.7:
        return (0, 255, 0)    # green
    if conf >= 0.4:
        return (0, 255, 255)  # yellow
    return (0, 0, 255)        # red


def _draw_crosshair(
    img: np.ndarray,
    pt: tuple[int, int],
    size: int = 12,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    u, v = pt
    cv2.line(img, (u - size, v), (u + size, v), color, 1, cv2.LINE_AA)
    cv2.line(img, (u, v - size), (u, v + size), color, 1, cv2.LINE_AA)
    cv2.circle(img, pt, 4, color, 1, cv2.LINE_AA)


def _make_aggregator(calib) -> DepthAggregator:
    return DepthAggregator(
        calib=calib,
        buffer_size=_BUFFER_SIZE,
        min_confidence=0.6,
        roi_radius=8,
    )


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
    if matcher_name == "cuda_bm":
        from stereo_depth.adapters.matcher.cuda_bm_matcher import CudaBmMatcher  # noqa: PLC0415
        return CudaBmMatcher()
    raise ValueError(f"Unknown matcher '{matcher_name}'. Valid options: sgbm | retinify | cuda_bm")


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
    fill_holes: bool = False,
    fill_radius: int = 3,
) -> None:
    """Open the UVC camera and run the depth pipeline on every frame.

    Displays a 2-panel window:
        left-rect (with aggregator overlay)  |  disparity (JET)

    Keymap:
        Q / ESC — quit
        S       — save left_rect + disparity PNG to outputs/stream/<timestamp>/
        R       — reset the aggregator ring buffer (useful after scene change)
    """
    calib   = YamlCalibrationRepo().load(str(calib_yaml))
    cap     = open_source(device=device, width=width, height=height)
    source  = UVCSource(cap, SBSSplitter())
    matcher = _build_matcher(matcher_name, preset, calib.image_size)

    post_processors = [HoleFillPostProcessor(radius=fill_radius)] if fill_holes else []

    pipeline = StereoPipeline(
        rectifier=OpenCVRectifier(),
        matcher=matcher,
        depth_estimator=OpenCVDepthEstimator(),
        calib=calib,
        post_processors=post_processors,
    )

    aggregator = _make_aggregator(calib)

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

    fps    = 0.0
    t_prev = time.perf_counter()

    # Keep the last rendered panels for the S-key save
    last_left_rect: np.ndarray | None = None
    last_disp_vis:  np.ndarray | None = None

    try:
        while True:
            pair      = source.grab()
            depth_map = pipeline.process(pair)

            # FPS: exponential moving average (α = 0.1)
            t_now  = time.perf_counter()
            fps    = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # --- Push disparity into aggregator ---
            aggregator.push(depth_map.disparity)
            stats = aggregator.stats()

            # --- Build left panel (BGR) ---
            left_rect = (
                depth_map.left_rect
                if depth_map.left_rect is not None
                else pair.left
            )
            if left_rect.ndim == 2:
                left_rect = cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)
            left_panel = left_rect.copy()

            # Crosshair — always shown
            _draw_crosshair(left_panel, _CROSSHAIR)

            # Aggregator overlay — shown once enough frames are buffered
            if stats.buffer_fill >= 5:
                results = aggregator.query([_CROSSHAIR])
                wp   = results[0]
                conf = wp.confidence
                col  = _conf_color(conf)

                if wp.xyz_m is not None:
                    depth_str = f"depth: {wp.xyz_m[2]:.2f} m  conf: {conf * 100:.0f}%"
                else:
                    depth_str = f"depth: N/A  conf: {conf * 100:.0f}%"

                cv2.putText(left_panel, depth_str, (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

                if conf < 0.4:
                    cv2.putText(left_panel, "LOW CONFIDENCE", (8, 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            # --- Build right panel (JET disparity) ---
            disp_panel = _disp_colormap(depth_map.disparity)

            # --- Combine panels ---
            display = np.hstack([left_panel, disp_panel])

            # --- HUD: bottom-left of combined frame ---
            hud = (
                f"FPS: {fps:.0f}  |  "
                f"buffer: {stats.buffer_fill}/{_BUFFER_SIZE}  |  "
                f"preset: {preset}  |  "
                f"Q quit"
            )
            cv2.putText(display, hud, (8, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(_WIN, display)

            # Stash for S-key save
            last_left_rect = left_rect
            last_disp_vis  = disp_panel

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break

            if key in (ord("s"), ord("S")):
                ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("outputs/stream") / ts
                out_dir.mkdir(parents=True, exist_ok=True)
                if last_left_rect is not None:
                    cv2.imwrite(str(out_dir / "left_rect.png"),  last_left_rect)
                if last_disp_vis is not None:
                    cv2.imwrite(str(out_dir / "disparity.png"), last_disp_vis)
                print(f"Saved to {out_dir}/")

            if key in (ord("r"), ord("R")):
                aggregator = _make_aggregator(calib)
                print("Aggregator ring buffer reset.")

    finally:
        source.release()
        cv2.destroyAllWindows()
