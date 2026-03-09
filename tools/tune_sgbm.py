"""Standalone SGBM tuning script.

Usage:
    python tools/tune_sgbm.py --left <left.png> --right <right.png> \
        --calib <calib.yaml> [--preset indoor]

Keymap in the viewer window:
    Q  quit
    S  save panels to outputs/tune/<timestamp>/
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Ensure the package is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stereo_depth.adapters.calibration.yaml_repo import YamlCalibrationRepo
from stereo_depth.adapters.depth.opencv_depth_estimator import OpenCVDepthEstimator
from stereo_depth.adapters.matcher.sgbm_matcher import SgbmMatcher
from stereo_depth.adapters.rectifier.opencv_rectifier import OpenCVRectifier
from stereo_depth.entities.frame import FramePair
from stereo_depth.infrastructure.depth_aggregator import DepthAggregator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGBM tuning viewer")
    p.add_argument("--left",   required=True, help="Left image path")
    p.add_argument("--right",  required=True, help="Right image path")
    p.add_argument("--calib",  required=True, help="Calibration YAML path")
    p.add_argument("--preset", default="indoor",
                   choices=["indoor", "outdoor", "high_quality"],
                   help="SGBM preset name (default: indoor)")
    p.add_argument("--scale", type=float, default=0.4,
                   help="Resize factor applied to rectified images before SGBM "
                        "(default: 0.4 ≈ 512×288; disparity is scaled back to full-res). "
                        "SGBM on 1280×720 CPU peaks at ~2 FPS regardless of params; "
                        "0.4x reaches ~20 FPS.")
    p.add_argument("--no-wls", dest="no_wls", action="store_true",
                   help="Disable WLS hole-filling filter (faster, for real-time testing).")
    return p.parse_args()


def colorize(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize arr to [0,255] and apply JET colormap."""
    clipped = np.clip(arr, vmin, vmax)
    normed = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return cv2.applyColorMap(normed, cv2.COLORMAP_JET)


def _check_rectification(left_rect: np.ndarray, right_rect: np.ndarray) -> None:
    """Check epipolar alignment and print a warning if rectification is poor.

    Samples 20 feature points from left_rect, finds each match in right_rect
    via template matching restricted to the same row ±2 px (epipolar band),
    then reports mean vertical pixel error.
    """
    to_gray = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    left_gray  = to_gray(left_rect)
    right_gray = to_gray(right_rect)

    pts = cv2.goodFeaturesToTrack(left_gray, maxCorners=20, qualityLevel=0.01,
                                  minDistance=30)
    if pts is None or len(pts) == 0:
        print("Rectification row error: n/a  (no features found)")
        return

    H, W = left_gray.shape
    half = 15          # half-size of the matching patch
    search_vy = 2      # vertical search radius (px) around the expected epipolar row
    errors: list[float] = []

    for pt in pts:
        x, y = int(pt[0, 0]), int(pt[0, 1])
        # Skip points too close to borders for a full patch
        if x < half or y < half or x >= W - half or y >= H - half:
            continue

        patch = left_gray[y - half : y + half + 1, x - half : x + half + 1]

        # Search strip in right image: same row ±search_vy, full image width
        sy0 = max(0, y - half - search_vy)
        sy1 = min(H, y + half + search_vy + 1)
        strip = right_gray[sy0:sy1, :]

        if strip.shape[0] < patch.shape[0] or strip.shape[1] < patch.shape[1]:
            continue

        result = cv2.matchTemplate(strip, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.5:   # weak match — skip
            continue

        # Centre of the matched patch in original image coordinates
        match_y = sy0 + max_loc[1] + half
        errors.append(abs(match_y - y))

    if not errors:
        print("Rectification row error: n/a  (no reliable matches)")
        return

    mean_err = float(np.mean(errors))
    status = "GOOD" if mean_err < 1.0 else "WARNING: re-calibrate"
    print(f"Rectification row error: {mean_err:.2f} px  [{status}]")

    if mean_err >= 1.0:
        print(
            "Horizontal streaking detected. Likely cause: calibration was done at a\n"
            "different physical camera position. Re-run:\n"
            "  stereo-depth calibrate --data data/calib/<session> "
            "--out outputs/calib/calib.yaml"
        )


def _wls_filter(disp: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Apply WLS filter to fill holes on flat/low-texture surfaces.

    WLS propagates depth from edges into textureless regions.
    Falls back to medianBlur if cv2.ximgproc is not available.
    """
    try:
        wls = cv2.ximgproc.createDisparityWLSFilterGeneric(use_confidence=False)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        # WLS expects 16-bit fixed-point (before /16 scaling)
        disp16 = (disp * 16).astype(np.int16)
        filtered = wls.filter(disp16, left)
        return filtered.astype(np.float32) / 16.0
    except AttributeError:
        print("WARNING: cv2.ximgproc not available; falling back to medianBlur(5)")
        return cv2.medianBlur(disp, 5)


def run(args: argparse.Namespace) -> None:
    # --- Load inputs ---
    left_img  = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    if left_img is None:
        raise FileNotFoundError(f"Cannot read left image: {args.left}")
    if right_img is None:
        raise FileNotFoundError(f"Cannot read right image: {args.right}")

    calib = YamlCalibrationRepo().load(args.calib)

    # --- Build pipeline components ---
    rectifier = OpenCVRectifier()
    matcher   = SgbmMatcher(preset_name=args.preset)
    estimator = OpenCVDepthEstimator()

    pair = FramePair(left=left_img, right=right_img)

    # --- Rectify (always at full resolution, timed) ---
    t0 = time.perf_counter()
    rect = rectifier.rectify(pair, calib)
    rect_ms = (time.perf_counter() - t0) * 1000

    # --- Rectification quality check (Issue 1) ---
    _check_rectification(rect.left, rect.right)

    # --- Optional downscale before SGBM ---
    # SGBM on 1280×720 CPU peaks at ~2 FPS regardless of parameter tuning.
    # Downscaling to 0.4x (512×288) reaches ~20 FPS; disparity is rescaled back.
    scale = args.scale
    if scale != 1.0:
        h_s = int(rect.left.shape[0] * scale)
        w_s = int(rect.left.shape[1] * scale)
        left_s  = cv2.resize(rect.left,  (w_s, h_s), interpolation=cv2.INTER_AREA)
        right_s = cv2.resize(rect.right, (w_s, h_s), interpolation=cv2.INTER_AREA)
    else:
        left_s, right_s = rect.left, rect.right

    # --- SGBM ---
    t1 = time.perf_counter()
    disp_s = matcher.compute(left_s, right_s)
    sgbm_ms = (time.perf_counter() - t1) * 1000

    # --- WLS (Issue 2: optional via --no-wls) ---
    t2 = time.perf_counter()
    if not args.no_wls:
        disp_s = _wls_filter(disp_s, left_s)
    wls_ms = (time.perf_counter() - t2) * 1000

    # Scale disparity back to full resolution if downscaled
    if scale != 1.0:
        disp = cv2.resize(disp_s, (rect.left.shape[1], rect.left.shape[0]),
                          interpolation=cv2.INTER_LINEAR) / scale
    else:
        disp = disp_s

    depth = estimator.to_depth(disp, calib)
    total_ms = rect_ms + sgbm_ms + wls_ms

    # --- Depth Aggregator setup ---
    _BUFFER_SIZE = 10
    # Hardcoded test waypoints in rectified-image pixel coordinates (u, v)
    _WAYPOINTS = [
        (rect.left.shape[1] // 2,       rect.left.shape[0] // 2),  # centre
        (rect.left.shape[1] // 4,       rect.left.shape[0] // 2),  # left-centre
        (3 * rect.left.shape[1] // 4,   rect.left.shape[0] // 2),  # right-centre
    ]
    aggregator = DepthAggregator(calib, buffer_size=_BUFFER_SIZE, min_confidence=0.5)

    # --- Stats ---
    depth_data = depth.data
    valid_mask = np.isfinite(depth_data) & (depth_data > 0)
    valid_pct  = valid_mask.mean() * 100
    median_depth = float(np.nanmedian(depth_data[valid_mask])) if valid_mask.any() else float("nan")

    h_in, w_in = left_s.shape[:2]
    print(f"preset:          {args.preset}")
    print(f"scale:           {scale}x  ({w_in}x{h_in})")
    print(f"valid pixels:    {valid_pct:.1f}%")
    print(f"median depth:    {median_depth:.2f} m")
    print(f"rectify:         {rect_ms:.0f} ms")
    print(f"sgbm:            {sgbm_ms:.0f} ms")
    if not args.no_wls:
        print(f"wls:             {wls_ms:.0f} ms")
        if wls_ms > 20:
            print(
                "WLS is bottlenecking FPS. Try --no-wls for real-time, or reduce image "
                "resolution in camera config."
            )
    print(f"total:           {total_ms:.0f} ms  ({1000/total_ms:.1f} FPS)")

    # --- Build display panels ---
    # Panel 1: rectified left (BGR)
    left_rect_bgr = rect.left

    # Panel 2: disparity (JET), mask invalid
    disp_vis = disp.copy()
    disp_vis[disp_vis <= 0] = 0
    d_max = float(np.percentile(disp_vis[disp_vis > 0], 95)) if (disp_vis > 0).any() else 1.0
    disp_color = colorize(disp_vis, 0, d_max)

    # Panel 3: depth (JET, clipped 0–5 m)
    depth_vis = depth_data.copy()
    depth_vis[~np.isfinite(depth_vis)] = 0
    depth_vis = np.clip(depth_vis, 0, 5)
    depth_color = colorize(depth_vis, 0, 5)

    # Resize all panels to the same height before hstack
    h = left_rect_bgr.shape[0]
    def _resize(img: np.ndarray) -> np.ndarray:
        if img.shape[0] != h:
            s = h / img.shape[0]
            img = cv2.resize(img, None, fx=s, fy=s)
        return img

    panel = np.hstack([
        _resize(left_rect_bgr),
        _resize(disp_color),
        _resize(depth_color),
    ])

    # Add text labels
    for i, label in enumerate(["left_rect", "disparity", "depth (0-5m)"]):
        x = i * (panel.shape[1] // 3) + 8
        cv2.putText(panel, label, (x, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # --- Show window ---
    win = "SGBM Tuner  [Q=quit  S=save]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_count = 0
    while True:
        # Accumulate the current disparity into the aggregator each iteration
        aggregator.push(disp)
        frame_count += 1

        display_panel = panel.copy()

        # Overlay waypoints on left-rect panel once buffer is full
        if aggregator.stats().buffer_fill == _BUFFER_SIZE:
            results = aggregator.query(_WAYPOINTS)
            for wp in results:
                u, v = wp.pixel
                color = (0, 255, 0)
                cv2.circle(display_panel, (u, v), 6, color, 2)
                if wp.xyz_m is not None:
                    label = f"{wp.xyz_m[2]:.2f}m {wp.confidence * 100:.0f}%"
                else:
                    label = f"N/A {wp.confidence:.2f}"
                cv2.putText(
                    display_panel, label, (u + 8, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )

        # Print AggregatorStats every 30 frames
        if frame_count % 30 == 0:
            s = aggregator.stats()
            print(
                f"[DepthAggregator] fill={s.buffer_fill}/{_BUFFER_SIZE}  "
                f"mean_conf={s.mean_confidence:.2f}  "
                f"query_lat={s.query_latency_ms:.1f} ms  "
                f"push_lat={s.push_latency_ms:.2f} ms"
            )

        cv2.imshow(win, display_panel)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or Esc
            break
        if key in (ord("s"), ord("S")):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("outputs/tune") / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "left_rect.png"),  left_rect_bgr)
            cv2.imwrite(str(out_dir / "disparity.png"),  disp_color)
            cv2.imwrite(str(out_dir / "depth.png"),      depth_color)
            cv2.imwrite(str(out_dir / "panel.png"),      display_panel)
            print(f"Saved panels to {out_dir}/")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())
